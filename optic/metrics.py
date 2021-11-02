import numpy as np
from numba import njit, prange


@njit
def signal_power(x):
    """
    Computes the average power of x

    :param x: input signal [np array]

    :return: P = mean(abs(x)**2)
    """
    return np.mean(x * np.conj(x)).real


@njit(parallel=True)
def hardDecision(rxSymb, constSymb, bitMap):
    """
    Euclidean distance based symbol decision

    :param rxSymb: received symbol sequence
    :param constSymb: constellation symbols [M x 1 array]
    :param bitMap: bit mapping [M x log2(M) array]

    :return: sequence of bits decided
    """

    M = len(constSymb)
    b = int(np.log2(M))

    decBits = np.zeros(len(rxSymb) * b)

    for i in prange(0, len(rxSymb)):
        indSymb = np.argmin(np.abs(rxSymb[i] - constSymb))
        decBits[i * b : i * b + b] = bitMap[indSymb, :]

    return decBits


def fastBERcalc(rx, tx, mod):
    """
    BER calculation

    :param rx: received symbol sequence
    :param tx: transmitted symbol sequence
    :param mod: commpy modem object

    :return BER: bit-error-rate
    :return SER: symbol-error-rate
    :return SNR: estimated SNR
    """
    # constellation parameters
    constSymb = mod.constellation
    M = mod.m
    Es = mod.Es

    # We want all the signal sequences to be disposed in columns:
    try:
        if rx.shape[1] > rx.shape[0]:
            rx = rx.T
    except IndexError:
        rx = rx.reshape(len(rx), 1)

    try:
        if tx.shape[1] > tx.shape[0]:
            tx = tx.T
    except IndexError:
        tx = tx.reshape(len(tx), 1)

    nModes = int(tx.shape[1])  # number of sinal modes
    SNR = np.zeros(nModes)
    BER = np.zeros(nModes)
    SER = np.zeros(nModes)
    b = int(np.log2(M))

    bitMap = mod.demodulate(constSymb, demod_type="hard")
    bitMap = bitMap.reshape(-1, b)

    # pre-processing
    for k in range(0, nModes):
        # symbol normalization
        rx[:, k] = rx[:, k] / np.sqrt(signal_power(rx[:, k]))
        tx[:, k] = tx[:, k] / np.sqrt(signal_power(tx[:, k]))

        # correct (possible) phase ambiguity
        rot = np.mean(tx[:, k] / rx[:, k])
        rx[:, k] = rot * rx[:, k]

        # estimate SNR of the received constellation
        SNR[k] = 10 * np.log10(
            signal_power(tx[:, k]) / signal_power(rx[:, k] - tx[:, k])
        )

    for k in range(0, nModes):
        # hard decision demodulation of the received symbols
        brx = hardDecision(np.sqrt(Es) * rx[:, k], constSymb, bitMap)
        btx = hardDecision(np.sqrt(Es) * tx[:, k], constSymb, bitMap)

        err = np.logical_xor(brx, btx)
        BER[k] = np.mean(err)
        SER[k] = np.mean(np.sum(err.reshape(-1, b), axis=1) > 0)

    return BER, SER, SNR


@njit(parallel=True)
def calcLLR(rxSymb, σ2, constSymb, bitMap):
    """
    LLR calculation (circular AGWN channel)

    :param rxSymb: received symbol sequence
    :param σ2: noise variance
    :param constSymb: constellation symbols [M x 1 array]
    :param bitMap: bit mapping [M x log2(M)]

    :return: sequence of calculated LLRs
    """
    M = len(constSymb)
    b = int(np.log2(M))

    LLRs = np.zeros(len(rxSymb) * b)

    for i in prange(0, len(rxSymb)):
        prob = np.exp((-np.abs(rxSymb[i] - constSymb) ** 2) / σ2)

        for indBit in range(0, b):
            p0 = np.sum(prob[bitMap[:, indBit] == 0])
            p1 = np.sum(prob[bitMap[:, indBit] == 1])

            LLRs[i * b + indBit] = np.log(p0) - np.log(p1)

    return LLRs


def monteCarloGMI(rx, tx, mod):
    """
    GMI calculation

    :param rx: received symbol sequence
    :param tx: transmitted symbol sequence
    :param mod: commpy modem object

    :return: estimated GMI
    """
    # constellation parameters
    constSymb = mod.constellation
    M = mod.m
    Es = mod.Es

    # We want all the signal sequences to be disposed in columns:
    try:
        if rx.shape[1] > rx.shape[0]:
            rx = rx.T
    except IndexError:
        rx = rx.reshape(len(rx), 1)

    try:
        if tx.shape[1] > tx.shape[0]:
            tx = tx.T
    except IndexError:
        tx = tx.reshape(len(tx), 1)

    nModes = int(tx.shape[1])  # number of sinal modes
    GMI = np.zeros(nModes)

    noiseVar = np.var(rx - tx, axis=0)

    bitMap = mod.demodulate(constSymb, demod_type="hard")
    bitMap = bitMap.reshape(-1, int(np.log2(M)))

    # symbol normalization
    for k in range(0, nModes):
        rx[:, k] = rx[:, k] / np.sqrt(signal_power(rx[:, k]))
        tx[:, k] = tx[:, k] / np.sqrt(signal_power(tx[:, k]))

        # correct (possible) phase ambiguity
        rot = np.mean(tx[:, k] / rx[:, k])
        rx[:, k] = rot * rx[:, k]

    for k in range(0, nModes):
        # set the noise variance
        σ2 = noiseVar[k]

        # hard decision demodulation of the transmitted symbols
        btx = hardDecision(np.sqrt(Es) * tx[:, k], constSymb, bitMap)
        # soft demodulation of the received symbols
        LLRs = calcLLR(rx[:, k], σ2, constSymb / np.sqrt(Es), bitMap)

        LLRs[LLRs == np.inf] = 500
        LLRs[LLRs == -np.inf] = -500

        # Compute bitwise MIs and their sum
        b = int(np.log2(M))

        MIperBitPosition = np.zeros(b)

        for n in range(0, b):
            MIperBitPosition[n] = 1 - np.mean(
                np.log2(1 + np.exp((2 * btx[n::b] - 1) * LLRs[n::b]))
            )

        GMI[k] = np.sum(MIperBitPosition)

    return GMI, MIperBitPosition
