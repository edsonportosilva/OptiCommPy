import numpy as np
from numba import njit
from scipy.special import erf


@njit
def signal_power(x):
    """
    Computes the average power of x

    :param x: input signal [np array]

    :return: P = mean(abs(x)**2)
    """
    return np.mean(x * np.conj(x)).real


@njit
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

    for i in range(0, len(rxSymb)):
        indSymb = np.argmin(np.abs(rxSymb[i] - constSymb))
        decBits[i * b: i * b + b] = bitMap[indSymb, :]

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


@njit
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

    for i in range(0, len(rxSymb)):
        prob = np.exp((-np.abs(rxSymb[i] - constSymb) ** 2) / σ2)

        for indBit in range(0, b):
            p0 = np.sum(prob[bitMap[:, indBit] == 0])
            p1 = np.sum(prob[bitMap[:, indBit] == 1])

            LLRs[i * b + indBit] = np.log(p0) - np.log(p1)

    return LLRs


def monteCarloGMI(rx, tx, mod):
    """
    GMI estimation

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

        # LLR clipping
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


def monteCarloMI(rx, tx, mod, px=[]):
    """
    MI estimation

    :param rx: received symbol sequence
    :param tx: transmitted symbol sequence
    :param mod: commpy modem object
    :param px: probability mass function of constellation symbols

    :return: estimated MI
    """

    # constellation parameters
    M = mod.m
    Es = mod.Es
    constSymb = mod.constellation / np.sqrt(Es)

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

    nModes = int(rx.shape[1])  # number of sinal modes
    MI = np.zeros(nModes)

    for k in range(0, nModes):
        rx[:, k] = rx[:, k] / np.sqrt(signal_power(rx[:, k]))
        tx[:, k] = tx[:, k] / np.sqrt(signal_power(tx[:, k]))

    # Estimate noise variance from the data
    noiseVar = np.var(rx - tx, axis=0)

    if len(px) == 0:  # if px is not defined
        px = 1 / M * np.ones(M)  # assume uniform distribution

    for k in range(0, nModes):
        σ2 = noiseVar[k]
        MI[k] = calcMI(rx[:, k], tx[:, k], σ2, constSymb, px)

    return MI


@njit
def calcMI(rx, tx, σ2, constSymb, pX):
    """
    Mutual information (MI) calculation (circular AGWN channel)

    :param rx: received symbol sequence
    :param tx: transmitted symbol sequence
    :param σ2: noise variance
    :param constSymb: constellation symbols [M x 1 array]
    :param pX: prob. mass function (pmf) of constSymb [M x 1 array]

    :return: estimated MI
    """
    N = len(rx)
    H_XgY = np.zeros(1, dtype=np.float64)
    H_X = np.sum(-pX * np.log2(pX))

    for k in range(0, N):
        indSymb = np.argmin(np.abs(tx[k] - constSymb))

        pYgX = np.exp(-(1 / σ2) * np.abs(rx[k] - tx[k]) ** 2)  # p(Y|X)
        pXY = (
            np.exp(-(1 / σ2) * np.abs(rx[k] - constSymb) ** 2) * pX
        )  # p(Y,X) = p(Y|X)*p(X)

        # p(X|Y) = p(Y|X)*p(X)/p(Y), where p(Y) = sum(q(Y|X)*p(X)) in X

        pY = np.sum(pXY)

        H_XgY -= np.log2((pYgX * pX[indSymb]) / pY)

    H_XgY = H_XgY / N

    return H_X - H_XgY


def Qfunc(x):
    return 0.5 - 0.5 * erf(x / np.sqrt(2))


def theoryBER(M, EbN0, constType):
    """
    Theoretical bit error probability for QAM/PSK equiprobable constellations
    in AWGN channel (approximated calculation)

    :param M: order of the modulation
    :param EbN0: signal-to-noise ratio per bit [dB]
    :param constType: 'qam','psk'

    :return: estimated probability of error (Pb)
    """
    EbN0lin = 10 ** (EbN0 / 10)
    k = np.log2(M)

    if constType == "qam":
        L = np.sqrt(M)
        Pb = (
            2
            * (1 - 1 / L)
            / np.log2(L)
            * Qfunc(np.sqrt(3 * np.log2(L) / (L ** 2 - 1) * (2 * EbN0lin)))
        )

    elif constType == "psk":
        Ps = 2 * Qfunc(np.sqrt(2 * k * EbN0lin) * np.sin(np.pi / M))
        Pb = Ps / k

    return Pb
