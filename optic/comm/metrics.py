"""
================================================================================
Metrics for signal and performance characterization (:mod:`optic.comm.metrics`)
================================================================================

.. autosummary::
   :toctree: generated/

   bert                     -- Calculate BER and Q-factor for optical communication using On-Off Keying (OOK).
   fastBERcalc              -- Monte Carlo BER/SER/SNR calculation
   calcLLR                  -- LLR calculation assuming a circular AGWN channel model
   calcExtrLLR              -- Calculate the extrinsic bit LLRs assuming an auxiliary Gaussian channel model
   monteCarloGMI            -- Monte Carlo based generalized mutual information (GMI) estimation
   monteCarloMI             -- Monte Carlo based mutual information (MI) estimation
   calcMI                   -- Mutual information (MI) calculation for AWGN channels
   Qfunc                    -- Calculate function :math:`Q(x)`
   calcEVM                  -- Calculate error vector magnitude (EVM) metrics
   theoryBER                -- Theoretical (approx.) bit error probability for PAM/QAM/PSK in AWGN channel
   theoryMI                 -- Calculate mutual information for the DCMC AWGN channel
   theoryGMI                -- Calculate generalized mutual information for the DCMC AWGN channel
   calcLinOSNR              -- Calculate the OSNR evolution in a multi-span fiber transmission system
"""

"""Metrics for signal and performance characterization."""
import logging as logg
from collections import defaultdict

import numpy as np
import scipy.constants as const
from numba import njit, prange
from scipy.integrate import dblquad, quad
from scipy.special import erf

from optic.comm.modulation import demodulateGray, grayMapping, minEuclid
from optic.dsp.core import pnorm, signalPower
from optic.utils import dB2lin, dec2bitarray, llr2bitProb


def bert(Irx, bitsTx=None, seed=123):
    """
    Calculate Bit Error Rate (BER) and Q-factor for optical communication using On-Off Keying (OOK).

    Parameters
    ----------
    Irx : np.array
        Received signal intensity values.

    bitsTx : np.array, optional
        Transmitted bit sequence. If not provided, a random bit sequence is generated.

    seed : int, optional
        Random seed for bit sequence generation when bitsTx is not provided.

    Returns
    -------
    BER : float
        Bit Error Rate, a measure of the number of erroneous bits relative to the total bits.

    Q : float
        Q-factor, a measure of signal quality in the communication system.

    Notes
    -----
    This function calculates the BER and Q-factor for an optical communication system using On-Off Keying (OOK) modulation.
    The received signal intensity `Irx` and an optional transmitted bit sequence `bitsTx` are required. If `bitsTx` is not provided,
    a random bit sequence is generated using the specified `seed`.

    The following statistics are calculated for the received signal:
    - :math:`I_1`: The average value of the signal when the transmitted bit is 1.
    - :math:`I_0`: The average value of the signal when the transmitted bit is 0.
    - :math:`\\sigma_1`: The standard deviation of the signal when the transmitted bit is 1.
    - :math:`\\sigma_0`: The standard deviation of the signal when the transmitted bit is 0.

    The optimal decision threshold `Id` and the Q-factor are calculated based on the signal statistics.

    The function then applies the optimal decision rule to estimate the received bit sequence `bitsRx`. The Bit Error Rate (BER) is calculated
    by comparing `bitsRx` to `bitsTx`.

    References
    ----------
    [1] Agrawal, Govind P. Fiber-optic communication systems. John Wiley & Sons, 2012.

    """
    if bitsTx is None:
        np.random.seed(seed=seed)  # fixing the seed

        # generate reference pseudo-random bit sequence
        bitsTx = np.random.randint(2, size=Irx.size)

    # get received signal statistics
    I1 = np.mean(Irx[bitsTx == 1])  # average value of I1
    I0 = np.mean(Irx[bitsTx == 0])  # average value of I0

    std1 = np.std(Irx[bitsTx == 1])  # standard deviation std1 of I1
    std0 = np.std(Irx[bitsTx == 0])  # standard deviation std0 of I0

    Id = (std1 * I0 + std0 * I1) / (std1 + std0)  # optimal decision threshold
    Q = (I1 - I0) / (std1 + std0)  # Qfactor

    # apply the optimal decision rule
    bitsRx = np.empty(bitsTx.size)
    bitsRx[Irx > Id] = 1
    bitsRx[Irx <= Id] = 0

    # calculate the BER
    err = np.logical_xor(bitsRx, bitsTx)

    BER = np.mean(err)

    return BER, Q


def fastBERcalc(rx, tx, M, constType, px=None):
    """
    Monte Carlo BER/SER/SNR calculation.

    Parameters
    ----------
    rx : np.array
        Received symbol sequence.
    tx : np.array
        Transmitted symbol sequence.
    M : int
        Modulation order.
    constType : string
        Modulation type: 'qam', 'psk', 'pam' or 'ook'.
    px : (M, 1) np.array
        Prior symbol probabilities.

    Returns
    -------
    BER : np.array
        Bit-error-rate.
    SER : np.array
        Symbol-error-rate.
    SNR : np.array
        Estimated SNR from the received constellation.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.

    """
    if M != 2 and constType == "ook":
        logg.warning("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2

    # constellation parameters
    if px is None:
        px = []
    if len(px) == 0:  # if px is not defined
        px = 1 / M * np.ones(M)  # assume uniform distribution

    # constellation parameters
    constSymb = grayMapping(M, constType)
    Es = np.sum(np.abs(constSymb) ** 2 * px)

    # make copies of inputs
    tx = tx.copy()
    rx = rx.copy()

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

    # pre-processing
    for k in range(nModes):
        if constType in ["qam", "psk"]:
            # correct (possible) phase ambiguity
            rot = np.mean(tx[:, k] / rx[:, k])
            rx[:, k] = rot * rx[:, k]
        # symbol normalization
        rx[:, k] = pnorm(rx[:, k])
        tx[:, k] = pnorm(tx[:, k])

        # estimate SNR of the received constellation
        SNR[k] = 10 * np.log10(signalPower(tx[:, k]) / signalPower(rx[:, k] - tx[:, k]))
    for k in range(nModes):
        brx = demodulateGray(np.sqrt(Es) * rx[:, k], M, constType)
        btx = demodulateGray(np.sqrt(Es) * tx[:, k], M, constType)

        err = np.logical_xor(brx, btx)
        BER[k] = np.mean(err)
        SER[k] = np.mean(np.sum(err.reshape(-1, int(np.log2(M))), axis=1) > 0)
    return BER, SER, SNR


@njit(parallel=True, cache=True)
def calcLLR(rxSymb, σ2, constSymb, bitMap, px, maxLog=False):
    """
    LLR calculation assuming a circular AGWN channel model.

    Parameters
    ----------
    rxSymb : np.array
        Received symbol sequence.
    σ2 : scalar
        Noise variance.
    constSymb : (M, 1) np.array
        Constellation symbols.
    bitMap : (M, log2(M)) np.array
            Bit-to-symbol mapping.
    px : (M, 1) np.array
        Prior symbol probabilities.
    maxLog : bool, optional
        If True, use the Max-Log approximation for LLR calculation.

    Returns
    -------
    LLRs : np.array
        sequence of calculated LLRs.

    References
    ----------
    [1] A. Alvarado, T. Fehenberger, B. Chen, e F. M. J. Willems, “Achievable Information Rates for Fiber Optics: Applications and Computations”, Journal of Lightwave Technology, vol. 36, nº 2, p. 424–439, jan. 2018, doi: 10.1109/JLT.2017.2786351.

    """
    M = len(constSymb)
    b = int(np.log2(M))

    LLRs = np.zeros(len(rxSymb) * b)

    if maxLog:
        # Pre-compute log-priors (adding a small epsilon to avoid log(0) if any px is 0)
        logPx = np.log(px + 1e-12)

        for i in prange(len(rxSymb)):
            # Calculate the log-domain metric for all M symbols
            # metric = -|y - s|^2 / sigma^2 + ln(P(x))
            metric = -(np.abs(rxSymb[i] - constSymb) ** 2) / σ2 + logPx

            for indBit in range(b):
                # Find the maximum metric for bits 0 and 1 (Max-Log approximation)
                max0 = np.max(metric[bitMap[:, indBit] == 0])
                max1 = np.max(metric[bitMap[:, indBit] == 1])

                # LLR = max(metric_0) - max(metric_1)
                LLRs[i * b + indBit] = max0 - max1
    else:
        for i in prange(len(rxSymb)):
            # Calculate the probability of each symbol
            prob = np.exp((-np.abs(rxSymb[i] - constSymb) ** 2) / σ2) * px

            for indBit in range(b):
                p0 = np.sum(prob[bitMap[:, indBit] == 0])
                p1 = np.sum(prob[bitMap[:, indBit] == 1])

                LLRs[i * b + indBit] = np.log(p0) - np.log(p1)
    return LLRs


@njit(parallel=True, cache=True)
def calcExtrLLR(bitLLR, x, xMu, xNu, M, constSymb, bitMap, px=None, prec=np.float32):
    """
    Calculate the extrinsic LLRs assuming an auxiliary Gaussian channel model.

    Parameters
    -----------
    bitLLR : np.array of shape (q*numSymb,)
        received bit LLRs
    x : np.array of shape (numSymb,)
        received symbols
    xMu : np.array of shape (numSymb,)
        mean of the received symbols
    xNu : np.array of shape (numSymb,)
        variance of the received symbols
    M : int
        modulation order
    constSymb : np.array of shape (M,)
        constellation symbols
    bitMap : np.array of shape (M, q)
        bit mapping of the constellation symbols
    px : np.array of shape (M,), optional
        prior probabilities of the constellation symbols, if None, uniform distribution is used

    Returns
    -------
    LLRe : np.array of shape (q*numSymb,)
        extrinsic LLRs for each bit
    """
    numFloor = 1e-3  # minimum variance to avoid division by zero
    probFloor = 1e-4  # minimum probability to avoid log(0)

    q = int(np.log2(M))
    numSymb = len(x)

    if px is None:
        px = np.ones(M, dtype=prec) / M

    LLRe = np.zeros((numSymb, q), dtype=prec)
    constBits1 = bitMap.astype(prec)
    constBits0 = 1.0 - constBits1

    Pb1 = llr2bitProb(bitLLR.reshape((numSymb, q))).astype(prec)
    Pb1 = np.clip(Pb1, probFloor, 1 - probFloor)
    Pb0 = 1.0 - Pb1

    for indSymb in prange(numSymb):
        mu = xMu[indSymb]
        var = max(xNu[indSymb], numFloor)

        # Gaussian likelihood
        psi_gsi = np.empty(M, dtype=prec)
        for m in range(M):
            diff_real = x[indSymb].real - (mu * constSymb[m]).real
            diff_imag = x[indSymb].imag - (mu * constSymb[m]).imag
            diff_abs2 = diff_real**2 + diff_imag**2
            psi_gsi[m] = (1.0 / (np.pi * var)) * np.exp(-diff_abs2 / var) * px[m]

        # Compute symbol prior from bit probabilities
        priorProbSymb = np.ones(M, dtype=prec)
        probProd = np.empty((M, q), dtype=prec)
        for m in range(M):
            for b in range(q):
                probProd[m, b] = (
                    Pb1[indSymb, b] * constBits1[m, b]
                    + Pb0[indSymb, b] * constBits0[m, b]
                )
                priorProbSymb[m] *= probProd[m, b]

        # Compute extrinsic LLRs
        for b in range(q):
            Pe1 = 0.0
            Pe0 = 0.0
            for m in range(M):
                extrPrior = priorProbSymb[m] / probProd[m, b]
                if bitMap[m, b] == 1:
                    Pe1 += psi_gsi[m] * extrPrior
                else:
                    Pe0 += psi_gsi[m] * extrPrior

            Pe1 = min(max(Pe1, probFloor), 1 - probFloor)
            Pe0 = min(max(Pe0, probFloor), 1 - probFloor)
            LLRe[indSymb, b] = np.log(Pe0 / Pe1)

    return LLRe.flatten()


def monteCarloGMI(rx, tx, M, constType, px=None, bitMap=None):
    """
    Monte Carlo based generalized mutual information (GMI) estimation.

    Parameters
    ----------
    rx : np.array
        Received symbol sequence.
    tx : np.array
        Transmitted symbol sequence.
    M : int
        Modulation order.
    constType : string
        Modulation type: 'qam' or 'psk'
    px : (M, 1) np.array
        Prior symbol probabilities. The default is [].
    bitMap : (M, b) np.array
        Bit mapping matrix. The default is None.

    Returns
    -------
    GMI : np.array
        Generalized mutual information values.
    NGMI : np.array
        Normalized mutual information.

    References
    ----------
    [1] A. Alvarado, T. Fehenberger, B. Chen, e F. M. J. Willems, “Achievable Information Rates for Fiber Optics: Applications and Computations”, Journal of Lightwave Technology, vol. 36, nº 2, p. 424–439, jan. 2018, doi: 10.1109/JLT.2017.2786351.

    """
    if px is None:
        px = []
    # constellation parameters
    constSymb = grayMapping(M, constType)
    b = int(np.log2(M))

    # if bitMap is not provided, consider Gray mapping
    if bitMap is None:
        bitMap = dec2bitarray(np.arange(len(constSymb)), b)
        bitMap = bitMap.reshape(-1, b)

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
    NGMI = np.zeros(nModes)

    if len(px) == 0:  # if px is not defined, assume uniform distribution
        px = 1 / M * np.ones(constSymb.shape)

    # Normalize constellation
    Es = np.sum(np.abs(constSymb) ** 2 * px)
    constSymb = constSymb / np.sqrt(Es)

    # Calculate source entropy
    H = np.sum(-px * np.log2(px))

    # symbol normalization
    for k in range(nModes):
        if constType in ["qam", "psk"]:
            # correct (possible) phase ambiguity
            rot = np.mean(tx[:, k] / rx[:, k])
            rx[:, k] = rot * rx[:, k]
        # symbol normalization
        rx[:, k] = pnorm(rx[:, k])
        tx[:, k] = pnorm(tx[:, k])
    for k in range(nModes):
        # set the noise variance
        noiseVar = np.var(rx[:, k] - tx[:, k], axis=0)

        if constType in ["pam", "ook"]:
            noiseVar *= 2

        # demodulate transmitted symbol sequence
        btx = demodulateGray(np.sqrt(Es) * tx[:, k], M, constType)

        # soft demodulation of the received symbols
        LLRs = calcLLR(rx[:, k], noiseVar, constSymb, bitMap, px)

        # LLR clipping
        LLRs[LLRs == np.inf] = 500
        LLRs[LLRs == -np.inf] = -500

        # Compute bitwise MIs and their sum
        MIperBitPosition = np.zeros(b)

        for n in range(b):
            MIperBitPosition[n] = H / b - np.mean(
                np.log2(1 + np.exp((2 * btx[n::b] - 1) * LLRs[n::b]))
            )
        GMI[k] = np.sum(MIperBitPosition)
        NGMI[k] = GMI[k] / H
    return GMI, NGMI


def monteCarloMI(rx, tx, M, constType, px=None):
    """
    Monte Carlo based mutual information (MI) estimation.

    Parameters
    ----------
    rx : np.array
        Received symbol sequence.
    tx : np.array
        Transmitted symbol sequence.
    M : int
        Modulation order.
    constType : string
        Modulation type: 'qam' or 'psk'
    px : (M, 1) np.array
        p.m.f. of the constellation symbols. The default is [].

    Returns
    -------
    MI : np.array
        Estimated MI values.

    References
    ----------
    [1] A. Alvarado, T. Fehenberger, B. Chen, e F. M. J. Willems, “Achievable Information Rates for Fiber Optics: Applications and Computations”, Journal of Lightwave Technology, vol. 36, nº 2, p. 424–439, jan. 2018, doi: 10.1109/JLT.2017.2786351.

    """
    if px is None:
        px = []
    if len(px) == 0:  # if px is not defined
        px = 1 / M * np.ones(M)  # assume uniform distribution
    # constellation parameters
    constSymb = grayMapping(M, constType)
    Es = np.sum(np.abs(constSymb) ** 2 * px)
    constSymb = constSymb / np.sqrt(Es)

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

    for k in range(nModes):
        if constType in ["qam", "psk"]:
            # correct (possible) phase ambiguity
            rot = np.mean(tx[:, k] / rx[:, k])
            rx[:, k] = rot * rx[:, k]
        # symbol normalization
        rx[:, k] = pnorm(rx[:, k])
        tx[:, k] = pnorm(tx[:, k])

    # Estimate noise variance from the data
    noiseVar = np.var(rx - tx, axis=0)

    for k in range(nModes):
        σ2 = noiseVar[k]
        MI[k] = calcMI(rx[:, k], tx[:, k], σ2, constSymb, px)
    return MI


@njit(cache=True)
def calcMI(rx, tx, σ2, constSymb, pX):
    """
    Mutual information (MI) calculation for AWGN channels.

    Parameters
    ----------
    rx : np.array
        Received symbol sequence.
    tx : np.array
        Transmitted symbol sequence.
    σ2 : scalar
        Noise variance.
    constSymb : (M,) np.array
        Constellation symbols.
    pX : (M,) np.array
        Prob. mass function (p.m.f.) of the constellation symbols.

    Returns
    -------
    scalar
        Estimated mutual information.
    """
    N = len(rx)
    H_XgY = 0.0

    # Unconditional Entropy H(X)
    H_X = np.sum(-pX * np.log2(np.maximum(pX, 1e-50)))

    # Dimensionality Toggle: Adjust variance scale based on data type
    is_real = not (np.iscomplexobj(rx) or np.iscomplexobj(constSymb))
    var_scale = 2.0 * σ2 if is_real else σ2

    for k in range(N):
        # Index of the actually transmitted symbol
        indSymb = np.argmin(np.abs(tx[k] - constSymb))

        # log2 p(Y|X) for the transmitted symbol
        log2_pYgX = -(1.0 / var_scale) * np.abs(rx[k] - tx[k]) ** 2 * np.log2(np.exp(1))

        # p(Y) = sum_x( p(Y|X=x)*p(X=x) ) over all constellation points
        pXY = np.exp(-(1.0 / var_scale) * np.abs(rx[k] - constSymb) ** 2) * pX
        pY = np.sum(pXY)

        # H(X|Y) Accumulation: -( log2(p(Y|X)) + log2(p(X)) - log2(p(Y)) )
        # Using max() to prevent log2(0) underflow at very high SNRs
        H_XgY -= log2_pYgX + np.log2(pX[indSymb]) - np.log2(max(pY, 1e-300))

    H_XgY = H_XgY / N

    return H_X - H_XgY


def Qfunc(x):
    """
    Calculate function Q(x).

    Parameters
    ----------
    x : scalar
        function input.

    Returns
    -------
    scalar
        value of Q(x).

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.

    """
    return 0.5 - 0.5 * erf(x / np.sqrt(2))


def calcEVM(symb, M, constType, symbTx=None):
    """
    Calculate error vector magnitude (EVM) metrics.

    Parameters
    ----------
    symb : np.array
        Sequence of noisy symbols.
    M : int
        Constellation order.
    constType : string
        Modulation type: 'pam', 'qam' or 'psk'
    symbTx : np.array, optional
        Sequence of transmitted symbols (noiseless). The default is [].

    Returns
    -------
    EVM : np.array
        Error vector magnitude (EVM) per signal dimension.

    References
    ----------
    [1] R. A. Shafik, et al, “On the error vector magnitude as a performance metric and comparative analysis”, em 2006 International Conference on Emerging Technologies, 2006, p. 27–31. doi: 10.1109/ICET.2006.335992.

    [2] H. A. Mahmoud e H. Arslan, “Error vector magnitude to SNR conversion for nondata-aided receivers”, IEEE Transactions on Wireless Communications, vol. 8, nº 5, p. 2694–2704, 2009, doi: 10.1109/TWC.2009.080862.

    """
    if symbTx is None:
        symbTx = []
    symb = pnorm(symb)

    # We want all the signal sequences to be disposed in columns:
    try:
        if symb.shape[1] > symb.shape[0]:
            symb = symb.T
    except IndexError:
        symb = symb.reshape(len(symb), 1)
    if len(symbTx):  # if symbTx is provided
        try:
            if symbTx.shape[1] > symbTx.shape[0]:
                symbTx = symbTx.T
        except IndexError:
            symbTx = symbTx.reshape(len(symbTx), 1)
        symbTx = pnorm(symbTx)
    # constellation parameters
    constSymb = grayMapping(M, constType)
    constSymb = pnorm(constSymb)

    EVM = np.zeros(symb.shape[1])

    for ii in range(symb.shape[1]):
        if not len(symbTx):
            decided = np.zeros(symb.shape[0], dtype="complex")
            ind = minEuclid(symb[:, ii], constSymb)  # min. dist. decision
            decided = constSymb[ind]
        else:
            if constType in ["qam", "psk"]:
                # correct (possible) phase ambiguity
                rot = np.mean(symbTx[:, ii] / symb[:, ii])
                symb[:, ii] = rot * symb[:, ii]
            decided = symbTx[:, ii]

        EVM[ii] = np.mean(np.abs(symb[:, ii] - decided) ** 2) / np.mean(
            np.abs(decided) ** 2
        )
    return EVM


def theoryBER(M, EbN0, constType):
    """
    Theoretical (approx.) bit error probability for PAM/QAM/PSK in AWGN channel.

    Parameters
    ----------
    M : int
        Modulation order.
    EbN0 : scalar
        Signal-to-noise ratio (SNR) per bit in dB.
    constType : string
        Modulation type: 'pam', 'qam' or 'psk'

    Returns
    -------
    Pb : scalar
        Theoretical probability of bit error.

    Notes
    -----
    The values of error probability obtained with this function are good approximations for moderate to high SNR regime (see [1]).
    All cases assume Gray mapped constellations. For low SNR values and high constellation cardinalities (:math:`P_b`>1e-1), the results
    should underestimate the real error probability.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.

    """
    EbN0lin = 10 ** (EbN0 / 10)
    k = np.log2(M)

    if constType == "qam":
        L = np.sqrt(M)
        Pb = (
            2
            * (1 - 1 / L)
            / np.log2(L)
            * Qfunc(np.sqrt(3 * np.log2(L) / (L**2 - 1) * (2 * EbN0lin)))
        )
    elif constType == "psk":
        Ps = 2 * Qfunc(np.sqrt(2 * k * EbN0lin) * np.sin(np.pi / M))
        Pb = Ps / k
    elif constType == "pam":
        Ps = (2 * (M - 1) / M) * Qfunc(np.sqrt(6 * np.log2(M) / (M**2 - 1) * EbN0lin))
        Pb = Ps / k
    return Pb


@njit(cache=True, fastmath=True)
def condEntropy(yI, yQ, const, pX, ind, σ):
    """
    Calculate conditional entropy :math:`H(X|Y=y)` for the DCMC AWGN channel.

    Parameters
    ----------
    yI, yQ : float
        Real and imaginary parts of the received signal Y.
    const : array_like
        Constellation of complex-valued transmitted symbols.
    pX : array_like
        Probability of each transmitted symbol.
    ind : int
        Index of the transmitted symbol.
    σ : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    float
        conditional entropy :math:`H(X|Y=y)` for the DCMC AWGN channel.

    References
    ----------
    [1] A. Alvarado, T. Fehenberger, B. Chen, e F. M. J. Willems, “Achievable Information Rates for Fiber Optics: Applications and Computations”, Journal of Lightwave Technology, vol. 36, nº 2, p. 424–439, jan. 2018, doi: 10.1109/JLT.2017.2786351.
    """
    π = np.pi
    prob = 0
    M = len(const)

    for ii in prange(len(const)):
        xI = const[ii].real
        xQ = const[ii].imag
        prob += (
            1
            / (2 * π * σ**2)
            * np.exp(-((yI - xI) ** 2 + (yQ - xQ) ** 2) / (2 * σ**2))
            * pX[ii]
        )

    log2pY = np.log2(max([prob, 1e-50]))

    xI = const[ind].real
    xQ = const[ind].imag

    expTerm = (
        1 / (2 * π * σ**2) * np.exp(-((yI - xI) ** 2 + (yQ - xQ) ** 2) / (2 * σ**2))
    )

    int1 = expTerm * np.log2(max(expTerm, 1e-50))  # p(Y|X)*log2(p(Y|X))

    int2 = expTerm * np.log2(pX[ind])  # p(Y|X)*log2(p(X))

    int3 = expTerm * log2pY  # p(Y|X)*log2(p(Y))

    return (
        -(int1 + int2 - int3) * pX[ind]
    )  # integral of p(Y,X)*log2(p(Y|X)p(X)/p(Y)) = H(X|Y)


@njit(cache=True)
def minR(R, x):
    """
    Find the index of the minimum absolute difference between an array R and a value x.

    Parameters
    ----------
    R : array_like
        Array of values.
    x : float
        Value for comparison.

    Returns
    -------
    int
        Index of the minimum absolute difference.
    """
    return np.argmin(np.abs(R - np.abs(x)))


def theoryMI(M, constType, SNR, pX=None, symmetry=True, lim=np.inf, tol=1e-3):
    """
    Calculate mutual information for discrete input continuous output the memoryless AWGN channel (DCMC).

    Parameters
    ----------
    M : int
        Number of symbols in the constellation.
    constType : str
        Type of constellation ('qam', 'psk').
    SNR : float
        Signal-to-noise ratio in dB.
    pX : array_like, optional
        Probability of each transmitted symbol (default is None).
    symmetry : bool, optional
        Flag to exploit rotational symmetry of the constellation (default is True).
    lim : int, optional
        Limit for numerical integration (default is np.inf).
    tol : float, optional
        Tolerance for numerical integration error (default is 1e-3).

    Returns
    -------
    float
        Mutual information for the given parameters.

    References
    ----------
    [1] A. Alvarado, T. Fehenberger, B. Chen, e F. M. J. Willems, “Achievable Information Rates for Fiber Optics: Applications and Computations”, Journal of Lightwave Technology, vol. 36, nº 2, p. 424–439, jan. 2018, doi: 10.1109/JLT.2017.2786351.
    """
    if pX is None:
        pX = 1 / M * np.ones(M)

    constSymb = grayMapping(M, constType)  # get constellation
    Es = np.sum(np.abs(constSymb) ** 2 * pX)  # calculate average symbol energy
    constSymb = constSymb / np.sqrt(Es)  # normalize average symbol energy

    if constType in ["pam", "ook"]:
        σ = np.sqrt(1 / dB2lin(SNR))  # noise std per dimension
    else:
        σ = np.sqrt((1 / 2) * 1 / dB2lin(SNR))  # noise std per dimension

    MI = -np.sum(pX * np.log2(pX))

    if symmetry:
        # Exploit rotational symmetry of the constellation to speed up calculations
        groups = defaultdict(list)
        for i, x in enumerate(constSymb):
            key = round(abs(x) / 1e-14) * 1e-14
            groups[key].append(i)

        for key, indices in groups.items():
            # Choose a representative index for the group
            rep_index = indices[0]
            # Number of elements sharing the same |constSymb|
            count = len(indices)
            MI -= (
                dblquad(
                    condEntropy,
                    -lim,
                    lim,
                    -lim,
                    lim,
                    args=(constSymb, pX, rep_index, σ),
                    epsabs=tol,
                )[0]
                * count
            )

    else:
        for ind in range(len(constSymb)):
            MI -= dblquad(
                condEntropy,
                -lim,
                lim,
                -lim,
                lim,
                args=(constSymb, pX, ind, σ),
                epsabs=tol,
            )[0]

    return MI


def theoryGMI(M, constType, SNR, bitMap=None, pX=None, lim=None, tol=1e-3):
    """
    Calculate generalized mutual information (GMI) for discrete input continuous output the memoryless AWGN channel (DCMC).

    Parameters
    ----------
    M : int
        Number of symbols in the constellation.
    constType : str
        Type of constellation ('qam', 'psk').
    SNR : float
        Signal-to-noise ratio in dB.
    bitMap : array_like
        Bit mapping of the constellation symbols (shape: M x log2(M)).
    pX : array_like, optional
        Probability of each transmitted symbol (default is None).
    lim : float, optional
        Limit for numerical integration (default is np.inf).
    tol : float, optional
        Tolerance for numerical integration error (default is 1e-3).

    Returns
    -------
    float
        Theoretical GMI for the given parameters.

    References
    ----------
    [1] A. Alvarado, T. Fehenberger, B. Chen, e F. M. J. Willems, “Achievable Information Rates for Fiber Optics: Applications and Computations”, Journal of Lightwave Technology, vol. 36, nº 2, p. 424–439, jan. 2018, doi: 10.1109/JLT.2017.2786351.
    """
    # if pX is not provided, assume uniform distribution
    if pX is None:
        pX = np.ones(M) / M

    constSymb = grayMapping(M, constType)
    b = int(np.log2(M))

    # if bitMap is not provided, consider Gray mapping
    if bitMap is None:
        bitMap = dec2bitarray(np.arange(len(constSymb)), b)
        bitMap = bitMap.reshape(-1, b)

    Es = np.sum(np.abs(constSymb) ** 2 * pX)
    constSymb = constSymb / np.sqrt(Es)

    if constType in ["pam", "ook"]:
        σ = np.sqrt(1 / dB2lin(SNR))
    else:
        σ = np.sqrt((1 / 2) * 1 / dB2lin(SNR))

    if lim is None:
        lim = np.max(np.abs(constSymb)) + 6 * σ

    GMI = 0.0

    if constType in ["pam", "ook"]:
        is1D = True
    else:
        is1D = False

    # Iterate over each parallel bit channel
    for i in range(b):
        #  Marginal bit probabilities P(B_i = 0) and P(B_i = 1)
        pB0 = np.sum(pX[bitMap[:, i] == 0])
        pB1 = np.sum(pX[bitMap[:, i] == 1])

        # Bit entropy H(B_i)
        Hb = 0.0
        if pB0 > 0:
            Hb -= pB0 * np.log2(max(pB0, 1e-50))
        if pB1 > 0:
            Hb -= pB1 * np.log2(max(pB1, 1e-50))

        if is1D:
            # Conditional entropy H(B_i | Y) via numerical integration (1D case)
            H_b_y = quad(
                condEntropyBit1D,
                -lim,
                lim,
                args=(constSymb.real, bitMap, pX, i, σ),
                epsabs=tol,
            )[0]
        else:
            # Conditional entropy H(B_i | Y) via numerical integration
            H_b_y = dblquad(
                condEntropyBit2D,
                -lim,
                lim,
                -lim,
                lim,
                args=(constSymb, bitMap, pX, i, σ),
                epsabs=tol,
            )[0]

        # Accumulate I(B_i; Y)
        GMI += Hb - H_b_y

    return GMI


@njit(cache=True)
def condEntropyBit2D(yI, yQ, const, bitMap, pX, i, σ):
    """
    Calculate conditional bit entropy :math:`H(B_i|Y=y)` for the DCMC AWGN channel

    Parameters
    ----------
    yI, yQ : float
        Real and imaginary parts of the received signal Y.
    const : array_like
        Constellation of complex-valued transmitted symbols.
    bitMap : array_like
        Bit mapping matrix of shape (M, q).
    pX : array_like
        Probability of each transmitted symbol.
    i : int
        Index of the bit channel being evaluated.
    σ : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    float
        conditional bit entropy :math:`H(B_i|Y=y)` for the given bit channel.
    """
    π = np.pi
    M = len(const)

    # Accumulate joint probabilities for B_i = 0 and B_i = 1
    p_y_b0 = 0.0
    p_y_b1 = 0.0
    pB0 = 0.0
    pB1 = 0.0

    for ii in prange(M):
        xI = const[ii].real
        xQ = const[ii].imag

        # Joint probability p(y, x) = p(y|x) * p(x)
        p_y_x = (
            1
            / (2 * π * σ**2)
            * np.exp(-((yI - xI) ** 2 + (yQ - xQ) ** 2) / (2 * σ**2))
            * pX[ii]
        )

        if bitMap[ii, i] == 0:
            p_y_b0 += p_y_x
            pB0 += pX[ii]
        else:
            p_y_b1 += p_y_x
            pB1 += pX[ii]

    # Total probability p(y)
    p_y = p_y_b0 + p_y_b1
    log2pY = np.log2(max([p_y, 1e-50]))

    entropy = 0.0

    # Compute integrand for B_i = 0
    if pB0 > 0:
        expTermB0 = p_y_b0 / pB0  # p(Y | B_i=0)

        int1_b0 = expTermB0 * np.log2(
            max([expTermB0, 1e-50])
        )  # p(Y|B_0)*log2(p(Y|B_0))
        int2_b0 = expTermB0 * np.log2(pB0)  # p(Y|B_0)*log2(p(B_0))
        int3_b0 = expTermB0 * log2pY  # p(Y|B_0)*log2(p(Y))

        entropy += -(int1_b0 + int2_b0 - int3_b0) * pB0

    # Compute integrand for B_i = 1
    if pB1 > 0:
        expTermB1 = p_y_b1 / pB1  # p(Y | B_i=1)

        int1_b1 = expTermB1 * np.log2(
            max([expTermB1, 1e-50])
        )  # p(Y|B_1)*log2(p(Y|B_1))
        int2_b1 = expTermB1 * np.log2(pB1)  # p(Y|B_1)*log2(p(B_1))
        int3_b1 = expTermB1 * log2pY  # p(Y|B_1)*log2(p(Y))

        entropy += -(int1_b1 + int2_b1 - int3_b1) * pB1

    return entropy  # sum over bits of p(Y, B_i) * log2(p(Y|B_i)p(B_i)/p(Y))


@njit(cache=True)
def condEntropyBit1D(y, const, bitMap, pX, i, σ):
    """
    Calculate conditional bit entropy :math:`H(B_i|Y=y)` for the DCMC AWGN channel (1D case).

    Parameters
    ----------
    y : float
        Received signal Y (real).
    const : array_like
        Constellation of transmitted symbols.
    bitMap : array_like
        Bit mapping matrix of shape (M, q).
    pX : array_like
        Probability of each transmitted symbol.
    i : int
        Index of the bit channel being evaluated.
    σ : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    float
        conditional bit entropy :math:`H(B_i|Y=y)` for the given bit channel (1D case).
    """
    π = np.pi
    M = len(const)

    p_y_b0 = 0.0
    p_y_b1 = 0.0
    pB0 = 0.0
    pB1 = 0.0

    # Normalization factor for the Gaussian PDF
    normFactor = 1.0 / np.sqrt(2 * π * σ**2)

    for ii in prange(M):
        x = const[ii]

        # PDF Gaussiana 1D
        p_y_x = normFactor * np.exp(-((y - x) ** 2) / (2 * σ**2)) * pX[ii]

        if bitMap[ii, i] == 0:
            p_y_b0 += p_y_x
            pB0 += pX[ii]
        else:
            p_y_b1 += p_y_x
            pB1 += pX[ii]

    p_y = p_y_b0 + p_y_b1
    log2pY = np.log2(max(p_y, 1e-50))
    entropy = 0.0

    if pB0 > 0:
        expTermB0 = p_y_b0 / pB0
        int1_b0 = expTermB0 * np.log2(max(expTermB0, 1e-50))
        int2_b0 = expTermB0 * np.log2(pB0)
        int3_b0 = expTermB0 * log2pY
        entropy += -(int1_b0 + int2_b0 - int3_b0) * pB0

    if pB1 > 0:
        expTermB1 = p_y_b1 / pB1
        int1_b1 = expTermB1 * np.log2(max(expTermB1, 1e-50))
        int2_b1 = expTermB1 * np.log2(pB1)
        int3_b1 = expTermB1 * log2pY
        entropy += -(int1_b1 + int2_b1 - int3_b1) * pB1

    return entropy


def GN_Model_NyquistWDM(Rs, Nch, Δf, α, γ, Ls, Ns, Ptx_dBm, D, Bref, Fc):
    # Reference: [1] P. Poggiolini, "The GN Model of Non-Linear Propagation in
    # Uncompensated Coherent Optical Systems," in Journal of Lightwave
    # Technology, vol. 30, no. 24, pp. 3857-3879, Dec.15, 2012,
    # doi: 10.1109/JLT.2012.2217729.

    # Channel parameters:
    λ = const.c / Fc * 1e-3  # wavelength km
    # λ = λ * 1e-3  # wavelength km
    c = const.c / 1.5 * 1e-3  # speed of light km/s
    α = α / (10 * np.log10(np.exp(1)))  # fiber attenuation coefficient
    Leff = (1 - np.exp(-2 * α * Ls)) / (2 * α)  # fiber effective length
    Leffa = 1 / (2 * α)  # the asymptotic effective length [km]
    Ptx = 10 ** (Ptx_dBm / 10) * 1e-3  # input power per channel dBm to W
    β2 = -D * λ**2 / (2 * np.pi * c)

    # Calculate NLIN variance using the GN-Model (see reference):
    # [1], Eq.(15)
    var_NLI = (
        (8 / 27)
        * (γ**2)
        * Leff**2
        * (Ptx / Rs) ** 3
        * (
            np.arcsinh(
                (np.pi**2) / 2 * np.abs(β2) * Leffa * Nch ** (2 * Rs / Δf) * Rs**2
            )
        )
        / (np.pi * np.abs(β2) * Leffa)
        * Bref
    )

    epsilon = (3 / 10) * np.log(
        1
        + 6
        / Ls
        * Leffa
        / np.arcsinh(
            (np.pi**2 / 2) * np.abs(β2) * Leffa * (Nch**2) ** (2 * Rs / Δf) * Rs**2
        )
    )
    # epsilon = 0.1
    # epsilon = 0;
    var_NLI = 2 * (Ns ** (1 + epsilon)) * var_NLI  # FIXME: is there a
    # multiplication by two here? without the multiplication by two, var_NLI
    # does not match the split-step simulation.

    return var_NLI


def ASE_NyquistWDM(α, Ls, Ns, NF, Bref, Fc):
    # ASE noise power calculation:
    G = α * Ls  # amplifier gain (dB)

    NF_lin = 10 ** (NF / 10)  # amplifier noise figure (linear)
    G_lin = 10 ** (G / 10)  # amplifier gain (linear)
    nsp = (G_lin * NF_lin - 1) / (2 * (G_lin - 1))

    # ASE noise power calculation:
    # Ref. Eq.(54) of R. -J. Essiambre,et al, "Capacity Limits of Optical Fiber
    # Networks," in Journal of Lightwave Technology, vol. 28, no. 4,
    # pp. 662-701, Feb.15, 2010, doi: 10.1109/JLT.2009.2039464.
    N_ase = Ns * (G_lin - 1) * nsp * const.h * Fc
    return 2 * N_ase * Bref


def GNmodel_OSNR(Rs, Nch, Δf, Ptx, paramCh=None, Bref=12.5e9):
    if paramCh is None:
        paramCh = []
    # check input parameters
    Ltotal = getattr(paramCh, "Ltotal", 800)
    Ls = getattr(paramCh, "Lspan", 50)
    α = getattr(paramCh, "alpha", 0.2)
    D = getattr(paramCh, "D", 16)
    γ = getattr(paramCh, "gamma", 1.3)
    Fc = getattr(paramCh, "Fc", 193.1e12)
    NF = getattr(paramCh, "NF", 4.5)

    Ns = Ltotal // Ls

    OSNR = np.zeros(len(Ptx))
    P_nli = np.zeros(len(Ptx))
    P_ase = np.zeros(len(Ptx))

    for k, Ptx_dBm in enumerate(Ptx):
        P_nli[k] = GN_Model_NyquistWDM(Rs, Nch, Δf, α, γ, Ls, Ns, Ptx_dBm, D, Bref, Fc)
        P_ase[k] = ASE_NyquistWDM(α, Ls, Ns, NF, Bref, Fc)
        OSNR[k] = 10 ** (Ptx_dBm / 10) * 1e-3 / (P_nli[k] + P_ase[k])
    return OSNR, P_nli, P_ase


def calcLinOSNR(Ns, Pin, α, Ls, OSNRin, NF=4.5, Fc=193.1e12, Bref=12.5e9):
    """
    Calculate the OSNR evolution in a multi-span fiber transmission system.

    Parameters
    ----------
    Ns : int
        Number of spans of fiber + EDFA.
    Pin : scalar
        Fiber launch power.
    α : scalar
        Fiber attenuation coefficient in dB/km.
    Ls : scalar
        Length of fiber spans in km.
    OSNRin : scalar
        OSNR at the input of the first span.
    NF : scalar, optional
        Noise figure of the EDFA amplifiers. The default is 4.5.
    Fc : scalar, optional
        Optical central frequency. The default is 193.1e12.
    Bref : scalar, optional
        Reference bandwidth for OSNR measurement. The default is 12.5e9.

    Returns
    -------
    OSNR : np.array
        OSNR values in dB at the output of each fiber span.

    References
    ----------
    [1] J. G. Proakis; M. Salehi, Communication Systems Engineering, 2nd Edition. Pearson, 2002.

    [2] R. -J. Essiambre, et al, "Capacity Limits of Optical Fiber Networks,"  Journal of Lightwave Technology, vol. 28, no. 4, p. 662-701, 2010, doi: 10.1109/JLT.2009.2039464.

    [3] R. Schober, P. Bayvel, e F. D. Pasquale, “Analytical model for the calculation of the optical signal-to-noise ratio (SNR) of WDM EDFA chains”, Optical and Quantum Electronics, vol. 31, no 3, p. 237–241. 1999, doi: 10.1023/A:1006948826091.

    """
    G = α * Ls
    NF_lin = 10 ** (NF / 10)  # amplifier noise figure (linear)
    G_lin = 10 ** (G / 10)  # amplifier gain (linear)
    nsp = (G_lin * NF_lin - 1) / (2 * (G_lin - 1))

    # ASE noise power calculation:
    # Ref. Eq.(54) of R. -J. Essiambre,et al, "Capacity Limits of Optical Fiber
    # Networks," in Journal of Lightwave Technology, vol. 28, no. 4,
    # pp. 662-701, Feb.15, 2010, doi: 10.1109/JLT.2009.2039464.
    N_ase = (G_lin - 1) * nsp * const.h * Fc
    P_ase = (2 * N_ase * Bref) / 1e-3  # in mW

    P_ase_dBm = 10 * np.log10(P_ase)  # ASE power in dBm generated per EDFA

    Pn_in_edfa = (Pin - OSNRin) - α * Ls  # ASE power sent to the 1st EDFA
    OSNR = np.zeros(Ns + 1)
    OSNR[0] = OSNRin

    # Calculate OSNR at the output of each span
    for spanN in range(1, Ns + 1):
        Pn_out_edfa = 10 * np.log10(
            10 ** ((Pn_in_edfa + G) / 10) + 10 ** (P_ase_dBm / 10)
        )  # Total ASE power at the output of the spanN-th EDFA
        OSNR[spanN] = Pin - Pn_out_edfa  # current OSNR
        Pn_in_edfa = Pn_out_edfa - α * Ls  # ASE power sent to the next EDFA

    return OSNR
