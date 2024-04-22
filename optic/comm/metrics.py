"""
================================================================================
Metrics for signal and performance characterization (:mod:`optic.comm.metrics`)
================================================================================

.. autosummary::
   :toctree: generated/

   bert                     -- Calculate BER and Q-factor for optical communication using On-Off Keying (OOK).
   fastBERcalc              -- Monte Carlo BER/SER/SNR calculation
   calcLLR                  -- LLR calculation (circular AGWN channel)
   monteCarloGMI            -- Monte Carlo based generalized mutual information (GMI) estimation
   monteCarloMI             -- Monte Carlo based mutual information (MI) estimation
   Qfunc                    -- Calculate function Q(x)
   calcEVM                  -- Calculate error vector magnitude (EVM) metrics
   theoryBER                -- Theoretical (approx.) bit error probability for PAM/QAM/PSK in AWGN channel
   theoryMI                 -- Calculate mutual information for the DCMC AWGN channel
   calcLinOSNR              -- Calculate the OSNR evolution in a multi-span fiber transmission system
"""


"""Metrics for signal and performance characterization."""
import logging as logg

import numpy as np
from numba import njit, prange
from scipy.special import erf
from scipy.integrate import dblquad
import scipy.constants as const

from optic.utils import dB2lin
from optic.dsp.core import pnorm, signal_power
from optic.comm.modulation import grayMapping, demodulateGray, minEuclid


def bert(Irx, bitsTx=None, seed=123):
    """
    Calculate Bit Error Rate (BER) and Q-factor for optical communication using On-Off Keying (OOK).

    Parameters
    ----------
    Irx : numpy.ndarray
        Received signal intensity values.

    bitsTx : numpy.ndarray, optional
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
    - `I1`: The average value of the signal when the transmitted bit is 1.
    - `I0`: The average value of the signal when the transmitted bit is 0.
    - `std1`: The standard deviation of the signal when the transmitted bit is 1.
    - `std0`: The standard deviation of the signal when the transmitted bit is 0.

    The optimal decision threshold `Id` and the Q-factor are calculated based on the signal statistics.

    The function then applies the optimal decision rule to estimate the received bit sequence `bitsRx`. The Bit Error Rate (BER) is calculated
    by comparing `bitsRx` to `bitsTx`.

    Example
    -------
    >>> Irx = np.array([0.1, 0.8, 0.2, 0.7, 0.3])
    >>> bitsTx = np.array([0, 1, 0, 1, 0])
    >>> BER, Q = ook_BERT(Irx, bitsTx)
    >>> print(f"BER: {BER}, Q-factor: {Q}")

    References
    ----------
    Agrawal, Govind P. Fiber-optic communication systems. John Wiley & Sons, 2012.

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


def fastBERcalc(rx, tx, M, constType):
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

    Returns
    -------
    BER : np.array
        Bit-error-rate.
    SER : np.array
        Symbol-error-rate.
    SNR : np.array
        Estimated SNR from the received constellation.

    """
    if M != 2 and constType == "ook":
        logg.warn("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2
    # constellation parameters
    constSymb = grayMapping(M, constType)
    Es = np.mean(np.abs(constSymb) ** 2)

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
        SNR[k] = 10 * np.log10(
            signal_power(tx[:, k]) / signal_power(rx[:, k] - tx[:, k])
        )
    for k in range(nModes):
        brx = demodulateGray(np.sqrt(Es) * rx[:, k], M, constType)
        btx = demodulateGray(np.sqrt(Es) * tx[:, k], M, constType)

        err = np.logical_xor(brx, btx)
        BER[k] = np.mean(err)
        SER[k] = np.mean(np.sum(err.reshape(-1, int(np.log2(M))), axis=1) > 0)
    return BER, SER, SNR


@njit(parallel=True)
def calcLLR(rxSymb, σ2, constSymb, bitMap, px):
    """
    LLR calculation (circular AGWN channel).

    Parameters
    ----------
    rxSymb : np.array
        Received symbol sequence.
    σ2 : scalar
        Noise variance.
    constSymb : (M, 1) np.array
        Constellation symbols.
    px : (M, 1) np.array
        Prior symbol probabilities.
    bitMap : (M, log2(M)) np.array
        Bit-to-symbol mapping.

    Returns
    -------
    LLRs : np.array
        sequence of calculated LLRs.

    """
    M = len(constSymb)
    b = int(np.log2(M))

    LLRs = np.zeros(len(rxSymb) * b)

    for i in prange(len(rxSymb)):
        prob = np.exp((-np.abs(rxSymb[i] - constSymb) ** 2) / σ2) * px

        for indBit in range(b):
            p0 = np.sum(prob[bitMap[:, indBit] == 0])
            p1 = np.sum(prob[bitMap[:, indBit] == 1])

            LLRs[i * b + indBit] = np.log(p0) - np.log(p1)
    return LLRs


def monteCarloGMI(rx, tx, M, constType, px=None):
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

    Returns
    -------
    GMI : np.array
        Generalized mutual information values.
    NGMI : np.array
        Normalized mutual information.

    """
    if px is None:
        px = []
    # constellation parameters
    constSymb = grayMapping(M, constType)

    # get bit mapping
    b = int(np.log2(M))
    bitMap = demodulateGray(constSymb, M, constType)
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
        σ2 = np.var(rx[:, k] - tx[:, k], axis=0)

        # demodulate transmitted symbol sequence
        btx = demodulateGray(np.sqrt(Es) * tx[:, k], M, constType)

        # soft demodulation of the received symbols
        LLRs = calcLLR(rx[:, k], σ2, constSymb, bitMap, px)

        # LLR clipping
        LLRs[LLRs == np.inf] = 500
        LLRs[LLRs == -np.inf] = -500

        # Compute bitwise MIs and their sum
        b = int(np.log2(M))

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
    pX : (M, 1) np.array
        p.m.f. of the constellation symbols. The default is [].

    Returns
    -------
    MI : np.array
        Estimated MI values.

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
        MI[k] = calcMI(rx[:, k], tx[:, k], σ2, constSymb, px)[0]
    return MI


@njit
def calcMI(rx, tx, σ2, constSymb, pX):
    """
    Mutual information (MI) calculation (circular AGWN channel).

    Parameters
    ----------
    rx : np.array
        Received symbol sequence.
    tx : np.array
        Transmitted symbol sequence.
    σ2 : scalar
        Noise variance.
    constSymb : (M, 1) np.array
        Constellation symbols.
    pX : (M, 1) np.array
        prob. mass function (p.m.f.) of the constellation symbols.

    Returns
    -------
    scalar
        Estimated mutual information.

    """
    N = len(rx)
    H_XgY = np.zeros(1, dtype=np.float64)
    H_X = np.sum(-pX * np.log2(pX))

    for k in range(N):
        indSymb = np.argmin(np.abs(tx[k] - constSymb))

        log2_pYgX = (
            -(1 / σ2) * np.abs(rx[k] - tx[k]) ** 2 * np.log2(np.exp(1))
        )  # log2 p(Y|X)
        # print('pYgX:', pYgX)
        pXY = (
            np.exp(-(1 / σ2) * np.abs(rx[k] - constSymb) ** 2) * pX
        )  # p(Y,X) = p(Y|X)*p(X)
        # print('pXY:', pXY)
        # p(X|Y) = p(Y|X)*p(X)/p(Y), where p(Y) = sum(q(Y|X)*p(X)) in X

        pY = np.sum(pXY)

        # print('pY:', pY)
        H_XgY -= log2_pYgX + np.log2(pX[indSymb]) - np.log2(pY)
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
    constType : TYPE
        DESCRIPTION.
    symbTx : np.array, optional
        Sequence of transmitted symbols (noiseless). The default is [].

    Returns
    -------
    EVM : np.array
        Error vector magnitude (EVM) per signal dimension.

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


@njit
def condEntropy(yI, yQ, const, pX, ind, σ):
    """
    Calculate conditional entropy H(X|Y=y)

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
        conditional entropy H(X|Y=y).
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

    int1 = expTerm * np.log2(max([expTerm, 1e-50]))  # p(Y|X)*log2(p(Y|X))

    int2 = expTerm * np.log2(pX[ind])  # p(Y|X)*log2(p(X))

    int3 = expTerm * log2pY  # p(Y|X)*log2(p(Y))

    return (
        -(int1 + int2 - int3) * pX[ind]
    )  # integral of p(Y,X)*log2(p(Y|X)p(X)/p(Y)) = H(X|Y)


@njit
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


def theoryMI(M, constType, SNR, pX=None, symetry=True, lim=np.inf, tol=1e-3):
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
    symetry : bool, optional
        Flag to exploit rotational symmetry of the constellation (default is True).
    lim : int, optional
        Limit for numerical integration (default is np.inf).
    tol : float, optional
        Tolerance for numerical integration error (default is 1e-3).

    Returns
    -------
    float
        Mutual information for the given parameters.
    """
    constSymb = grayMapping(M, constType)  # get constellation
    Es = signal_power(constSymb)  # calculate average symbol energy
    constSymb = constSymb / np.sqrt(Es)  # normalize average symbol energy

    σ = np.sqrt((1 / 2) * 1 / dB2lin(SNR))  # noise variance per dimension

    if pX is None:
        pX = 1 / M * np.ones(M)

    MI = -np.sum(pX * np.log2(pX))

    if symetry:
        # Exploit rotational symmetry of the constellation to speed up calculations
        constR = np.abs(constSymb)
        R = np.unique(constR)
        MI_R = np.zeros(R.shape)
        symbCount = np.zeros(R.shape)

        for ind in range(len(R)):
            symbCount[ind] = np.sum(constR == R[ind])

        for ind in range(len(constSymb)):
            indR = minR(R, constSymb[ind])

            if MI_R[indR] == 0:
                MI_R[indR] = dblquad(
                    condEntropy,
                    -lim,
                    lim,
                    -lim,
                    lim,
                    args=(constSymb, pX, ind, σ),
                    epsabs=tol,
                )[0]
        MI -= np.sum(MI_R * symbCount)

    else:
        MI = 0
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
            (np.pi**2 / 2)
            * np.abs(β2)
            * Leffa
            * (Nch**2) ** (2 * Rs / Δf)
            * Rs**2
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
