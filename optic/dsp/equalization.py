"""
===============================================================
DSP algorithms for equalization (:mod:`optic.dsp.equalization`)
===============================================================

.. autosummary::
   :toctree: generated/

   edc                 -- Electronic chromatic dispersion compensation (EDC).
   mimoAdaptEqualizer  -- General :math:`N \\times N` MIMO adaptive equalizer with several adaptive filtering algorithms available.
   manakovDBP          -- Manakov SSF digital backpropagation (DBP) algorithm.
   dfe                 -- Decision feedback adaptive equalizer (DFE) for SISO receivers.
   ffe                 -- Decision-directed feedforward adaptive equalizer (FFE) for SISO receivers.
   volterra            -- Decision-directed Volterra equalizer implementation up to 3rd order for SISO receivers.
"""

"""Functions for adaptive and static equalization."""
import logging as logg

import numpy as np
import scipy.constants as const
from numba import njit
from numpy.fft import fft, fftfreq, ifft
from tqdm.notebook import tqdm

from optic.comm.modulation import grayMapping
from optic.dsp.core import blockwiseFFTConv, pnorm, anorm
from optic.models.channels import convergenceCondition, nlinPhaseRot

# try:
#     from optic.dsp.coreGPU import blockwiseFFTConv
# except ImportError:
#     from optic.dsp.core import blockwiseFFTConv


def edc(Ei, param):
    """
    Electronic chromatic dispersion compensation (EDC).

    Parameters
    ----------
    Ei : np.array
        Input optical field.
    param : optic.utils.parameters object
        Parameters of the optical channel.

        - param.L: total fiber length [km][default: 50 km]
        - param.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
        - param.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]
        - param.Fs: sampling frequency [Hz] [default: []]
        - param.Rs: symbol rate [baud] [default: 32e9]
        - param.NfilterCoeffs: number of filter coefficients [default: []]
        - param.Nfft: FFT size [default: []]

    Returns
    -------
    np.array
        CD compensated signal.

    References
    ----------

    [1] S. J. Savory, “Digital coherent optical receivers: Algorithms and subsystems”, IEEE Journal on Selected Topics in Quantum Electronics, vol. 16, nº 5, p. 1164–1179, set. 2010, doi: 10.1109/JSTQE.2010.2044751.

    [2] K. Kikuchi, “Fundamentals of Coherent Optical Fiber Communications”, J. Lightwave Technol., JLT, vol. 34, nº 1, p. 157–179, jan. 2016.

    """
    try:
        Fs = param.Fs
    except AttributeError:
        logg.error("Simulation sampling frequency (Fs) not provided.")

    try:
        nModes = Ei.shape[1]
        input1D = False
    except IndexError:
        nModes = 1
        Ei = Ei.reshape(Ei.size, nModes)
        input1D = True

    # check input parameters
    L = getattr(param, "L", 50)
    D = getattr(param, "D", 16)
    Fc = getattr(param, "Fc", 193.1e12)
    Rs = getattr(param, "Rs", 32e9)
    NfilterCoeffs = getattr(param, "NfilterCoeffs", None)
    Nfft = getattr(param, "Nfft", None)

    # c  = 299792458   # speed of light [m/s](vacuum)
    c_kms = const.c / 1e3
    λ = c_kms / Fc
    β2 = -(D * λ**2) / (2 * np.pi * c_kms)

    # If number of filter coefficients is not provided, calculate it
    # based on the dispersion parameter, the fiber length and the symbol rate
    if NfilterCoeffs is None:
        NfilterCoeffs = int(2 * np.ceil(6.67 * np.abs(β2) * L * Rs**2 * (Fs / Rs)))

    # If FFT size is not provided, calculate it based on the number of filter coefficients
    if Nfft is None:
        Nfft = 2 ** int(np.ceil(np.log2(NfilterCoeffs)))

    ω = 2 * np.pi * Fs * fftfreq(NfilterCoeffs)  # angular frequency vector

    H = np.exp(-1j * (β2 / 2) * (ω**2) * L)  # frequency response of the CD filter

    logg.info(f"Running CD compensation...")
    logg.info(f"CD filter length: {NfilterCoeffs} taps, FFT size: {Nfft}")

    Eo = np.zeros(Ei.shape, dtype=Ei.dtype)

    # Apply CD compensation to each mode
    for indMode in range(nModes):
        Eo[:, indMode] = blockwiseFFTConv(
            Ei[:, indMode], H, NFFT=Nfft, freqDomainFilter=True
        )

    if input1D:
        # If the input was 1D, return a 1D array
        Eo = Eo.flatten()

    return Eo


def mimoAdaptEqualizer(x, param=None, dx=None):
    """
    General :math:`N \times N` MIMO adaptive equalizer with several adaptive filtering algorithms available.

    Parameters
    ----------
    x : np.array
        Input array.
    dx : np.array, optional
        Syncronized exact symbol sequence corresponding to the received input array x.
    param : optic.utils.parameters object, optional
        Parameter object containing the following attributes:

        - numIter : int, number of pre-convergence iterations [default: 1]
        - nTaps : int, number of filter taps [default: 15]
        - mu : float or list of floats, step size parameter(s) [default: [1e-3]]
        - lambdaRLS : float, RLS forgetting factor [default: 0.99]
        - SpS : int, samples per symbol [default: 2]
        - H : np.array, coefficient matrix [default: []]
        - L : int or list of ints, length of the output of the training section [default: []]
        - Hiter : list, history of coefficient matrices [default: []]
        - storeCoeff : bool, flag indicating whether to store coefficient matrices [default: False]
        - runWL: bool, flag indicating whether to run the equalizer in the widely-linear mode [default: False]
        - alg : str or list of strs, specifying the equalizer algorithm(s) [default: ['nlms']]
        - constType : str, constellation type [default: 'qam']
        - M : int, modulation order [default: 4]
        - prgsBar : bool, flag indicating whether to display progress bar [default: True]
        - prec: data type, precision of the computations [default: np.complex64]

    Returns
    -------
    yEq : np.array
        Equalized output array.
    H : np.array
        Coefficient matrix.
    errSq : np.array
        Squared absolute error array.
    Hiter : list
        History of coefficient matrices.

    Notes
    -----
    Algorithms available: 'cma', 'rde', 'nlms', 'dd-lms', 'da-rde', 'rls', 'dd-rls', 'static'.

    References
    ----------
    [1] P. S. R. Diniz, Adaptive Filtering: Algorithms and Practical Implementation. Springer US, 2012.

    [2] S. J. Savory, “Digital coherent optical receivers: Algorithms and subsystems”, IEEE Journal on Selected Topics in Quantum Electronics, vol. 16, nº 5, p. 1164–1179, set. 2010, doi: 10.1109/JSTQE.2010.2044751.

    [3] K. Kikuchi, “Fundamentals of Coherent Optical Fiber Communications”, J. Lightwave Technol., JLT, vol. 34, nº 1, p. 157–179, jan. 2016.

    [4] E. P. Da Silva e D. Zibar, “Widely Linear Equalization for IQ Imbalance and Skew Compensation in Optical Coherent Receivers”, Journal of Lightwave Technology, vol. 34, nº 15, p. 3577–3586, ago. 2016, doi: 10.1109/JLT.2016.2577716.
    """
    if dx is None:
        dx = []
    if param is None:
        param = []

    # check input parameters
    numIter = getattr(param, "numIter", 1)
    nTaps = getattr(param, "nTaps", 15)
    mu = getattr(param, "mu", [1e-3])
    lambdaRLS = getattr(param, "lambdaRLS", 0.99)
    SpS = getattr(param, "SpS", 2)
    H = getattr(param, "H", [])
    H_ = getattr(param, "H_", [])
    L = getattr(param, "L", [])
    Hiter = getattr(param, "Hiter", [])
    storeCoeff = getattr(param, "storeCoeff", False)
    runWL = getattr(param, "runWL", False)
    alg = getattr(param, "alg", ["nlms"])
    constType = getattr(param, "constType", "qam")
    M = getattr(param, "M", 4)
    shapingFactor = getattr(param, "shapingFactor", 0)
    prgsBar = getattr(param, "prgsBar", True)
    returnResults = getattr(param, "returnResults", False)
    prec = getattr(param, "prec", np.complex64)

    # We want all the signal sequences to be disposed in columns:
    if not len(dx):
        dx = x.copy()
    try:
        if x.shape[1] > x.shape[0]:
            x = x.T
        input1D = False
    except IndexError:
        x = x.reshape(len(x), 1)
        input1D = True
    try:
        if dx.shape[1] > dx.shape[0]:
            dx = dx.T
    except IndexError:
        dx = dx.reshape(len(dx), 1)
    nModes = int(x.shape[1])  # number of sinal modes (order of the MIMO equalizer)

    dx = dx.astype(prec)
    x = x.astype(prec)
    mu = np.array(mu).astype(np.float32)
    lambdaRLS = np.array([lambdaRLS]).astype(prec)[0]

    Lpad = int(np.floor(nTaps / 2))
    zeroPad = np.zeros((Lpad, nModes), dtype=prec)
    x = np.concatenate(
        (zeroPad, x, zeroPad)
    )  # pad start and end of the signal with zeros

    # Defining training parameters:
    constSymb = grayMapping(M, constType).astype(prec)  # constellation

    # Calculate MB distribution
    px = np.exp(-shapingFactor * np.abs(constSymb) ** 2)
    px = px / np.sum(px)

    # normalize reference constellation accouting for the probability mass function
    constSymb /= np.sqrt(np.sum(np.abs(constSymb) ** 2 * px))

    totalNumSymb = int(np.fix((len(x) - nTaps) / SpS + 1))

    if not L:  # if L is not defined
        L = [
            totalNumSymb
        ]  # Length of the output (1 sample/symbol) of the training section
    if not H:  # if H is not defined
        H = np.zeros((nModes**2, nTaps), dtype=prec)

        for initH in range(nModes):  # initialize filters' taps
            H[initH + initH * nModes, int(np.floor(H.shape[1] / 2))] = (
                1 + 1j * 0  # Central spike initialization
            )
    if not H_:  # if H_ is not defined
        H_ = np.zeros((nModes**2, nTaps), dtype=prec)

    logg.info(f"Running adaptive equalizer...")
    # Equalizer training:
    if type(alg) == list:
        yEq = np.zeros((totalNumSymb, x.shape[1]), dtype=prec)
        errSq = np.zeros((totalNumSymb, x.shape[1]), dtype=prec).T

        nStart = 0
        for indstage, runAlg in enumerate(alg):
            logg.info(f"{runAlg} - training stage #%d", indstage)

            nEnd = nStart + L[indstage]

            if indstage == 0:
                for indIter in tqdm(range(numIter), disable=not (prgsBar)):
                    logg.info(
                        f"{runAlg} pre-convergence training iteration #%d", indIter
                    )
                    yEq[nStart:nEnd, :], H, H_, errSq[:, nStart:nEnd], Hiter = (
                        coreAdaptEq(
                            x[nStart * SpS : (nEnd + 2 * Lpad) * SpS, :],
                            dx[nStart:nEnd, :],
                            SpS,
                            H,
                            H_,
                            L[indstage],
                            mu[indstage],
                            lambdaRLS,
                            nTaps,
                            storeCoeff,
                            runWL,
                            runAlg,
                            constSymb,
                            prec,
                        )
                    )
                    logg.info(
                        f"{runAlg} MSE = %.6f.", np.nanmean(errSq[:, nStart:nEnd]).real
                    )
            else:
                yEq[nStart:nEnd, :], H, H_, errSq[:, nStart:nEnd], Hiter = coreAdaptEq(
                    x[nStart * SpS : (nEnd + 2 * Lpad) * SpS, :],
                    dx[nStart:nEnd, :],
                    SpS,
                    H,
                    H_,
                    L[indstage],
                    mu[indstage],
                    lambdaRLS,
                    nTaps,
                    storeCoeff,
                    runWL,
                    runAlg,
                    constSymb,
                    prec,
                )
                logg.info(
                    f"{runAlg} MSE = %.6f.", np.nanmean(errSq[:, nStart:nEnd]).real
                )
            nStart = nEnd
    else:
        for indIter in tqdm(range(numIter), disable=not (prgsBar)):
            logg.info(f"{alg}training iteration #%d", indIter)
            yEq, H, errSq, Hiter = coreAdaptEq(
                x, dx, SpS, H, L, mu, nTaps, storeCoeff, alg, constSymb, prec
            )
            logg.info(f"{alg}MSE = %.6f.", np.nanmean(errSq).real)

    if input1D:
        # If the input was 1D, return a 1D array
        yEq = yEq.flatten()

    if returnResults:
        if runWL:
            return yEq, H, H_, errSq, Hiter
        else:
            return yEq, H, errSq, Hiter
    else:
        return yEq


@njit(fastmath=True)
def coreAdaptEq(
    x, dx, SpS, H, H_, L, mu, lambdaRLS, nTaps, storeCoeff, runWL, alg, constSymb, prec
):
    """
    Adaptive equalizer core processing function

    Parameters
    ----------
    x : np.array
        Input array.
    dx : np.array
        Exact constellation radius array.
    SpS : int
        Samples per symbol.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    L : int
        Length of the output.
    mu : float
        Step size parameter.
    lambdaRLS : float
        RLS forgetting factor.
    nTaps : int
        Number of taps.
    storeCoeff : bool
        Flag indicating whether to store coefficient matrices.
    runWL : bool
        Run widely-linear mode
    alg : str
        Equalizer algorithm.
    constSymb : np.array
        Constellation symbols.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    yEq : np.array
        Equalized output array.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    errSq : np.array
        Squared absolute error array.
    Hiter : np.array
        History of coefficient matrices.

    """
    # allocate variables
    nModes = int(x.shape[1])
    indTaps = np.arange(0, nTaps)
    indMode = np.arange(0, nModes)

    errSq = np.empty((nModes, L))
    x = x.astype(prec)
    H = H.astype(prec)
    H_ = H_.astype(prec)

    yEq = x[:L].copy()
    yEq[:] = np.nan
    outEq = np.array([[0 + 1j * 0]]).repeat(nModes).reshape(nModes, 1).astype(prec)

    if storeCoeff:
        Hiter = (
            np.array([[0 + 1j * 0]])
            .repeat((nModes**2) * nTaps * L)
            .reshape(nModes**2, nTaps, L)
            .astype(prec)
        )
    else:
        Hiter = (
            np.array([[0 + 1j * 0]])
            .repeat((nModes**2) * nTaps)
            .reshape(nModes**2, nTaps, 1)
            .astype(prec)
        )
    if alg == "rls":
        Sd = np.eye(nTaps, dtype=prec)
        a = Sd.copy()
        for _ in range(nTaps - 1):
            Sd = np.concatenate((Sd, a))
    # Radii cma, rde
    Rcma = (
        np.mean(np.abs(constSymb) ** 4) / np.mean(np.abs(constSymb) ** 2)
    ) * np.ones((1, nModes)).astype(prec)
    Rrde = np.unique(np.abs(constSymb)).astype(prec)

    for ind in range(L):
        outEq[:] = 0

        indIn = indTaps + ind * SpS  # simplify indexing and improve speed

        # pass signal sequence through the equalizer:
        for N in range(nModes):
            inEq = x[indIn, N : N + 1]  # slice input coming from the Nth mode
            outEq += (
                H[indMode + N * nModes, :] @ inEq
            )  # add contribution from the Nth mode to the equalizer's output
            if runWL:
                outEq += H_[indMode + N * nModes, :] @ inEq.conjugate()
                # add augmented contribution from the Nth mode to the equalizer's output

        yEq[ind, :] = outEq.T

        # update equalizer taps acording to the specified
        # algorithm and save squared error:
        if alg == "nlms":
            H, H_, errSq[:, ind] = nlmsUp(
                x[indIn, :], dx[ind, :], outEq, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "cma":
            H, H_, errSq[:, ind] = cmaUp(
                x[indIn, :], Rcma, outEq, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "dd-lms":
            H, H_, errSq[:, ind] = ddlmsUp(
                x[indIn, :], constSymb, outEq, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "rde":
            H, H_, errSq[:, ind] = rdeUp(
                x[indIn, :], Rrde, outEq, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "da-rde":
            H, H_, errSq[:, ind] = dardeUp(
                x[indIn, :], dx[ind, :], outEq, mu, H, H_, nModes, runWL, prec
            )
        elif alg == "rls":
            H, Sd, errSq[:, ind] = rlsUp(
                x[indIn, :], dx[ind, :], outEq, lambdaRLS, H, Sd, nModes, prec
            )
        elif alg == "dd-rls":
            H, Sd, errSq[:, ind] = ddrlsUp(
                x[indIn, :], constSymb, outEq, lambdaRLS, H, Sd, nModes, prec
            )
        elif alg == "static":
            errSq[:, ind] = errSq[:, ind - 1]
        else:
            raise ValueError(
                "Equalization algorithm not specified (or incorrectly specified)."
            )
        if storeCoeff:
            Hiter[:, :, ind] = H
        else:
            Hiter[:, :, 0] = H

    return yEq, H, H_, errSq, Hiter


@njit(fastmath=True)
def nlmsUp(x, dx, outEq, mu, H, H_, nModes, runWL, prec):
    """
    Coefficient update with the NLMS algorithm.

    Parameters
    ----------
    x : np.array
        Input array.
    dx : np.array
        Desired output array.
    outEq : np.array
        Equalized output array.
    mu : float
        Step size for the update.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec: data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H : np.array
        Updated augmented coefficient matrix.
    err_sq : np.array
        Squared absolute error.

    """
    indMode = np.arange(0, nModes)
    err = dx - outEq.T  # calculate output error for the NLMS algorithm

    errDiag = np.diag(err[0]).astype(prec)  # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing and improve speed
        inAdapt = x[:, N].T / np.linalg.norm(x[:, N]) ** 2  # NLMS normalization
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * errDiag @ inAdaptPar.conjugate()
        )  # gradient descent update
        if runWL:
            H_[indUpdTaps, :] += mu * errDiag @ inAdaptPar  # gradient descent update
    return H, H_, np.abs(err) ** 2


@njit(fastmath=True)
def rlsUp(x, dx, outEq, λ, H, Sd, nModes, prec):
    """
    Coefficient update with the RLS algorithm.

    Parameters
    ----------
    x : np.array
        Input array.
    dx : np.array
        Desired output array.
    outEq : np.array
        Equalized output array.
    λ : float
        Forgetting factor.
    H : np.array
        Coefficient matrix.
    Sd : np.array
        Inverse correlation matrix.
    nModes : int
        Number of modes.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    Sd : np.array
        Updated inverse correlation matrix.
    err_sq : np.array
        Squared absolute error.

    """
    nTaps = H.shape[1]
    indMode = np.arange(0, nModes)
    indTaps = np.arange(0, nTaps)
    Sd = Sd.astype(prec)

    err = dx - outEq.T  # calculate output error for the NLMS algorithm

    errDiag = np.diag(err[0]).astype(prec)  # define diagonal matrix from error array

    # update equalizer taps
    Sd = Sd.astype(prec)

    for N in range(nModes):
        indUpdModes = indMode + N * nModes
        indUpdTaps = indTaps + N * nTaps

        Sd_ = Sd[indUpdTaps, :]

        inAdapt = x[:, N].conjugate().reshape(-1, 1).astype(prec)  # input samples
        inAdaptPar = ((inAdapt.T).repeat(nModes).reshape(len(x), -1).T).astype(
            prec
        )  # expand input to parallelize tap adaptation

        A = (Sd_ @ inAdapt).astype(prec)
        B = (inAdapt.conjugate().astype(prec).T @ Sd_).astype(prec)
        C = (inAdapt.conjugate().astype(prec).T @ A).astype(prec)
        num = (A @ B).astype(prec)

        Sd_ = ((1 / λ) * (Sd_ - num / (λ + C))).astype(prec)

        Y = (Sd_ @ inAdaptPar.T).astype(prec).T

        H[indUpdModes, :] += errDiag @ Y

        Sd[indUpdTaps, :] = Sd_
    return H, Sd, np.abs(err) ** 2


@njit(fastmath=True)
def ddlmsUp(x, constSymb, outEq, mu, H, H_, nModes, runWL, prec):
    """
    Coefficient update with the DD-LMS algorithm.

    Parameters
    ----------
    x : np.array
        Input array.
    constSymb : np.array
        Array of constellation symbols.
    outEq : np.array
        Equalized output array.
    mu : float
        Step size for the update.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    err_sq : np.array
        Squared absolute error.

    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    decided = np.zeros(outEq.shape, dtype=prec)
    x = x.astype(prec)

    for k in range(nModes):
        indSymb = np.argmin(np.abs(outEq[0, k] - constSymb))
        decided[0, k] = constSymb[indSymb]
    err = decided - outEq  # calculate output error for the DDLMS algorithm

    err = err.astype(prec)
    errDiag = np.diag(err[0])  # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * errDiag @ inAdaptPar.conjugate()
        )  # gradient descent update
        if runWL:
            H_[indUpdTaps, :] += mu * errDiag @ inAdaptPar  # gradient descent update
    return H, H_, np.abs(err) ** 2


@njit(fastmath=True)
def ddrlsUp(x, constSymb, outEq, λ, H, Sd, nModes, prec):
    """
    Coefficient update with the DD-RLS algorithm.

    Parameters
    ----------
    x : np.array
        Input array.
    constSymb : np.array
        Array of constellation symbols.
    outEq : np.array
        Equalized output array.
    λ : float
        Forgetting factor.
    H : np.array
        Coefficient matrix.
    Sd : np.array
        Inverse correlation matrix.
    nModes : int
        Number of modes.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    Sd : np.array
        Updated inverse correlation matrix.
    err_sq : np.array
        Squared absolute error.

    """
    nTaps = H.shape[1]
    indMode = np.arange(0, nModes)
    indTaps = np.arange(0, nTaps)

    outEq = outEq.T
    decided = np.zeros(outEq.shape, dtype=prec)

    for k in range(nModes):
        indSymb = np.argmin(np.abs(outEq[0, k] - constSymb))
        decided[0, k] = constSymb[indSymb]
    err = decided - outEq  # calculate output error for the DDLMS algorithm

    errDiag = np.diag(err[0])  # define diagonal matrix from error array

    # update equalizer taps
    Sd = Sd.astype(prec)

    for N in range(nModes):
        indUpdModes = indMode + N * nModes
        indUpdTaps = indTaps + N * nTaps

        Sd_ = Sd[indUpdTaps, :]

        inAdapt = x[:, N].conjugate().reshape(-1, 1).astype(prec)  # input samples
        inAdaptPar = ((inAdapt.T).repeat(nModes).reshape(len(x), -1).T).astype(
            prec
        )  # expand input to parallelize tap adaptation

        A = (Sd_ @ inAdapt).astype(prec)
        B = (inAdapt.conjugate().astype(prec).T @ Sd_).astype(prec)
        C = (inAdapt.conjugate().astype(prec).T @ A).astype(prec)
        num = (A @ B).astype(prec)

        Sd_ = ((1 / λ) * (Sd_ - num / (λ + C))).astype(prec)

        Y = (Sd_ @ inAdaptPar.T).astype(prec).T

        H[indUpdModes, :] += errDiag @ Y

        Sd[indUpdTaps, :] = Sd_
    return H, Sd, np.abs(err) ** 2


@njit(fastmath=True)
def cmaUp(x, R, outEq, mu, H, H_, nModes, runWL, prec):
    """
    Coefficient update with the CMA algorithm.

    Parameters
    ----------
    x : np.array
        Input array.
    R : np.array
        Correlation array.
    outEq : np.array
        Equalized output array.
    mu : float
        Step size parameter.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    err_sq : np.array
        Squared absolute error.

    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    err = R - np.abs(outEq) ** 2  # calculate output error for the CMA algorithm
    err = err.astype(prec)

    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])  # define diagonal matrix

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * prodErrOut @ inAdaptPar.conjugate()
        )  # gradient descent update
        if runWL:
            H_[indUpdTaps, :] += mu * prodErrOut @ inAdaptPar  # gradient descent update
    return H, H_, np.abs(err) ** 2


@njit(fastmath=True)
def rdeUp(x, R, outEq, mu, H, H_, nModes, runWL, prec):
    """
    Coefficient update with the RDE algorithm.

    Parameters
    ----------
    x : np.array
        Input array.
    R : np.array
        Constellation radius array.
    outEq : np.array
        Equalized output array.
    mu : float
        Step size parameter.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    err_sq : np.array
        Squared absolute error.

    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    decidedR = np.zeros(outEq.shape, dtype=prec)

    # find closest constellation radius
    for k in range(nModes):
        indR = np.argmin(np.abs(R - np.abs(outEq[0, k])))
        decidedR[0, k] = R[indR]
    err = (
        decidedR**2 - np.abs(outEq) ** 2
    )  # calculate output error for the RDE algorithm

    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])  # define diagonal matrix

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * prodErrOut @ inAdaptPar.conjugate()
        )  # gradient descent update
        if runWL:
            H_[indUpdTaps, :] += mu * prodErrOut @ inAdaptPar  # gradient descent update

    return H, H_, np.abs(err) ** 2


@njit(fastmath=True)
def dardeUp(x, dx, outEq, mu, H, H_, nModes, runWL, prec):
    """
    Coefficient update with the data-aided RDE algorithm.

    Parameters
    ----------
    x : np.array
        Input array.
    dx : np.array
        Exact constellation radius array.
    outEq : np.array
        Equalized output array.
    mu : float
        Step size parameter.
    H : np.array
        Coefficient matrix.
    H_ : np.array
        Augmented coefficient matrix.
    nModes : int
        Number of modes.
    runWL: bool
        Run widely-linear mode.
    prec : data type
        Precision of the computations [default: np.complex64].

    Returns
    -------
    H : np.array
        Updated coefficient matrix.
    H_ : np.array
        Updated augmented coefficient matrix.
    err_sq : np.array
        Squared absolute error.

    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    decidedR = np.zeros(outEq.shape, dtype=prec)

    # find exact constellation radius
    for k in range(nModes):
        decidedR[0, k] = np.abs(dx[k])
    err = (
        decidedR**2 - np.abs(outEq) ** 2
    )  # calculate output error for the RDE algorithm

    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])  # define diagonal matrix

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * prodErrOut @ inAdaptPar.conjugate()
        )  # gradient descent update
        if runWL:
            H_[indUpdTaps, :] += mu * prodErrOut @ inAdaptPar  # gradient descent update
    return H, H_, np.abs(err) ** 2


def manakovDBP(Ei, param):
    """
    Run the Manakov SSF digital backpropagation (symmetric, dual-pol.).

    Parameters
    ----------
    Ei : np.array
        Input optical signal field.
    param : optic.utils.parameters object
        Physical/simulation parameters of the optical channel.

        - param.Ltotal: total fiber length [km][default: 400 km]
        - param.Lspan: span length [km][default: 80 km]
        - param.hz: step-size for the split-step Fourier method [km][default: 0.5 km]
        - param.alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
        - param.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
        - param.gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
        - param.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]
        - param.Fs: simulation sampling frequency [samples/second][default: None]
        - param.prec: numerical precision [default: np.complex128]
        - param.amp: 'edfa', 'ideal', or 'None. [default:'edfa']
        - param.maxIter: max number of iter. in the trap. integration [default: 10]
        - param.tol: convergence tol. of the trap. integration.[default: 1e-5]
        - param.nlprMethod: adap step-size based on nonl. phase rot. [default: True]
        - param.maxNlinPhaseRot: max nonl. phase rot. tolerance [rad][default: 2e-2]
        - param.prgsBar: display progress bar? bolean variable [default:True]
        - param.saveSpanN: specify the span indexes to be outputted [default:[]]
        - param.returnParameters: bool, return channel parameters [default: False]


    Returns
    -------
    Ech : np.array
        Optical signal after nonlinear backward propagation.
    param : parameter object  (struct)
        Object with physical/simulation parameters used in the split-step alg.

    References
    ----------
    [1] E. Ip e J. M. Kahn, “Compensation of dispersion and nonlinear impairments using digital backpropagation”, Journal of Lightwave Technology, vol. 26, nº 20, p. 3416–3425, 2008, doi: 10.1109/JLT.2008.927791.

    [2] E. Ip, “Nonlinear compensation using backpropagation for polarization-multiplexed transmission”, Journal of Lightwave Technology, vol. 28, nº 6, p. 939–951, mar. 2010, doi: 10.1109/JLT.2010.2040135.

    """
    try:
        Fs = param.Fs
    except AttributeError:
        logg.error("Simulation sampling frequency (Fs) not provided.")

    # check input parameters
    param.Ltotal = getattr(param, "Ltotal", 400)
    param.Lspan = getattr(param, "Lspan", 80)
    param.hz = getattr(param, "hz", 0.5)
    param.alpha = getattr(param, "alpha", 0.2)
    param.D = getattr(param, "D", 16)
    param.gamma = getattr(param, "gamma", 1.3)
    param.Fc = getattr(param, "Fc", 193.1e12)
    param.prec = getattr(param, "prec", np.complex128)
    param.amp = getattr(param, "amp", "edfa")
    param.maxIter = getattr(param, "maxIter", 10)
    param.tol = getattr(param, "tol", 1e-5)
    param.nlprMethod = getattr(param, "nlprMethod", True)
    param.maxNlinPhaseRot = getattr(param, "maxNlinPhaseRot", 2e-2)
    param.prgsBar = getattr(param, "prgsBar", True)
    param.saveSpanN = getattr(param, "saveSpanN", [param.Ltotal // param.Lspan])
    param.returnParameters = getattr(param, "returnParameters", False)

    Ltotal = param.Ltotal
    Lspan = param.Lspan
    hz = param.hz
    alpha = param.alpha
    D = param.D
    gamma = param.gamma
    amp = param.amp
    Fc = param.Fc
    prec = param.prec
    maxIter = param.maxIter
    tol = param.tol
    prgsBar = param.prgsBar
    saveSpanN = param.saveSpanN
    nlprMethod = param.nlprMethod
    maxNlinPhaseRot = param.maxNlinPhaseRot
    returnParameters = param.returnParameters

    Nspans = int(np.floor(Ltotal / Lspan))

    # channel parameters
    c_kms = const.c / 1e3  # speed of light (vacuum) in km/s
    λ = c_kms / Fc
    α = alpha / (10 * np.log10(np.exp(1)))
    β2 = -(D * λ**2) / (2 * np.pi * c_kms)
    γ = gamma

    # generate frequency axis
    Nfft = len(Ei)
    ω = 2 * np.pi * Fs * fftfreq(Nfft).astype(prec)

    Ech_x = Ei[:, 0::2].T
    Ech_y = Ei[:, 1::2].T

    # define static part of the linear operator
    argLimOp = np.array((α / 2) - 1j * (β2 / 2) * (ω**2)).astype(prec)

    if Ech_x.shape[0] > 1:
        argLimOp = np.tile(argLimOp, (Ech_x.shape[0], 1))
    else:
        argLimOp = argLimOp.reshape(1, -1)

    if saveSpanN:
        Ech_spans = np.zeros((Ei.shape[0], Ei.shape[1] * len(saveSpanN))).astype(prec)
        indRecSpan = 0

    for spanN in tqdm(range(1, Nspans + 1), disable=not (prgsBar)):
        # reverse amplification step
        if amp in {"edfa", "ideal"}:
            Ech_x = Ech_x * np.exp(-α / 2 * Lspan)
            Ech_y = Ech_y * np.exp(-α / 2 * Lspan)
        elif amp is None:
            Ech_x = Ech_x * np.exp(0)
            Ech_y = Ech_y * np.exp(0)

        Ex_conv = Ech_x.copy()
        Ey_conv = Ech_y.copy()
        z_current = 0

        # reverse fiber propagation steps
        while z_current < Lspan:
            Pch = Ech_x * np.conj(Ech_x) + Ech_y * np.conj(Ech_y)

            phiRot = nlinPhaseRot(Ex_conv, Ey_conv, Pch, γ)

            if nlprMethod:
                hz_ = (
                    maxNlinPhaseRot / np.max(phiRot)
                    if Lspan - z_current >= maxNlinPhaseRot / np.max(phiRot)
                    else Lspan - z_current
                )
            elif Lspan - z_current < hz:
                hz_ = Lspan - z_current  # check that the remaining
                # distance is not less than hz (due to non-integer
                # steps/span)
            else:
                hz_ = hz

            # define the linear operator
            linOperator = np.exp(argLimOp * (hz_ / 2))

            # First linear step (frequency domain)
            Ex_hd = ifft(fft(Ech_x) * linOperator)
            Ey_hd = ifft(fft(Ech_y) * linOperator)

            # Nonlinear step (time domain)
            for nIter in range(maxIter):
                rotOperator = np.exp(-1j * phiRot * hz_)

                Ech_x_fd = Ex_hd * rotOperator
                Ech_y_fd = Ey_hd * rotOperator

                # Second linear step (frequency domain)
                Ech_x_fd = ifft(fft(Ech_x_fd) * linOperator)
                Ech_y_fd = ifft(fft(Ech_y_fd) * linOperator)

                # check convergence o trapezoidal integration in phiRot
                lim = convergenceCondition(Ech_x_fd, Ech_y_fd, Ex_conv, Ey_conv)

                Ex_conv = Ech_x_fd.copy()
                Ey_conv = Ech_y_fd.copy()

                if lim < tol:
                    break
                elif nIter == maxIter - 1:
                    logg.warning(
                        f"Warning: target SSFM error tolerance was not achieved in {maxIter} iterations"
                    )

                phiRot = nlinPhaseRot(Ex_conv, Ey_conv, Pch, γ)

            Ech_x = Ech_x_fd.copy()
            Ech_y = Ech_y_fd.copy()

            z_current += hz_  # update propagated distance

        if spanN in saveSpanN:
            Ech_spans[:, 2 * indRecSpan : 2 * indRecSpan + 1] = Ech_x.T
            Ech_spans[:, 2 * indRecSpan + 1 : 2 * indRecSpan + 2] = Ech_y.T
            indRecSpan += 1

    if saveSpanN:
        Ech = Ech_spans
    else:
        Ech_x = Ech_x
        Ech_y = Ech_y

        Ech = Ei.copy()
        Ech[:, 0::2] = Ech_x.T
        Ech[:, 1::2] = Ech_y.T

    return (Ech, param) if returnParameters else Ech


def dfe(x, dx, param):
    """
    Decision feedback adaptive equalizer (DFE) for SISO receivers.

    Parameters
    ----------
    x : np.array
        Input signal to be equalized.
    dx : np.array
        Desired (reference) signal.
    param : optic.utils.parameters object
        DFE parameters:

        - param.nTapsFF: number of feedforward taps [default: 5]
        - param.nTapsFB: number of feedback taps [default: 5]
        - param.mu: step size [default: 0.0001]
        - param.nTrain: number of training symbols [default: 1000]
        - param.prec: precision [default: np.float32]
        - param.M: modulation order [default: 4]
        - param.constType: constellation type ('pam', 'qam', etc.) [default: 'pam']

    Returns
    -------
    yEq : np.array
        Equalized output signal.
    f : np.array
        Final feedforward filter coefficients.
    b : np.array
        Final feedback filter coefficients.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.

    """
    nTapsFF = getattr(param, "nTapsFF", 5)  # number of feedforward taps
    nTapsFB = getattr(param, "nTapsFB", 5)  # number of feedback taps
    mu = getattr(param, "mu", 0.0001)  # step size
    nTrain = getattr(param, "nTrain", 1000)  # number of training symbols
    prec = getattr(param, "prec", np.float32)  # precision
    M = getattr(param, "M", 4)  # modulation order
    constType = getattr(param, "constType", "pam")  # constellation type

    constSymb = grayMapping(M, constType).astype(prec)  # constellation
    constSymb = pnorm(constSymb)  # power-normalize constellation

    x = x.astype(prec)
    dx = dx.astype(prec)
    dx = dx.flatten()

    if constType == "pam":
        yEq, f, b, mse = realValuedDFECore(
            x, dx, nTapsFF, nTapsFB, mu, nTrain, prec, constSymb
        )
    else:
        yEq, f, b, mse = complexValuedDFECore(
            x, dx, nTapsFF, nTapsFB, mu, nTrain, prec, constSymb
        )

    return yEq, f, b, mse


@njit(fastmath=True)
def realValuedDFECore(
    x,
    dx,
    nTapsFF=5,
    nTapsFB=5,
    mu=0.0001,
    nTrain=1000,
    prec=np.float32,
    constSymb=None,
):
    """
    Decision feedback equalizer (DFE) core implementation.

    Parameters
    ----------
    x : np.array
        Input signal to be equalized.
    dx : np.array
        Desired (reference) signal.
    nTapsFF : int
        Number of feedforward taps
    nTapsFB : int
        Number of feedback taps
    mu : float
        Step size
    nTrain : int
        Number of training symbols
    prec : data type
        Precision
    M : int
        Modulation order
    constType : str
        Constellation type ('pam', 'qam', etc.)

    Returns
    -------
    yEq : np.array
        Equalized output signal.
    f : np.array
        Final feedforward filter coefficients.
    b : np.array
        Final feedback filter coefficients.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.

    """
    N = len(x)  # number of input samples

    # Initialize filters (center the main tap roughly in the middle of FF)
    f = np.zeros(nTapsFF, dtype=prec)
    f[0] = 1.0
    b = np.zeros(nTapsFB, dtype=prec)

    constSymb = constSymb.astype(prec)

    # Buffers
    xbuf = x[0:nTapsFF].astype(prec)  # past input samples
    dbuf = np.zeros(nTapsFB, dtype=prec)  # past decisions
    yEq = np.zeros(N, dtype=prec)
    mse = np.zeros(N, dtype=prec)

    for k in range(N):
        # Update FF buffer: newest sample at index 0
        xbuf = np.roll(xbuf, -1)
        xbuf[nTapsFF - 1] = x[k + nTapsFF - 1] if k + nTapsFF - 1 < N else 0.0

        # Compute output
        yEq[k] = np.dot(f, xbuf) + np.dot(b, dbuf)

        # Reference for adaptation: training then decision-directed
        if k < nTrain:
            d_ref = dx[k]
        else:
            indSymb = np.argmin(np.abs(yEq[k] - constSymb))
            d_ref = constSymb[indSymb]

        # Error
        ek = d_ref - yEq[k]
        mse[k] = ek**2

        # LMS updates
        f += mu * ek * xbuf
        b += mu * ek * dbuf

        # Update feedback buffer with the new decision
        if nTapsFB > 0:
            dbuf = np.roll(dbuf, 1)
            dbuf[0] = d_ref

    return yEq, f, b, mse


@njit(fastmath=True)
def complexValuedDFECore(
    x,
    dx,
    nTapsFF=5,
    nTapsFB=5,
    mu=0.0001,
    nTrain=1000,
    prec=np.complex64,
    constSymb=None,
):
    """
    Decision feedback equalizer (DFE) core implementation for complex-valued signals.

    Parameters
    ----------
    x : np.array
        Input signal to be equalized.
    dx : np.array
        Desired (reference) signal.
    nTapsFF : int
        Number of feedforward taps
    nTapsFB : int
        Number of feedback taps
    mu : float
        Step size
    nTrain : int
        Number of training symbols
    prec : data type
        Precision
    constSymb : np.array
        Constellation symbols

    Returns
    -------
    yEq : np.array
        Equalized output signal.
    f : np.array
        Final feedforward filter coefficients.
    b : np.array
        Final feedback filter coefficients.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.

    """
    N = len(x)  # number of input samples

    # Initialize filters (center the main tap roughly in the middle of FF)
    f = np.zeros(nTapsFF, dtype=prec)
    f[0] = 1.0 + 0j
    b = np.zeros(nTapsFB, dtype=prec)

    # Buffers
    xbuf = x[0:nTapsFF].astype(prec)  # past input samples
    dbuf = np.zeros(nTapsFB, dtype=prec)  # past decisions
    yEq = np.zeros(N, dtype=prec)
    mse = np.zeros(N, dtype=prec)

    for k in range(N):
        # Update FF buffer: newest sample at index 0
        xbuf = np.roll(xbuf, -1)
        xbuf[nTapsFF - 1] = x[k + nTapsFF - 1] if k + nTapsFF - 1 < N else 0.0 + 0j

        # Compute output
        yEq[k] = np.dot(f, xbuf) + np.dot(b, dbuf)

        # Reference for adaptation: training then decision-directed
        if k < nTrain:
            d_ref = dx[k]
        else:
            indSymb = np.argmin(np.abs(yEq[k] - constSymb))
            d_ref = constSymb[indSymb]

        # Error
        ek = d_ref - yEq[k]
        mse[k] = np.abs(ek) ** 2

        # LMS updates
        f += mu * ek * xbuf.conjugate()
        b += mu * ek * dbuf.conjugate()

        # Update feedback buffer with the new decision
        if nTapsFB > 0:
            dbuf = np.roll(dbuf, 1)
            dbuf[0] = d_ref

    return yEq, f, b, mse


def ffe(x, dx, param):
    """
    Decision-directed feedforward adaptive equalizer (FFE) for SISO receivers.

    Parameters
    ----------
    x : np.array
        Input signal to be equalized.
    dx : np.array
        Desired (reference) signal.
    param : optic.utils.parameters object
        FFE parameters:

        - param.nTaps: number of feedforward taps [default: 5]
        - param.mu: step size [default: 0.0001]
        - param.nTrain: number of training symbols [default: 1000]
        - param.prec: precision [default: np.float32]
        - param.M: modulation order [default: 4]
        - param.constType: constellation type ('pam', 'qam', etc.) [default: 'pam']

    Returns
    -------
    yEq : np.array
        Equalized output signal.
    f : np.array
        Final feedforward filter coefficients.

    References
    ----------
    [1] S. Haykin, "Adaptive Filter Theory," 5th ed., Pearson, 2013.

    """
    nTaps = getattr(param, "nTaps", 5)  # number of feedforward taps
    mu = getattr(param, "mu", 0.0001)  # step size
    nTrain = getattr(param, "nTrain", 1000)  # number of training symbols
    prec = getattr(param, "prec", np.float32)  # precision
    M = getattr(param, "M", 4)  # modulation order
    constType = getattr(param, "constType", "pam")  # constellation type

    constSymb = grayMapping(M, constType).astype(prec)  # constellation
    constSymb = pnorm(constSymb)  # power-normalize constellation

    x = x.astype(prec)
    dx = dx.astype(prec)
    dx = dx.flatten()

    x = np.pad(x, (nTaps // 2, nTaps // 2), "constant", constant_values=(0, 0))

    if constType == "pam":
        yEq, f, mse = realValuedFFECore(x, dx, nTaps, mu, nTrain, prec, constSymb)
    else:
        yEq, f, mse = complexValuedFFECore(
            x, dx, nTaps, mu, nTrain, prec, constSymb
        )

    return yEq, f, mse


@njit(fastmath=True)
def realValuedFFECore(
    x,
    dx,
    nTaps=5,
    mu=0.0001,
    nTrain=1000,
    prec=np.float32,
    constSymb=None,
):
    """
    Decision-directed feedforward equalizer (FFE) core implementation.

    Parameters
    ----------
    x : np.array
        Input signal to be equalized.
    dx : np.array
        Desired (reference) signal.
    nTaps : int
        Number of feedforward taps
    mu : float
        Step size
    nTrain : int
        Number of training symbols
    prec : data type
        Precision
    constSymb : np.array
        Constellation symbols

    Returns
    -------
    yEq : np.array
        Equalized output signal.
    f : np.array
        Final feedforward filter coefficients.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.

    """
    N = len(x) - nTaps + nTaps % 2  # number of input samples

    # Initialize filter (center the main tap roughly in the middle)
    f = np.zeros(nTaps, dtype=prec)
    f[nTaps // 2] = 1.0

    constSymb = constSymb.astype(prec)

    # Buffer
    yEq = np.zeros(N, dtype=prec)
    mse = np.zeros(N, dtype=prec)

    for k in range(N):
        # Update buffer: newest sample at index 0
        xbuf = x[k : k + nTaps]

        # Compute output
        yEq[k] = np.dot(f, xbuf)

        # Reference for adaptation: training then decision-directed
        if k < nTrain:
            d_ref = dx[k]
        else:
            indSymb = np.argmin(np.abs(yEq[k] - constSymb))
            d_ref = constSymb[indSymb]

        # Error
        ek = d_ref - yEq[k]
        mse[k] = ek**2

        # LMS update
        f += mu * ek * xbuf
    return yEq, f, mse


@njit(fastmath=True)
def complexValuedFFECore(
    x,
    dx,
    nTaps=5,
    mu=0.0001,
    nTrain=1000,
    prec=np.complex64,
    constSymb=None,
):
    """
    Decision-directed feedforward equalizer (FFE) core implementation for complex-valued signals.

    Parameters
    ----------
    x : np.array
        Input signal to be equalized.
    dx : np.array
        Desired (reference) signal.
    nTaps : int
        Number of feedforward taps
    mu : float
        Step size
    nTrain : int
        Number of training symbols
    prec : data type
        Precision
    constSymb : np.array
        Constellation symbols

    Returns
    -------
    yEq : np.array
        Equalized output signal.
    f : np.array
        Final feedforward filter coefficients.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.
    """
    N = len(x) - nTaps + nTaps % 2  # number of input samples

    # Initialize filter (center the main tap roughly in the middle)
    f = np.zeros(nTaps, dtype=prec)
    f[nTaps // 2] = 1.0 + 0j

    constSymb = constSymb.astype(prec)

    # Buffer
    yEq = np.zeros(N, dtype=prec)
    mse = np.zeros(N, dtype=prec)

    for k in range(N):
        # Update buffer: newest sample at index 0
        xbuf = x[k : k + nTaps]

        # Compute output
        yEq[k] = np.dot(f, xbuf)

        # Reference for adaptation: training then decision-directed
        if k < nTrain:
            d_ref = dx[k]
        else:
            indSymb = np.argmin(np.abs(yEq[k] - constSymb))
            d_ref = constSymb[indSymb]

        # Error
        ek = d_ref - yEq[k]
        mse[k] = np.abs(ek) ** 2

        # LMS update
        f += mu * ek * xbuf.conjugate()
    return yEq, f, mse

def volterra(x, dx, param):
    """
    Decision-directed Volterra equalizer implementation up to 3rd order for SISO receivers..

    Parameters
    ----------
    x : np.array
        Input signal to be equalized.
    dx : np.array
        Desired (reference) signal.
    param : optic.utils.parameters object
        Volterra equalizer parameters:

        - param.n1Taps: number of taps of linear part [default: 5]
        - param.n2Taps: number of taps of quadratic part [default: 3]
        - param.n3Taps: number of taps of cubic part [default: 2]
        - param.mu: step size [default: 0.001]
        - param.nTrain: number of training symbols [default: 1000]
        - param.order: Volterra series order (2 for quadratic) [default: 2]
        - param.prec: precision [default: np.float32]
        - param.M: modulation order [default: 4]
        - param.constType: constellation type ('pam', 'qam', etc.) [default: 'pam']        

    Returns
    -------
    yEq : np.array
        Equalized output signal.
    h : list of np.array
        Final Volterra filter coefficients [h1, h2, h3].

    References
    ----------
    [1] Diniz, P. R., da Silva, E. A. B., & Netto, S. L. (2010). Adaptive Filtering: Algorithms and Practical Implementation. Springer Science & Business Media.

    """
    n1Taps = getattr(param, "n1Taps", 5)  # number of taps of linear part
    n2Taps = getattr(param, "n2Taps", 3)  # number of taps of quadratic part
    n3Taps = getattr(param, "n3Taps", 2)  # number of taps of cubic part
    mu = getattr(param, "mu", 0.001)  # step size
    nTrain = getattr(param, "nTrain", 1000)  # number of training symbols
    order = getattr(param, "order", 2)  # Volterra series order
    prec = getattr(param, "prec", np.float32)  # precision
    M = getattr(param, "M", 4)  # modulation order
    constType = getattr(param, "constType", "pam")  # constellation type

    constSymb = grayMapping(M, constType).astype(prec)  # constellation    
    constSymb = pnorm(constSymb)  # amplitude-normalize constellation   

    x = x.astype(prec)
    dx = dx.astype(prec)
    dx = dx.flatten()

    nTaps = max(n1Taps, n2Taps, n3Taps)

    x = anorm(x)  

    x = np.pad(x, (nTaps // 2, nTaps // 2), "constant", constant_values=(0, 0))

    yEq, h1, h2, h3, mse = volterraCore(
        x, dx, n1Taps, n2Taps, n3Taps, order, mu, nTrain, prec, constSymb
    )

    h = [h1, h2, h3]

    yEq = pnorm(yEq)
    
    return yEq, h, mse

@njit(fastmath=True)
def volterraCore(
    x,
    dx,
    n1Taps=7,
    n2Taps=5,
    n3Taps=3,
    order=2,
    mu=0.0001,
    nTrain=1000,
    prec=np.float32,
    constSymb=None,
):
    """
    Decision-directed Volterra equalizer core implementation.

    Parameters
    ----------
    x : np.array
        Input signal to be equalized.
    dx : np.array
        Desired (reference) signal.
    n1Taps : int
        Number of taps of linear part
    n2Taps : int
        Number of taps of quadratic part
    n3Taps : int
        Number of taps of cubic part
    order : int
        Volterra series order (2 for quadratic, 3 for cubic)
    mu : float
        Step size
    nTrain : int
        Number of training symbols
    prec : data type
        Precision
    constSymb : np.array
        Constellation symbols

    Returns
    -------
    yEq : np.array
        Equalized output signal.
    h1 : np.array
        Final linear filter coefficients.
    h2 : np.array
        Final quadratic filter coefficients.
    h3 : np.array
        Final cubic filter coefficients.

    References
    ----------
    [1] Diniz, P. R., da Silva, E. A. B., & Netto, S. L. (2010). Adaptive Filtering: Algorithms and Practical Implementation. Springer Science & Business Media.

    """
    nTaps = np.max(np.array([n1Taps, n2Taps, n3Taps]))

    N = len(x) - nTaps + nTaps % 2  # number of input samples

    # Initialize filters (center the main tap roughly in the middle)
    h1 = np.zeros(n1Taps, dtype=prec)
    h1[n1Taps // 2] = 1.0
       
    h2 = np.zeros((n2Taps, n2Taps), dtype=prec)
    h3 = np.zeros((n3Taps, n3Taps, n3Taps), dtype=prec)            
     
    constSymb = constSymb.astype(prec)

    # Buffer
    yEq = np.zeros(N, dtype=prec)
    mse = np.zeros(N, dtype=prec)

    t2 = int((n1Taps-n2Taps) //2)
    t3 = int((n1Taps-n3Taps) //2)

    for k in range(N):
        # Update buffer: newest sample at index 0
        xbuf = x[k : k + nTaps]

        # Compute output
        linearPart = np.dot(h1, xbuf)
        quadraticPart = 0.0
        cubicPart = 0.0

        for i in range(n2Taps):
            for j in range(n2Taps):
                quadraticPart += h2[i, j] * xbuf[t2+i] * xbuf[t2+j] 

        if order == 3:            
            for i in range(n3Taps):
                for j in range(n3Taps):
                    for l in range(n3Taps):
                        cubicPart += h3[i, j, l] * xbuf[t3+i] * xbuf[t3+j] * xbuf[t3+l]

        yEq[k] = linearPart + quadraticPart + cubicPart
         
        # Reference for adaptation: training then decision-directed
        if k < nTrain:
            d_ref = dx[k]
        else:
            indSymb = np.argmin(np.abs(yEq[k] - constSymb))
            d_ref = constSymb[indSymb]

        # Error
        ek = d_ref - yEq[k]
        mse[k] = ek**2

        # LMS updates
        h1 += mu * ek * xbuf
        for i in range(n2Taps):
            for j in range(n2Taps):
                h2[i, j] += mu/2 * ek * xbuf[t2+i] * xbuf[t2+j]

        if order == 3:            
            for i in range(n3Taps):
                for j in range(n3Taps):
                    for l in range(n3Taps):
                        h3[i, j, l] += mu/2 * ek * xbuf[t3+i] * xbuf[t3+j] * xbuf[t3+l]
      
    return yEq, h1, h2, h3, mse
