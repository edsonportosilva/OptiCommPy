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
from optic.dsp.adaptiveFiltering import (
    complexValuedDFECore,
    complexValuedFFECore,    
    coreAdaptEqBlockTD,
    coreAdaptEqBlockFD,
    realValuedDFECore,
    realValuedFFECore,
    volterraCore,
)
from optic.dsp.core import anorm, blockwiseFFTConv, pnorm
from optic.models.channels import convergenceCondition, nlinPhaseRot

# try:
#     from optic.dsp.coreGPU import blockwiseFFTConv
# except ImportError:
#     from optic.dsp.core import blockwiseFFTConv


def edc(sigIn, param):
    """
    Electronic chromatic dispersion compensation (EDC).

    Parameters
    ----------
    sigIn : np.array
        Dispersed input signal.
    param : optic.utils.parameters object
        Parameters of the optical channel.

        - param.L : total fiber length [km][default: 50 km]
        - param.D : chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
        - param.Fc : carrier frequency [Hz] [default: 193.1e12 Hz]
        - param.Fs : sampling frequency [Hz] [default: []]
        - param.Rs : symbol rate [baud] [default: 32e9]
        - param.NfilterCoeffs : number of filter coefficients [default: []]
        - param.Nfft : FFT size [default: []]

    Returns
    -------
    sigOut : np.array
        Dispersion compensated output signal.

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
        nModes = sigIn.shape[1]
        input1D = False
    except IndexError:
        nModes = 1
        sigIn = sigIn.reshape(sigIn.size, nModes)
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

    sigOut = np.zeros(sigIn.shape, dtype=sigIn.dtype)

    # Apply CD compensation to each mode
    for indMode in range(nModes):
        sigOut[:, indMode] = blockwiseFFTConv(
            sigIn[:, indMode], H, NFFT=Nfft, freqDomainFilter=True
        )

    if input1D:
        # If the input was 1D, return a 1D array
        sigOut = sigOut.flatten()

    return sigOut


def mimoAdaptEqualizer(sigIn, param=None, symbRef=None):
    """
    General :math:`N \\times N` MIMO adaptive equalizer with several adaptive filtering algorithms available.

    Parameters
    ----------
    sigIn : np.array
        Input signal array.
    symbRef : np.array, optional
        Reference symbol sequence synchronized to sigIn.
    param : optic.utils.parameters object, optional
        Parameter object containing the following attributes:

        - param.numIter : int, number of pre-convergence iterations [default: 1]
        - param.nTaps : int, number of filter taps [default: 15]
        - param.mu : float or list of floats, step size parameter(s) [default: [1e-3]]
        - param.lambdaRLS : float, RLS forgetting factor [default: 0.99]
        - param.SpS : int, samples per symbol [default: 2]
        - param.H : np.array, coefficient matrix [default: []]
        - param.L : int or list of ints, length of the output of the training section [default: []]
        - param.Hiter : list, history of coefficient matrices [default: []]
        - param.storeCoeff : bool, flag indicating whether to store coefficient matrices [default: False]
        - param.runWL: bool, flag indicating whether to run the equalizer in the widely-linear mode [default: False]
        - param.alg : str or list of strs, specifying the equalizer algorithm(s) [default: ['nlms']]
        - param.constType : str, constellation type [default: 'qam']
        - param.M : int, modulation order [default: 4]
        - param.prgsBar : bool, flag indicating whether to display progress bar [default: True]
        - param.returnResults : bool, flag indicating whether to return all results [default: False]
        - param.prec : data type, precision of the computations [default: np.complex64]

    Returns
    -------
    sigOut : np.array
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
    if symbRef is None:
        symbRef = []
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
    domain = getattr(param, "domain", "freq")
    Nfft = getattr(param, "Nfft", 128)
    blockSize = getattr(param, "blockSize", 1)

    # We want all the signal sequences to be disposed in columns:
    if not len(symbRef):
        symbRef = sigIn.copy()
    try:
        if sigIn.shape[1] > sigIn.shape[0]:
            sigIn = sigIn.T
        input1D = False
    except IndexError:
        sigIn = sigIn.reshape(len(sigIn), 1)
        input1D = True
    try:
        if symbRef.shape[1] > symbRef.shape[0]:
            symbRef = symbRef.T
    except IndexError:
        symbRef = symbRef.reshape(len(symbRef), 1)
    nModes = int(sigIn.shape[1])  # number of signal modes (order of the MIMO equalizer)

    symbRef = symbRef.astype(prec)
    sigIn = sigIn.astype(prec)
    mu = np.array(mu).astype(np.float32)
    lambdaRLS = np.array([lambdaRLS]).astype(prec)[0]

    Lpad = int(np.floor(nTaps / 2))
    zeroPad = np.zeros((Lpad, nModes), dtype=prec)
    sigIn = np.concatenate(
        (zeroPad, sigIn, zeroPad)
    )  # pad start and end of the signal with zeros

    # Defining training parameters:
    constSymb = grayMapping(M, constType).astype(prec)  # constellation

    # Calculate MB distribution
    px = np.exp(-shapingFactor * np.abs(constSymb) ** 2)
    px = px / np.sum(px)

    # normalize reference constellation accouting for the probability mass function
    constSymb /= np.sqrt(np.sum(np.abs(constSymb) ** 2 * px))
    totalNumSymb = int(np.fix((len(sigIn) - nTaps) / SpS + 1))

    if not L:  # if L is not defined
        L = [
            totalNumSymb
        ]  # Length of the output (1 sample/symbol) of the training section
    if not H:  # if H is not defined
        if domain in ["time"]:
            H = np.zeros((nModes**2, nTaps), dtype=prec)
            for initH in range(nModes):  # initialize filters' taps
                H[initH + initH * nModes, int(np.floor(H.shape[1] / 2))] = (
                    1 + 1j * 0  # Central spike initialization
                )
        elif domain in ["freq"]:
            H = np.zeros((nModes**2, nTaps), dtype=prec)

            for initH in range(nModes):  # initialize filters' taps
                H[initH + initH * nModes, int(np.floor(H.shape[1] / 2))] = (
                    1 + 1j * 0  # Central spike initialization
                )
            # H = fft(H, n=Nfft, axis=1)  # FFT of the filters' taps
    if not H_:  # if H_ is not defined
        H_ = np.zeros((nModes**2, nTaps), dtype=prec)

    logg.info(f"Running adaptive equalizer...")
    # Equalizer training:
    if type(alg) == list:
        sigOut = np.zeros((totalNumSymb, sigIn.shape[1]), dtype=prec)
        errSq = np.zeros((totalNumSymb, sigIn.shape[1]), dtype=prec).T

        nStart = 0
        for indstage, runAlg in enumerate(alg):
            logg.info(f"{runAlg} - training stage #%d", indstage)

            nEnd = nStart + L[indstage]

            if indstage == 0:
                for indIter in tqdm(range(numIter), disable=not (prgsBar)):
                    logg.info(
                        f"{runAlg} pre-convergence training iteration #%d", indIter
                    )
                    if domain == "freq":
                        sigOut[nStart:nEnd, :], H, H_, errSq[:, nStart:nEnd], Hiter = (
                            coreAdaptEqBlockFD(
                                sigIn[nStart * SpS : (nEnd + 2 * Lpad) * SpS, :],
                                symbRef[nStart:nEnd, :],
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
                                Nfft,
                            )
                        )
                    elif domain == "time":
                        sigOut[nStart:nEnd, :], H, H_, errSq[:, nStart:nEnd], Hiter = (
                            coreAdaptEqBlockTD(
                                sigIn[nStart * SpS : (nEnd + 2 * Lpad) * SpS, :],
                                symbRef[nStart:nEnd, :],
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
                                blockSize,
                            )
                        )
                    logg.info(
                        f"{runAlg} MSE = %.6f.", np.nanmean(errSq[:, nStart:nEnd]).real
                    )
            else:
                if domain == "freq":
                    sigOut[nStart:nEnd, :], H, H_, errSq[:, nStart:nEnd], Hiter = (
                        coreAdaptEqBlockFD(
                            sigIn[nStart * SpS : (nEnd + 2 * Lpad) * SpS, :],
                            symbRef[nStart:nEnd, :],
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
                            Nfft,
                        )
                    )
                elif domain == "time":
                    sigOut[nStart:nEnd, :], H, H_, errSq[:, nStart:nEnd], Hiter = (
                        coreAdaptEqBlockTD(
                            sigIn[nStart * SpS : (nEnd + 2 * Lpad) * SpS, :],
                            symbRef[nStart:nEnd, :],
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
                            blockSize,
                        )
                    )
                logg.info(
                        f"{runAlg} MSE = %.6f.", np.nanmean(errSq[:, nStart:nEnd]).real
                    )
            nStart = nEnd
    else:
        for indIter in tqdm(range(numIter), disable=not (prgsBar)):
            logg.info(f"{alg}training iteration #%d", indIter)
            if domain == "freq":
                sigOut, H, H_, errSq, Hiter = coreAdaptEqBlockFD(
                    sigIn,
                    symbRef,
                    SpS,
                    H,
                    H_,
                    L,
                    mu,
                    lambdaRLS,
                    nTaps,                    
                    storeCoeff,
                    runWL,
                    alg,
                    constSymb,
                    prec,
                    Nfft,
                )            
            elif domain == "time":
                sigOut, H, H_, errSq, Hiter = coreAdaptEqBlockTD(
                    sigIn,
                    symbRef,
                    SpS,
                    H,
                    H_,
                    L,
                    mu,
                    lambdaRLS,
                    nTaps,
                    storeCoeff,
                    runWL,
                    alg,
                    constSymb,
                    prec,
                    blockSize,
                )
            logg.info(f"{alg}MSE = %.6f.", np.nanmean(errSq).real)

    if input1D:
        # If the input was 1D, return a 1D array
        sigOut = sigOut.flatten()

    if returnResults:
        if runWL:
            return sigOut, H, H_, errSq, Hiter
        else:
            return sigOut, H, errSq, Hiter
    else:
        return sigOut


def manakovDBP(Ei, param):
    """
    Run the Manakov SSF digital backpropagation (symmetric, dual-pol.).

    Parameters
    ----------
    Ei : np.array
        Input optical signal field.
    param : optic.utils.parameters object
        Physical/simulation parameters of the optical channel.

        - param.Ltotal : total fiber length [km][default: 400 km]
        - param.Lspan : span length [km][default: 80 km]
        - param.hz : step-size for the split-step Fourier method [km][default: 0.5 km]
        - param.alpha : fiber attenuation parameter [dB/km][default: 0.2 dB/km]
        - param.D : chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
        - param.gamma : fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
        - param.Fc : carrier frequency [Hz] [default: 193.1e12 Hz]
        - param.Fs : simulation sampling frequency [samples/second][default: None]
        - param.prec : numerical precision [default: np.complex128]
        - param.amp : 'edfa', 'ideal', or 'None. [default:'edfa']
        - param.maxIter : max number of iter. in the trap. integration [default: 10]
        - param.tol : convergence tol. of the trap. integration.[default: 1e-5]
        - param.nlprMethod : adap step-size based on nonl. phase rot. [default: True]
        - param.maxNlinPhaseRot : max nonl. phase rot. tolerance [rad][default: 2e-2]
        - param.prgsBar : display progress bar? bolean variable [default:True]
        - param.saveSpanN : specify the span indexes to be outputted [default:[]]
        - param.returnParameters : bool, return channel parameters [default: False]


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


def dfe(sigIn, symbRef, param):
    """
    Decision feedback adaptive equalizer (DFE) for SISO receivers.

    Parameters
    ----------
    sigIn : np.array
        Input signal to be equalized.
    symbRef : np.array
        Desired (reference) signal.
    param : optic.utils.parameters object
        DFE parameters:

        - param.nTapsFF : number of feedforward taps [default: 5]
        - param.nTapsFB : number of feedback taps [default: 5]
        - param.SpS : samples per symbol [default: 1]
        - param.mu : step size [default: 0.0001]
        - param.nTrain : number of training symbols [default: 1000]
        - param.prec : precision [default: np.float32]
        - param.M : modulation order [default: 4]
        - param.constType : constellation type ('pam', 'qam', etc.) [default: 'pam']
        - param.f : initial feedforward coeffs [default: None]
        - param.b : initial feedback coeffs [default: None]
        - param.trainingMode : operation mode ('data-aided', 'fulltime') [default: 'data-aided']
        - param.preconvIters : number of pre-convergence iterations [default: 1]

    Returns
    -------
    sigOut : np.array
        Equalized output signal.
    f : np.array
        Final feedforward filter coefficients.
    b : np.array
        Final feedback filter coefficients.

    Notes
    -----
    - Training mode 'data-aided' uses the known training symbols for adaptation, while 'fulltime' continues to adapt using decision-directed mode even after the training phase.
    - Pre-convergence iterations can help the algorithm to converge better by restarting the adaptation process after the initial training phase.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.

    """
    nTapsFF = getattr(param, "nTapsFF", 5)  # number of feedforward taps
    nTapsFB = getattr(param, "nTapsFB", 5)  # number of feedback taps
    SpS = getattr(param, "SpS", 1)  # samples per symbol
    mu = getattr(param, "mu", 0.0001)  # step size
    nTrain = getattr(param, "nTrain", 1000)  # number of training symbols
    prec = getattr(param, "prec", None)  # precision
    M = getattr(param, "M", 4)  # modulation order
    constType = getattr(param, "constType", "pam")  # constellation type
    f = getattr(param, "f", None)  # initial feedforward coeffs
    b = getattr(param, "b", None)  # initial feedback coeffs
    trainingMode = getattr(param, "trainingMode", "data-aided")  # operation mode
    preconvIters = getattr(param, "preconvIters", 1)  # pre-convergence iterations

    if prec is None:
        prec = sigIn.dtype  # infer precision from input signal if not provided

    constSymb = grayMapping(M, constType).astype(prec)  # constellation
    constSymb = pnorm(constSymb)  # power-normalize constellation

    # Make copies to avoid modifying original arrays
    sigIn = sigIn.copy()
    symbRef = symbRef.copy()

    # normalize imput signal
    sigIn = pnorm(sigIn)  # power-normalize input signal
    symbRef = pnorm(symbRef)  # power-normalize desired signal

    # Ensure correct data types
    sigIn = sigIn.astype(prec)
    symbRef = symbRef.astype(prec)
    symbRef = symbRef.flatten()

    # Initialize filters (center the main tap roughly in the middle of FF)
    if f is None:
        f = np.zeros(nTapsFF, dtype=prec)
        f[nTapsFF // 2] = 1.0

    if b is None:
        b = np.zeros(nTapsFB, dtype=prec)

    sigIn = np.pad(
        sigIn, (nTapsFF // 2, nTapsFF // 2), "constant", constant_values=(0, 0)
    )

    if constType == "pam":
        sigOut, f, b, mse = realValuedDFECore(
            sigIn,
            symbRef,
            nTapsFF,
            nTapsFB,
            SpS,
            mu,
            nTrain,
            prec,
            constSymb,
            f,
            b,
            trainingMode,
            preconvIters,
        )
    else:
        sigOut, f, b, mse = complexValuedDFECore(
            sigIn,
            symbRef,
            nTapsFF,
            nTapsFB,
            SpS,
            mu,
            nTrain,
            prec,
            constSymb,
            f,
            b,
            trainingMode,
            preconvIters,
        )

    return sigOut, f, b, mse


def ffe(sigIn, symbRef, param):
    """
    Decision-directed feedforward adaptive equalizer (FFE) for SISO receivers.

    Parameters
    ----------
    sigIn : np.array
        Input signal to be equalized.
    symbRef : np.array
        Desired (reference) signal.
    param : optic.utils.parameters object
        FFE parameters:

        - param.nTaps : number of feedforward taps [default: 5]
        - param.mu : step size [default: 0.0001]
        - param.SpS : samples per symbol [default: 1]
        - param.nTrain : number of training symbols [default: 1000]
        - param.prec : precision [default: np.float32]
        - param.M : modulation order [default: 4]
        - param.constType : constellation type ('pam', 'qam', etc.) [default: 'pam']
        - param.f : initial feedforward coeffs [default: None]
        - param.trainingMode : operation mode ('data-aided', 'fulltime') [default: 'data-aided']
        - param.preconvIters : number of pre-convergence iterations [default: 1]

    Returns
    -------
    sigOut : np.array
        Equalized output signal.
    f : np.array
        Final feedforward filter coefficients.

    Notes
    -----
    - Training mode 'data-aided' uses the known training symbols for adaptation, while 'fulltime' continues to adapt using decision-directed mode even after the training phase.
    - Pre-convergence iterations can help the algorithm to converge better by restarting the adaptation process after the initial training phase.

    References
    ----------
    [1] S. Haykin, "Adaptive Filter Theory," 5th ed., Pearson, 2013.

    """
    nTaps = getattr(param, "nTaps", 5)  # number of feedforward taps
    mu = getattr(param, "mu", 0.0001)  # step size
    SpS = getattr(param, "SpS", 1)  # samples per symbol
    nTrain = getattr(param, "nTrain", 1000)  # number of training symbols
    prec = getattr(param, "prec", None)  # precision
    M = getattr(param, "M", 4)  # modulation order
    constType = getattr(param, "constType", "pam")  # constellation type
    f = getattr(param, "f", None)  # initial feedforward coeffs
    trainingMode = getattr(param, "trainingMode", "data-aided")  # operation mode
    preconvIters = getattr(param, "preconvIters", 1)  # pre-convergence iterations

    if prec is None:
        prec = sigIn.dtype  # infer precision from input signal if not provided

    constSymb = grayMapping(M, constType).astype(prec)  # constellation
    constSymb = pnorm(constSymb)  # power-normalize constellation

    # Make copies to avoid modifying original arrays
    sigIn = sigIn.copy()
    symbRef = symbRef.copy()

    # normalize input signal
    sigIn = pnorm(sigIn)  # power-normalize input signal
    symbRef = pnorm(symbRef)  # power-normalize desired signal

    # Ensure correct data types
    sigIn = sigIn.astype(prec)
    symbRef = symbRef.astype(prec)
    symbRef = symbRef.flatten()

    sigIn = np.pad(sigIn, (nTaps // 2, nTaps // 2), "constant", constant_values=(0, 0))

    if f is None:
        # Initialize filter (center the main tap roughly in the middle)
        f = np.zeros(nTaps, dtype=prec)
        f[nTaps // 2] = 1.0

    if constType == "pam":
        sigOut, f, mse = realValuedFFECore(
            sigIn,
            symbRef,
            nTaps,
            SpS,
            mu,
            nTrain,
            prec,
            constSymb,
            f,
            trainingMode,
            preconvIters,
        )
    else:
        sigOut, f, mse = complexValuedFFECore(
            sigIn,
            symbRef,
            nTaps,
            SpS,
            mu,
            nTrain,
            prec,
            constSymb,
            f,
            trainingMode,
            preconvIters,
        )

    return sigOut, f, mse


def volterra(sigIn, symbRef, param):
    """
    Decision-directed Volterra equalizer implementation up to 3rd order for SISO receivers

    Parameters
    ----------
    sigIn : np.array
        Input signal to be equalized.
    symbRef : np.array
        Desired (reference) signal.
    param : optic.utils.parameters object
        Volterra equalizer parameters:

        - param.n1Taps : number of taps of linear part [default: 5]
        - param.n2Taps : number of taps of quadratic part [default: 3]
        - param.n3Taps : number of taps of cubic part [default: 2]
        - param.h : list of initial filter coefficients [default: None]
        - param.SpS : samples per symbol [default: 1]
        - param.mu : step size [default: 0.001]
        - param.nTrain : number of training symbols [default: 1000]
        - param.order : Volterra series order (2 for quadratic) [default: 2]
        - param.prec : precision [default: np.float32]
        - param.M : modulation order [default: 4]
        - param.constType : constellation type ('pam', 'qam', etc.) [default: 'pam']
        - param.trainingMode : operation mode ('data-aided', 'fulltime') [default: 'data-aided']
        - param.preconvIters : number of pre-convergence iterations [default: 1]

    Returns
    -------
    sigOut : np.array
        Equalized output signal.
    h : list of np.array
        Final Volterra filter coefficients [h1, h2, h3].

    Notes
    -----
    - Training mode 'data-aided' uses the known training symbols for adaptation, while 'fulltime' continues to adapt using decision-directed mode even after the training phase.
    - Pre-convergence iterations can help the algorithm to converge better by restarting the adaptation process after the initial training phase.

    References
    ----------
    [1] Diniz, P. R., da Silva, E. A. B., & Netto, S. L. (2010). Adaptive Filtering: Algorithms and Practical Implementation. Springer Science & Business Media.

    """
    n1Taps = getattr(param, "n1Taps", 5)  # number of taps of linear part
    n2Taps = getattr(param, "n2Taps", 3)  # number of taps of quadratic part
    n3Taps = getattr(param, "n3Taps", 2)  # number of taps of cubic part
    h = getattr(param, "h", None)  # initial filter coeffs
    SpS = getattr(param, "SpS", 1)  # samples per symbol
    mu = getattr(param, "mu", 0.001)  # step size
    nTrain = getattr(param, "nTrain", 1000)  # number of training symbols
    order = getattr(param, "order", 2)  # Volterra series order
    prec = getattr(param, "prec", np.float32)  # precision
    M = getattr(param, "M", 4)  # modulation order
    constType = getattr(param, "constType", "pam")  # constellation type
    trainingMode = getattr(param, "trainingMode", "data-aided")  # operation mode
    preconvIters = getattr(param, "preconvIters", 1)  # pre-convergence iterations

    if n1Taps < n2Taps or n1Taps < n3Taps:
        logg.error("n1Taps must be greater than or equal to n2Taps and n3Taps.")

    if h is None:
        # Initialize filters (center the main tap roughly in the middle)
        h1 = np.zeros(n1Taps, dtype=prec)
        h1[n1Taps // 2] = 1.0

        h2 = np.zeros((n2Taps, n2Taps), dtype=prec)
        h3 = np.zeros((n3Taps, n3Taps, n3Taps), dtype=prec)
    else:
        h1 = h[0]
        h2 = h[1]
        h3 = h[2]

    constSymb = grayMapping(M, constType).astype(prec)  # constellation
    constSymb = pnorm(constSymb)  # amplitude-normalize constellation

    # Make copies to avoid modifying original arrays
    sigIn = sigIn.copy()
    symbRef = symbRef.copy()

    # normalize input signal
    sigIn = pnorm(sigIn)  # power-normalize input signal
    symbRef = pnorm(symbRef)  # power-normalize desired signal

    # Ensure correct data types
    sigIn = sigIn.astype(prec)
    symbRef = symbRef.astype(prec)
    symbRef = symbRef.flatten()

    nTaps = max(n1Taps, n2Taps, n3Taps)

    sigIn = anorm(sigIn)

    sigIn = np.pad(sigIn, (nTaps // 2, nTaps // 2), "constant", constant_values=(0, 0))

    sigOut, h1, h2, h3, mse = volterraCore(
        sigIn,
        symbRef,
        order,
        SpS,
        mu,
        nTrain,
        h1,
        h2,
        h3,
        prec,
        constSymb,
        trainingMode,
        preconvIters,
    )

    h = [h1, h2, h3]

    sigOut = pnorm(sigOut)

    return sigOut, h, mse
