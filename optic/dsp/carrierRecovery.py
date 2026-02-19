"""
==========================================================================================
DSP algorithms for carrier phase and frequency recovery (:mod:`optic.dsp.carrierRecovery`)
==========================================================================================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   bps            -- Blind phase search (BPS) carrier phase recovery algorithm.
   ddpll          -- Decision-directed phase-locked loop (DD-PLL) carrier phase recovery algorithm.
   viterbi        -- Viterbi & Viterbi carrier phase recovery algorithm.
   fourthPowerFOE -- Frequency offset (FO) estimation and compensation with the 4th-power method.
   cpr            -- General function to call and configure any of the CPR algorithms in this module.
"""

import logging as logg

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numpy.fft import fft, fftfreq, fftshift

from optic.comm.modulation import grayMapping
from optic.dsp.core import movingAverage, pnorm

try:
    from optic.dsp.coreGPU import checkGPU

    if checkGPU():
        from optic.dsp.carrierRecoveryGPU import bpsGPU
    else:
        pass
except ImportError:
    pass


def cpr(Ei, param=None, symbTx=None):
    """
    Carrier phase recovery function (CPR)

    Parameters
    ----------
    Ei : complex-valued np.array
        received constellation symbols.
    param : optic.utils.parameter object, optional
        Configuration parameters [default: None].

        - param.alg: CPR algorithm to be used ['bps', 'bpsGPU', 'ddpll', or 'viterbi'] [default: 'bps'].
        - param.shapingFactor: shaping factor, for probabilistic shaped QAM with MB dististribution.[default: 0]

        BPS params:

        - param.M: constellation order. [default: 4]
        - param.N: length of BPS the moving average window. [default: 35]
        - param.B: number of BPS test phases. [default: 64]

        DDPLL params:

        - param.tau1: DDPLL loop filter param. 1. [default: 1/2*pi*10e6]
        - param.tau2: DDPLL loop filter param. 2. [default: 1/2*pi*10e6]
        - param.Kv: DDPLL loop filter gain. [default: 0.1]
        - param.Ts: symbol period. [default: 1/32e9]
        - param.pilotInd: indexes of pilot-symbol locations.

        Viterbi params:

        - param.N: length of the moving average window. [default: 35]

    symbTx :complex-valued np.array, optional
        Transmitted symbol sequence. [default: None]

    Returns
    -------
    Eo : complex-valued np.array
        Phase-compensated signal.
    phaseEst : real-valued np.array
        Time-varying estimated phase-shifts.

    References
    ----------
    [1] T. Pfau, S. Hoffmann, e R. Noé, “Hardware-efficient coherent digital receiver concept with feedforward carrier recovery for M-QAM constellations”, Journal of Lightwave Technology, vol. 27, nº 8, p. 989–999, 2009, doi: 10.1109/JLT.2008.2010511.

    [2] S. J. Savory, “Digital coherent optical receivers: Algorithms and subsystems”, IEEE Journal on Selected Topics in Quantum Electronics, vol. 16, nº 5, p. 1164–1179, set. 2010, doi: 10.1109/JSTQE.2010.2044751.

    [3] H. Meyer, Digital Communication Receivers: Synchronization, Channel estimation, and Signal Processing, Wiley 1998. Section 5.8 and 5.9.
    """
    if symbTx is None:
        symbTx = np.zeros(Ei.shape)
    if param is None:
        param = []

    # check input parameters
    alg = getattr(param, "alg", "bps")
    M = getattr(param, "M", 4)
    constType = getattr(param, "constType", "qam")
    shapingFactor = getattr(param, "shapingFactor", 0)
    B = getattr(param, "B", 64)
    N = getattr(param, "N", 35)
    Kv = getattr(param, "Kv", 0.1)
    tau1 = getattr(param, "tau1", 1 / (2 * np.pi * 10e6))
    tau2 = getattr(param, "tau2", 1 / (2 * np.pi * 10e6))
    Ts = getattr(param, "Ts", 1 / 32e9)
    pilotInd = getattr(param, "pilotInd", np.array([len(Ei) + 1]))
    returnPhases = getattr(param, "returnPhases", False)

    try:
        Ei.shape[1]
        input1D = False
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)
        input1D = True

    # constellation parameters
    constSymb = grayMapping(M, constType)
    px = np.exp(-shapingFactor * np.abs(constSymb) ** 2)
    px = px / np.sum(px)
    constSymb /= np.sqrt(np.sum(np.abs(constSymb) ** 2 * px))

    # 4th power frequency offset estimation/compensation
    logg.info(f"Running frequency offset compensation...")
    Ei, fo = fourthPowerFOE(Ei, 1 / Ts)
    Ei = pnorm(Ei)
    logg.info(f"Estimated frequency offset (MHz): {np.round(fo/1e6, 3)}")

    if alg == "ddpll":
        logg.info(f"Running DDPLL carrier phase recovery...")
        phaseEst = ddpll(Ei, Ts, Kv, tau1, tau2, constSymb, symbTx, pilotInd)
    elif alg == "bps":
        logg.info(f"Running BPS carrier phase recovery...")
        phaseEst = bps(Ei, N // 2, constSymb, B)
    elif alg == "bpsGPU":
        try:
            logg.info("Running GPU-based BPS carrier phase recovery...")
            phaseEst = bpsGPU(Ei, N // 2, constSymb, B)
        except NameError:
            logg.warning("GPU unavailable, switching to CPU processing...")
            phaseEst = bps(Ei, N // 2, constSymb, B)
    elif alg == "viterbi":
        logg.info(f"Running Viterbi&Viterbi carrier phase recovery...")
        phaseEst = viterbi(Ei, N)
    else:
        raise ValueError("CPR algorithm incorrectly specified.")
    phaseEst = np.unwrap(4 * phaseEst, axis=0) / 4

    discard = (
        phaseEst.shape[0] // 4
    )  # discard 1/4 of the symbols at the beginning and end
    sigmaPhase = np.mean(np.var(np.diff(phaseEst[discard:-discard, :], axis=0), axis=0))
    logg.info(f"Estimated linewidth: {sigmaPhase/(2 * np.pi* Ts)/1e3:.3f} kHz")

    Eo = pnorm(Ei * np.exp(1j * phaseEst))

    if input1D:
        # If input was 1D, return a 1D array
        Eo = Eo.flatten()
        phaseEst = phaseEst.flatten()

    return (Eo, phaseEst) if returnPhases else Eo


@njit
def bps(Ei, N, constSymb, B):
    """
    Blind phase search (BPS) algorithm

    Parameters
    ----------
    Ei : complex-valued np.array
        Received constellation symbols.
    N : int
        Half of the 2*N+1 average window.
    constSymb : complex-valued np.array
        Complex-valued constellation.
    B : int
        number of test phases.

    Returns
    -------
    phaseEst : real-valued np.array
        Time-varying estimated phase-shifts.

    References
    ----------
    [1] T. Pfau, S. Hoffmann, e R. Noé, “Hardware-efficient coherent digital receiver concept with feedforward carrier recovery for M-QAM constellations”, Journal of Lightwave Technology, vol. 27, nº 8, p. 989–999, 2009, doi: 10.1109/JLT.2008.2010511.
    """
    nModes = Ei.shape[1]

    testPhases = np.arange(0, B) * (np.pi / 2) / B  # test phases

    phaseEst = np.zeros(Ei.shape, dtype="float")

    zeroPad = np.zeros((N, nModes), dtype="complex")
    x = np.concatenate(
        (zeroPad, Ei, zeroPad)
    )  # pad start and end of the signal with zeros

    L = x.shape[0]

    for n in range(nModes):
        dist = np.zeros((B, constSymb.shape[0]), dtype="float")
        dmin = np.zeros((B, 2 * N + 1), dtype="float")

        for k in range(L):
            for indPhase, phi in enumerate(testPhases):
                dist[indPhase, :] = np.abs(x[k, n] * np.exp(1j * phi) - constSymb) ** 2
                dmin[indPhase, -1] = np.min(dist[indPhase, :])
            if k >= 2 * N:
                sumDmin = np.sum(dmin, axis=1)
                indRot = np.argmin(sumDmin)
                phaseEst[k - 2 * N, n] = testPhases[indRot]
            dmin = np.roll(dmin, -1)
    return phaseEst


@njit
def ddpll(Ei, Ts, Kv, tau1, tau2, constSymb, symbTx, pilotInd):
    """
    Decision-directed Phase-locked Loop (DDPLL) algorithm

    Parameters
    ----------
    Ei : complex-valued np.array
        Received constellation symbols.
    Ts : float scalar
        Symbol period.
    Kv : float scalar
        Loop filter gain.
    tau1 : float scalar
        Loop filter parameter 1.
    tau2 : float scalar
        Loop filter parameter 2.
    constSymb : complex-valued np.array
        Complex-valued ideal constellation symbols.
    symbTx : complex-valued np.array
        Transmitted symbol sequence.
    pilotInd : int np.array
        Indexes of pilot-symbol locations.

    Returns
    -------
    phaseEst : real-valued np.array
        Time-varying estimated phase-shifts.

    References
    ----------
    [1] H. Meyer, Digital Communication Receivers: Synchronization, Channel estimation, and Signal Processing, Wiley 1998. Section 5.8 and 5.9.
    """
    nSymbols, nModes = Ei.shape

    phaseEst = np.zeros((nSymbols, nModes), dtype=np.float64)

    # Loop filter coefficients
    a1b = np.array(
        [
            1,
            Ts / (2 * tau1) * (1 - 1 / np.tan(Ts / (2 * tau2))),
            Ts / (2 * tau1) * (1 + 1 / np.tan(Ts / (2 * tau2))),
        ]
    )

    u = np.zeros(3, dtype=np.float64)  # [u_f, u_d1, u_d]

    for n in range(nModes):
        u[2] = 0  # Output of phase detector (residual phase error)
        u[0] = 0  # Output of loop filter

        for k in range(Ei.shape[0]):
            u[1] = u[2]

            # Remove estimate of phase error from input symbol
            Eo = Ei[k, n] * np.exp(1j * phaseEst[k, n])

            # Slicer (perform hard decision on symbol)
            if k in pilotInd:
                # phase estimation with pilot symbol
                # Generate phase error signal (also called x_n (Meyer))
                u[2] = np.imag(Eo * np.conj(symbTx[k, n]))
            else:
                # find closest constellation symbol
                decided = np.argmin(np.abs(Eo - constSymb))
                # Generate phase error signal (also called x_n (Meyer))
                u[2] = np.imag(Eo * np.conj(constSymb[decided]))
            # Pass phase error signal in Loop Filter (also called e_n (Meyer))
            u[0] = np.sum(a1b * u)

            # Estimate the phase error for the next symbol
            if k < Ei.shape[0] - 1:
                phaseEst[k + 1, n] = phaseEst[k, n] - Kv * u[0]
    return phaseEst


def viterbi(Ei, N=35, M=4):
    """
    Viterbi & Viterbi carrier phase recovery algorithm.

    Parameters
    ----------
    Ei : np.array
        Input signal.
    N : int, optional
        Size of the moving average window.
    M : int, optional
        M-th power order.

    Returns
    -------
    np.array, float
        Estimated phase error.

    References
    ----------
    [1] S. J. Savory, “Digital coherent optical receivers: Algorithms and subsystems”, IEEE Journal on Selected Topics in Quantum Electronics, vol. 16, nº 5, p. 1164–1179, set. 2010, doi: 10.1109/JSTQE.2010.2044751.
    """
    return (
        -np.unwrap(np.angle(movingAverage(Ei**M, N)) / M, period=2 * np.pi / M, axis=0)
        - np.pi / 4
    )


def fourthPowerFOE(Ei, Fs, plotSpec=False):  # sourcery skip: extract-method
    """
    Estimate the frequency offset (FO) with the 4th-power method.

    Parameters
    ----------
    Ei : np.array
        Input signal.
    Fs : float
        Sampling frequency.
    plotSpec : bool, optional
        Whether to plot the spectrum. Default is False.

    Returns
    -------
    np.array, float
        - The output signal after applying frequency offset correction.
        - The estimated frequency offset.

    References
    ----------
    [1] S. J. Savory, “Digital coherent optical receivers: Algorithms and subsystems”, IEEE Journal on Selected Topics in Quantum Electronics, vol. 16, nº 5, p. 1164–1179, set. 2010, doi: 10.1109/JSTQE.2010.2044751.
    """
    Nfft = Ei.shape[0]

    f = Fs * fftfreq(Nfft)
    f = fftshift(f)

    nModes = Ei.shape[1]
    Eo = Ei.copy()
    t = np.arange(0, Eo.shape[0]) * 1 / Fs
    fo = np.zeros(nModes)
    for n in range(nModes):
        f4 = 10 * np.log10(np.abs(fftshift(fft(Ei[:, n] ** 4))))
        indFO = np.argmax(f4)
        fo[n] = f[indFO] / 4
        Eo[:, n] = Ei[:, n] * np.exp(-1j * 2 * np.pi * fo[n] * t)

    if plotSpec:
        plotSpectrum(f, f4, indFO)
    return Eo, fo


def plotSpectrum(f, f4, indFO):
    plt.figure()
    plt.plot(f, f4, label="$|FFT(s[k]^4)|[dB]$")
    plt.plot(f[indFO], f4[indFO], "x", label="$4f_o$")
    plt.legend()
    plt.xlim(min(f), max(f))
    plt.grid()
