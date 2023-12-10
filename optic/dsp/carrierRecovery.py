"""
==========================================================================================
DSP algorithms for carrier phase and frequency recovery (:mod:`optic.dsp.carrierRecovery`)
==========================================================================================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   bps            -- Blind phase search (BPS) carrier phase recovery algorithm
   ddpll          -- Decision-directed phase-locked loop (DD-PLL) carrier phase recovery algorithm
   viterbi        -- Viterbi & Viterbi carrier phase recovery algorithm
   fourthPowerFOE -- Frequency offset (FO) estimation and compensation with the 4th-power method
   cpr            -- General function to call and configure any of the CPR algorithms in this module   
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numpy.fft import fft, fftfreq, fftshift

from optic.dsp.core import pnorm, movingAverage
from optic.comm.modulation import GrayMapping


def cpr(Ei, param=None, symbTx=None):
    """
    Carrier phase recovery function (CPR)

    Parameters
    ----------
    Ei : complex-valued ndarray
        received constellation symbols.
    param : core.param object, optional
        configuration parameters. The default is [].

        - param.alg: CPR algorithm to be used ['bps', 'ddpll', or 'viterbi']

        BPS params:

        - param.M: constellation order. The default is 4.

        - param.N: length of BPS the moving average window. The default is 35.

        - param.B: number of BPS test phases. The default is 64.

        DDPLL params:

        - param.tau1: DDPLL loop filter param. 1. The default is 1/2*pi*10e6.

        - param.tau2: DDPLL loop filter param. 2. The default is 1/2*pi*10e6.

        - param.Kv: DDPLL loop filter gain. The default is 0.1.

        - param.Ts: symbol period. The default is 1/32e9.

        - param.pilotInd: indexes of pilot-symbol locations.

        Viterbi params:

        - param.N: length of the moving average window. The default is 35.

    symbTx :complex-valued ndarray, optional
        Transmitted symbol sequence. The default is [].

    Raises
    ------
    ValueError
        Error is generated if the CPR algorithm is not correctly
        passed.

    Returns
    -------
    Eo : complex-valued ndarray
        Phase-compensated signal.
    θ : real-valued ndarray
        Time-varying estimated phase-shifts.

    """
    if symbTx is None:
        symbTx = np.zeros(Ei.shape)
    if param is None:
        param = []

    # check input parameters
    alg = getattr(param, "alg", "bps")
    M = getattr(param, "M", 4)
    constType = getattr(param, "constType", "qam")
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
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)

    # constellation parameters
    constSymb = GrayMapping(M, constType)
    constSymb = pnorm(constSymb)

    # 4th power frequency offset estimation/compensation
    Ei, _ = fourthPowerFOE(Ei, 1 / Ts)
    Ei = pnorm(Ei)

    if alg == "ddpll":
        θ = ddpll(Ei, Ts, Kv, tau1, tau2, constSymb, symbTx, pilotInd)
    elif alg == "bps":
        θ = bps(Ei, N // 2, constSymb, B)
    elif alg == "viterbi":
        θ = viterbi(Ei, N)
    else:
        raise ValueError("CPR algorithm incorrectly specified.")
    θ = np.unwrap(4 * θ, axis=0) / 4

    Eo = pnorm(Ei * np.exp(1j * θ))

    if Eo.shape[1] == 1:
        Eo = Eo[:]
        θ = θ[:]

    return (Eo, θ) if returnPhases else Eo


@njit
def bps(Ei, N, constSymb, B):
    """
    Blind phase search (BPS) algorithm

    Parameters
    ----------
    Ei : complex-valued ndarray
        Received constellation symbols.
    N : int
        Half of the 2*N+1 average window.
    constSymb : complex-valued ndarray
        Complex-valued constellation.
    B : int
        number of test phases.

    Returns
    -------
    θ : real-valued ndarray
        Time-varying estimated phase-shifts.

    """
    nModes = Ei.shape[1]

    ϕ_test = np.arange(0, B) * (np.pi / 2) / B  # test phases

    θ = np.zeros(Ei.shape, dtype="float")

    zeroPad = np.zeros((N, nModes), dtype="complex")
    x = np.concatenate(
        (zeroPad, Ei, zeroPad)
    )  # pad start and end of the signal with zeros

    L = x.shape[0]

    for n in range(nModes):
        dist = np.zeros((B, constSymb.shape[0]), dtype="float")
        dmin = np.zeros((B, 2 * N + 1), dtype="float")

        for k in range(L):
            for indPhase, ϕ in enumerate(ϕ_test):
                dist[indPhase, :] = np.abs(x[k, n] * np.exp(1j * ϕ) - constSymb) ** 2
                dmin[indPhase, -1] = np.min(dist[indPhase, :])
            if k >= 2 * N:
                sumDmin = np.sum(dmin, axis=1)
                indRot = np.argmin(sumDmin)
                θ[k - 2 * N, n] = ϕ_test[indRot]
            dmin = np.roll(dmin, -1)
    return θ


@njit
def ddpll(Ei, Ts, Kv, tau1, tau2, constSymb, symbTx, pilotInd):
    """
    Decision-directed Phase-locked Loop (DDPLL) algorithm

    Parameters
    ----------
    Ei : complex-valued ndarray
        Received constellation symbols.
    Ts : float scalar
        Symbol period.
    Kv : float scalar
        Loop filter gain.
    tau1 : float scalar
        Loop filter parameter 1.
    tau2 : float scalar
        Loop filter parameter 2.
    constSymb : complex-valued ndarray
        Complex-valued ideal constellation symbols.
    symbTx : complex-valued ndarray
        Transmitted symbol sequence.
    pilotInd : int ndarray
        Indexes of pilot-symbol locations.

    Returns
    -------
    θ : real-valued ndarray
        Time-varying estimated phase-shifts.

    References
    ----------
    [1] H. Meyer, Digital Communication Receivers: Synchronization, Channel
    estimation, and Signal Processing, Wiley 1998. Section 5.8 and 5.9.

    """
    nSymbols, nModes = Ei.shape

    θ = np.zeros((nSymbols, nModes), dtype=np.float64)

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
            Eo = Ei[k, n] * np.exp(1j * θ[k, n])

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
                θ[k + 1, n] = θ[k, n] - Kv * u[0]
    return θ


def viterbi(Ei, N=35, M=4):
    """
    Viterbi & Viterbi carrier phase recovery algorithm.

    Parameters
    ----------
    Ei : ndarray
        Input signal.
    N : int, optional
        Size of the moving average window.
    M : int, optional
        M-th power order.

    Returns
    -------
    ndarray, float
        Estimated phase error.
    """
    return (
        -np.unwrap(
            np.angle(movingAverage(Ei**M, N)) / M, period=2 * np.pi / M, axis=0
        )
        - np.pi / 4
    )


def fourthPowerFOE(Ei, Fs, plotSpec=False):  # sourcery skip: extract-method
    """
    Estimate the frequency offset (FO) with the 4th-power method.

    Parameters
    ----------
    Ei : ndarray
        Input signal.
    Fs : float
        Sampling frequency.
    plotSpec : bool, optional
        Whether to plot the spectrum. Default is False.

    Returns
    -------
    ndarray, float
        - The output signal after applying frequency offset correction.
        - The estimated frequency offset.

    """
    Nfft = Ei.shape[0]

    f = Fs * fftfreq(Nfft)
    f = fftshift(f)

    nModes = Ei.shape[1]
    Eo = Ei.copy()
    t = np.arange(0, Eo.shape[0]) * 1 / Fs

    for n in range(nModes):
        f4 = 10 * np.log10(np.abs(fftshift(fft(Ei[:, n] ** 4))))
        indFO = np.argmax(f4)
        fo = f[indFO] / 4
        Eo[:, n] = Ei[:, n] * np.exp(-1j * 2 * np.pi * fo * t)

    if plotSpec:
        plotSpectrum(f, f4, indFO)
    return Eo, f[indFO] / 4


def plotSpectrum(f, f4, indFO):
    plt.figure()
    plt.plot(f, f4, label="$|FFT(s[k]^4)|[dB]$")
    plt.plot(f[indFO], f4[indFO], "x", label="$4f_o$")
    plt.legend()
    plt.xlim(min(f), max(f))
    plt.grid()
