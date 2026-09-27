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

import numpy as np
from numba import njit
from numpy.fft import fft, fftfreq

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


def cpr(sigIn, param=None, symbTx=None):
    """
    Carrier phase recovery function (CPR)

    Parameters
    ----------
    sigIn : complex-valued np.array
        received constellation symbols.
    param : optic.utils.parameter object, optional
        Configuration parameters [default: None].

        - param.alg : CPR algorithm to be used ['bps', 'bpsGPU', 'ddpll', or 'viterbi'] [default: 'bps'].
        - param.shapingFactor : shaping factor, for probabilistic shaped QAM with MB dististribution.[default: 0]
        - param.constType : constellation type ['qam' or 'psk']. [default: 'qam']
        - param.M : constellation order. [default: 4]
        - param.returnPhases : whether to return the estimated phase shifts along with the output signal. [default: False]
        - param.runFOE : whether to run the Mth-power frequency offset estimation and compensation before CPR. [default: True]

        BPS params:

        - param.N : length of BPS the moving average window. [default: 35]
        - param.B : number of BPS test phases. [default: 64]

        DDPLL params:

        - param.tau1 : DDPLL loop filter param. 1. [default: 1/2*pi*10e6]
        - param.tau2 : DDPLL loop filter param. 2. [default: 1/2*pi*10e6]
        - param.Kv : DDPLL loop filter gain. [default: 0.1]
        - param.Ts : symbol period. [default: 1/32e9]
        - param.pilotInd : indexes of pilot-symbol locations.

        Viterbi params:

        - param.N : length of the moving average window. [default: 35]

    symbTx :complex-valued np.array, optional
        Transmitted symbol sequence. [default: None]

    Returns
    -------
    sigOut : complex-valued np.array
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
        symbTx = np.zeros(sigIn.shape)
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
    pilotInd = getattr(param, "pilotInd", np.array([len(sigIn) + 1]))
    runFOE = getattr(param, "runFOE", True)
    returnPhases = getattr(param, "returnPhases", False)

    try:
        sigIn.shape[1]
        input1D = False
    except IndexError:
        sigIn = sigIn.reshape(len(sigIn), 1)
        input1D = True

    # constellation parameters
    constSymb = grayMapping(M, constType)
    px = np.exp(-shapingFactor * np.abs(constSymb) ** 2)
    px = px / np.sum(px)
    constSymb /= np.sqrt(np.sum(np.abs(constSymb) ** 2 * px))

    # 4th power frequency offset estimation/compensation
    if runFOE:
        logg.info(f"Running frequency offset compensation...")
        if constType in ["psk", "apsk"]:
            sigIn, fo = fourthPowerFOE(sigIn, 1 / Ts, M)
        else:
            sigIn, fo = fourthPowerFOE(sigIn, 1 / Ts, 4)
        sigIn = pnorm(sigIn)
        logg.info(f"Estimated frequency offset (MHz): {np.round(fo/1e6, 3)}")

    if alg == "ddpll":
        logg.info(f"Running DDPLL carrier phase recovery...")
        phaseEst = ddpll(sigIn, Ts, Kv, tau1, tau2, constSymb, symbTx, pilotInd)
    elif alg == "bps":
        logg.info(f"Running BPS carrier phase recovery...")
        phaseEst = bps(sigIn, N // 2, constSymb, B)
    elif alg == "bpsGPU":
        try:
            logg.info("Running GPU-based BPS carrier phase recovery...")
            phaseEst = bpsGPU(sigIn, N // 2, constSymb, B)
        except NameError:
            logg.warning("GPU unavailable, switching to CPU processing...")
            phaseEst = bps(sigIn, N // 2, constSymb, B)
    elif alg == "viterbi":
        logg.info(f"Running Viterbi&Viterbi carrier phase recovery...")
        if constType in ["psk"]:
            phaseEst = viterbi(sigIn, N, M) + np.pi / 4
        else:
            phaseEst = viterbi(sigIn, N)
    else:
        logg.error("CPR algorithm incorrectly specified.")
    phaseEst = np.unwrap(4 * phaseEst, axis=0) / 4

    discard = (
        phaseEst.shape[0] // 4
    )  # discard 1/4 of the symbols at the beginning and end
    sigmaPhase = np.mean(np.var(np.diff(phaseEst[discard:-discard, :], axis=0), axis=0))
    logg.info(f"Estimated linewidth: {sigmaPhase/(2 * np.pi* Ts)/1e3:.3f} kHz")

    sigOut = pnorm(sigIn * np.exp(1j * phaseEst))

    if input1D:
        # If input was 1D, return a 1D array
        sigOut = sigOut.flatten()
        phaseEst = phaseEst.flatten()

    return (sigOut, phaseEst) if returnPhases else sigOut


@njit(fastmath=True, cache=True)
def bps(sigIn, N, constSymb, B):
    r"""
    Blind phase search (BPS) algorithm

    Parameters
    ----------
    sigIn : complex-valued np.array
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

    Notes
    -----
    The blind phase search (BPS) algorithm is a feedforward carrier phase estimator
    that works with any constellation. Due to the rotational symmetry of square QAM
    constellations, the phase is only identifiable modulo :math:`\pi/2`, so
    :math:`B` test phases are distributed within this interval,

    .. math::
        \varphi_b = \frac{b}{B}\,\frac{\pi}{2}, \qquad b = 0, 1, \ldots, B-1. \tag{1}

    For each received symbol :math:`y[k]` and test phase, the squared distance
    between the rotated symbol and the closest constellation point is computed,

    .. math::
        d_b[k] = \min_{x \in \mathcal{X}}\left|y[k]e^{j\varphi_b} - x\right|^2. \tag{2}

    To reduce the influence of the noise, the distances are summed over a window of
    :math:`2N+1` consecutive symbols, and the test phase with the smallest sum is
    selected,

    .. math::
        \hat{\varphi}[k] = \arg\min_{\varphi_b} \sum_{n=-N}^{N} d_b[k+n]. \tag{3}

    The phase-corrected symbols are :math:`y[k]e^{j\hat{\varphi}[k]}`. The
    :math:`\pi/2` ambiguity of the estimates is removed later by phase unwrapping
    (see :func:`cpr`).

    References
    ----------
    [1] T. Pfau, S. Hoffmann, e R. Noé, “Hardware-efficient coherent digital receiver concept with feedforward carrier recovery for M-QAM constellations”, Journal of Lightwave Technology, vol. 27, nº 8, p. 989–999, 2009, doi: 10.1109/JLT.2008.2010511.
    """
    nModes = sigIn.shape[1]
    windowLen = 2 * N + 1

    testPhases = np.arange(0, B) * (np.pi / 2) / B  # test phases
    rotations = np.exp(1j * testPhases)

    constReal = constSymb.real.copy()
    constImag = constSymb.imag.copy()

    phaseEst = np.zeros(sigIn.shape, dtype="float")

    zeroPad = np.zeros((N, nModes), dtype="complex")
    x = np.concatenate(
        (zeroPad, sigIn, zeroPad)
    )  # pad start and end of the signal with zeros

    L = x.shape[0]

    for n in range(nModes):
        # circular buffer with the min. distances inside the averaging window
        dmin = np.zeros((windowLen, B), dtype="float")
        sumDmin = np.zeros(B, dtype="float")  # running sum over the window

        for k in range(L):
            slot = k % windowLen
            for indPhase in range(B):
                xRot = x[k, n] * rotations[indPhase]

                # squared distance to the closest constellation symbol
                dminNew = np.inf
                for indSymb in range(constReal.shape[0]):
                    dist = (xRot.real - constReal[indSymb]) ** 2 + (
                        xRot.imag - constImag[indSymb]
                    ) ** 2
                    if dist < dminNew:
                        dminNew = dist

                # replace the oldest distance in the window by the newest one
                sumDmin[indPhase] += dminNew - dmin[slot, indPhase]
                dmin[slot, indPhase] = dminNew

            if k >= 2 * N:
                indRot = np.argmin(sumDmin)
                phaseEst[k - 2 * N, n] = testPhases[indRot]
    return phaseEst


@njit(fastmath=True, cache=True)
def ddpll(sigIn, Ts, Kv, tau1, tau2, constSymb, symbTx, pilotInd):
    r"""
    Decision-directed Phase-locked Loop (DDPLL) algorithm

    Parameters
    ----------
    sigIn : complex-valued np.array
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

    Notes
    -----
    The decision-directed phase-locked loop (DD-PLL) tracks the carrier phase
    symbol by symbol. The received symbol :math:`y[k]` is first corrected by the
    current phase estimate :math:`\hat{\theta}[k]`, and a phase error signal is
    generated by comparing it with the decided symbol :math:`\hat{x}[k]` (or with the
    known pilot symbol, at pilot positions),

    .. math::
        u_d[k] = \mathrm{Im}\left\{y[k]e^{j\hat{\theta}[k]}\,\hat{x}^*[k]\right\}
        \approx |\hat{x}[k]|^2 \sin\left(\theta[k] + \hat{\theta}[k]\right), \tag{1}

    which, for small errors, is proportional to the residual phase error. The error
    signal is smoothed by a second-order (proportional-integral) loop filter,

    .. math::
        u_f[k] = u_f[k-1] + b_1 u_d[k-1] + b_2 u_d[k], \tag{2}

    with :math:`b_{1,2} = \frac{T_s}{2\tau_1}\left[1 \mp
    \cot\left(\frac{T_s}{2\tau_2}\right)\right]`, given by the loop filter
    time constants :math:`\tau_1` and :math:`\tau_2`, and the phase estimate for the next symbol is updated as

    .. math::
        \hat{\theta}[k+1] = \hat{\theta}[k] - K_v u_f[k], \tag{3}

    where :math:`K_v` is the loop gain.

    References
    ----------
    [1] H. Meyer, Digital Communication Receivers: Synchronization, Channel estimation, and Signal Processing, Wiley 1998. Section 5.8 and 5.9.
    """
    nSymbols, nModes = sigIn.shape

    phaseEst = np.zeros((nSymbols, nModes), dtype=np.float64)

    # Loop filter coefficients
    b1 = Ts / (2 * tau1) * (1 - 1 / np.tan(Ts / (2 * tau2)))
    b2 = Ts / (2 * tau1) * (1 + 1 / np.tan(Ts / (2 * tau2)))

    # Boolean mask of pilot-symbol locations (O(1) lookup inside the loop)
    isPilot = np.zeros(nSymbols, dtype=np.bool_)
    for ind in pilotInd:
        if 0 <= ind < nSymbols:
            isPilot[int(ind)] = True

    for n in range(nModes):
        u_d = 0.0  # Output of phase detector (residual phase error)
        u_f = 0.0  # Output of loop filter

        for k in range(nSymbols):
            u_d1 = u_d

            # Remove estimate of phase error from input symbol
            sigOut = sigIn[k, n] * np.exp(1j * phaseEst[k, n])

            # Slicer (perform hard decision on symbol)
            if isPilot[k]:
                # phase estimation with pilot symbol
                decided = symbTx[k, n]
            else:
                # find closest constellation symbol
                decided = constSymb[np.argmin(np.abs(sigOut - constSymb))]

            # Generate phase error signal (also called x_n (Meyer))
            u_d = np.imag(sigOut * np.conj(decided))

            # Pass phase error signal in Loop Filter (also called e_n (Meyer))
            u_f = u_f + b1 * u_d1 + b2 * u_d

            # Estimate the phase error for the next symbol
            if k < nSymbols - 1:
                phaseEst[k + 1, n] = phaseEst[k, n] - Kv * u_f
    return phaseEst


def viterbi(sigIn, N=35, M=4):
    r"""
    Viterbi & Viterbi carrier phase recovery algorithm.

    Parameters
    ----------
    sigIn : np.array
        Input signal.
    N : int, optional
        Size of the moving average window.
    M : int, optional
        M-th power order.

    Returns
    -------
    np.array, float
        Estimated phase error.

    Notes
    -----
    The Viterbi & Viterbi algorithm is a feedforward carrier phase estimator for
    constellations with :math:`M`-fold rotational symmetry. Raising the received
    symbols :math:`y[k] = x[k]e^{j\theta[k]} + n[k]` to the :math:`M`-th power
    removes the phase modulation, since :math:`x^M[k]` has a constant argument (for
    QPSK with points at :math:`\pm\pi/4` and :math:`\pm 3\pi/4`, :math:`x^4 = -1`),
    leaving :math:`M\theta[k]`. The noise is reduced by averaging over a window of
    :math:`N` symbols, and the phase correction returned is

    .. math::
        \hat{\varphi}[k] = -\frac{1}{M}\arg\left\{\sum_{n} y^M[k+n]\right\}
        - \frac{\pi}{4}, \tag{1}

    which is unwrapped with period :math:`2\pi/M` along the sequence. The corrected
    symbols :math:`y[k]e^{j\hat{\varphi}[k]}` are aligned with the constellation up
    to the :math:`2\pi/M` ambiguity inherent to its rotational symmetry. For square
    QAM constellations, the fourth power is used as well, since the average of
    :math:`x^4` is also a negative real number (e.g. :math:`-0.68E_s^2` for 16-QAM).

    References
    ----------
    [1] S. J. Savory, “Digital coherent optical receivers: Algorithms and subsystems”, IEEE Journal on Selected Topics in Quantum Electronics, vol. 16, nº 5, p. 1164–1179, set. 2010, doi: 10.1109/JSTQE.2010.2044751.
    """
    return (
        -np.unwrap(
            np.angle(movingAverage(sigIn**M, N)) / M, period=2 * np.pi / M, axis=0
        )
        - np.pi / 4
    )


def fourthPowerFOE(sigIn, Fs, M=4):
    r"""
    Estimate the frequency offset (FO) with the 4th-power method.

    Parameters
    ----------
    sigIn : np.array
        Input signal.
    Fs : float
        Sampling frequency.
    M : int, optional
        M-th power order. Default is 4.

    Returns
    -------
    np.array, float
        - The output signal after applying frequency offset correction.
        - The estimated frequency offset.

    Notes
    -----
    A frequency offset :math:`\Delta f` between the transmitter laser and the local
    oscillator rotates the received symbols as :math:`y[k] = x[k]e^{j2\pi\Delta f
    kT_s}`. Raising the signal to the :math:`M`-th power (:math:`M = 4` for QPSK and
    square QAM) largely removes the modulation, producing a strong spectral line at
    :math:`M\Delta f`. The frequency offset is estimated from the location of the
    peak of the spectrum of :math:`y^M[k]`,

    .. math::
        \Delta\hat{f} = \frac{1}{M}\arg\max_{f}
        \left|\mathrm{DFT}\left\{y^M[k]\right\}(f)\right|, \tag{1}

    and compensated as

    .. math::
        y_c[k] = y[k]\,e^{-j2\pi\Delta\hat{f}\,kT_s}. \tag{2}

    The estimation range is :math:`|\Delta f| < F_s/(2M)`, and the resolution is
    :math:`F_s/(MN)`, where :math:`F_s` is the sampling rate and :math:`N` the number
    of samples.

    References
    ----------
    [1] S. J. Savory, “Digital coherent optical receivers: Algorithms and subsystems”, IEEE Journal on Selected Topics in Quantum Electronics, vol. 16, nº 5, p. 1164–1179, set. 2010, doi: 10.1109/JSTQE.2010.2044751.
    """
    Nfft = sigIn.shape[0]

    f = Fs * fftfreq(Nfft)
    t = np.arange(0, Nfft) * 1 / Fs

    # spectral peak of the M-th power signal of each mode
    indFO = np.argmax(np.abs(fft(sigIn**M, axis=0)), axis=0)
    fo = f[indFO] / M

    sigOut = sigIn * np.exp(-1j * 2 * np.pi * np.outer(t, fo))

    return sigOut.astype(sigIn.dtype, copy=False), fo
