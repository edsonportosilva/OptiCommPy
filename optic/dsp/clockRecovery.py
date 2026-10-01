"""
===============================================================================
DSP algorithms for clock and timming recovery (:mod:`optic.dsp.clockRecovery`)
===============================================================================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   gardnerTED             -- Calculate the timing error using the Gardner timing error detector.
   gardnerTEDnyquist      -- Modified Gardner timing error detector for Nyquist pulses.
   interpolator           -- Perform cubic interpolation using the Farrow structure.
   gardnerClockRecovery   -- Perform clock recovery using Gardner's algorithm with a loop PI filter.
   calcClockDrift         -- Estimate clock drift from relative time delays fed to the interpolator.
"""

import logging as logg

import numpy as np
from numba import njit
from scipy.signal import find_peaks


@njit
def gardnerTED(x):
    r"""
    Calculate the timing error using the Gardner timing error detector.

    Parameters
    ----------
    x : numpy.np.array
        Input array of size 3 representing a segment of the received signal.

    Returns
    -------
    float
        Gardner timing error detector (TED) value.

    Notes
    -----
    The Gardner timing error detector (TED) works with two samples per symbol. Given
    the samples at the current and the previous symbol instants, :math:`y_k` and
    :math:`y_{k-1}`, and the sample halfway between them, :math:`y_{k-1/2}`, the timing
    error is

    .. math::
        e_k = \mathrm{Re}\left\{y_{k-1/2}^*\left(y_k - y_{k-1}\right)\right\}. \tag{1}

    When a symbol transition occurs, the midpoint sample is close to zero if the
    sampling instants are correct; otherwise, its sign relative to the slope of the
    transition indicates whether the sampling is early or late. The TED does not
    depend on symbol decisions and is insensitive to the carrier phase.
    """
    return np.real(np.conj(x[1]) * (x[2] - x[0]))


@njit
def gardnerTEDnyquist(x):
    r"""
    Modified Gardner timing error detector for Nyquist pulses.

    Parameters
    ----------
    x : numpy.np.array
        Input array of size 3 representing a segment of the received signal.

    Returns
    -------
    float
        Gardner timing error detector (TED) value.

    Notes
    -----
    For Nyquist pulses with small roll-off factors, the S-curve of the classical
    Gardner TED (see :func:`gardnerTED`) vanishes. This modified detector uses the
    power of the samples instead,

    .. math::
        e_k = |y_{k-1/2}|^2\left(|y_{k-1}|^2 - |y_k|^2\right), \tag{1}

    where :math:`y_{k-1}` and :math:`y_k` are consecutive symbol-spaced samples and
    :math:`y_{k-1/2}` is the sample halfway between them.
    """
    return np.abs(x[1]) ** 2 * (np.abs(x[0]) ** 2 - np.abs(x[2]) ** 2)


@njit
def interpolator(x, t):
    r"""
    Perform cubic interpolation using the Farrow structure.

    Parameters
    ----------
    x : numpy.np.array
        Input array of size 4 representing the values for cubic interpolation.
    t : float
        Interpolation parameter.

    Returns
    -------
    y : float
        Interpolated signal value.

    Notes
    -----
    The interpolator computes the value of the signal at a fractional position
    :math:`t \in [-1, 1]` with respect to the sample :math:`x[2]`, by cubic Lagrange
    interpolation of the four samples :math:`x[0], \ldots, x[3]`, located at the
    positions :math:`-2, -1, 0` and :math:`1`,

    .. math::
        y(t) = \sum_{i=0}^{3} x[i]\,\ell_i(t), \qquad
        \ell_i(t) = \prod_{j \neq i}\frac{t - t_j}{t_i - t_j}, \tag{1}

    which is implemented with the Farrow structure, i.e. with the polynomial
    coefficients

    .. math::
        :nowrap:

        \begin{equation}
            \begin{aligned}
                \ell_0(t) &= -\tfrac{1}{6}t^3 + \tfrac{1}{6}t, &
                \ell_1(t) &= \tfrac{1}{2}t^3 + \tfrac{1}{2}t^2 - t, \\
                \ell_2(t) &= -\tfrac{1}{2}t^3 - t^2 + \tfrac{1}{2}t + 1, &
                \ell_3(t) &= \tfrac{1}{6}t^3 + \tfrac{1}{2}t^2 + \tfrac{1}{3}t.
            \end{aligned} \tag{2}
        \end{equation}

    In this form, the fractional delay :math:`t` can be changed at every output
    sample without recomputing filter coefficients.
    """
    return (
        x[0] * (-1 / 6 * t**3 + 1 / 6 * t)
        + x[1] * (1 / 2 * t**3 + 1 / 2 * t**2 - 1 * t)
        + x[2] * (-1 / 2 * t**3 - 1 * t**2 + 1 / 2 * t + 1)
        + x[3] * (1 / 6 * t**3 + 1 / 2 * t**2 + 1 / 3 * t)
    )


def gardnerClockRecovery(sigIn, param=None):
    r"""
    Perform clock recovery using Gardner's algorithm with a loop PI filter.

    Parameters
    ----------
    sigIn : numpy.np.array
        Input array representing the received signal.
    param : core.parameter
        Clock recovery parameters:

            - param.kp : Proportional gain for the loop filter. [default: 1e-3]
            - param.ki : Integral gain for the loop filter. [default: 1e-6]
            - param.isNyquist : is the pulse shape a Nyquist pulse? [default: True]
            - param.returnTiming : return estimated timing values. [default: False]
            - param.lpad : length of zero padding at the end of the input vector. [default: 1]
            - param.maxPPM : maximum clock rate expected deviation in PPM. [default: 500]

    Returns
    -------
    tuple
        Tuple containing the recovered signal (sigOut) and the timing values.

    Notes
    -----
    The clock recovery is a feedback loop composed of an interpolator, a timing
    error detector (TED), a loop filter and a numerically controlled oscillator
    (NCO). At each output sample, the interpolator computes the signal at the
    fractional delay :math:`\tau` given by the NCO (see :func:`interpolator`). Once
    per symbol, the TED computes the timing error :math:`e_k` (see
    :func:`gardnerTED` and :func:`gardnerTEDnyquist`), which is filtered by a
    proportional-integral (PI) loop filter,

    .. math::
        v_k = k_p e_k + k_i\sum_{l \le k} e_l, \tag{1}

    and the NCO updates the fractional delay as

    .. math::
        \tau \leftarrow \tau - v_k. \tag{2}

    Whenever :math:`\tau` crosses :math:`\pm 1`, it is wrapped back into
    :math:`[-1, 1]` and one input sample is skipped or repeated, which allows the
    loop to track a clock frequency offset (drift) between the transmitter and the
    receiver. The integral term removes the steady-state timing error caused by such
    a drift.
    """
    # Check and set default values for input parameters
    kp = getattr(param, "kp", 1e-3)
    ki = getattr(param, "ki", 1e-6)
    isNyquist = getattr(param, "isNyquist", True)
    returnTiming = getattr(param, "returnTiming", False)
    lpad = getattr(param, "lpad", 1)
    maxPPM = getattr(param, "maxPPM", 500)

    try:
        sigIn.shape[1]
        input1D = False
    except IndexError:
        input1D = True
        sigIn = sigIn.reshape(len(sigIn), 1)

    sigIn = np.pad(sigIn, ((0, lpad), (0, 0)))

    # Initializing variables:
    nModes = sigIn.shape[1]
    nSamples = sigIn.shape[0]

    # Initiate output vector according with a maximum estimate of clock deviation
    sigOut = np.zeros((int((1 - maxPPM / 1e6) * nSamples), nModes), dtype=np.complex64)

    Ln = sigOut.shape[0]

    t_nco_values = np.zeros(sigOut.shape, dtype=np.float64)
    last_n = 0
    logg.info(f"Running clock recovery...")

    for indMode in range(nModes):
        intPart = 0
        t_nco = 0

        n = 2
        m = 2

        while n < Ln - 1 and m < nSamples - 2:
            sigOut[n, indMode] = interpolator(sigIn[m - 2 : m + 2, indMode], t_nco)

            if n % 2 == 0:
                if isNyquist:
                    ted = gardnerTEDnyquist(sigOut[n - 2 : n + 1, indMode])
                else:
                    ted = gardnerTED(sigOut[n - 2 : n + 1, indMode])

                # Loop PI Filter:
                intPart = ki * ted + intPart
                propPart = kp * ted
                loopFilterOut = propPart + intPart

                t_nco -= loopFilterOut

            # NCO clock gap
            if t_nco > 1:
                t_nco -= 1  # shift t_nco backward by one sample
                n -= 1  # shift index of next vector for TED calculation backward by one sample
            elif t_nco < -1:
                t_nco += 1  # shift t_nco foward by one sample
                n += 2  # shift index of next vector for TED calculation forward by two samples
                m += 1  # shift index of next interpolating vector forward by one sample
            else:
                n += 1
                m += 1

            t_nco_values[n, indMode] = t_nco

        if n > last_n:
            last_n = n

        logg.info(
            f"Estimated clock drift mode {indMode}: {calcClockDrift(t_nco_values[:, indMode])[0]:.2f} ppm"
        )

    sigOut = sigOut[0:last_n, :]

    if input1D:
        # If input was 1D, return a 1D array
        sigOut = sigOut.flatten()

    if returnTiming:
        return sigOut, t_nco_values
    else:
        return sigOut


def calcClockDrift(t_nco_values):
    """
    Calculate the clock drift in parts per million (ppm) from t_nco values.

    Parameters
    ----------
    t_nco_values : np.array
        An array containing the relative time delay values provided to the NCO.

    Returns
    -------
    float
        The clock deviation in parts per million (ppm).
    """
    try:
        t_nco_values.shape[1]
        input1D = False
    except IndexError:
        t_nco_values = t_nco_values.reshape(len(t_nco_values), 1)
        input1D = True

    timingError = t_nco_values - np.mean(t_nco_values)

    t = np.arange(timingError.shape[0])

    nModes = t_nco_values.shape[1]
    ppm = np.zeros(nModes)

    for indMode in range(nModes):
        peaks, _ = find_peaks(np.abs(np.diff(timingError[:, indMode])), height=0.5)
        mean_period = np.mean(np.diff(t[peaks]))  # mean period of t_nco_values
        fo = 1 / mean_period
        ppm[indMode] = np.sign(np.mean(t_nco_values)) * fo * 1e6

    if input1D:
        # If input was 1D, return a 1D array
        ppm = ppm.flatten()

    return ppm
