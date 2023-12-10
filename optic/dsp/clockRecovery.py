"""
===============================================================================
DSP algorithms for clock and timming recovery (:mod:`optic.dsp.clockRecovery`)
===============================================================================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   gardnerTED             -- Calculate the timing error using the Gardner timing error detector
   gardnerTEDnyquist      -- Modified Gardner timing error detector for Nyquist pulses
   interpolator           -- Perform cubic interpolation using the Farrow structure
   gardnerClockRecovery   -- Perform clock recovery using Gardner's algorithm with a loop PI filter   
"""
import numpy as np
from numba import njit


@njit
def gardnerTED(x):
    """
    Calculate the timing error using the Gardner timing error detector.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of size 3 representing a segment of the received signal.

    Returns
    -------
    float
        Gardner timing error detector (TED) value.
    """
    return np.real(np.conj(x[1]) * (x[2] - x[0]))


@njit
def gardnerTEDnyquist(x):
    """
    Modified Gardner timing error detector for Nyquist pulses.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of size 3 representing a segment of the received signal.

    Returns
    -------
    float
        Gardner timing error detector (TED) value.
    """
    return np.abs(x[1]) ** 2 * (np.abs(x[0]) ** 2 - np.abs(x[2]) ** 2)


@njit
def interpolator(x, t):
    """
    Perform cubic interpolation using the Farrow structure.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of size 4 representing the values for cubic interpolation.
    t : float
        Interpolation parameter.

    Returns
    -------
    y : float
        Interpolated signal value.
    """
    return (
        x[0] * (-1 / 6 * t**3 + 1 / 6 * t)
        + x[1] * (1 / 2 * t**3 + 1 / 2 * t**2 - 1 * t)
        + x[2] * (-1 / 2 * t**3 - 1 * t**2 + 1 / 2 * t + 1)
        + x[3] * (1 / 6 * t**3 + 1 / 2 * t**2 + 1 / 3 * t)
    )


def gardnerClockRecovery(Ei, param=None):
    """
    Perform clock recovery using Gardner's algorithm with a loop PI filter.

    Parameters
    ----------
    Ei : numpy.ndarray
        Input array representing the received signal.
    param : core.parameter
        Resampling parameters:
            - kp : Proportional gain for the loop filter. Default is 1e-3.

            - ki : Integral gain for the loop filter. Default is 1e-6.

            - isNyquist: is the pulse shape a Nyquist pulse? Default is True.

            - returnTiming: return estimated timing values. Default is False.

    Returns
    -------
    tuple
        Tuple containing the recovered signal (Eo) and the timing values.
    """
    # Check and set default values for input parameters
    kp = getattr(param, "kp", 1e-3)
    ki = getattr(param, "ki", 1e-6)
    isNyquist = getattr(param, "isNyquist", True)
    returnTiming = getattr(param, "returnTiming", False)

    try:
        Ei.shape[1]
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)

    # Initializing variables:
    nModes = Ei.shape[1]

    Eo = Ei.copy()
    Ei = np.pad(Ei, ((0, 2)), "constant")

    L = Ei.shape[0]

    timing_values = []

    for indMode in range(nModes):
        intPart = 0
        t_nco = 0
        timing_values_mode = []

        n = 2
        m = 2

        while n < L - 1 and m < L - 2:
            Eo[n, indMode] = interpolator(Ei[m - 2 : m + 2, indMode], t_nco)

            if n % 2 == 0:
                if isNyquist:
                    ted = gardnerTEDnyquist(Eo[n - 2 : n + 1, indMode])
                else:
                    ted = gardnerTED(Eo[n - 2 : n + 1, indMode])

                # Loop PI Filter:
                intPart = ki * ted + intPart
                propPart = kp * ted
                loopFilterOut = propPart + intPart

                t_nco -= loopFilterOut

            n += 1
            m += 1

            # NCO
            if t_nco > 0:
                t_nco -= 1
                m -= 1
                n -= 2
            elif t_nco < -1:
                t_nco += 1
                m += 1
                n += 2

            timing_values_mode.append(t_nco)

        timing_values.append(timing_values_mode)

    Eo = Eo[0:n, :]

    if returnTiming:
        return Eo, np.asarray(timing_values).astype("float32").T
    else:
        return Eo
