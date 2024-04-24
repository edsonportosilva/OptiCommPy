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
   calcClockDrift         -- Estimate clock drift from relative time delays fed to the interpolator
"""
import logging as logg

import numpy as np
from numba import njit
from scipy.signal import find_peaks


@njit
def gardnerTED(x):
    """
    Calculate the timing error using the Gardner timing error detector.

    Parameters
    ----------
    x : numpy.np.array
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
    x : numpy.np.array
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
    x : numpy.np.array
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
    Ei : numpy.np.array
        Input array representing the received signal.
    param : core.parameter
        Clock recovery parameters:
            - kp : Proportional gain for the loop filter. Default is 1e-3.

            - ki : Integral gain for the loop filter. Default is 1e-6.

            - isNyquist: is the pulse shape a Nyquist pulse? Default is True.

            - returnTiming: return estimated timing values. Default is False.

            - lpad: length of zero padding at the end of the input vector. Default is 1.

            - maxPPM: maximum clock rate expected deviation in PPM. Default is 500.

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
    lpad = getattr(param, "lpad", 1)
    maxPPM = getattr(param, "maxPPM", 500)

    try:
        Ei.shape[1]
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)

    Ei = np.pad(Ei, ((0, lpad), (0, 0)))

    # Initializing variables:
    nModes = Ei.shape[1]
    nSamples = Ei.shape[0]

    # Initiate output vector according with a maximum estimate of clock deviation
    Eo = np.zeros((int((1 - maxPPM / 1e6) * nSamples), nModes), dtype=np.complex64)

    Ln = Eo.shape[0]

    t_nco_values = np.zeros(Eo.shape, dtype=np.float64)
    last_n = 0
    logg.info(f"Running clock recovery...")
    
    for indMode in range(nModes):
        intPart = 0
        t_nco = 0

        n = 2
        m = 2

        while n < Ln - 1 and m < nSamples - 2:
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
        
        logg.info(f"Estimated clock drift mode {indMode}: {calcClockDrift(t_nco_values[:, indMode])[0]:.2f} ppm")

    Eo = Eo[0:last_n, :]   

    if returnTiming:
        return Eo, t_nco_values
    else:
        return Eo


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
    except IndexError:
        t_nco_values = t_nco_values.reshape(len(t_nco_values), 1)

    timingError = t_nco_values - np.mean(t_nco_values)

    t = np.arange(timingError.shape[0])

    nModes = t_nco_values.shape[1]
    ppm = np.zeros(nModes)

    for indMode in range(nModes):
        peaks, _ = find_peaks(np.abs(np.diff(timingError[:,indMode])), height=0.5)
        mean_period = np.mean(np.diff(t[peaks])) # mean period of t_nco_values
        fo = 1/mean_period
        ppm[indMode] = np.sign(np.mean(t_nco_values))*fo*1e6

    return ppm
