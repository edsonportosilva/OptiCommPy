"""
==================================================
Core digital signal processing utilities (:mod:`optic.dsp.core`)
==================================================

.. autosummary::
   :toctree: generated/

   sigPow                 -- Calculate the average power of x
   signal_power           -- Calculate the total average power of x
   firFilter              -- Perform FIR filtering and compensate for filter delay
   pulseShape             -- Generate a pulse shaping filter
   lowPassFIR             -- Calculate FIR coefficients of a lowpass filter
   decimate               -- Decimate signal
   resample               -- Signal resampling
   symbolSync             -- Synchronizer delayed sequences of symbols
   finddelay              -- Estimate the delay between sequences of symbols
   pnorm                  -- Normalize the average power of each componennt of x
"""
"""Digital signal processing utilities."""
import numpy as np
from numba import njit
from scipy import signal


@njit
def sigPow(x):
    """
    Calculate the average power of x per mode.

    Parameters
    ----------
    x : np.array
        Signal.

    Returns
    -------
    scalar
        Average power of x: P = mean(abs(x)**2).

    """
    return np.mean(np.abs(x) ** 2)


def signal_power(x):
    """
    Calculate the total power of x.

    Parameters
    ----------
    x : np.array
        Signal.

    Returns
    -------
    scalar
        Total power of x: P = sum(abs(x)**2).

    """
    return np.sum(np.mean(x * np.conj(x), axis=0).real)


def firFilter(h, x):
    """
    Perform FIR filtering and compensate for filter delay.

    Parameters
    ----------
    h : np.array
        Coefficients of the FIR filter (impulse response, symmetric).
    x : np.array
        Input signal.
    prec: cp.dtype
        Size of the complex representation.

    Returns
    -------
    y : np.array
        Output (filtered) signal.
    """
    try:
        x.shape[1]
    except IndexError:
        x = x.reshape(len(x), 1)
    y = x.copy()
    nModes = x.shape[1]

    for n in range(nModes):
        y[:, n] = np.convolve(x[:, n], h, mode="same")
    if y.shape[1] == 1:
        y = y[:, 0]
    return y


import numpy as np


@njit
def rrcFilterTaps(t, alpha, Ts):
    """
    Generate Root-Raised Cosine (RRC) filter coefficients.

    Parameters:
    -----------
    t : array-like
        Time values.
    alpha : float
        RRC roll-off factor.
    Ts : float
        Symbol period.

    Returns:
    --------
    coeffs : ndarray
        RRC filter coefficients.

    References:
    -----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.
    """
    coeffs = np.zeros(len(t), dtype=np.float64)

    for i, t_i in enumerate(t):
        t_abs = abs(t_i)
        if t_i == 0:
            coeffs[i] = (1 / Ts) * (1 + alpha * (4 / np.pi - 1))
        elif t_abs == Ts / (4 * alpha):
            term1 = (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
            term2 = (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            coeffs[i] = (alpha / (Ts * np.sqrt(2))) * (term1 + term2)
        else:
            t1 = np.pi * t_i / Ts
            t2 = 4 * alpha * t_i / Ts
            coeffs[i] = (
                (1 / Ts)
                * (
                    np.sin(t1 * (1 - alpha))
                    + 4 * alpha * t_i / Ts * np.cos(t1 * (1 + alpha))
                )
                / (np.pi * t_i * (1 - t2**2))
            )

    return coeffs


@njit
def rrcFilterTaps(t, alpha, Ts):
    """
    Generate Raised Cosine (RC) filter coefficients.

    Parameters:
    -----------
    t : array-like
        Time values.
    alpha : float
        RC roll-off factor.
    Ts : float
        Symbol period.

    Returns:
    --------
    coeffs : ndarray
        RC filter coefficients.

    References:
    -----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.
    """
    coeffs = np.zeros(len(t), dtype=np.float64)
    π = np.pi

    for i, t_i in enumerate(t):
        if t_i == 0:
            coeffs[i] = 1 / (4 * Ts) * np.sinc(1 / (2 * alpha * Ts))
        else:
            coeffs[i] = (
                (1 / Ts)
                * np.sinc(t_i / Ts)
                * np.cos(π * alpha * t_i / Ts)
                / (1 - 4 * alpha**2 * t_i**2 / Ts**2)
            )

    return coeffs


def pulseShape(pulseType, SpS=2, N=1024, alpha=0.1, Ts=1):
    """
    Generate a pulse shaping filter.

    Parameters
    ----------
    pulseType : string ('rect','nrz','rrc')
        type of pulse shaping filter.
    SpS : int, optional
        Number of samples per symbol of input signal. The default is 2.
    N : int, optional
        Number of filter coefficients. The default is 1024.
    alpha : float, optional
        Rolloff of RRC filter. The default is 0.1.
    Ts : float, optional
        Symbol period in seconds. The default is 1.

    Returns
    -------
    filterCoeffs : np.array
        Array of filter coefficients (normalized).

    """
    fa = (1 / Ts) * SpS

    if pulseType == "rect":
        filterCoeffs = np.concatenate(
            (np.zeros(int(SpS / 2)), np.ones(SpS), np.zeros(int(SpS / 2)))
        )
    elif pulseType == "nrz":
        t = np.linspace(-2, 2, SpS)
        Te = 1
        filterCoeffs = np.convolve(
            np.ones(SpS),
            2 / (np.sqrt(np.pi) * Te) * np.exp(-(t**2) / Te),
            mode="full",
        )
    elif pulseType == "rrc":
        t = np.linspace(-N // 2, N // 2, N) * (1 / fa)
        filterCoeffs = rrcFilterTaps(t, alpha, Ts)

    elif pulseType == "rc":
        t = np.linspace(-N // 2, N // 2, N) * (1 / fa)
        filterCoeffs = rrcFilterTaps(N, alpha, Ts)

    filterCoeffs = filterCoeffs / np.sqrt(np.sum(filterCoeffs**2))

    return filterCoeffs


def sincInterp(x, fa):
    fa_sinc = 32 * fa
    Ta_sinc = 1 / fa_sinc
    Ta = 1 / fa
    t = np.arange(0, x.size * 32) * Ta_sinc

    plt.figure()
    y = upsample(x, 32)
    y[y == 0] = np.nan
    plt.plot(t, y.real, "ko", label="x[k]")

    x_sum = 0
    for k in range(x.size):
        xk_interp = x[k] * np.sinc((t - k * Ta) / Ta)
        x_sum += xk_interp
        plt.plot(t, xk_interp)
    plt.legend(loc="upper right")
    plt.xlim(min(t), max(t))
    plt.grid()

    return x_sum, t


def lowPassFIR(fc, fa, N, typeF="rect"):
    """
    Calculate FIR coefficients of a lowpass filter.

    Parameters
    ----------
    fc : float
        Cutoff frequency.
    fa : float
        Sampling frequency.
    N : int
        Number of filter coefficients.
    typeF : string, optional
        Type of response ('rect', 'gauss'). The default is "rect".

    Returns
    -------
    h : np.array
        Filter coefficients.

    """
    fu = fc / fa
    d = (N - 1) / 2
    n = np.arange(0, N)

    # calculate filter coefficients
    if typeF == "rect":
        h = (2 * fu) * np.sinc(2 * fu * (n - d))
    elif typeF == "gauss":
        h = (
            np.sqrt(2 * np.pi / np.log(2))
            * fu
            * np.exp(-(2 / np.log(2)) * (np.pi * fu * (n - d)) ** 2)
        )
    return h


def upsample(x, factor):
    """
    Upsample a signal by inserting zeros between samples.

    Parameters
    ----------
    x : array-like
        Input signal to upsample.
    factor : int
        Upsampling factor. The signal will be upsampled by inserting
        `factor - 1` zeros between each original sample.

    Returns
    -------
    xUp : array-like
        The upsampled signal with zeros inserted between samples.

    Notes
    -----
    This function inserts zeros between the samples of the input signal to
    increase its sampling rate. The upsampling factor determines how many
    zeros are inserted between each original sample.

    If the input signal is a 2D array, the upsampling is performed
    column-wise.
    """
    try:
        xUp = np.zeros((factor * x.shape[0], x.shape[1]), dtype=x.dtype)
        xUp[0::factor, :] = x
    except IndexError:
        xUp = np.zeros(factor * x.shape[0], dtype=x.dtype)
        xUp[0::factor] = x

    return xUp


def decimate(Ei, param):
    """
    Decimate signal.

    Parameters
    ----------
    Ei : np.array
        Input signal.
    param : core.parameter
        Decimation parameters:

        - param.SpS_in  : samples per symbol of the input signal.

        - param.SpS_out : samples per symbol of the output signal.

    Returns
    -------
    Eo : np.array
        Decimated signal.

    """
    decFactor = int(param.SpS_in / param.SpS_out)

    # simple timing recovery
    sampDelay = np.zeros(Ei.shape[1])

    # finds best sampling instant
    # (maximum variance sampling time)
    for k in range(Ei.shape[1]):
        a = Ei[:, k].reshape(Ei.shape[0], 1)
        varVector = np.var(a.reshape(-1, param.SpS_in), axis=0)
        sampDelay[k] = np.where(varVector == np.amax(varVector))[0][0]
    # downsampling
    Eo = Ei[::decFactor, :].copy()

    for k in range(Ei.shape[1]):
        Ei[:, k] = np.roll(Ei[:, k], -int(sampDelay[k]))
        Eo[:, k] = Ei[0::decFactor, k]
    return Eo


def resample(Ei, param):
    """
    Resample signal to a given sampling rate.

    Parameters
    ----------
    Ei : ndarray
        Input signal.
    param : core.parameter
        Resampling parameters:
            param.Rs      : symbol rate of the signal
            param.SpS_in  : samples per symbol of the input signal.
            param.SpS_out : samples per symbol of the output signal.

    Returns
    -------
    Eo : ndarray
        Resampled signal.

    """
    try:
        Ei.shape[1]
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)
    nModes = Ei.shape[1]
    inFs = param.SpS_in * param.Rs
    outFs = param.SpS_out * param.Rs

    tin = np.arange(0, Ei.shape[0]) * (1 / inFs)
    tout = np.arange(0, Ei.shape[0] * (1 / inFs), 1 / outFs)

    Eo = np.zeros((len(tout), Ei.shape[1]), dtype="complex")

    # Anti-aliasing filters:
    N = 2048
    hi = lowPassFIR(inFs / 2, inFs, N, typeF="rect")
    ho = lowPassFIR(outFs / 2, outFs, N, typeF="rect")

    Ei = firFilter(hi, Ei)

    if nModes == 1:
        Ei = Ei.reshape(len(Ei), 1)
    for k in range(nModes):
        Eo[:, k] = np.interp(tout, tin, Ei[:, k])
    Eo = firFilter(ho, Eo)

    return Eo


def symbolSync(rx, tx, SpS, mode="amp"):
    """
    Symbol synchronizer.

    Parameters
    ----------
    rx : np.array
        Received symbol sequence.
    tx : np.array
        Transmitted symbol sequence.
    SpS : int
        Samples per symbol of input signals.

    Returns
    -------
    tx : ndarray
        Transmitted sequence synchronized to rx.

    """
    nModes = rx.shape[1]

    rx = rx[0::SpS, :]

    # calculate time delay
    delay = np.zeros(nModes)

    corrMatrix = np.zeros((nModes, nModes))

    if mode == "amp":
        for n in range(nModes):
            for m in range(nModes):
                corrMatrix[m, n] = np.max(
                    np.abs(signal.correlate(np.abs(tx[:, m]), np.abs(rx[:, n])))
                )
        swap = np.argmax(corrMatrix, axis=0)

        tx = tx[:, swap]

        for k in range(nModes):
            delay[k] = finddelay(np.abs(tx[:, k]), np.abs(rx[:, k]))
    elif mode == "real":
        for n in range(nModes):
            for m in range(nModes):
                corrMatrix[m, n] = np.max(
                    np.abs(signal.correlate(np.real(tx[:, m]), np.real(rx[:, n])))
                )
        swap = np.argmax(corrMatrix, axis=0)

        tx = tx[:, swap]

        for k in range(nModes):
            delay[k] = finddelay(np.real(tx[:, k]), np.real(rx[:, k]))

    # compensate time delay
    for k in range(nModes):
        tx[:, k] = np.roll(tx[:, k], -int(delay[k]))
    return tx


def finddelay(x, y):
    """
    Find delay between x and y.

    Parameters
    ----------
    x : np.array
        Signal 1.
    y : np.array
        Signal 2.

    Returns
    -------
    d : int
        Delay between x and y, in samples.

    """
    return np.argmax(signal.correlate(x, y)) - x.shape[0] + 1


@njit
def pnorm(x):
    """
    Normalize the average power of each componennt of x.

    Parameters
    ----------
    x : np.array
        Signal.

    Returns
    -------
    np.array
        Signal x with each component normalized in power.

    """
    return x / np.sqrt(np.mean(x * np.conj(x)).real)


@njit
def gaussianComplexNoise(shapeOut, σ2=1.0):
    """
    Generate complex circular Gaussian noise.

    Parameters:
    -----------
    shapeOut : tuple of int
        Shape of ndarray to be generated.
    σ2 : float, optional
        Variance of the noise (default is 1).

    Returns:
    --------
    noise : ndarray
        Generated complex circular Gaussian noise.
    """
    return np.random.normal(0, np.sqrt(σ2 / 2), shapeOut) + 1j * np.random.normal(
        0, np.sqrt(σ2 / 2), shapeOut
    )


@njit
def gaussianNoise(shapeOut, σ2=1.0):
    """
    Generate Gaussian noise.

    Parameters:
    -----------
    shapeOut : tuple of int
        Shape of ndarray to be generated.
    σ2 : float, optional
        Variance of the noise (default is 1).

    Returns:
    --------
    noise : ndarray
        Generated Gaussian noise.
    """
    return np.random.normal(0, np.sqrt(σ2), shapeOut)
