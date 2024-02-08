"""
================================================================
Core digital signal processing utilities (:mod:`optic.dsp.core`)
================================================================

.. autosummary::
   :toctree: generated/

   sigPow                 -- Calculate the average power of x
   signal_power           -- Calculate the total average power of x
   firFilter              -- Perform FIR filtering and compensate for filter delay
   rrcFilterTaps          -- Generate Root-Raised Cosine (RRC) filter coefficients
   rcFilterTaps           -- Generate Raised Cosine (RC) filter coefficients
   pulseShape             -- Generate a pulse shaping filter
   clockSamplingInterp    -- Interpolate signal to a given sampling rate
   quantizer              -- Quantize the input signal using a uniform quantizer 
   lowPassFIR             -- Calculate FIR coefficients of a lowpass filter
   decimate               -- Decimate signal
   resample               -- Signal resampling
   upsample               -- Upsample a signal by inserting zeros between samples
   symbolSync             -- Synchronizer delayed sequences of symbols
   finddelay              -- Estimate the delay between sequences of symbols
   pnorm                  -- Normalize the average power of each componennt of x
   gaussianComplexNoise   -- Generate complex-valued circular Gaussian noise
   gaussianNoise          -- Generate Gaussian noise
   phaseNoise             -- Generate realization of a random-walk phase-noise process
   movingAverage          -- Calculate the sliding window moving average
"""

"""Digital signal processing utilities."""
import numpy as np
from numba import njit, prange
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

    Parameters
    ----------
    t : array-like
        Time values.
    alpha : float
        RRC roll-off factor.
    Ts : float
        Symbol period.

    Returns
    -------
    coeffs : ndarray
        RRC filter coefficients.

    References
    ----------
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
def rcFilterTaps(t, alpha, Ts):
    """
    Generate Raised Cosine (RC) filter coefficients.

    Parameters
    ----------
    t : array-like
        Time values.
    alpha : float
        RC roll-off factor.
    Ts : float
        Symbol period.

    Returns
    -------
    coeffs : ndarray
        RC filter coefficients.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th Edition). McGraw-Hill Education.
    """
    coeffs = np.zeros(len(t), dtype=np.float64)
    π = np.pi

    for i, t_i in enumerate(t):
        t_abs = abs(t_i)
        if t_abs == Ts / (2 * alpha):
            coeffs[i] = π / (4 * Ts) * np.sinc(1 / (2 * alpha))
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
        filterCoeffs = rcFilterTaps(t, alpha, Ts)

    filterCoeffs = filterCoeffs / np.sqrt(np.sum(filterCoeffs**2))

    return filterCoeffs


@njit(parallel=True)
def clockSamplingInterp(x, Fs_in=1, Fs_out=1, jitter_rms=1e-9):
    """
    Interpolate signal to a given sampling rate.

    Parameters
    ----------
    x : ndarray
        Input signal.
    param : core.parameter
        Resampling parameters:
            - param.Fs_in  : sampling frequency of the input signal.

            - param.Fs_out : sampling frequency of the output signal.

            - param.jitter_rms: standard deviation of the time jitter.

    Returns
    -------
    y : ndarray
        Resampled signal.

    """
    nModes = x.shape[1]

    inTs = 1 / Fs_in
    outTs = 1 / Fs_out

    tin = np.arange(0, x.shape[0]) * inTs
    tout = np.arange(0, x.shape[0] * inTs, outTs)

    jitter = np.random.normal(0, jitter_rms, tout.shape)
    tout += jitter

    y = np.zeros((len(tout), x.shape[1]), dtype=x.dtype)

    for k in prange(nModes):
        y[:, k] = np.interp(tout, tin, x[:, k])

    return y


@njit(parallel=True)
def quantizer(x, nBits=16, maxV=1, minV=-1):
    """
    Quantize the input signal using a uniform quantizer with the specified precision.

    Parameters
    ----------
    x : np.array
        The input signal to be quantized.
    nBits : int
        Number of bits used for quantization. The quantizer will have 2^nBits levels.
    maxV : float, optional
        Maximum value for the quantizer's full-scale range (default is 1).
    minV : float, optional
        Minimum value for the quantizer's full-scale range (default is -1).

    Returns
    -------
    np.array
        The quantized output signal with the same shape as 'x', quantized using 'nBits' levels.

    """
    Δ = (maxV - minV) / (2**nBits - 1)

    d = np.arange(minV, maxV + Δ, Δ)

    y = np.zeros(x.shape, dtype=np.float64)

    for indMode in prange(x.shape[1]):
        for idx in prange(len(x)):
            y[idx, indMode] = d[int(np.argmin(np.abs(x[idx, indMode] - d)))]

    return y


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
    try:
        Ei.shape[1]
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)

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
            - param.Rs      : symbol rate of the signal.

            - param.SpS_in  : samples per symbol of the input signal.
            
            - param.SpS_out : samples per symbol of the output signal.

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
    N = min(Ei.shape[0], 202)
    # hi = lowPassFIR(inFs / 2, inFs, N, typeF="rect")
    hi = lowPassFIR(outFs / 2, inFs, N, typeF="rect")

    Ei = firFilter(hi, Ei)

    if nModes == 1:
        Ei = Ei.reshape(len(Ei), 1)
    for k in range(nModes):
        Eo[:, k] = np.interp(tout, tin, Ei[:, k])
    # Eo = firFilter(ho, Eo)

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

    Parameters
    ----------
    shapeOut : tuple of int
        Shape of ndarray to be generated.
    σ2 : float, optional
        Variance of the noise (default is 1).

    Returns
    -------
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

    Parameters
    ----------
    shapeOut : tuple of int
        Shape of ndarray to be generated.
    σ2 : float, optional
        Variance of the noise (default is 1).

    Returns
    -------
    noise : ndarray
        Generated Gaussian noise.
    """
    return np.random.normal(0, np.sqrt(σ2), shapeOut)


@njit
def phaseNoise(lw, Nsamples, Ts):
    """
    Generate realization of a random-walk phase-noise process.

    Parameters
    ----------
    lw : scalar
        laser linewidth.
    Nsamples : scalar
        number of samples to be draw.
    Ts : scalar
        sampling period.

    Returns
    -------
    phi : np.array
        realization of the phase noise process.

    """
    σ2 = 2 * np.pi * lw * Ts
    phi = np.zeros(Nsamples)

    for ind in range(Nsamples - 1):
        phi[ind + 1] = phi[ind] + np.random.normal(0, np.sqrt(σ2))

    return phi


def movingAverage(x, N):
    """
    Calculate the sliding window moving average of a 2D NumPy array along each column.

    Parameters
    ----------
    x : numpy.ndarray
        Input 2D array with shape (M, N), where M is the number of samples and N is the number of columns.
    N : int
        Size of the sliding window.

    Returns
    -------
    numpy.ndarray
        2D array containing the sliding window moving averages along each column.

    Notes
    -----
    The function pads the signal with zeros at both ends to compensate for the lag between the output
    of the moving average and the original signal.

    """
    nCol = x.shape[1]
    y = np.zeros(x.shape, dtype=x.dtype)

    startInd = N//2

    endInd = -N//2+1 if N%2 else -N//2
    for indCol in range(nCol):
        # Pad the signal with zeros at both ends
        padded_x = np.pad(x[:, indCol], (N//2, N//2), mode='constant')

        # Calculate moving average using convolution
        h = np.ones(N) / N
        ma = np.convolve(padded_x, h, 'same')
        y[:, indCol] = ma[startInd:endInd]

    return y

