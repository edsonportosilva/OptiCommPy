"""Digital signal processing utilities."""
import matplotlib.pyplot as plt
import numpy as np
from commpy.filters import rcosfilter, rrcosfilter
from commpy.utilities import upsample
from numba import njit
from scipy import signal


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

    t = np.linspace(-2, 2, SpS)
    Te = 1

    if pulseType == "rect":
        filterCoeffs = np.concatenate(
            (np.zeros(int(SpS / 2)), np.ones(SpS), np.zeros(int(SpS / 2)))
        )
    elif pulseType == "nrz":
        filterCoeffs = np.convolve(
            np.ones(SpS),
            2 / (np.sqrt(np.pi) * Te) * np.exp(-(t ** 2) / Te),
            mode="full",
        )
    elif pulseType == "rrc":
        tindex, filterCoeffs = rrcosfilter(N, alpha, Ts, fa)
    elif pulseType == "rc":
        tindex, filterCoeffs = rcosfilter(N, alpha, Ts, fa)
    filterCoeffs = filterCoeffs / np.sqrt(np.sum(filterCoeffs ** 2))

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


def decimate(Ei, param):
    """
    Decimate signal.

    Parameters
    ----------
    Ei : np.array
        Input signal.
    param : core.parameter
    Decimation parameters:
            param.SpS_in  : samples per symbol of the input signal.
            param.SpS_out : samples per symbol of the output signal.

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


def symbolSync(rx, tx, SpS, mode='amp'):
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
    
    if mode == 'amp':
        for n in range(nModes):
            for m in range(nModes):
                corrMatrix[m, n] = np.max(
                    np.abs(signal.correlate(np.abs(tx[:, m]), np.abs(rx[:, n])))
                )
        swap = np.argmax(corrMatrix, axis=0)
    
        tx = tx[:, swap]
    
        for k in range(nModes):
            delay[k] = finddelay(np.abs(tx[:, k]), np.abs(rx[:, k]))
    elif mode == 'real':
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
    Normalize the average power of x.

    Parameters
    ----------
    x : np.array
        Signal.

    Returns
    -------
    np.array
        Signal x with normalized power.

    """
    return x / np.sqrt(np.mean(x * np.conj(x)).real)
