import matplotlib.pyplot as plt
import numpy as np
from commpy.filters import rcosfilter, rrcosfilter
from commpy.utilities import upsample
from scipy import signal
from scipy.signal import lfilter


def firFilter(h, x):
    """
    Implements FIR filtering and compensates filter delay
    (assuming the impulse response is symmetric)

    :param h: impulse response (symmetric)
    :param x: input signal

    :return y: output filtered signal
    """

    try:
        x.shape[1]
    except IndexError:
        x = x.reshape(len(x), 1)

    y = x.copy()
    nModes = x.shape[1]
    N = h.size

    for n in range(0, nModes):
        x_ = x[:, n]
        x_ = np.pad(x_, (0, int(N / 2)), "constant")
        y_ = lfilter(h, 1, x_)
        y[:, n] = y_[int(N / 2): y_.size]

    if y.shape[1] == 1:
        y = y[:, 0]

    return y


def pulseShape(pulseType, SpS=2, N=1024, alpha=0.1, Ts=1):
    """
    Generate pulse shaping filters

    :param pulseType: 'rect','nrz','rrc'
    :param SpS: samples per symbol
    :param N: number of filter coefficients
    :param alpha: RRC rolloff factor
    :param Ts: symbol period

    :return filterCoeffs: normalized filter coefficients
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

    return filterCoeffs / np.sqrt(np.sum(filterCoeffs ** 2))


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
    for k in range(0, x.size):
        xk_interp = x[k] * np.sinc((t - k * Ta) / Ta)
        x_sum += xk_interp
        plt.plot(t, xk_interp)

    plt.legend(loc="upper right")
    plt.xlim(min(t), max(t))
    plt.grid()

    return x_sum, t


def lowPassFIR(fc, fa, N, typeF="rect"):
    """
    Calculate FIR coeffs for a lowpass filter

    :param fc : cutoff frequency
    :param fa : sampling frequency
    :param N  : number of coefficients
    :param typeF : 'rect' or 'gauss'

    :return h : FIR filter coefficients
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

    decFactor = int(param.SpS_in / param.SpS_out)

    # simple timing recovery
    sampDelay = np.zeros(Ei.shape[1])

    # finds best sampling instant
    # (maximum variance sampling time)
    for k in range(0, Ei.shape[1]):
        a = Ei[:, k].reshape(Ei.shape[0], 1)
        varVector = np.var(a.reshape(-1, param.SpS_in), axis=0)
        sampDelay[k] = np.where(varVector == np.amax(varVector))[0][0]

    # downsampling
    Eo = Ei[::decFactor, :]

    for k in range(0, Ei.shape[1]):
        Ei[:, k] = np.roll(Ei[:, k], int(sampDelay[k]))
        Eo[:, k] = Ei[0::decFactor, k]

    return Eo


def symbolSync(rx, tx, SpS):

    nModes = rx.shape[1]

    rx = rx[0::SpS, :]

    # calculate time delay
    delay = np.zeros(nModes)

    corrMatrix = np.zeros((nModes, nModes))

    for n in range(0, nModes):
        for m in range(0, nModes):
            corrMatrix[m, n] = np.max(
                np.abs(signal.correlate(np.abs(tx[:, m]), np.abs(rx[:, n])))
            )

    swap = np.argmax(corrMatrix, axis=0)

    tx = tx[:, swap]

    for k in range(0, nModes):
        delay[k] = finddelay(np.abs(tx[:, k]), np.abs(rx[:, k]))

    # compensate time delay
    for k in range(nModes):
        tx[:, k] = np.roll(tx[:, k], -int(delay[k]))

    return tx


def finddelay(x, y):

    d = np.argmax(signal.correlate(x, y)) - x.shape[0]+1

    return d
