import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from commpy.filters import rcosfilter, rrcosfilter
from commpy.modulation import QAMModem
from commpy.utilities import upsample
from numba import njit
from numpy.fft import fft, fftfreq, fftshift, ifft
from scipy import signal
from scipy.signal import lfilter
from scipy.stats.kde import gaussian_kde
from tqdm.notebook import tqdm

from optic.models import linFiberCh


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


def eyediagram(sig, Nsamples, SpS, n=3, ptype="fast", plotlabel=None):
    """
    Plots the eye diagram of a modulated signal waveform

    :param Nsamples: number os samples to be plotted
    :param SpS: samples per symbol
    :param n: number of symbol periods
    :param type: 'fast' or 'fancy'
    :param plotlabel: label for the plot legend
    """

    if np.iscomplex(sig).any():
        d = 1
        plotlabel_ = plotlabel + " [real]"
    else:
        d = 0
        plotlabel_ = plotlabel

    for ind in range(0, d + 1):
        if ind == 0:
            y = sig[0:Nsamples].real
            x = np.arange(0, y.size, 1) % (n * SpS)
        else:
            y = sig[0:Nsamples].imag
            plotlabel_ = plotlabel + " [imag]"

        plt.figure()
        if ptype == "fancy":
            k = gaussian_kde(np.vstack([x, y]))
            k.set_bandwidth(bw_method=k.factor / 5)

            xi, yi = (
                1.1
                * np.mgrid[
                    x.min(): x.max(): x.size ** 0.5 * 1j,
                    y.min(): y.max(): y.size ** 0.5 * 1j,
                ]
            )
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=1, shading="auto")
            plt.show()
        elif ptype == "fast":
            y[x == n * SpS] = np.nan
            y[x == 0] = np.nan

            plt.plot(x / SpS, y, color="blue", alpha=0.8, label=plotlabel_)
            plt.xlim(min(x / SpS), max(x / SpS))
            plt.xlabel("symbol period (Ts)")
            plt.ylabel("amplitude")
            plt.title("eye diagram")

            if plotlabel is not None:
                plt.legend(loc="upper left")

            plt.grid()
            plt.show()
    return None


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


def edc(Ei, L, D, Fc, Fs):
    """
    Electronic chromatic dispersion compensation (EDC)

    :param Ei: dispersed signal
    :param L: fiber length [km]
    :param D: chromatic dispersion parameter [ps/nm/km]
    :param Fc: carrier frequency [Hz]
    :param Fs: sampling frequency [Hz]

    :return Eo: CD compensated signal
    """
    Eo = linFiberCh(Ei, L, 0, -D, Fc, Fs)

    return Eo


@njit
def ddpll(Ei, N, constSymb, symbTx, pilotInd):
    """
    DDPLL
    """
    nModes = Ei.shape[1]

    ϕ = np.zeros(Ei.shape)
    θ = np.zeros(Ei.shape)

    for n in range(0, nModes):
        # correct (possible) initial phase rotation
        rot = np.mean(symbTx[:, n] / Ei[:, n])
        Ei[:, n] = rot * Ei[:, n]

        for k in range(0, len(Ei)):

            decided = np.argmin(
                np.abs(Ei[k, n] * np.exp(1j * θ[k - 1, n]) - constSymb)
            )  # find closest constellation symbol

            if k in pilotInd:
                ϕ[k, n] = np.angle(
                    symbTx[k, n] / (Ei[k, n])
                )  # phase estimation with pilot symbol
            else:
                ϕ[k, n] = np.angle(
                    constSymb[decided] / (Ei[k, n])
                )  # phase estimation after symbol decision

            if k > N:
                θ[k, n] = np.mean(ϕ[k - N: k, n])  # moving average filter
            else:
                θ[k, n] = np.angle(symbTx[k, n] / (Ei[k, n]))

    Eo = Ei * np.exp(1j * θ)  # compensate phase rotation

    return Eo, ϕ, θ


def cpr(Ei, N, M, symbTx, pilotInd=[]):
    """
    Carrier phase recovery (CPR)

    """
    try:
        Ei.shape[1]
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)

    mod = QAMModem(m=M)
    constSymb = mod.constellation / np.sqrt(mod.Es)

    Eo, ϕ, θ = ddpll(Ei, N, constSymb, symbTx, pilotInd)

    if Eo.shape[1] == 1:
        Eo = Eo[:]
        ϕ = ϕ[:]
        θ = θ[:]

    return Eo, ϕ, θ


def fourthPowerFOE(Ei, Ts, plotSpec=False):
    """
    4th power frequency offset estimator (FOE)
    """

    Fs = 1 / Ts
    Nfft = len(Ei)

    f = Fs * fftfreq(Nfft)
    f = fftshift(f)

    f4 = 10 * np.log10(np.abs(fftshift(fft(Ei ** 4))))
    indFO = np.argmax(f4)

    if plotSpec:
        plt.figure()
        plt.plot(f, f4, label="$|FFT(s[k]^4)|[dB]$")
        plt.plot(f[indFO], f4[indFO], "x", label="$4f_o$")
        plt.legend()
        plt.xlim(min(f), max(f))
        plt.grid()

    return f[indFO] / 4


def dbp(Ei, Fs, Ltotal, Lspan, hz=0.5, alpha=0.2, gamma=1.3, D=16, Fc=193.1e12):
    """
    Digital backpropagation (symmetric, single-pol.)

    :param Ei: input signal
    :param Ltotal: total fiber length [km]
    :param Lspan: span length [km]
    :param hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :param alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :param D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :param gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :param Fc: carrier frequency [Hz][default: 193.1e12 Hz]
    :param Fs: sampling frequency [Hz]

    :return Ech: backpropagated signal
    """
    # c = 299792458   # speed of light (vacuum)
    c_kms = const.c / 1e3
    λ = c_kms / Fc
    α = -alpha / (10 * np.log10(np.exp(1)))
    β2 = (D * λ ** 2) / (2 * np.pi * c_kms)
    γ = -gamma

    Nfft = len(Ei)

    ω = 2 * np.pi * Fs * fftfreq(Nfft)

    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(np.floor(Lspan / hz))

    Ech = Ei.reshape(len(Ei),)
    Ech = fft(Ech)  # single-polarization field

    linOperator = np.exp(-(α / 2) * (hz / 2) + 1j * (β2 / 2) * (ω ** 2) * (hz / 2))

    for spanN in tqdm(range(0, Nspans)):

        Ech = Ech * np.exp((α / 2) * Nsteps * hz)

        for stepN in range(0, Nsteps):
            # First linear step (frequency domain)
            Ech = Ech * linOperator

            # Nonlinear step (time domain)
            Ech = ifft(Ech)
            Ech = Ech * np.exp(1j * γ * (Ech * np.conj(Ech)) * hz)

            # Second linear step (frequency domain)
            Ech = fft(Ech)
            Ech = Ech * linOperator

    Ech = ifft(Ech)

    return Ech.reshape(len(Ech),)


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
