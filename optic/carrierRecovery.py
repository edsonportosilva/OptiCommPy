import matplotlib.pyplot as plt
import numpy as np
from commpy.modulation import QAMModem
from numba import njit
from numpy.fft import fft, fftfreq, fftshift


def cpr(Ei, symbTx=[], paramCPR=[]):
    """
    Carrier phase recovery (CPR)

    :param Ei: received symbols
    :param symbTx: transmitted symbols [only for pilot-aided CPR]
    :param paramCPR.alg: CPR algorithm to be used ['bps' or 'ddpll']
    :param paramCPR.N: length of the moving average window
    :param paramCPR.M: constellation
    :param paramCPR.B: number of BPS test phases

    :return θ: estimated phases
    """
    # check input parameters
    alg = getattr(paramCPR, "alg", "bps")
    M = getattr(paramCPR, "M", 4)
    B = getattr(paramCPR, "B", 64)
    N = getattr(paramCPR, "N", 35)
    pilotInd = getattr(paramCPR, "pilotInd", np.array([len(Ei)+1]))

    try:
        Ei.shape[1]
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)

    mod = QAMModem(m=M)
    constSymb = mod.constellation / np.sqrt(mod.Es)
    
    θ = np.zeros(Ei.shape)
    ϕ = np.zeros(Ei.shape)
    
    if alg == "ddpll":
        ϕ[:], θ[:] = ddpll(Ei, N, constSymb, symbTx, pilotInd)
    elif alg == "bps":
        θ[:] = bps(Ei, int(N/2), constSymb, B)
        ϕ[:] = np.copy(θ)
        
        ϕ = np.unwrap(4 * ϕ, axis=0) / 4
        θ = np.unwrap(4 * θ, axis=0) / 4        
    else:
        raise ValueError("CPR algorithm incorrectly specified.")

    Eo = Ei * np.exp(1j * θ)

    if Eo.shape[1] == 1:
        Eo = Eo[:]
        ϕ = ϕ[:]
        θ = θ[:]

    return Eo, ϕ, θ


@njit
def bps(Ei, N, constSymb, B):
    """
    Blind phase search (BPS) algorithm

    :param Ei: received symbols
    :param N: half of the 2*N+1 average window
    :param constSymb: constellation
    :param B: number of test phases

    :return θ: estimated phases
    """
    nModes = Ei.shape[1]

    ϕ_test = np.arange(0, B) * (np.pi / 2) / B  # test phases

    θ = np.zeros(Ei.shape, dtype="float")

    zeroPad = np.zeros((N, nModes), dtype="complex")
    x = np.concatenate(
        (zeroPad, Ei, zeroPad)
    )  # pad start and end of the signal with zeros

    L = x.shape[0]

    for n in range(0, nModes):

        dist = np.zeros((B, constSymb.shape[0]), dtype="float")
        dmin = np.zeros((B, 2 * N + 1), dtype="float")

        for k in range(0, L):
            for indPhase, ϕ in enumerate(ϕ_test):
                dist[indPhase, :] = np.abs(x[k, n]*np.exp(1j * ϕ) - constSymb) ** 2
                dmin[indPhase, -1] = np.min(dist[indPhase, :])

            if k >= 2 * N:
                sumDmin = np.sum(dmin, axis=1)
                indRot = np.argmin(sumDmin)
                θ[k - 2 * N, n] = ϕ_test[indRot]

            dmin = np.roll(dmin, -1)

    return θ


@njit
def ddpll(Ei, N, constSymb, symbTx, pilotInd):
    """
    Decision directed phase-locked loop (DDPLL)

    :param Ei: received symbols
    :param N: moving average window size
    :param constSymb: constellation
    :param symbTx: sequence of transmitted symbols
    :param pilotInd: indices of pilot symbol positions

    :return ϕ: estimated phases (before the moving average filter)
    :return θ: estimated phases (after the moving average filter)
    """
    nModes = Ei.shape[1]

    ϕ = np.zeros(Ei.shape)
    θ = np.zeros(Ei.shape)

    for n in range(0, nModes):
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

    return ϕ, θ


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
