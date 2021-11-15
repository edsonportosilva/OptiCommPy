import numpy as np
from commpy.modulation import QAMModem
from commpy.utilities import signal_power
from numba import njit
from tqdm.notebook import tqdm

from optic.core import parameters



def cpr(Ei, symbTx=[], param=[]):
    """
    Carrier phase recovery (CPR)

    """
    # check input parameters
    alg        = getattr(param, 'alg', 'bps')
    M          = getattr(param, 'M', 4)
    B          = getattr(param, 'B', 16)
    N          = getattr(param, 'N', 15)    
    pilotInd   = getattr(param, 'pilotInd', [])
    
    try:
        Ei.shape[1]
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)

    mod = QAMModem(m=M)
    constSymb = mod.constellation / np.sqrt(mod.Es)

    if alg == 'ddpll':
        ϕ, θ = ddpll(Ei, N, constSymb, symbTx, pilotInd)      
        Eo = Ei*np.exp(1j*np.unwrap(4*θ, axis=0)/4)

    elif alg == 'bps':
        θ = bps(Ei, N, constSymb, B)
        ϕ = θ.copy()
        Eo = Ei*np.exp(1j*np.unwrap(4*θ, axis=0)/4)
    else:
        raise ValueError('CPR algorithm not specified (or incorrectly specified).')

    if Eo.shape[1] == 1:
        Eo = Eo[:]
        ϕ = ϕ[:]
        θ = θ[:]

    return Eo, ϕ, θ


@njit
def bps(Ei, N, constSymb, B):
    """
    blind phase search (BPS) algorithm  
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
    decision directed phase-locked loop (DDPLL)
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


    return ϕ, θ