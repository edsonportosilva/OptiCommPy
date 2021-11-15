import numpy as np
from commpy.modulation import QAMModem
from commpy.utilities import signal_power
from numba import njit
from tqdm.notebook import tqdm

from optic.core import parameters


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
    if alg == 'ddpll':
        Eo, ϕ, θ = ddpll(Ei, N, constSymb, symbTx, pilotInd)
    elif: 
         Eo, ϕ, θ = ddpll(Ei, N, constSymb, symbTx, pilotInd)
         
    if Eo.shape[1] == 1:
        Eo = Eo[:]
        ϕ = ϕ[:]
        θ = θ[:]

    return Eo, ϕ, θ


@njit
def bps(Ei, constSymb, B, N):
    """
    blind phase search (BPS) algorithm  
    """          
    ϕ_test = np.arange(0, B)*(np.pi/2)/B # test phases
    
    dist = np.zeros((constSymb.shape[0], B), dtype=np.float)
    L = Ei.shape[0]
    
    ϕ_dec = np.zeros(L, dtype=np.float)
    θ = np.zeros(L, dtype=np.float)
    
    for k in range(0, L):                
        for indPhase, ϕ in enumerate(ϕ_test):     
            
            dist[:, indPhase] = np.abs(Ei[k]-constSymb*np.exp(1j*ϕ))
        
        indRot = np.argwhere( dist == np.min(dist) )[0, 1]
        
        ϕ_dec[k] = ϕ_test[indRot]
        
        if k > 2*N+1:
            θ[k] = np.mean(ϕ_dec[k - N: k + N])  # moving average filter
        else:
            θ[k] = ϕ_dec[k]
            
    Eo = Ei * np.exp(1j * θ)  # compensate phase rotation
        
    return Eo, θ, ϕ


