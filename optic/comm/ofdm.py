"""
==================================================
OFDM utilities (:mod:`optic.comm.ofdm`)
==================================================

.. autosummary::
   :toctree: generated/

   hermit                   -- Hermitian simmetry block.
   calcSymbolRate           -- Calculate the symbol rate of a given OFDM configuration
   modulateOFDM             -- OFDM symbols modulator
   demodulateOFDM           -- OFDM symbols demodulator   
"""

import numpy as np

from numpy.fft  import fft, ifft, fftshift
from scipy.interpolate import interp1d
from optic.dsp.core import upsample


def hermit(V):
    """
    Hermitian simmetry block.

    Parameters
    ----------
    V : complex-valued np.array
        input array
        
    Returns
    -------
    Vh : complex-valued np.array
        vector with hermitian simmetry
    """
    
    Vh = np.zeros(2*len(V) + 2, complex)
    
    Vh[1:len(V)+1] = V 
    
    for j in range(len(V)):
        Vh[len(Vh) - j - 1] = np.conjugate(V[j])
    
    return Vh

def padOFDM(x, L):
    return np.pad(x,(L, L),'constant', constant_values=0)

def calcSymbolRate(M, Rb, Nfft, Np, G, hermitSym):
    """    
    Calculate the symbol rate of a given OFDM configuration.

    Parameters
    ----------
    M         : scalar
                constellation order
    Rb        : scalar
                bit rate
    Nfft      : scalar
                size of FFT
    Np        : scalar
                number of pilot subcarriers
    G         : scalar
                cyclic prefix length
    hermitSym : boolean
                True: Real OFDM symbols / False: Complex OFDM symbols

    Returns
    -------
    Rs        : scalar
                OFDM symbol rate
    """    
    nDataSymbols = (Nfft//2 - 1 - Np) if hermitSym else (Nfft - Np)
    return Rb / (nDataSymbols/(Nfft + G) * np.log2(M))


def modulateOFDM(symbTx, param):
    """
    OFDM symbols modulator.

    Parameters
    ----------
    Nfft          : scalar
                    size of FFT
    G             : scalar
                    cyclic prefix length
    pilot         : complex-valued scalar
                    pilot symbol
    pilotCarriers : np.array
                    indexes of pilot subcarriers
    symbTx        : complex-valued array
                    symbols sequency transmitted
    hermitSym     : boolean
                    True-> Real OFDM symbols / False: Complex OFDM symbols 
    SpS           : int
                    oversampling factor
    
    Returns
    -------
    symbTx_OFDM   : complex-valued np.array
                    OFDM symbols sequency transmitted
    """
    # Check and set default values for input parameters
    Nfft = getattr(param, "Nfft", 512)
    G = getattr(param, "G", 4)
    hermitSymmetry = getattr(param, "hermitSymmetry", False)
    pilot = getattr(param, "pilot", 1+1j)
    pilotCarriers = getattr(param, "pilotCarriers", np.array([], dtype = int))
    SpS = getattr(param, "SpS", 2)
        
    # Number of pilot subcarriers
    Np = len(pilotCarriers)

    # Number of subcarriers
    Ns = Nfft//2 - 1 if hermitSymmetry else Nfft
    numSymb  = len(symbTx)
    numOFDMframes = numSymb//(Ns - Np)

    Carriers = np.arange(0, Ns)
    dataCarriers  = np.array(list(set(Carriers) - set(pilotCarriers)))

    # Serial to parallel
    symbTx_par = np.reshape(symbTx, (numOFDMframes, Ns - Np))   
    symbTx_OFDM_par = np.zeros( (numOFDMframes, SpS*(Nfft + G)), dtype=np.complex64)
    
    for indFrame in range(numOFDMframes):
        # Start OFDM frame with zeros
        frameOFDM = np.zeros(Ns, dtype=np.complex64) 

        # Insert data and pilot subcarriers                    
        frameOFDM[dataCarriers]  = symbTx_par[indFrame, :]
        frameOFDM[pilotCarriers] = pilot

        # Hermitian symmetry
        if hermitSymmetry:
            frameOFDM = hermit(frameOFDM)
       
        # IFFT operation       
        symbTx_OFDM_par[indFrame, SpS*G : SpS*(G + Nfft)] = ifft(fftshift(padOFDM(frameOFDM, Nfft*(SpS-1)//2))) * np.sqrt(SpS*Nfft)
        
        # Cyclic prefix addition
        symbTx_OFDM_par[indFrame, 0 : SpS*G] = symbTx_OFDM_par[indFrame, Nfft*SpS : SpS*(Nfft + G)].copy()

    return symbTx_OFDM_par.ravel()


def demodulateOFDM(symbRx_OFDM, param):
    """
    OFDM symbols demodulator.

    Parameters
    ----------
    Nfft          : scalar
                    size of FFT
    N             : scalar
                    number of transmitted subcarriers
    G             : scalar
                    cyclic prefix length
    pilot         : complex-valued scalar
                    pilot symbol
    pilotCarriers : np.array
                    indexes of pilot subcarriers
    symbRx_OFDM   : complex-valued array
                    OFDM symbols sequency received
    
    Returns
    -------
    symbRx        : complex np.array
                    demodulated symbols sequency received
    """
    # Check and set default values for input parameters
    Nfft = getattr(param, "Nfft", 512)
    G = getattr(param, "G", 4)
    hermitSymmetry = getattr(param, "hermitSymmetry", False)
    pilot = getattr(param, "pilot", 1+1j)
    pilotCarriers = getattr(param, "pilotCarriers", np.array([], dtype = int))
    
    # Number of pilot subcarriers
    Np = len(pilotCarriers)

    # Number of subcarriers
    N = Nfft//2 - 1 if hermitSymmetry else Nfft
    Carriers      = np.arange(0, N)
    dataCarriers  = np.array(list(set(Carriers) - set(pilotCarriers)))

    H_abs = 0
    H_pha = 0

    numSymb       = len(symbRx_OFDM)
    numOFDMframes = numSymb//(Nfft + G)

    symbRx_OFDM_par = np.reshape(symbRx_OFDM, (numOFDMframes, Nfft + G))

    # Cyclic prefix extraction
    symbRx_OFDM_par = symbRx_OFDM_par[:, G : G + Nfft]

    # FFT operation
    for indFrame in range(numOFDMframes):
        symbRx_OFDM_par[indFrame, :] = fftshift(fft(symbRx_OFDM_par[indFrame,:])) / np.sqrt(Nfft)

    if hermitSymmetry:
        # Removal of hermitian symmetry
        symbRx_OFDM_par = symbRx_OFDM_par[:, 1 : 1 + N]

    # Equalization
    if Np != 0:
        # Channel estimation
        for indFrame in range(numOFDMframes):
            H_est = symbRx_OFDM_par[indFrame, :][pilotCarriers] / pilot

            H_abs += interp1d(pilotCarriers, np.abs(H_est), kind = 'linear', fill_value = "extrapolate")(Carriers)
            H_pha += interp1d(pilotCarriers, np.angle(H_est), kind = 'linear', fill_value = "extrapolate")(Carriers)

            if(indFrame == numOFDMframes - 1):
                H_abs = H_abs/numOFDMframes
                H_pha = H_pha/numOFDMframes

        for indFrame in range(numOFDMframes):
            symbRx_OFDM_par[indFrame, :] = symbRx_OFDM_par[indFrame, :] / (H_abs*np.exp(H_pha))

        # Pilot extraction
        symbRx_OFDM_par = symbRx_OFDM_par[:, dataCarriers]

    return symbRx_OFDM_par.ravel()