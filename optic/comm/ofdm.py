"""
==================================================
OFDM utilities (:mod:`optic.comm.ofdm`)
==================================================

.. autosummary::
   :toctree: generated/

   hermit                   -- Hermitian simmetry block
   zeroPad                  -- Pad an input array with zeros on both sides
   calcSymbolRate           -- Calculate the symbol rate of a given OFDM configuration
   modulateOFDM             -- OFDM symbols modulator
   demodulateOFDM           -- OFDM symbols demodulator
"""

import numpy as np
from numpy.fft import fft, fftshift, ifft
from scipy.interpolate import interp1d


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

    Vh = np.zeros(2 * len(V) + 2, complex)

    Vh[1 : len(V) + 1] = V

    for j in range(len(V)):
        Vh[len(Vh) - j - 1] = np.conjugate(V[j])

    return Vh


def zeroPad(x, L):
    """
    Pad an input array with zeros on both sides.

    Parameters
    ----------
    x : array_like
        Input array to be padded.
    L : int
        Number of zeros to pad on each side of the array.

    Returns
    -------
    padded_array : np.array
        Padded array with zeros added at the beginning and end.

    Notes
    -----
    This function pads the input array `x` with `L` zeros on both sides, effectively increasing
    its length by `2*L`.

    """
    return np.pad(x, (L, L), "constant", constant_values=0)


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
    nDataSymbols = (Nfft // 2 - 1 - Np) if hermitSym else (Nfft - Np)
    return Rb / (nDataSymbols / (Nfft + G) * np.log2(M))


def modulateOFDM(symb, param=None):
    """
    Modulate OFDM signal.

    Parameters
    ----------
    symb : np.np.array
        Complex-valued array of modulation symbols representing the symbols sequence to be transmitted.
    param : optic.utils.parameters object, optional
        Parameters for OFDM modulation.

        - param.Nfft : scalar, optional. Size of the FFT. [default: 512].
        - param.G : scalar, optional. Cyclic prefix length. [default: 4].
        - param.hermitSymmetry : bool, optional. If True, indicates real OFDM symbols; if False, indicates complex OFDM symbols. [default: False].
        - param.pilot : complex-valued scalar, optional. Pilot symbol. [default: 1 + 1j].
        - param.pilotCarriers : np.array, optional. Indexes of pilot subcarriers. [default: empty array].
        - param.nullCarriers : np.array, optional. Indexes of null subcarriers. [default: empty array].
        - param.SpS : int, optional. Oversampling factor. [default: 2].

    Returns
    -------
    np.array
        Complex-valued array representing the OFDM symbols sequence transmitted.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """

    # Check and set default values for input parameters
    Nfft = getattr(param, "Nfft", 512)
    G = getattr(param, "G", 4)
    hermitSymmetry = getattr(param, "hermitSymmetry", False)
    pilot = getattr(param, "pilot", 0.25 + 0.25j)
    SpS = getattr(param, "SpS", 2)
    pilotCarriers = getattr(param, "pilotCarriers", np.array([], dtype=np.int64))
    nullCarriers = getattr(param, "nullCarriers", np.array([], dtype=np.int64))

    Ns = Nfft // 2 - 1 if hermitSymmetry else Nfft
    Np = len(pilotCarriers)
    Nz = len(nullCarriers)
    Ni = Ns - Np - Nz

    Carriers = np.arange(0, Ns)
    dataCarriers = np.setdiff1d(Carriers, np.union1d(pilotCarriers, nullCarriers))

    numSymb = len(symb)

    if numSymb % Ni != 0:
        raise ValueError(
            f"Number of symbols ({numSymb}) is not divisible by number of data carriers per OFDM frame ({Ni})."
        )

    numOFDMframes = numSymb // Ni

    # Serial to parallel
    symb_par = np.reshape(symb, (numOFDMframes, Ni))
    sigOFDM_par = np.zeros((numOFDMframes, SpS * (Nfft + G)), dtype=np.complex64)

    for indFrame in range(numOFDMframes):
        # Start OFDM frame with zeros
        frameOFDM = np.zeros(Ns, dtype=np.complex64)

        # Insert data, pilot and null subcarriers
        frameOFDM[dataCarriers] = symb_par[indFrame, :]
        frameOFDM[pilotCarriers] = pilot
        frameOFDM[nullCarriers] = 0

        # Hermitian symmetry
        if hermitSymmetry:
            frameOFDM = hermit(frameOFDM)

        # IFFT operation
        sigOFDM_par[indFrame, SpS * G : SpS * (G + Nfft)] = ifft(
            fftshift(zeroPad(frameOFDM, (Nfft * (SpS - 1)) // 2))
        ) * np.sqrt(SpS * Nfft)

        # Cyclic prefix addition
        if G > 0:
            sigOFDM_par[indFrame, 0 : SpS * G] = sigOFDM_par[
                indFrame, Nfft * SpS : SpS * (Nfft + G)
            ].copy()

    return sigOFDM_par.ravel()


def demodulateOFDM(sig, param=None):
    """
    Demodulate OFDM signal.

    Parameters
    ----------
    sig : np.np.array
        Complex-valued array representing the OFDM signal sequence received at one sample per symbol.
    param : optic.utils.parameters object, optional
        Parameters for OFDM demodulation.

        - param.Nfft : scalar, optional. Size of the FFT [default: 512].
        - param.G : scalar, optional. Cyclic prefix length [default: 4].
        - param.hermitSymmetry : bool, optional. If True, indicates real OFDM symbols; if False, indicates complex OFDM symbols [default: False].
        - param.pilot : complex-valued scalar, optional. Pilot symbol [default: 1 + 1j].
        - param.pilotCarriers : np.array, optional. Indexes of pilot subcarriers [default: an empty array].
        - param.nullCarriers : np.array, optional. Indexes of null subcarriers [default: an empty array].
        - param.returnChannel : bool, optional. If True, return the estimated channel [default: False].

    Returns
    -------
    np.array or tuple
        If `returnChannel` is False, returns a complex-valued array representing the demodulated symbols sequence received.
        If `returnChannel` is True, returns a tuple containing the demodulated symbols sequence received and the estimated channel.

    Notes
    -----
    - The input signal must be sampled at one sample per symbol.
    - This function performs demodulation of the OFDM signal according to the provided parameters, including channel estimation and single tap equalization.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """

    # Check and set default values for input parameters
    Nfft = getattr(param, "Nfft", 512)
    G = getattr(param, "G", 4)
    hermitSymmetry = getattr(param, "hermitSymmetry", False)
    pilot = getattr(param, "pilot", 0.25 + 0.25j)
    returnChannel = getattr(param, "returnChannel", False)
    pilotCarriers = getattr(param, "pilotCarriers", np.array([], dtype=np.int64))
    nullCarriers = getattr(param, "nullCarriers", np.array([], dtype=np.int64))

    Ns = Nfft // 2 - 1 if hermitSymmetry else Nfft
    Np = len(pilotCarriers)
    Nz = len(nullCarriers)
    Ni = Ns - Np - Nz

    Carriers = np.arange(0, Ns)
    dataCarriers = np.setdiff1d(Carriers, np.union1d(pilotCarriers, nullCarriers))

    numSymb = len(sig)

    if numSymb % (Nfft + G) != 0:
        raise ValueError(
            f"Number of received symbols ({numSymb}) is not divisible by Nfft + G ({Nfft + G})."
        )

    numOFDMframes = numSymb // (Nfft + G)

    H_abs = 0
    H_pha = 0

    sig_par = np.reshape(sig, (numOFDMframes, Nfft + G))

    # Cyclic prefix removal
    sig_par = sig_par[:, G : G + Nfft]

    # FFT operation
    for indFrame in range(numOFDMframes):
        sig_par[indFrame, :] = fftshift(fft(sig_par[indFrame, :])) / np.sqrt(Nfft)

    if hermitSymmetry:
        # Removal of hermitian symmetry
        sig_par = sig_par[:, 1 : 1 + Ns]

    # Channel estimation and single tap equalization
    if Np != 0:
        # Channel estimation
        for indFrame in range(numOFDMframes):
            H_est = sig_par[indFrame, :][pilotCarriers] / pilot

            H_abs += interp1d(
                pilotCarriers, np.abs(H_est), kind="linear", fill_value="extrapolate"
            )(Carriers)
            H_pha += interp1d(
                pilotCarriers, np.angle(H_est), kind="linear", fill_value="extrapolate"
            )(Carriers)

            if indFrame == numOFDMframes - 1:
                H_abs = H_abs / numOFDMframes
                H_pha = H_pha / numOFDMframes

        for indFrame in range(numOFDMframes):
            sig_par[indFrame, :] = sig_par[indFrame, :] / (H_abs * np.exp(1j * H_pha))

    # Data carriers
    sig_par = sig_par[:, dataCarriers]

    if returnChannel:
        return sig_par.ravel(), H_abs * np.exp(1j * H_pha)
    else:
        return sig_par.ravel()
