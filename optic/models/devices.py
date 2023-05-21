"""Basic physical models for optical devices."""
import logging as logg

import numpy as np
import scipy.constants as const
from scipy.linalg import norm
from numba import njit
from numpy.fft import fft, fftfreq, ifft
from numpy.random import normal
from tqdm.notebook import tqdm

from optic.dsp.core import lowPassFIR, signal_power

try:
    from optic.dsp.coreGPU import firFilter
except ImportError:
    from optic.dsp.core import firFilter

def pm(Ai, u, Vπ):
    """
    Optical Phase Modulator (PM).

    Parameters
    ----------
    Ai : scalar or np.array
        Amplitude of the optical field at the input of the PM.
    u : np.array
        Electrical driving signal.
    Vπ : scalar
        PM's Vπ voltage.
    Returns
    -------
    Ao : np.array
        Modulated optical field at the output of the PM.

    """
    try:
        u.shape
    except AttributeError:
        u = np.array([u])

    try:
        if Ai.shape == () and u.shape != ():
            Ai = Ai * np.ones(u.shape)
        else:
            assert (
                Ai.shape == u.shape
            ), "Ai and u need to have the same dimensions"
    except AttributeError:
        Ai = Ai * np.ones(u.shape)

    π = np.pi
    return Ai * np.exp(1j* (u / Vπ) * π)

def mzm(Ai, u, Vπ, Vb):
    """
    Optical Mach-Zehnder Modulator (MZM).

    Parameters
    ----------
    Ai : scalar or np.array
        Amplitude of the optical field at the input of the MZM.
    u : np.array
        Electrical driving signal.
    Vπ : scalar
        MZM's Vπ voltage.
    Vb : scalar
        MZM's bias voltage.

    Returns
    -------
    Ao : np.array
        Modulated optical field at the output of the MZM.

    """
    try:
        u.shape
    except AttributeError:
        u = np.array([u])

    try:
        if Ai.shape == () and u.shape != ():
            Ai = Ai * np.ones(u.shape)
        else:
            assert (
                Ai.shape == u.shape
            ), "Ai and u need to have the same dimensions"
    except AttributeError:
        Ai = Ai * np.ones(u.shape)

    π = np.pi
    return Ai * np.cos(0.5 / Vπ * (u + Vb) * π)


def iqm(Ai, u, Vπ, VbI, VbQ):
    """
    Optical In-Phase/Quadrature Modulator (IQM).

    Parameters
    ----------
    Ai : scalar or np.array
        Amplitude of the optical field at the input of the IQM.
    u : complex-valued np.array
        Modulator's driving signal (complex-valued baseband).
    Vπ : scalar
        MZM Vπ-voltage.
    VbI : scalar
        I-MZM's bias voltage.
    VbQ : scalar
        Q-MZM's bias voltage.

    Returns
    -------
    Ao : complex-valued np.array
        Modulated optical field at the output of the IQM.

    """
    try:
        u.shape
    except AttributeError:
        u = np.array([u])

    try:
        if Ai.shape == () and u.shape != ():
            Ai = Ai * np.ones(u.shape)
        else:
            assert (
                Ai.shape == u.shape
            ), "Ai and u need to have the same dimensions"
    except AttributeError:
        Ai = Ai * np.ones(u.shape)

    return mzm(Ai / np.sqrt(2), u.real, Vπ, VbI) + 1j * mzm(
        Ai / np.sqrt(2), u.imag, Vπ, VbQ
    )


def pbs(E, θ=0):
    """
    Polarization beam splitter (PBS).

    Parameters
    ----------
    E : (N,2) np.array
        Input pol. multiplexed optical field.
    θ : scalar, optional
        Rotation angle of input field in radians. The default is 0.

    Returns
    -------
    Ex : (N,) np.array
        Ex output single pol. field.
    Ey : (N,) np.array
        Ey output single pol. field.

    """
    try:
        assert E.shape[1] == 2, "E need to be a (N,2) or a (N,) np.array"
    except IndexError:
        E = np.repeat(E, 2).reshape(-1, 2)
        E[:, 1] = 0

    rot = np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]]) + 1j * 0

    E = E @ rot

    Ex = E[:, 0]
    Ey = E[:, 1]

    return Ex, Ey


def photodiode(E, paramPD=None):
    """
    Pin photodiode (PD).

    Parameters
    ----------
    E : np.array
        Input optical field.
    paramPD : parameter object (struct), optional
        Parameters of the photodiode.

    paramPD.R: photodiode responsivity [A/W][default: 1 A/W]
    paramPD.Tc: temperature [°C][default: 25°C]
    paramPD.Id: dark current [A][default: 5e-9 A]
    paramPD.RL: impedance load [Ω] [default: 50Ω]
    paramPD.B bandwidth [Hz][default: 30e9 Hz]
    paramPD.Fs: sampling frequency [Hz] [default: 60e9 Hz]
    paramPD.fType: frequency response type [default: 'rect']
    paramPD.N: number of the frequency resp. filter taps. [default: 8001]
    paramPD.ideal: ideal PD?(i.e. no noise, no frequency resp.) [default: True]

    Returns
    -------
    ipd : np.array
          photocurrent.

    """
    if paramPD is None:
        paramPD = []
    kB = const.value("Boltzmann constant")
    q = const.value("elementary charge")

    # check input parameters
    R = getattr(paramPD, "R", 1)
    Tc = getattr(paramPD, "Tc", 25)
    Id = getattr(paramPD, "Id", 5e-9)
    RL = getattr(paramPD, "RL", 50)
    B = getattr(paramPD, "B", 30e9)
    Fs = getattr(paramPD, "Fs", 60e9)
    N = getattr(paramPD, "N", 8000)
    fType = getattr(paramPD, "fType", "rect")
    ideal = getattr(paramPD, "ideal", True)

    assert R > 0, "PD responsivity should be a positive scalar"
    assert (
        Fs >= 2 * B
    ), "Sampling frequency Fs needs to be at least twice of B."

    ipd = R * E * np.conj(E)  # ideal fotodetected current

    if not (ideal):

        Pin = (np.abs(E) ** 2).mean()

        # shot noise
        σ2_s = 2 * q * (R * Pin + Id) * B  # shot noise variance

        # thermal noise
        T = Tc + 273.15  # temperature in Kelvin
        σ2_T = 4 * kB * T * B / RL  # thermal noise variance

        # add noise sources to the p-i-n receiver
        Is = normal(0, np.sqrt(Fs * (σ2_s / (2 * B))), ipd.size)
        It = normal(0, np.sqrt(Fs * (σ2_T / (2 * B))), ipd.size)

        ipd += Is + It

        # lowpass filtering
        h = lowPassFIR(B, Fs, N, typeF=fType)
        ipd = firFilter(h, ipd)

    return ipd.real


def balancedPD(E1, E2, paramPD=None):
    """
    Balanced photodiode (BPD).

    Parameters
    ----------
    E1 : np.array
        Input optical field.
    E2 : np.array
        Input optical field.
    paramPD : parameter object (struct), optional
        Parameters of the photodiodes.

    paramPD.R: photodiode responsivity [A/W][default: 1 A/W]
    paramPD.Tc: temperature [°C][default: 25°C]
    paramPD.Id: dark current [A][default: 5e-9 A]
    paramPD.RL: impedance load [Ω] [default: 50Ω]
    paramPD.B bandwidth [Hz][default: 30e9 Hz]
    paramPD.Fs: sampling frequency [Hz] [default: 60e9 Hz]
    paramPD.fType: frequency response type [default: 'rect']
    paramPD.N: number of the frequency resp. filter taps. [default: 8001]
    paramPD.ideal: ideal PD?(i.e. no noise, no frequency resp.) [default: True]

    Returns
    -------
    ibpd : np.array
           Balanced photocurrent.

    """
    if paramPD is None:
        paramPD = []
    assert E1.shape == E2.shape, "E1 and E2 need to have the same shape"

    i1 = photodiode(E1, paramPD)
    i2 = photodiode(E2, paramPD)
    return i1 - i2


def hybrid_2x4_90deg(Es, Elo):
    """
    Optical 2 x 4 90° hybrid.

    Parameters
    ----------
    Es : np.array
        Input signal optical field.
    Elo : np.array
        Input LO optical field.

    Returns
    -------
    Eo : np.array
        Optical hybrid outputs.

    """
    assert Es.shape == (len(Es),), "Es need to have a (N,) shape"
    assert Elo.shape == (len(Elo),), "Elo need to have a (N,) shape"
    assert Es.shape == Elo.shape, "Es and Elo need to have the same (N,) shape"

    # optical hybrid transfer matrix
    T = np.array(
        [
            [1 / 2, 1j / 2, 1j / 2, -1 / 2],
            [1j / 2, -1 / 2, 1 / 2, 1j / 2],
            [1j / 2, 1 / 2, -1j / 2, -1 / 2],
            [-1 / 2, 1j / 2, -1 / 2, 1j / 2],
        ]
    )

    Ei = np.array([Es, np.zeros((Es.size,)), np.zeros((Es.size,)), Elo])

    return T @ Ei


def coherentReceiver(Es, Elo, paramPD=None):
    """
    Single polarization coherent optical front-end.

    Parameters
    ----------
    Es : np.array
        Input signal optical field.
    Elo : np.array
        Input LO optical field.
    paramPD : parameter object (struct), optional
        Parameters of the photodiodes.

    Returns
    -------
    s : np.array
        Downconverted signal after balanced detection.

    """
    if paramPD is None:
        paramPD = []
    assert Es.shape == (len(Es),), "Es need to have a (N,) shape"
    assert Elo.shape == (len(Elo),), "Elo need to have a (N,) shape"
    assert Es.shape == Elo.shape, "Es and Elo need to have the same (N,) shape"

    # optical 2 x 4 90° hybrid
    Eo = hybrid_2x4_90deg(Es, Elo)

    # balanced photodetection
    sI = balancedPD(Eo[1, :], Eo[0, :], paramPD)
    sQ = balancedPD(Eo[2, :], Eo[3, :], paramPD)

    return sI + 1j * sQ


def pdmCoherentReceiver(Es, Elo, θsig=0, paramPD=None):
    """
    Polarization multiplexed coherent optical front-end.

    Parameters
    ----------
    Es : np.array
        Input signal optical field.
    Elo : np.array
        Input LO optical field.
    θsig : scalar, optional
        Input polarization rotation angle in rad. The default is 0.
    paramPD : parameter object (struct), optional
        Parameters of the photodiodes.

    Returns
    -------
    S : np.array
        Downconverted signal after balanced detection.

    """
    if paramPD is None:
        paramPD = []
    assert len(Es) == len(Elo), "Es and Elo need to have the same length"

    Elox, Eloy = pbs(Elo, θ=np.pi / 4)  # split LO into two orth. polarizations
    Esx, Esy = pbs(Es, θ=θsig)  # split signal into two orth. polarizations

    Sx = coherentReceiver(Esx, Elox, paramPD)  # coherent detection of pol.X
    Sy = coherentReceiver(Esy, Eloy, paramPD)  # coherent detection of pol.Y

    return np.array([Sx, Sy]).T


def edfa(Ei, Fs, G=20, NF=4.5, Fc=193.1e12):
    """
    Implement simple EDFA model.

    Parameters
    ----------
    Ei : np.array
        Input signal field.
    Fs : scalar
        Sampling frequency in Hz.
    G : scalar, optional
        Amplifier gain in dB. The default is 20.
    NF : scalar, optional
        EDFA noise figure in dB. The default is 4.5.
    Fc : scalar, optional
        Central optical frequency. The default is 193.1e12.

    Returns
    -------
    Eo : np.array
        Amplified noisy optical signal.

    """
    assert G > 0, "EDFA gain should be a positive scalar"
    assert NF >= 3, "The minimal EDFA noise figure is 3 dB"

    NF_lin = 10 ** (NF / 10)
    G_lin = 10 ** (G / 10)
    nsp = (G_lin * NF_lin - 1) / (2 * (G_lin - 1))

    # ASE noise power calculation:
    # Ref. Eq.(54) of R. -J. Essiambre,et al, "Capacity Limits of Optical Fiber
    # Networks," in Journal of Lightwave Technology, vol. 28, no. 4,
    # pp. 662-701, Feb.15, 2010, doi: 10.1109/JLT.2009.2039464.

    N_ase = (G_lin - 1) * nsp * const.h * Fc
    p_noise = N_ase * Fs

    noise = normal(0, np.sqrt(p_noise / 2), Ei.shape) + 1j * normal(
        0, np.sqrt(p_noise / 2), Ei.shape
    )
    return Ei * np.sqrt(G_lin) + noise