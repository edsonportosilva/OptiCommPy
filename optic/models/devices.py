"""
========================================================
Models for optical devices (:mod:`optic.models.devices`)
========================================================

.. autosummary::
   :toctree: generated/

   pm                    -- Optical phase modulator
   mzm                   -- Optical Mach-Zhender modulator
   iqm                   -- Optical In-Phase/Quadrature Modulator (IQM)
   pbs                   -- Polarization beam splitter (PBS)
   hybrid_2x4_90deg      -- Optical 2 x 4 90° hybrid
   photodiode            -- Pin photodiode
   balancedPD            -- Balanced photodiode pair
   coherentReceiver      -- Optical coherent receiver (single polarization)
   pdmCoherentReceiver   -- Optical polarization-multiplexed coherent receiver
   edfa                  -- Simple EDFA model (gain + AWGN noise)
"""


"""Basic physical models for optical devices."""
import logging as logg

import numpy as np
import scipy.constants as const
from optic.utils import parameters
from optic.dsp.core import lowPassFIR, gaussianComplexNoise, gaussianNoise, phaseNoise

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
            assert Ai.shape == u.shape, "Ai and u need to have the same dimensions"
    except AttributeError:
        Ai = Ai * np.ones(u.shape)

    π = np.pi
    return Ai * np.exp(1j * (u / Vπ) * π)


def mzm(Ai, u, param=None):
    """
    Optical Mach-Zehnder Modulator (MZM).

    Parameters
    ----------
    Ai : scalar or np.array
        Amplitude of the optical field at the input of the MZM.
    u : np.array
        Electrical driving signal.
    param : parameter object  (struct)
        Object with physical/simulation parameters of the mzm.

        - param.Vpi: MZM's Vpi voltage [V][default: 2 V]

        - param.Vb: MZM's bias voltage [V][default: -1 V]

    Returns
    -------
    Ao : np.array
        Modulated optical field at the output of the MZM.

    """
    if param is None:
        param = []

    # check input parameters
    Vpi = getattr(param, "Vpi", 2)
    Vb = getattr(param, "Vb", -1)

    try:
        u.shape
    except AttributeError:
        u = np.array([u])

    try:
        if Ai.shape == () and u.shape != ():
            Ai = Ai * np.ones(u.shape)
        else:
            assert Ai.shape == u.shape, "Ai and u need to have the same dimensions"
    except AttributeError:
        Ai = Ai * np.ones(u.shape)

    π = np.pi
    return Ai * np.cos(0.5 / Vpi * (u + Vb) * π)


def iqm(Ai, u, param=None):
    """
    Optical In-Phase/Quadrature Modulator (IQM).

    Parameters
    ----------
    Ai : scalar or np.array
        Amplitude of the optical field at the input of the IQM.
    u : complex-valued np.array
        Modulator's driving signal (complex-valued baseband).
    param : parameter object  (struct)
        Object with physical/simulation parameters of the mzm.

        - param.Vpi: MZM's Vpi voltage [V][default: 2 V]

        - param.VbI: I-MZM's bias voltage [V][default: -2 V]

        - param.VbQ: Q-MZM's bias voltage [V][default: -2 V]

        - param.Vphi: PM bias voltage [V][default: 1 V]

    Returns
    -------
    Ao : complex-valued np.array
        Modulated optical field at the output of the IQM.

    """
    if param is None:
        param = []

    # check input parameters
    Vpi = getattr(param, "Vpi", 2)
    VbI = getattr(param, "VbI", -2)
    VbQ = getattr(param, "VbQ", -2)
    Vphi = getattr(param, "Vphi", 1)

    try:
        u.shape
    except AttributeError:
        u = np.array([u])

    try:
        if Ai.shape == () and u.shape != ():
            Ai = Ai * np.ones(u.shape)
        else:
            assert Ai.shape == u.shape, "Ai and u need to have the same dimensions"
    except AttributeError:
        Ai = Ai * np.ones(u.shape)

    # define parameters for the I-MZM:
    paramI = parameters()
    paramI.Vpi = Vpi
    paramI.Vb = VbI

    # define parameters for the Q-MZM:
    paramQ = parameters()
    paramQ.Vpi = Vpi
    paramQ.Vb = VbQ

    return mzm(Ai / np.sqrt(2), u.real, paramI) + pm(
        mzm(Ai / np.sqrt(2), u.imag, paramQ), Vphi * np.ones(u.shape), Vpi
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


def photodiode(E, param=None):
    """
    Pin photodiode (PD).

    Parameters
    ----------
    E : np.array
        Input optical field.
    param : parameter object (struct), optional
        Parameters of the photodiode.

        - param.R: photodiode responsivity [A/W][default: 1 A/W]

        - param.Tc: temperature [°C][default: 25°C]

        - param.Id: dark current [A][default: 5e-9 A]

        - param.Ipd_sat: saturation value of the photocurrent [A][default: 5e-3 A]

        - param.RL: impedance load [Ω] [default: 50Ω]

        - param.B bandwidth [Hz][default: 30e9 Hz]

        - param.Fs: sampling frequency [Hz] [default: None]

        - param.fType: frequency response type [default: 'rect']

        - param.N: number of the frequency resp. filter taps. [default: 8001]

        - param.ideal: ideal PD?(i.e. no noise, no frequency resp.) [default: True]

    Returns
    -------
    ipd : np.array
          photocurrent.

    """
    if param is None:
        param = []
    kB = const.value("Boltzmann constant")
    q = const.value("elementary charge")

    # check input parameters
    R = getattr(param, "R", 1)
    Tc = getattr(param, "Tc", 25)
    Id = getattr(param, "Id", 5e-9)
    RL = getattr(param, "RL", 50)
    B = getattr(param, "B", 30e9)
    Ipd_sat = getattr(param, "Ipd_sat", 5e-3)
    N = getattr(param, "N", 8000)
    fType = getattr(param, "fType", "rect")
    ideal = getattr(param, "ideal", True)

    assert R > 0, "PD responsivity should be a positive scalar"

    ipd = R * E * np.conj(E)  # ideal photocurrent

    if not (ideal):
        try:
            Fs = param.Fs
        except AttributeError:
            logg.error("Simulation sampling frequency (Fs) not provided.")

        assert Fs >= 2 * B, "Sampling frequency Fs needs to be at least twice of B."

        ipd[ipd > Ipd_sat] = Ipd_sat  # saturation of the photocurrent

        ipd_mean = ipd.mean().real

        # shot noise
        σ2_s = 2 * q * (ipd_mean + Id) * B  # shot noise variance

        # thermal noise
        T = Tc + 273.15  # temperature in Kelvin
        σ2_T = 4 * kB * T * B / RL  # thermal noise variance

        # add noise sources to the p-i-n receiver
        Is = np.random.normal(0, np.sqrt(Fs * (σ2_s / (2 * B))), ipd.size)
        It = np.random.normal(0, np.sqrt(Fs * (σ2_T / (2 * B))), ipd.size)

        ipd += Is + It

        # lowpass filtering
        h = lowPassFIR(B, Fs, N, typeF=fType)
        ipd = firFilter(h, ipd)

    return ipd.real


def balancedPD(E1, E2, param=None):
    """
    Balanced photodiode (BPD).

    Parameters
    ----------
    E1 : np.array
        Input optical field.
    E2 : np.array
        Input optical field.
    param : parameter object (struct), optional
        Parameters of the photodiodes.

        - param.R: photodiode responsivity [A/W][default: 1 A/W]
        - param.Tc: temperature [°C][default: 25°C]
        - param.Id: dark current [A][default: 5e-9 A]
        - param.RL: impedance load [Ω] [default: 50Ω]
        - param.B bandwidth [Hz][default: 30e9 Hz]
        - param.Fs: sampling frequency [Hz] [default: 60e9 Hz]
        - param.fType: frequency response type [default: 'rect']
        - param.N: number of the frequency resp. filter taps. [default: 8001]
        - param.ideal: ideal PD?(i.e. no noise, no frequency resp.) [default: True]

    Returns
    -------
    ibpd : np.array
           Balanced photocurrent.

    """
    assert E1.shape == E2.shape, "E1 and E2 need to have the same shape"

    i1 = photodiode(E1, param)
    i2 = photodiode(E2, param)
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


def coherentReceiver(Es, Elo, param=None):
    """
    Single polarization coherent optical front-end.

    Parameters
    ----------
    Es : np.array
        Input signal optical field.
    Elo : np.array
        Input LO optical field.
    param : parameter object (struct), optional
        Parameters of the photodiodes.

    Returns
    -------
    s : np.array
        Downconverted signal after balanced detection.

    """
    assert Es.shape == (len(Es),), "Es need to have a (N,) shape"
    assert Elo.shape == (len(Elo),), "Elo need to have a (N,) shape"
    assert Es.shape == Elo.shape, "Es and Elo need to have the same (N,) shape"

    # optical 2 x 4 90° hybrid
    Eo = hybrid_2x4_90deg(Es, Elo)

    # balanced photodetection
    sI = balancedPD(Eo[1, :], Eo[0, :], param)
    sQ = balancedPD(Eo[2, :], Eo[3, :], param)

    return sI + 1j * sQ


def pdmCoherentReceiver(Es, Elo, θsig=0, param=None):
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
    param : parameter object (struct), optional
        Parameters of the photodiodes.

    Returns
    -------
    S : np.array
        Downconverted signal after balanced detection.

    """
    assert len(Es) == len(Elo), "Es and Elo need to have the same length"

    Elox, Eloy = pbs(Elo, θ=np.pi / 4)  # split LO into two orth. polarizations
    Esx, Esy = pbs(Es, θ=θsig)  # split signal into two orth. polarizations

    Sx = coherentReceiver(Esx, Elox, param)  # coherent detection of pol.X
    Sy = coherentReceiver(Esy, Eloy, param)  # coherent detection of pol.Y

    return np.array([Sx, Sy]).T


def edfa(Ei, param=None):
    """
    Implement simple EDFA model.

    Parameters
    ----------
    Ei : np.array
        Input signal field.
    param : parameter object (struct), optional
        Parameters of the edfa.

        - param.G : amplifier gain in dB. The default is 20.
        - param.NF : EDFA noise figure in dB. The default is 4.5.
        - param.Fc : central optical frequency. The default is 193.1e12.
        - param.Fs : sampling frequency in samples/second.

    Returns
    -------
    Eo : np.array
        Amplified noisy optical signal.

    """
    try:
        Fs = param.Fs
    except AttributeError:
        logg.error("Simulation sampling frequency (Fs) not provided.")

    # check input parameters
    G = getattr(param, "G", 20)
    NF = getattr(param, "NF", 4.5)
    Fc = getattr(param, "Fc", 193.1e12)

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

    noise = gaussianComplexNoise(Ei.shape, p_noise)

    return Ei * np.sqrt(G_lin) + noise

def basicLaserModel(E, param=None):
    """
    Laser model with Maxwellian random walk phase noise and RIN.

    Parameters
    ----------
    E : np.array
        Input optical field.
    param : parameter object (struct), optional
        Parameters of the laser.

        - param.P: laser power [W]
        - param.lambda: laser wavelength [m]
        - param.lw: laser linewidth [Hz]
        - param.Fs: sampling rate [Hz]
        - param.Ns: number of signal samples

    Returns
    -------
    optical_signal : np.array
          Optical signal with phase noise and RIN.

    """
    try:
        Fs = param.Fs
    except AttributeError:
        logg.error("Simulation sampling frequency (Fs) not provided.")
  
    P = getattr(param,"P", 1e-3)              # Laser power in watts    
    lw = getattr(param,"lw", 1e3)             # Linewidth in Hz
    RIN_var = getattr(param,"RIN_var", 1e-20) # RIN variance    
    Ns = getattr(param,"Ns", 1000)            # Number of samples of the signal

    t = np.arange(0, Ns)* 1 / Fs
    
    # Simulate Maxwellian random walk phase noise    
    pn = phaseNoise(lw, Ns, 1/Fs)

    # Simulate relative intensity noise  (RIN)
    deltaP = gaussianNoise(pn.shape, RIN_var)
    
    # Return optical signal       
    return np.sqrt(P + deltaP)*np.exp(1j*pn)
