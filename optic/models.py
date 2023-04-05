"""Basic physical models for optical devices and optical channels."""
import logging as logg

import numpy as np
import scipy.constants as const
from scipy.linalg import norm
from numba import njit
from numpy.fft import fft, fftfreq, ifft
from numpy.random import normal
from tqdm.notebook import tqdm

from optic.dsp import lowPassFIR
from optic.metrics import signal_power

try:
    from optic.dspGPU import firFilter
except ImportError:
    from optic.dsp import firFilter


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


def linFiberCh(Ei, L, alpha, D, Fc, Fs):
    """
    Simulate signal propagation through a linear fiber channel.

    Parameters
    ----------
    Ei : np.array
        Input optical field.
    L : real scalar
        Length of the fiber.
    alpha : real scalar
        Fiber's attenuation coefficient in dB/km.
    D : real scalar
        Fiber's chromatic dispersion (2nd order) coefficient in ps/nm/km.
    Fc : real scalar
        Optical carrier frequency in Hz.
    Fs : real scalar
        Sampling rate of the simulation.

    Returns
    -------
    Eo : np.array
        Optical field at the output of the fiber.

    """
    # c  = 299792458   # speed of light [m/s](vacuum)
    c_kms = const.c / 1e3
    λ = c_kms / Fc
    α = alpha / (10 * np.log10(np.exp(1)))
    β2 = -(D * λ**2) / (2 * np.pi * c_kms)

    Nfft = len(Ei)

    ω = 2 * np.pi * Fs * fftfreq(Nfft)
    ω = ω.reshape(ω.size, 1)

    try:
        Nmodes = Ei.shape[1]
    except IndexError:
        Nmodes = 1
        Ei = Ei.reshape(Ei.size, Nmodes)

    ω = np.tile(ω, (1, Nmodes))
    Eo = ifft(
        fft(Ei, axis=0) * np.exp(-α * L + 1j * (β2 / 2) * (ω**2) * L), axis=0
    )

    if Nmodes == 1:
        Eo = Eo.reshape(
            Eo.size,
        )

    return Eo


def photodiode(E, paramPD=[]):
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


def balancedPD(E1, E2, paramPD=[]):
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


def coherentReceiver(Es, Elo, paramPD=[]):
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
    assert Es.shape == (len(Es),), "Es need to have a (N,) shape"
    assert Elo.shape == (len(Elo),), "Elo need to have a (N,) shape"
    assert Es.shape == Elo.shape, "Es and Elo need to have the same (N,) shape"

    # optical 2 x 4 90° hybrid
    Eo = hybrid_2x4_90deg(Es, Elo)

    # balanced photodetection
    sI = balancedPD(Eo[1, :], Eo[0, :], paramPD)
    sQ = balancedPD(Eo[2, :], Eo[3, :], paramPD)

    return sI + 1j * sQ


def pdmCoherentReceiver(Es, Elo, θsig=0, paramPD=[]):
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


def ssfm(Ei, Fs, paramCh):
    """
    Split-step Fourier method (symmetric, single-pol.).

    :param Ei: input signal
    :param Fs: sampling frequency of Ei [Hz]
    :param paramCh: object with physical parameters of the optical channel

    :paramCh.Ltotal: total fiber length [km][default: 400 km]
    :paramCh.Lspan: span length [km][default: 80 km]
    :paramCh.hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :paramCh.alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :paramCh.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :paramCh.gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :paramCh.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]
    :paramCh.amp: 'edfa', 'ideal', or 'None. [default:'edfa']
    :paramCh.NF: edfa noise figure [dB] [default: 4.5 dB]

    :return Ech: propagated signal
    """
    # check input parameters
    paramCh.Ltotal = getattr(paramCh, "Ltotal", 400)
    paramCh.Lspan = getattr(paramCh, "Lspan", 80)
    paramCh.hz = getattr(paramCh, "hz", 0.5)
    paramCh.alpha = getattr(paramCh, "alpha", 0.2)
    paramCh.D = getattr(paramCh, "D", 16)
    paramCh.gamma = getattr(paramCh, "gamma", 1.3)
    paramCh.Fc = getattr(paramCh, "Fc", 193.1e12)
    paramCh.amp = getattr(paramCh, "amp", "edfa")
    paramCh.NF = getattr(paramCh, "NF", 4.5)
    paramCh.prgsBar = getattr(paramCh, "prgsBar", True)

    Ltotal = paramCh.Ltotal
    Lspan = paramCh.Lspan
    hz = paramCh.hz
    alpha = paramCh.alpha
    D = paramCh.D
    gamma = paramCh.gamma
    Fc = paramCh.Fc
    amp = paramCh.amp
    NF = paramCh.NF
    prgsBar = paramCh.prgsBar

    # channel parameters
    c_kms = const.c / 1e3  # speed of light (vacuum) in km/s
    λ = c_kms / Fc
    α = alpha / (10 * np.log10(np.exp(1)))
    β2 = -(D * λ**2) / (2 * np.pi * c_kms)
    γ = gamma

    # generate frequency axis
    Nfft = len(Ei)
    ω = 2 * np.pi * Fs * fftfreq(Nfft)

    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(np.floor(Lspan / hz))

    Ech = Ei.reshape(
        len(Ei),
    )

    # define linear operator
    linOperator = np.exp(
        -(α / 2) * (hz / 2) + 1j * (β2 / 2) * (ω**2) * (hz / 2)
    )

    for _ in tqdm(range(1, Nspans + 1), disable=not (prgsBar)):
        Ech = fft(Ech)  # single-polarization field

        # fiber propagation step
        for _ in range(1, Nsteps + 1):
            # First linear step (frequency domain)
            Ech = Ech * linOperator

            # Nonlinear step (time domain)
            Ech = ifft(Ech)
            Ech = Ech * np.exp(1j * γ * (Ech * np.conj(Ech)) * hz)

            # Second linear step (frequency domain)
            Ech = fft(Ech)
            Ech = Ech * linOperator

        # amplification step
        Ech = ifft(Ech)
        if amp == "edfa":
            Ech = edfa(Ech, Fs, alpha * Lspan, NF, Fc)
        elif amp == "ideal":
            Ech = Ech * np.exp(α / 2 * Nsteps * hz)
        elif amp is None:
            Ech = Ech * np.exp(0)

    return (
        Ech.reshape(
            len(Ech),
        ),
        paramCh,
    )


def manakovSSF(Ei, Fs, paramCh, prec=np.complex128):
    """
    Run the Manakov split-step Fourier model (symmetric, dual-pol.).

    Parameters
    ----------
    Ei : np.array
        Input optical signal field.
    Fs : scalar
        Sampling frequency in Hz.
    paramCh : parameter object  (struct)
        Object with physical/simulation parameters of the optical channel.

    paramCh.Ltotal: total fiber length [km][default: 400 km]
    paramCh.Lspan: span length [km][default: 80 km]
    paramCh.hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    paramCh.alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    paramCh.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    paramCh.gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    paramCh.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]
    paramCh.amp: 'edfa', 'ideal', or 'None. [default:'edfa']
    paramCh.NF: edfa noise figure [dB] [default: 4.5 dB]
    paramCh.maxIter: max number of iter. in the trap. integration [default: 10]
    paramCh.tol: convergence tol. of the trap. integration.[default: 1e-5]
    paramCh.nlprMethod: adap step-size based on nonl. phase rot. [default: True]
    paramCh.maxNlinPhaseRot: max nonl. phase rot. tolerance [rad][default: 2e-2]
    paramCh.prgsBar: display progress bar? bolean variable [default:True]
    paramCh.saveSpanN: specify the span indexes to be output [default:[]]

    Returns
    -------
    Ech : np.array
        Optical signal after nonlinear propagation.
    paramCh : parameter object  (struct)
        Object with physical/simulation parameters used in the split-step alg.

    """
    # check input parameters
    paramCh.Ltotal = getattr(paramCh, "Ltotal", 400)
    paramCh.Lspan = getattr(paramCh, "Lspan", 80)
    paramCh.hz = getattr(paramCh, "hz", 0.5)
    paramCh.alpha = getattr(paramCh, "alpha", 0.2)
    paramCh.D = getattr(paramCh, "D", 16)
    paramCh.gamma = getattr(paramCh, "gamma", 1.3)
    paramCh.Fc = getattr(paramCh, "Fc", 193.1e12)
    paramCh.amp = getattr(paramCh, "amp", "edfa")
    paramCh.NF = getattr(paramCh, "NF", 4.5)
    paramCh.maxIter = getattr(paramCh, "maxIter", 10)
    paramCh.tol = getattr(paramCh, "tol", 1e-5)
    paramCh.nlprMethod = getattr(paramCh, "nlprMethod", True)
    paramCh.maxNlinPhaseRot = getattr(paramCh, "maxNlinPhaseRot", 2e-2)
    paramCh.prgsBar = getattr(paramCh, "prgsBar", True)
    paramCh.saveSpanN = getattr(
        paramCh, "saveSpanN", [int(paramCh.Ltotal / paramCh.Lspan)]
    )

    Ltotal = paramCh.Ltotal
    Lspan = paramCh.Lspan
    hz = paramCh.hz
    alpha = paramCh.alpha
    D = paramCh.D
    gamma = paramCh.gamma
    Fc = paramCh.Fc
    amp = paramCh.amp
    NF = paramCh.NF
    maxIter = paramCh.maxIter
    tol = paramCh.tol
    prgsBar = paramCh.prgsBar
    saveSpanN = paramCh.saveSpanN
    nlprMethod = paramCh.nlprMethod
    maxNlinPhaseRot = paramCh.maxNlinPhaseRot

    # channel parameters
    c_kms = const.c / 1e3  # speed of light (vacuum) in km/s
    λ = c_kms / Fc
    α = alpha / (10 * np.log10(np.exp(1)))
    β2 = -(D * λ**2) / (2 * np.pi * c_kms)
    γ = gamma

    # generate frequency axis
    Nfft = len(Ei)
    ω = 2 * np.pi * Fs * fftfreq(Nfft)

    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(np.floor(Lspan / hz))

    Ech_x = Ei[:, 0::2].T
    Ech_y = Ei[:, 1::2].T

    # define static part of the linear operator
    argLimOp = np.array(-(α / 2) + 1j * (β2 / 2) * (ω**2)).astype(prec)

    if Ech_x.shape[0] > 1:
        argLimOp = np.tile(argLimOp, (Ech_x.shape[0], 1))
    else:
        argLimOp = argLimOp.reshape(1, -1)

    if saveSpanN:
        Ech_spans = np.zeros(
            (Ei.shape[0], Ei.shape[1] * len(saveSpanN))
        ).astype(prec)
        indRecSpan = 0

    for spanN in tqdm(range(1, Nspans + 1), disable=not (prgsBar)):

        Ex_conv = Ech_x.copy()
        Ey_conv = Ech_y.copy()

        z_current = 0

        # fiber propagation steps
        while z_current < Lspan:

            Pch = Ech_x * np.conj(Ech_x) + Ech_y * np.conj(Ech_y)

            phiRot = nlinPhaseRot(Ex_conv, Ey_conv, Pch, γ)

            if nlprMethod:
                if Lspan - z_current >= maxNlinPhaseRot / np.max(phiRot):
                    hz_ = maxNlinPhaseRot / np.max(phiRot)
                else:
                    hz_ = Lspan - z_current
            else:
                if Lspan - z_current < hz:
                    hz_ = Lspan - z_current  # check that the remaining
                    # distance is not less than hz (due to non-integer
                    # steps/span)
                else:
                    hz_ = hz

            # define the linear operator
            linOperator = np.exp(argLimOp * (hz_ / 2))

            # First linear step (frequency domain)
            Ex_hd = ifft(fft(Ech_x) * linOperator)
            Ey_hd = ifft(fft(Ech_y) * linOperator)

            # Nonlinear step (time domain)
            for nIter in range(maxIter):
                rotOperator = np.exp(1j * phiRot * hz_)

                Ech_x_fd = Ex_hd * rotOperator
                Ech_y_fd = Ey_hd * rotOperator

                # Second linear step (frequency domain)
                Ech_x_fd = ifft(fft(Ech_x_fd) * linOperator)
                Ech_y_fd = ifft(fft(Ech_y_fd) * linOperator)

                # check convergence o trapezoidal integration in phiRot
                lim = convergenceCondition(
                    Ech_x_fd, Ech_y_fd, Ex_conv, Ey_conv
                )

                Ex_conv = Ech_x_fd.copy()
                Ey_conv = Ech_y_fd.copy()

                if lim < tol:
                    break
                elif nIter == maxIter - 1:
                    logg.warning(
                        f"Warning: target SSFM error tolerance was not achieved in {maxIter} iterations"
                    )

                phiRot = nlinPhaseRot(Ex_conv, Ey_conv, Pch, γ)

            Ech_x = Ech_x_fd.copy()
            Ech_y = Ech_y_fd.copy()

            z_current += hz_  # update propagated distance
        # amplification step
        if amp == "edfa":
            Ech_x = edfa(Ech_x, Fs, alpha * Lspan, NF, Fc)
            Ech_y = edfa(Ech_y, Fs, alpha * Lspan, NF, Fc)
        elif amp == "ideal":
            Ech_x = Ech_x * np.exp(α / 2 * Nsteps * hz)
            Ech_y = Ech_y * np.exp(α / 2 * Nsteps * hz)
        elif amp is None:
            Ech_x = Ech_x * np.exp(0)
            Ech_y = Ech_y * np.exp(0)

        if spanN in saveSpanN:
            Ech_spans[:, 2 * indRecSpan: 2 * indRecSpan + 1] = Ech_x.T
            Ech_spans[:, 2 * indRecSpan + 1: 2 * indRecSpan + 2] = Ech_y.T
            indRecSpan += 1

    if saveSpanN:
        Ech = Ech_spans
    else:
        Ech = Ei.copy()
        Ech[:, 0::2] = Ech_x.T
        Ech[:, 1::2] = Ech_y.T

    return Ech, paramCh


def nlinPhaseRot(Ex, Ey, Pch, γ):
    """
    Calculate nonlinear phase-shift per step for the Manakov SSFM.

    Parameters
    ----------
    Ex : np.array
        Input optical signal field of x-polarization.
    Ey : np.array
        Input optical signal field of y-polarization.
    Pch : np.array
        Input optical power.
    γ : real scalar
        fiber nonlinearity coefficient.

    Returns
    -------
    np.array
        nonlinear phase-shift of each sample of the signal.

    """
    return ((8 / 9) * γ * (Pch + Ex * np.conj(Ex) + Ey * np.conj(Ey)) / 2).real


def convergenceCondition(Ex_fd, Ey_fd, Ex_conv, Ey_conv):
    """
    Verify the convergence condition for the trapezoidal integration.

    Parameters
    ----------
    Ex_fd : np.array
        field of x-polarization fully dispersed and rotated.
    Ey_fd : np.array
        field of y-polarization fully dispersed and rotated.
    Ex_conv : np.array
        field of x-polarization at the begining of the step.
    Ey_conv : np.array
        Ifield of y-polarization at the begining of the step.

    Returns
    -------
    scalar
        squared root of the MSE normalized by the power of the fields.

    """
    return np.sqrt(
        norm(Ex_fd - Ex_conv) ** 2 + norm(Ey_fd - Ey_conv) ** 2
    ) / np.sqrt(norm(Ex_conv) ** 2 + norm(Ey_conv) ** 2)


@njit
def phaseNoise(lw, Nsamples, Ts):
    """
    Generate realization of a random-walk phase-noise process.

    Parameters
    ----------
    lw : scalar
        laser linewidth.
    Nsamples : scalar
        number of samples to be draw.
    Ts : scalar
        sampling period.

    Returns
    -------
    phi : np.array
        realization of the phase noise process.

    """
    σ2 = 2 * np.pi * lw * Ts
    phi = np.zeros(Nsamples)

    for ind in range(Nsamples - 1):
        phi[ind + 1] = phi[ind] + normal(0, np.sqrt(σ2))

    return phi


@njit
def awgn(sig, snr, Fs=1, B=1):
    """
    Implement an AWGN channel.

    Parameters
    ----------
    sig : np.array
        Input signal.
    snr : scalar
        Signal-to-noise ratio in dB.
    Fs : real scalar
        Sampling frequency. The default is 1.
    B : real scalar
        Signal bandwidth. The default is 1.

    Returns
    -------
    np.array
        Input signal plus noise.

    """
    snr_lin = 10 ** (snr / 10)
    noiseVar = signal_power(sig) / snr_lin
    σ = np.sqrt((Fs / B) * noiseVar)
    noise = normal(0, σ, sig.shape) + 1j * normal(0, σ, sig.shape)
    noise = 1 / np.sqrt(2) * noise

    return sig + noise
