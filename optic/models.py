import numpy as np
import scipy.constants as const
from numba import njit
from numpy.fft import fft, fftfreq, ifft
from numpy.random import normal
from tqdm.notebook import tqdm


def mzm(Ai, u, Vπ, Vb):
    """
    Optical Mach-Zehnder Modulator (MZM)

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
        assert Ai.shape == u.shape, "Ai and u need to have the same dimensions"
    except AttributeError:
        Ai = Ai * np.ones(u.shape)

    π = np.pi
    Ao = Ai * np.cos(0.5 / Vπ * (u + Vb) * π)

    return Ao


def iqm(Ai, u, Vπ, VbI, VbQ):
    """
    Optical In-Phase/Quadrature Modulator (IQM)

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
        assert Ai.shape == u.shape, "Ai and u need to have the same dimensions"
    except AttributeError:
        Ai = Ai * np.ones(u.shape)

    Ao = mzm(Ai / np.sqrt(2), u.real, Vπ, VbI) + 1j * mzm(
        Ai / np.sqrt(2), u.imag, Vπ, VbQ
    )

    return Ao


def pbs(E, θ=0):
    """
    Polarization beam splitter (PBS)

    Parameters
    ----------
    E : (N,2) nparray
        Input pol. multiplexed optical field.
    θ : scalar, optional
        Rotation angle of input field in radians. The default is 0.

    Returns
    -------
    Ex : (N,) nparray
        Ex output single pol. field.
    Ey : (N,) nparray
        Ey output single pol. field.

    """
    try:
        assert E.shape[1] == 2, "E need to be a (N,2) or a (N,) nparray"
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
    Linear fiber channel w/ loss and chromatic dispersion

    :param Ei: optical signal at the input of the fiber
    :param L: fiber length [km]
    :param alpha: loss coeficient [dB/km]
    :param D: chromatic dispersion parameter [ps/nm/km]
    :param Fc: carrier frequency [Hz]
    :param Fs: sampling frequency [Hz]

    :return Eo: optical signal at the output of the fiber
    """
    # c  = 299792458   # speed of light [m/s](vacuum)
    c_kms = const.c / 1e3
    λ = c_kms / Fc
    α = alpha / (10 * np.log10(np.exp(1)))
    β2 = -(D * λ ** 2) / (2 * np.pi * c_kms)

    Nfft = len(Ei)

    ω = 2 * np.pi * Fs * fftfreq(Nfft)
    ω = ω.reshape(ω.size, 1)

    try:
        Nmodes = Ei.shape[1]
    except IndexError:
        Nmodes = 1
        Ei = Ei.reshape(Ei.size, Nmodes)

    ω = np.tile(ω, (1, Nmodes))
    Eo = ifft(fft(Ei, axis=0) * np.exp(-α * L + 1j * (β2 / 2) * (ω ** 2) * L), axis=0)

    if Nmodes == 1:
        Eo = Eo.reshape(Eo.size,)

    return Eo


def balancedPD(E1, E2, R=1):
    """
    Balanced photodiode (BPD)

    Parameters
    ----------
    E1 : nparray
        Input optical field.
    E2 : nparray
        Input optical field.
    R : scalar, optional
        Photodiode responsivity in A/W. The default is 1.

    Returns
    -------
    ibpd : nparray
           Balanced photocurrent.

    """

    assert R > 0, "PD responsivity should be a positive scalar"
    assert E1.shape == E2.shape, "E1 and E2 need to have the same shape"

    i1 = R * E1 * np.conj(E1)
    i2 = R * E2 * np.conj(E2)
    ibpd = i1 - i2

    return ibpd


def hybrid_2x4_90deg(E1, E2):
    """
    Optical 2 x 4 90° hybrid

    :param E1: input signal field [nparray]
    :param E2: input LO field [nparray]

    :return: hybrid outputs
    """
    assert E1.shape == (len(E1),), "E1 need to have a (N,) shape"
    assert E2.shape == (len(E2),), "E2 need to have a (N,) shape"
    assert E1.shape == E2.shape, "E1 and E2 need to have the same (N,) shape"

    # optical hybrid transfer matrix
    T = np.array(
        [
            [1 / 2, 1j / 2, 1j / 2, -1 / 2],
            [1j / 2, -1 / 2, 1 / 2, 1j / 2],
            [1j / 2, 1 / 2, -1j / 2, -1 / 2],
            [-1 / 2, 1j / 2, -1 / 2, 1j / 2],
        ]
    )

    Ei = np.array([E1, np.zeros((E1.size,)), np.zeros((E1.size,)), E2])

    Eo = T @ Ei

    return Eo


def coherentReceiver(Es, Elo, Rd=1):
    """
    Single polarization coherent optical front-end

    :param Es: input signal field [nparray]
    :param Elo: input LO field [nparray]
    :param Rd: photodiode resposivity [scalar]

    :return: downconverted signal after balanced detection
    """
    assert Rd > 0, "PD responsivity should be a positive scalar"
    assert Es.shape == (len(Es),), "Es need to have a (N,) shape"
    assert Elo.shape == (len(Elo),), "Elo need to have a (N,) shape"
    assert Es.shape == Elo.shape, "Es and Elo need to have the same (N,) shape"

    # optical 2 x 4 90° hybrid
    Eo = hybrid_2x4_90deg(Es, Elo)

    # balanced photodetection
    sI = balancedPD(Eo[1, :], Eo[0, :], Rd)
    sQ = balancedPD(Eo[2, :], Eo[3, :], Rd)

    s = sI + 1j * sQ

    return s


def pdmCoherentReceiver(Es, Elo, θsig=0, Rdx=1, Rdy=1):
    """
    Polarization multiplexed coherent optical front-end

    :param Es: input signal field [2d nparray]
    :param Elo: input LO field [nparray]
    :param θsig: polarization rotation angle [rad][default: 0]
    :param Rdx: photodiode resposivity pol.X [scalar]
    :param Rdy: photodiode resposivity pol.Y [scalar]

    :return: downconverted signal after balanced detection
    """
    assert Rdx > 0 and Rdy > 0, "PD responsivity should be a positive scalar"
    assert len(Es) == len(Elo), "Es and Elo need to have the same number of samples"

    Elox, Eloy = pbs(Elo, θ=np.pi / 4)  # split LO into two orthogonal polarizations
    Esx, Esy = pbs(Es, θ=θsig)  # split signal into two orthogonal polarizations

    Sx = coherentReceiver(Esx, Elox, Rd=Rdx)  # coherent detection of pol.X
    Sy = coherentReceiver(Esy, Eloy, Rd=Rdy)  # coherent detection of pol.Y

    # Sx = Sx.reshape(len(Sx), 1)
    # Sy = Sy.reshape(len(Sy), 1)
    S  = np.array([Sx, Sy]) 
    
    return S #np.concatenate((Sx, Sy), axis=1)


def edfa(Ei, Fs, G=20, NF=4.5, Fc=193.1e12):
    """
    Simple EDFA model

    :param Ei: input signal field [nparray]
    :param Fs: sampling frequency [Hz][scalar]
    :param G: gain [dB][scalar, default: 20 dB]
    :param NF: EDFA noise figure [dB][scalar, default: 4.5 dB]
    :param Fc: optical center frequency [Hz][scalar, default: 193.1e12 Hz]

    :return: amplified noisy optical signal [nparray]
    """
    assert G > 0, "EDFA gain should be a positive scalar"
    assert NF >= 3, "The minimal EDFA noise figure is 3 dB"

    NF_lin = 10 ** (NF / 10)
    G_lin = 10 ** (G / 10)
    nsp = (G_lin * NF_lin - 1) / (2 * (G_lin - 1))
    N_ase = (G_lin - 1) * nsp * const.h * Fc
    p_noise = N_ase * Fs
    noise = normal(0, np.sqrt(p_noise / 2), Ei.shape) + 1j * normal(
        0, np.sqrt(p_noise / 2), Ei.shape
    )
    return Ei * np.sqrt(G_lin) + noise


def ssfm(Ei, Fs, paramCh):
    """
    Split-step Fourier method (symmetric, single-pol.)

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

    Ltotal = paramCh.Ltotal
    Lspan = paramCh.Lspan
    hz = paramCh.hz
    alpha = paramCh.alpha
    D = paramCh.D
    gamma = paramCh.gamma
    Fc = paramCh.Fc
    amp = paramCh.amp
    NF = paramCh.NF

    # channel parameters
    c_kms = const.c / 1e3  # speed of light (vacuum) in km/s
    λ = c_kms / Fc
    α = alpha / (10 * np.log10(np.exp(1)))
    β2 = -(D * λ ** 2) / (2 * np.pi * c_kms)
    γ = gamma

    # generate frequency axis
    Nfft = len(Ei)
    ω = 2 * np.pi * Fs * fftfreq(Nfft)

    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(np.floor(Lspan / hz))

    Ech = Ei.reshape(len(Ei),)

    # define linear operator
    linOperator = np.exp(-(α / 2) * (hz / 2) + 1j * (β2 / 2) * (ω ** 2) * (hz / 2))

    for spanN in tqdm(range(1, Nspans + 1)):
        Ech = fft(Ech)  # single-polarization field

        # fiber propagation step
        for stepN in range(1, Nsteps + 1):
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

    return Ech.reshape(len(Ech),), paramCh


def manakovSSF(Ei, Fs, paramCh):
    """
    Manakov model split-step Fourier (symmetric, dual-pol.)

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

    Ltotal = paramCh.Ltotal
    Lspan = paramCh.Lspan
    hz = paramCh.hz
    alpha = paramCh.alpha
    D = paramCh.D
    gamma = paramCh.gamma
    Fc = paramCh.Fc
    amp = paramCh.amp
    NF = paramCh.NF

    # channel parameters
    c_kms = const.c / 1e3  # speed of light (vacuum) in km/s
    λ = c_kms / Fc
    α = alpha / (10 * np.log10(np.exp(1)))
    β2 = -(D * λ ** 2) / (2 * np.pi * c_kms)
    γ = gamma

    # generate frequency axis
    Nfft = len(Ei)
    ω = 2 * np.pi * Fs * fftfreq(Nfft)

    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(np.floor(Lspan / hz))

    Ech_x = Ei[:, 0].reshape(len(Ei),)
    Ech_y = Ei[:, 1].reshape(len(Ei),)

    # define linear operator
    linOperator = np.exp(-(α / 2) * (hz / 2) + 1j * (β2 / 2) * (ω ** 2) * (hz / 2))

    for spanN in tqdm(range(1, Nspans + 1)):
        Ech_x = fft(Ech_x)  # polarization x field
        Ech_y = fft(Ech_y)  # polarization y field

        # fiber propagation step
        for stepN in range(1, Nsteps + 1):
            # First linear step (frequency domain)
            Ech_x = Ech_x * linOperator
            Ech_y = Ech_y * linOperator

            # Nonlinear step (time domain)
            Ex = ifft(Ech_x)
            Ey = ifft(Ech_y)
            Ech_x = Ex * np.exp(
                1j * (8 / 9) * γ * (Ex * np.conj(Ex) + Ey * np.conj(Ey)) * hz
            )
            Ech_y = Ey * np.exp(
                1j * (8 / 9) * γ * (Ex * np.conj(Ex) + Ey * np.conj(Ey)) * hz
            )

            # Second linear step (frequency domain)
            Ech_x = fft(Ech_x)
            Ech_y = fft(Ech_y)

            Ech_x = Ech_x * linOperator
            Ech_y = Ech_y * linOperator

        # amplification step
        Ech_x = ifft(Ech_x)
        Ech_y = ifft(Ech_y)

        if amp == "edfa":
            Ech_x = edfa(Ech_x, Fs, alpha * Lspan, NF, Fc)
            Ech_y = edfa(Ech_y, Fs, alpha * Lspan, NF, Fc)
        elif amp == "ideal":
            Ech_x = Ech_x * np.exp(α / 2 * Nsteps * hz)
            Ech_y = Ech_y * np.exp(α / 2 * Nsteps * hz)
        elif amp is None:
            Ech_x = Ech_x * np.exp(0)
            Ech_y = Ech_y * np.exp(0)

    Ech = np.array([Ech_x.reshape(len(Ei),), Ech_y.reshape(len(Ei),)]).T

    return Ech, paramCh


@njit
def phaseNoise(lw, Nsamples, Ts):

    σ2 = 2 * np.pi * lw * Ts
    phi = np.zeros(Nsamples)

    for ind in range(0, Nsamples - 1):
        phi[ind + 1] = phi[ind] + normal(0, np.sqrt(σ2))

    return phi
