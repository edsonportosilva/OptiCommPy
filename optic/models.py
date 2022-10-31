import numpy as np
import scipy.constants as const
from numba import njit
from numpy.fft import fft, fftfreq, ifft
from numpy.random import normal
from tqdm.notebook import tqdm
from optic.metrics import signal_power
from optic.dsp import lowPassFIR

try:
    from optic.dspGPU import firFilter
except:
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
        assert Ai.shape == u.shape, "Ai and u need to have the same dimensions"
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
        assert Ai.shape == u.shape, "Ai and u need to have the same dimensions"
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
    Linear fiber channel w/ loss and chromatic dispersion.

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
    paramPD : struct, optional

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
    assert Fs >= 2*B, "Sampling frequency Fs needs to be at least twice of B."

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


def balancedPD(E1, E2, R=1):
    """
    Balanced photodiode (BPD).

    Parameters
    ----------
    E1 : np.array
        Input optical field.
    E2 : np.array
        Input optical field.
    R : scalar, optional
        Photodiode responsivity in A/W. The default is 1.

    Returns
    -------
    ibpd : np.array
           Balanced photocurrent.

    """
    assert R > 0, "PD responsivity should be a positive scalar"
    assert E1.shape == E2.shape, "E1 and E2 need to have the same shape"

    i1 = R * E1 * np.conj(E1)
    i2 = R * E2 * np.conj(E2)
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


def coherentReceiver(Es, Elo, Rd=1):
    """
    Single polarization coherent optical front-end.

    Parameters
    ----------
    Es : np.array
        Input signal optical field.
    Elo : np.array
        Input LO optical field.
    Rd : scalar, optional
        Photodiodes responsivity in A/W. The default is 1.

    Returns
    -------
    s : np.array
        Downconverted signal after balanced detection.

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

    return sI + 1j * sQ


def pdmCoherentReceiver(Es, Elo, θsig=0, Rdx=1, Rdy=1):
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
    Rdx : scalar, optional
        Photodiode resposivity pol.X in A/W. The default is 1.
    Rdy : scalar, optional
        Photodiode resposivity pol.Y in A/W. The default is 1.

    Returns
    -------
    S : np.array
        Downconverted signal after balanced detection.

    """
    assert Rdx > 0 and Rdy > 0, "PD responsivity should be a positive scalar"
    assert len(Es) == len(Elo), "Es and Elo need to have the same length"

    Elox, Eloy = pbs(Elo, θ=np.pi / 4)  # split LO into two orth. polarizations
    Esx, Esy = pbs(Es, θ=θsig)  # split signal into two orth. polarizations

    Sx = coherentReceiver(Esx, Elox, Rd=Rdx)  # coherent detection of pol.X
    Sy = coherentReceiver(Esy, Eloy, Rd=Rdy)  # coherent detection of pol.Y

    return np.array([Sx, Sy]).T


def edfa(Ei, Fs, G=20, NF=4.5, Fc=193.1e12):
    """
    Implement simple EDFA model.

    Parameters
    ----------
    Ei : np.array
        Input signal field .
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


def manakovSSF(Ei, Fs, paramCh):
    """
    Manakov split-step Fourier model (symmetric, dual-pol.).

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

    Ech_x = Ei[:, 0].reshape(
        len(Ei),
    )
    Ech_y = Ei[:, 1].reshape(
        len(Ei),
    )

    # define linear operator
    linOperator = np.exp(
        -(α / 2) * (hz / 2) + 1j * (β2 / 2) * (ω**2) * (hz / 2)
    )

    for _ in tqdm(range(1, Nspans + 1), disable=not (prgsBar)):
        Ech_x = fft(Ech_x)  # polarization x field
        Ech_y = fft(Ech_y)  # polarization y field

        # fiber propagation step
        for _ in range(1, Nsteps + 1):
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

    Ech = np.array(
        [
            Ech_x.reshape(
                len(Ei),
            ),
            Ech_y.reshape(
                len(Ei),
            ),
        ]
    ).T

    return Ech, paramCh


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
        input signal.
    snr : scalar
        signal-to-noise ratio in dB.
    Fs : scalar
        sampling frequency. The default is 1.
    B : scalar
        signal bandwidth. The default is 1.

    Returns
    -------
    sigNoisy : np.array
        input signal plus noise.

    """
    snr_lin = 10 ** (snr / 10)
    noiseVar = signal_power(sig) / snr_lin
    σ = np.sqrt((Fs / B) * noiseVar)
    noise = normal(0, σ, sig.size) + 1j * normal(
        0, σ, sig.size
    )
    noise = 1 / np.sqrt(2) * noise

    return sig + noise
