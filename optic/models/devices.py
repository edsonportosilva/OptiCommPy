"""
===============================================================
Models for optoelectronic devices (:mod:`optic.models.devices`)
===============================================================

.. autosummary::
   :toctree: generated/

   pm                    -- Optical phase modulator.
   mzm                   -- Optical Mach-Zhender modulator.
   iqm                   -- Optical In-Phase/Quadrature Modulator (IQM).
   pbs                   -- Polarization beam splitter (PBS).
   opticalHybrid2x4      -- Optical hybrid 2 x 4 90°.
   voa                   -- Variable optical attenuator (VOA).
   photodiode            -- Pin photodiode.
   balancedPD            -- Balanced photodiode pair.
   coherentReceiver      -- Optical coherent receiver (single polarization).
   pdmCoherentReceiver   -- Optical polarization-multiplexed coherent receiver.
   edfa                  -- Simple EDFA model (gain + AWGN noise).
   basicLaserModel       -- Laser model with Maxwellian random walk phase noise and RIN.
   adc                   -- Analog-to-digital converter (ADC) model.
   dac                   -- Digital-to-analog converter (DAC) model.
"""

"""Basic physical models for optical/electronic devices."""
import logging as logg

import numpy as np
import scipy.constants as const

from optic.dsp.core import (
    calcMZM,
    calcPM,
    clockSamplingInterp,
    delaySignal,
    gaussianComplexNoise,
    gaussianNoise,
    iqMixing,
    lowPassFIR,
    phaseNoise,
    quantizer,
)
from optic.utils import dBm2W, parameters

try:
    from optic.dsp.coreGPU import checkGPU

    if checkGPU():
        from optic.dsp.coreGPU import firFilter
    else:
        from optic.dsp.core import firFilter
except ImportError:
    from optic.dsp.core import firFilter


def checkModulatorInputs(Ei, u):
    """
    Check the inputs of the optical modulator models.

    Parameters
    ----------
    Ei : scalar or np.array
        Optical field at the input of the modulator.
    u : scalar or np.array
        Electrical driving signal.

    Returns
    -------
    Ei : scalar or np.array
        Optical field (scalar fields are kept as scalars, since calcPM and
        calcMZM broadcast them to the shape of u).
    u : np.array
        Electrical driving signal as an array.

    """
    try:
        u.shape
    except AttributeError:
        u = np.array([u])

    if np.ndim(Ei) == 0:
        Ei = Ei[()] if isinstance(Ei, np.ndarray) else Ei  # 0-d array -> scalar
    else:
        assert Ei.shape == u.shape, "Ei and u need to have the same dimensions"

    return Ei, u


def pm(Ei, u, Vπ):
    r"""
    Optical Phase Modulator (PM).

    Parameters
    ----------
    Ei : scalar or np.array
        Optical field at the input of the PM.
    u : np.array
        Electrical driving signal.
    Vπ : scalar
        PM's Vπ voltage.
    Returns
    -------
    Ao : np.array
        Modulated optical field at the output of the PM.

    Notes
    -----
    The electro-optic (Pockels) effect in a material such as lithium niobate changes
    its refractive index proportionally to the applied electric field. As a result,
    the optical field that propagates through a phase modulator driven by the voltage
    :math:`u(t)` acquires a phase shift proportional to it,

    .. math::
        E_{out}(t) = E_{in}(t)\exp\left[j\pi\frac{u(t)}{V_\pi}\right], \tag{1}

    where :math:`V_\pi` is the voltage required to produce a phase shift of
    :math:`\pi` rad. The modulator changes only the phase of the field, so that
    :math:`|E_{out}(t)| = |E_{in}(t)|`.

    References
    ----------
    [1] G. P. Agrawal, Fiber-Optic Communication Systems. Wiley, 2021.

    """
    Ei, u = checkModulatorInputs(Ei, u)

    return calcPM(Ei, Vπ, u)


def mzm(Ei, u, param=None):
    r"""
    Optical Mach-Zhender Modulator (MZM).

    Parameters
    ----------
    Ei : scalar or np.array
        Optical field at the input of the MZM.
    u : np.array
        Electrical driving signal.
    param : optic.utils.parameters object, optional
        Parameters of the MZM model.

        - param.Vpi : MZM's Vpi voltage [V][default: 2 V]
        - param.Vb : MZM's bias voltage [V][default: -1 V]
        - param.ER : MZM extinction ratio [dB][default: 60 dB]

    Returns
    -------
    np.array
        Modulated optical field at the output of the MZM.

    Notes
    -----
    A Mach-Zehnder modulator (MZM) is an interferometer with a phase modulator in each
    of its arms. In push-pull operation, the arms are driven with opposite phase
    shifts, so that the output field is modulated in amplitude without residual phase
    modulation (chirp). For an ideal device, i.e. infinite extinction ratio,

    .. math::
        E_{out}(t) = E_{in}(t)\cos\left[\frac{\pi}{2}\frac{u(t) + V_b}{V_\pi}\right], \tag{1}

    where :math:`u(t)` is the driving signal, :math:`V_b` is the bias voltage, and
    :math:`V_\pi` is the voltage that switches the output from maximum to minimum
    transmission. The corresponding power transfer function is

    .. math::
        \frac{P_{out}(t)}{P_{in}(t)}
        = \cos^2\left[\frac{\pi}{2}\frac{u(t) + V_b}{V_\pi}\right]
        = \frac{1}{2}\left\{1 + \cos\left[\pi\frac{u(t) + V_b}{V_\pi}\right]\right\}. \tag{2}

    Biasing the modulator at :math:`V_b = -V_\pi/2` (quadrature point, the default)
    yields an output power that varies approximately linearly with small drive
    signals, as used in intensity modulation. Biasing at :math:`V_b = -V_\pi` (null
    point) makes the output field approximately linear with the drive signal, taking
    positive and negative values, as used in coherent modulation. A finite extinction
    ratio is also taken into account (see :func:`optic.dsp.core.calcMZM`).

    References
    ----------
    [1] G. P. Agrawal, Fiber-Optic Communication Systems. Wiley, 2021.

    [2] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    """
    if param is None:
        param = []

    # check input parameters
    Vpi = getattr(param, "Vpi", 2)
    Vb = getattr(param, "Vb", -1)
    ER = getattr(param, "ER", 60)  # extinction ratio in dB

    Ei, u = checkModulatorInputs(Ei, u)

    return calcMZM(Ei, Vpi, u, Vb, ER)

def ring_modulator(
    Ai: np.ndarray,  # optical field at the input of the ring
    u: np.ndarray,   # electrical driving signal
    param           # structure containing all other parameters
) -> np.ndarray:
    """
    Simulates the time-domain response of a photonic ring modulator to an input optical waveform and voltage.

    Parameters:
    -----------
    Ai : numpy.ndarray
        Array of complex input optical field samples.
    u : numpy.ndarray
        The voltage waveform applied to the modulator. Must have the same length as Ai.
    param : object
        Structure containing all other parameters:
        - dt: Time step in seconds between samples in input_waveform.
        - radius: Radius of the ring resonator in meters.
        - resonant_wavelength: Wavelength the ring is designed to resonate at, in meters.
        - n_eff: Effective refractive index at the resonant wavelength.
        - ng: Group index.
        - dn_dV: Change in effective index per volt.
        - loss_dB_m: Round-trip propagation loss in dB per meter.
        - kappa_power: Fraction of power coupled from bus waveguide into ring.
        - buffer_size_hint: Suggested initial size for the internal buffer.
        - rc_filter_enabled: Enable RC filter for voltage input.
        - rc_time_constant: Time constant for RC filter in seconds.
        - wavelength_offset: Wavelength offset from resonance in meters for the input light.

    Returns:
    --------
    output_waveform : numpy.ndarray
        Array of complex output optical field samples.

    References:
    -----------
    [1] W. Sacher and J. Poon, Dynamics of microring resonator modulators. Optics Express, 2008.
    """

    dt = param.dt
    radius = getattr(param, "radius", 10e-6)  # Default radius of the ring resonator
    resonant_wavelength = getattr(param, "resonant_wavelength", 1550e-9)  # Default resonant wavelength
    n_eff = getattr(param, "n_eff", 2.4)  # Default effective refractive index
    ng = getattr(param, "ng", 4.2)  # Default group index
    dn_dV = getattr(param, "dn_dV", 2E-4)  # Default change in effective index per volt
    loss_dB_m = getattr(param, "loss_dB_m", 4000)  # Default round-trip propagation loss in dB/m
    kappa_power = getattr(param, "kappa_power", 0.1)  # Default fraction of power coupled from bus waveguide into ring
    buffer_size_hint = getattr(param, "buffer_size_hint", 1000000)  # Default suggested initial size for the internal buffer
    rc_filter_enabled = getattr(param, "rc_filter_enabled", False)  # Default RC filter enabled
    rc_time_constant = getattr(param, "rc_time_constant", 5e-12)  # Default time constant for RC filter in seconds
    wavelength_offset = getattr(param, "wavelength_offset", -75e-12)  # Default wavelength offset from resonance in meters

    # --- Parameter Calculation ---
    kappa = np.sqrt(kappa_power)  # Field coupling coefficient
    sigma = np.sqrt(1 - kappa**2) # Field transmission coefficient (through-port)

    Lrt = 2 * np.pi * radius      # Round-trip length
    # Calculate amplitude loss factor 'a' from loss in dB/m
    a_loss = np.exp(-loss_dB_m * Lrt / (20 * np.log10(np.e))) # Round-trip field amplitude loss factor

    tau = ng * Lrt / const.c      # Round-trip time using group index for delay

    # --- Buffer Setup ---
    buffer_samples = int(np.ceil(tau / dt)) # Samples needed for round-trip delay
    actual_buffer_size = max(buffer_size_hint, buffer_samples + 1) # Ensure buffer is large enough

    if buffer_samples > buffer_size_hint:
        print(f"Note: Ring delay ({tau:.2e} s) requires {buffer_samples} samples. "
              f"Using buffer size {actual_buffer_size}.")

    # Initialize buffer for the internal field state (a_n(t)) - See Eq. (2) in Ref [1]
    # The buffer stores the field *inside* the ring just before the coupler.
    ring_field_buffer = np.zeros(actual_buffer_size, dtype=complex)
    buffer_idx = 0

    # --- Waveform Processing Setup ---
    n_samples = len(Ai)
    output_waveform = np.zeros(n_samples, dtype=complex)

    # Calculate the operating wavelength
    operating_wavelength = resonant_wavelength + wavelength_offset
    if operating_wavelength <= 0:
        raise ValueError("Operating wavelength must be positive.")

    # Pre-compute constant phase component (due to wavelength offset and static n_eff)
    base_phi = (2 * np.pi * n_eff / operating_wavelength) * Lrt - (2 * np.pi * n_eff / resonant_wavelength) * Lrt

    # Pre-compute voltage-dependent phase scaling factor
    # Delta_phi = (2 * pi * Delta_n_eff / lambda) * L = (2 * pi * dn_dV * V / lambda) * L
    voltage_phase_factor = (2 * np.pi * dn_dV / resonant_wavelength) * Lrt

    # --- RC Filter (if enabled) ---
    if u is not None and rc_filter_enabled:
        if len(u) != n_samples:
            raise ValueError("Voltage waveform must have the same length as the input waveform.")
        if rc_time_constant <= 0:
             raise ValueError("RC time constant must be positive.")

        alpha = dt / (rc_time_constant + dt) # Filter coefficient for IIR filter
        filtered_voltage = np.zeros_like(u)
        last_filtered_v = 0.0 # Initial condition for the filter

        # Apply first-order IIR filter: y[n] = alpha*x[n] + (1-alpha)*y[n-1]
        for i in range(n_samples):
            filtered_voltage[i] = alpha * u[i] + (1 - alpha) * last_filtered_v
            last_filtered_v = filtered_voltage[i]
    elif u is not None:
        filtered_voltage = u # Use voltage directly if filter is off
        if len(u) != n_samples:
            raise ValueError("Voltage waveform must have the same length as the input waveform.")
    else:
        filtered_voltage = np.zeros(n_samples) # No voltage applied

    # --- Simulation Loop ---
    for i in range(n_samples):
        # Get delayed ring field from buffer
        delayed_idx = (buffer_idx - buffer_samples + actual_buffer_size) % actual_buffer_size
        a_n_delayed = ring_field_buffer[delayed_idx]

        # Calculate total phase for this time step
        phi = base_phi + voltage_phase_factor * filtered_voltage[i]
        phase_term = a_loss * np.exp(-1j * phi) # Combined loss and phase shift

        # Calculate field inside the ring (a_n(t)) and store it
        # a_n(t) = kappa * s_in(t) + sigma * a_n(t - tau) * phase_term
        current_ring_field = kappa * Ai[i] + sigma * a_n_delayed * phase_term
        ring_field_buffer[buffer_idx] = current_ring_field

        # Calculate output field (s_out(t))
        # We need to store past values of a_n(t). Let's rename ring_field_buffer to a_n_buffer.
        a_n_delayed = ring_field_buffer[delayed_idx] # This is a_n(t-tau)

        # Calculate output field s_out(t)
        output_waveform[i] = sigma * Ai[i] + 1j * kappa * a_n_delayed * a_loss * np.exp(-1j * phi)

        # Calculate next internal field a_n(t) and store it
        a_n_current = sigma * a_n_delayed * a_loss * np.exp(-1j * phi) + 1j * kappa * Ai[i]
        ring_field_buffer[buffer_idx] = a_n_current

        # Update buffer index
        buffer_idx = (buffer_idx + 1) % actual_buffer_size

    return output_waveform

def iqm(Ei, u, param=None):
    r"""
    Optical In-Phase/Quadrature Modulator (IQM).

    Parameters
    ----------
    Ei : scalar or np.array
        Optical field at the input of the IQM.
    u : complex-valued np.array
        Modulator's driving signal (complex-valued baseband).
    param : optic.utils.parameters object, optional
        Parameters of the MZM models.

        - param.Vpi : MZM's Vpi voltage [V][default: 2 V]
        - param.VbI : I-MZM's bias voltage [V][default: -2 V]
        - param.VbQ : Q-MZM's bias voltage [V][default: -2 V]
        - param.Vphi : PM bias voltage [V][default: 1 V]
        - param.ERI : I-MZM extinction ratio [dB][default: 60 dB]
        - param.ERQ : Q-MZM extinction ratio [dB][default: 60 dB]

    Returns
    -------
    Eo : complex-valued np.array
        Modulated optical field at the output of the IQM.

    Notes
    -----
    An in-phase/quadrature modulator (IQM) is a nested structure with one MZM in each
    of its two arms, and a phase modulator that introduces a :math:`\pi/2` phase
    difference between them. The input field is split equally between the arms; the
    in-phase (I) MZM is driven by :math:`u_I(t) = \mathrm{Re}\{u(t)\}` and the
    quadrature (Q) MZM by :math:`u_Q(t) = \mathrm{Im}\{u(t)\}`, and the two fields are
    then recombined,

    .. math::
        E_{out}(t) = E_I(t) + E_Q(t)\,e^{j\pi V_\phi/V_\pi}, \tag{1}

    where :math:`E_I` and :math:`E_Q` are the outputs of the MZMs, each one fed with
    :math:`E_{in}/\sqrt{2}` (see :func:`mzm`), and :math:`V_\phi` is the bias of the
    phase modulator. With both MZMs biased at the null point, :math:`V_{b,I} =
    V_{b,Q} = -V_\pi`, and :math:`V_\phi = V_\pi/2` (the defaults), an ideal IQM
    produces

    .. math::
        E_{out}(t) = \frac{E_{in}(t)}{\sqrt{2}}\left\{
        \sin\left[\frac{\pi}{2}\frac{u_I(t)}{V_\pi}\right]
        + j\sin\left[\frac{\pi}{2}\frac{u_Q(t)}{V_\pi}\right]\right\}, \tag{2}

    which, for small drive signals, is proportional to the complex baseband signal
    :math:`u(t) = u_I(t) + ju_Q(t)`. This is how complex constellations such as QAM
    are imprinted on the optical carrier.

    References
    ----------
    [1] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    """
    if param is None:
        param = []

    # check input parameters
    Vpi = getattr(param, "Vpi", 2)
    VbI = getattr(param, "VbI", -2)
    VbQ = getattr(param, "VbQ", -2)
    Vphi = getattr(param, "Vphi", 1)
    ERI = getattr(param, "ERI", 60)
    ERQ = getattr(param, "ERQ", 60)

    Ei, u = checkModulatorInputs(Ei, u)

    # define parameters for the I-MZM:
    paramI = parameters()
    paramI.Vpi = Vpi
    paramI.Vb = VbI
    paramI.ER = ERI

    # define parameters for the Q-MZM:
    paramQ = parameters()
    paramQ.Vpi = Vpi
    paramQ.Vb = VbQ
    paramQ.ER = ERQ

    # Calculate MZMs outputs
    Ei_ = Ei / np.sqrt(2)  # split the input field between the I and Q branches

    EoI = mzm(Ei_, u.real, paramI)
    EoQ = mzm(Ei_, u.imag, paramQ)

    # Combine I and Q branches with the PM rotation to get the IQM output
    # (the PM bias Vphi is constant, so its phase rotation is a scalar)
    Eo = EoI + calcPM(EoQ, Vpi, Vphi)

    return Eo


def pbs(E, θ=0):
    r"""
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

    Notes
    -----
    The input field :math:`\mathbf{E} = [E_x, E_y]^T` is first rotated by the angle
    :math:`\theta` with respect to the principal axes of the polarization beam
    splitter (PBS), which then separates its two orthogonal components,

    .. math::
        :nowrap:

        \begin{equation}
            \begin{bmatrix} E_x' \\ E_y' \end{bmatrix} =
            \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}
            \begin{bmatrix} E_x \\ E_y \end{bmatrix}. \tag{1}
        \end{equation}

    A single-polarization input is assumed to be aligned with the :math:`x` axis,
    :math:`E_y = 0`, so that for :math:`\theta = \pi/4` its power is split equally
    between the two outputs.

    References
    ----------
    [1] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    """
    try:
        if E.shape[1] > 2:
            logg.error("E need to be a (N,2) or a (N,) np.array")
    except IndexError:
        E = np.repeat(E, 2).reshape(-1, 2)
        E[:, 1] = 0

    rot = np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]]) + 1j * 0

    E = E @ rot

    Ex = E[:, 0]
    Ey = E[:, 1]

    return Ex, Ey


def voa(E, A=0):
    r"""
    Variable optical attenuator (VOA).

    Parameters
    ----------
    E : np.array
        Input optical field.
    A : float
        attenuation [dB][default: 0 dB]

    Returns
    -------
    Eo : np.array
          Output optical field.

    Notes
    -----
    An attenuation of :math:`A` dB reduces the optical power by a factor
    :math:`10^{-A/10}`, which corresponds to scaling the optical field by

    .. math::
        E_{out}(t) = 10^{-A/20}\, E_{in}(t). \tag{1}

    References
    ----------
    [1] G. P. Agrawal, Fiber-Optic Communication Systems. Wiley, 2021.

    """
    assert A >= 0, "Attenuation should be a positive scalar"

    return E * 10 ** (-A / 20)


def photodiode(E, param=None):
    r"""
    Pin photodiode (PD).

    Parameters
    ----------
    E : np.array
        Input optical field.
    param : optic.utils.parameters object, optional
        Parameters of the photodiode model.

        - param.R : photodiode responsivity [A/W][default: 1 A/W]
        - param.Tc : temperature [°C][default: 25°C]
        - param.Id : dark current [A][default: 5e-9 A]
        - param.RL : impedance load [Ω] [default: 50Ω]
        - param.B : photodiode bandwidth [Hz][default: 30e9 Hz]
        - param.IpdSat : saturation value of the photocurrent [A][default: 5e-3 A]
        - param.N : number of the frequency resp. filter taps. [default: 255]
        - param.fType : frequency response type [default: 'rect']
        - param.ideal : bool enabling the ideal photodiode model (i.e. :math:`i_{pd}(t) = R|E(t)|^2`) [default: False]
        - param.shotNoise : bool enabling the addition of shot noise to photocurrent. [default: True]
        - param.thermalNoise : bool enabling the addition of thermal noise to photocurrent. [default: True]
        - param.currentSaturation : bool enabling the photocurrent saturation. [default: False]
        - param.bandwidthLimitation : bool enabling the bandwidth limitation. [default: True]
        - param.Fs : sampling frequency [Hz] [default: None]
        - param.seed : seed for the random number generator [default: None]

    Returns
    -------
    ipd : np.array
          photocurrent.

    Notes
    -----
    A PIN photodiode converts the incident optical power into an electric current.
    For an ideal photodiode, the photocurrent is proportional to the optical power,

    .. math::
        i_{pd}(t) = R\,|E(t)|^2 = R\,P(t), \tag{1}

    where :math:`R` is the responsivity in A/W. For a multimode field, the powers of
    all the modes are summed. Two noise sources are added to the photocurrent. The
    shot noise arises from the discrete nature of the photons and electrons, and has
    variance

    .. math::
        \sigma_s^2 = 2q\left[i_{pd}(t) + I_d\right]B, \tag{2}

    where :math:`q` is the elementary charge, :math:`I_d` is the dark current and
    :math:`B` is the photodiode bandwidth. The thermal (Johnson) noise is produced by
    the random motion of the electrons in the load resistor :math:`R_L`, at the
    absolute temperature :math:`T`, and has variance

    .. math::
        \sigma_T^2 = \frac{4k_B T B}{R_L}, \tag{3}

    where :math:`k_B` is the Boltzmann constant. Both are modeled as Gaussian noises,
    generated as white noises with power spectral density :math:`\sigma^2/(2B)` over
    the simulation bandwidth :math:`[-F_s/2, F_s/2]`, so that after the bandwidth
    limitation of the photodiode, which is modeled by a lowpass filter with cutoff
    frequency :math:`B`, their variances are those given by Eqs. (2) and (3). The
    photocurrent may also be limited to a saturation value :math:`I_{sat}`.

    References
    ----------
    [1] G. P. Agrawal, Fiber-Optic Communication Systems. Wiley, 2021.

    """
    if param is None:
        param = parameters()
    kB = const.value("Boltzmann constant")
    q = const.value("elementary charge")

    # check input parameters
    R = getattr(param, "R", 1)
    Tc = getattr(param, "Tc", 25)
    Id = getattr(param, "Id", 5e-9)
    RL = getattr(param, "RL", 50)
    B = getattr(param, "B", 30e9)
    IpdSat = getattr(param, "IpdSat", 5e-3)
    N = getattr(param, "N", 255)
    fType = getattr(param, "fType", "rect")
    ideal = getattr(param, "ideal", False)
    shotNoise = getattr(param, "shotNoise", True)
    thermalNoise = getattr(param, "thermalNoise", True)
    currentSaturation = getattr(param, "currentSaturation", False)
    bandwidthLimitation = getattr(param, "bandwidthLimitation", True)
    seed = getattr(param, "seed", None)

    assert R > 0, "PD responsivity should be a positive scalar"

    try:
        nModes = E.shape[1]
    except IndexError:
        nModes = 1

    # |E|^2 computed as a real-valued array, so that noise addition and
    # filtering are performed with real (instead of complex) arithmetic
    Pin = E.real**2 + E.imag**2

    if nModes > 1:
        ipd = R * np.sum(Pin, axis=1)  # ideal photocurrent with two or more modes
    else:
        ipd = R * Pin  # ideal photocurrent

    if N % 2 == 0:
        N += 1  # make sure N is odd
        logg.warning(
            "Number of filter taps (N) was even, incrementing by one to make it odd."
        )

    if seed is not None:
        np.random.seed(seed)  # set seed for reproducibility

    if not (ideal):
        try:
            Fs = param.Fs
        except AttributeError:
            logg.error("Simulation sampling frequency (Fs) not provided.")

        assert Fs >= 2 * B, "Sampling frequency Fs needs to be at least twice of B."

        if currentSaturation:
            ipd = np.minimum(ipd, IpdSat)  # saturation of the photocurrent

        if shotNoise:
            # shot noise
            σ2_s = 2 * q * (ipd + Id) * B  # shot noise variance
            Is = np.sqrt(Fs * (σ2_s / (2 * B))) * np.random.normal(0, 1, ipd.shape)
            # add shot noise to photocurrent
            ipd += Is
        if thermalNoise:
            # thermal noise
            T = Tc + 273.15  # temperature in Kelvin
            σ2_T = 4 * kB * T * B / RL  # thermal noise variance
            It = np.sqrt(Fs * (σ2_T / (2 * B))) * np.random.normal(0, 1, ipd.shape)
            # add thermal noise to photocurrent
            ipd += It
        if bandwidthLimitation:
            # lowpass filtering
            h = lowPassFIR(B, Fs, N, typeF=fType)
            ipd = firFilter(h, ipd)

    return ipd.real


def balancedPD(E1, E2, param=None):
    r"""
    Balanced photodiode pair (BPD).

    Parameters
    ----------
    E1 : np.array
        Input optical field.
    E2 : np.array
        Input optical field.
    param : optic.utils.parameters object, optional
        Parameters of the photodiode models.

        - param.R : photodiode responsivity [A/W][default: 1 A/W].
        - param.Tc : temperature [°C][default: 25°C].
        - param.Id : dark current [A][default: 5e-9 A].
        - param.RL : impedance load [Ω] [default: 50Ω].
        - param.B : photodiode bandwidth [Hz][default: 30e9 Hz].
        - param.Fs : sampling frequency [Hz] [default: 60e9 Hz].
        - param.fType : frequency response type [default: 'rect'].
        - param.N : number of the frequency resp. filter taps. [default: 255].
        - param.ideal : bool enabling the ideal photodiode model (i.e. no noise, no frequency resp.) [default: True].
        - param.seed : seed for the random number generator [default: None].

    Returns
    -------
    ibpd : np.array
           Balanced photocurrent.

    Notes
    -----
    A balanced photodetector consists of two photodiodes whose photocurrents are
    subtracted,

    .. math::
        i(t) = i_1(t) - i_2(t) = R\left(|E_1(t)|^2 - |E_2(t)|^2\right)
        + n_1(t) - n_2(t), \tag{1}

    where :math:`n_1` and :math:`n_2` are the independent noises of the photodiodes
    (see :func:`photodiode`). When the two inputs are the outputs of a coupler that
    combines a signal and a local oscillator, the terms :math:`|E_s|^2` and
    :math:`|E_{LO}|^2` cancel in Eq. (1), leaving only the beat term between the
    signal and the local oscillator.

    References
    ----------
    [1] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    [2] K. Kikuchi, “Fundamentals of Coherent Optical Fiber Communications”, J. Lightwave Technol., JLT, vol. 34, nº 1, p. 157–179, jan. 2016.
    """
    assert E1.shape == E2.shape, "E1 and E2 need to have the same shape"

    # check if input parameters are provided
    if param is None:
        paramPD1 = None
        paramPD2 = None
    else:
        # duplicate PD parameters:
        paramPD1 = param.copy()
        paramPD2 = param.copy()

        # check for seed in parameters
        if hasattr(paramPD1, "seed"):
            # in case the seed is provided, make sure to use different seeds for each photodiode
            if paramPD1.seed is not None:
                paramPD2.seed = (
                    paramPD1.seed + 1
                )  # to ensure different seeds for each photodiode

    i1 = photodiode(E1, paramPD1)
    i2 = photodiode(E2, paramPD2)

    return i1 - i2


def opticalHybrid2x4(Es, Elo):
    r"""
    Optical hybrid 2 x 4 90°.

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

    Notes
    -----
    The 2 x 4 90° optical hybrid combines the signal :math:`E_s` and the local
    oscillator :math:`E_{LO}` with relative phase shifts of 0, :math:`\pi`,
    :math:`\pi/2` and :math:`-\pi/2`. Its outputs are

    .. math::
        :nowrap:

        \begin{equation}
            \begin{bmatrix} E_1 \\ E_2 \\ E_3 \\ E_4 \end{bmatrix} = \frac{1}{2}
            \begin{bmatrix}
                E_s - E_{LO} \\
                j\left(E_s + E_{LO}\right) \\
                jE_s - E_{LO} \\
                -E_s + jE_{LO}
            \end{bmatrix}. \tag{1}
        \end{equation}

    Detecting the pairs :math:`(E_1, E_2)` and :math:`(E_3, E_4)` with balanced
    photodetectors gives currents proportional to the in-phase and quadrature
    components of :math:`E_s E_{LO}^*`, since

    .. math::
        |E_2|^2 - |E_1|^2 = \mathrm{Re}\left\{E_s E_{LO}^*\right\}, \qquad
        |E_3|^2 - |E_4|^2 = \mathrm{Im}\left\{E_s E_{LO}^*\right\}. \tag{2}

    References
    ----------
    [1] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    [2] K. Kikuchi, “Fundamentals of Coherent Optical Fiber Communications”, J. Lightwave Technol., JLT, vol. 34, nº 1, p. 157–179, jan. 2016.
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

    # only the first (Es) and last (Elo) hybrid inputs are non-zero
    return T[:, [0, 3]] @ np.array([Es, Elo])


def coherentReceiver(Es, Elo, paramFE=None, paramPD=None):
    r"""
    Single polarization coherent optical front-end.

    Parameters
    ----------
    Es : np.array
        Input signal optical field.
    Elo : np.array
        Input LO optical field.
    paramFE : parameter object (struct), optional
        Parameters of the optical frontend:

            - paramFE.Fs : simulation sampling frequency [samples/s].
            - paramFE.phaseImb : phase imbalance of the I/Q [rad].
            - paramFE.ampImb : amplitude imbalance of the I/Q [dB].
            - paramFE.timeSkew : delay of the I of the I/Q [s].

    paramPD : parameter object (struct), optional
        Parameters of the photodiodes

    Returns
    -------
    s : np.array
        Downconverted signal after balanced detection.

    Notes
    -----
    In a coherent receiver, the received signal is mixed with a local oscillator (LO)
    in a 90° optical hybrid, whose outputs are detected by two balanced
    photodetectors (see :func:`opticalHybrid2x4` and :func:`balancedPD`). The
    in-phase and quadrature photocurrents form the complex signal

    .. math::
        s(t) = i_I(t) + j\,i_Q(t) = R\,E_s(t)E_{LO}^*(t) + n(t), \tag{1}

    where :math:`R` is the responsivity of the photodiodes and :math:`n(t)` is the
    noise of the photodetectors. For an LO with constant amplitude and a frequency and
    phase aligned to those of the signal carrier, :math:`s(t)` is proportional to the
    complex envelope of the optical field, i.e., both its amplitude and its phase are
    recovered. Imperfections of the front-end (IQ imbalance and skew) are finally
    added to :math:`s(t)` (see :func:`optic.dsp.core.iqMixing`).

    References
    ----------
    [1] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    [2] K. Kikuchi, “Fundamentals of Coherent Optical Fiber Communications”, J. Lightwave Technol., JLT, vol. 34, nº 1, p. 157–179, jan. 2016.
    """
    assert Es.shape == (len(Es),), "Es need to have a (N,) shape"
    assert Elo.shape == (len(Elo),), "Elo need to have a (N,) shape"
    assert Es.shape == Elo.shape, "Es and Elo need to have the same (N,) shape"

    try:
        Fs = paramFE.Fs
    except AttributeError:
        logg.error("Simulation sampling frequency (Fs) not provided.")

    if paramPD is None:
        paramPD = parameters()
        paramPD.Fs = Fs

    paramPDchI = paramPD.copy()
    paramPDchQ = paramPD.copy()

    try:
        seed = paramPD.seed
        # make sure to use different seeds for each balanced pair of photodiodes
        paramPDchI.seed = seed
        paramPDchQ.seed = seed + 7
    except AttributeError:
        pass

    # optical hybrid 2 x 4 90°
    Eo = opticalHybrid2x4(Es, Elo)

    # balanced photodetection
    sI = balancedPD(Eo[1, :], Eo[0, :], paramPDchI)
    sQ = balancedPD(Eo[2, :], Eo[3, :], paramPDchQ)

    s = sI + 1j * sQ

    # add receiver front-end impairments
    s = iqMixing(s, paramFE)

    return s


def pdmCoherentReceiver(Es, Elo, paramFE, paramPD=None):
    r"""
    Polarization multiplexed coherent optical front-end.

    Parameters
    ----------
    Es : np.array
        Input signal optical field.
    Elo : np.array
        Input LO optical field.
    paramFE : parameter object (struct), optional
        Parameters of the optical frontend:

            - paramFE.Fs : simulation sampling frequency [samples/s].
            - paramFE.polRotation : input polarization rotation angle [rad].
            - paramFE.pdl : polarization dependent loss [dB]. If > 0, loss is on X polarization. If < 0, loss is on Y polarization.
            - paramFE.polDelay : polarization delay [s]. If > 0, delay is on X polarization. If < 0, delay is on Y polarization.
            - paramFE.polX.phaseImb : phase imbalance of the I/Q of the X polarization [rad].
            - paramFE.polX.ampImb : amplitude imbalance of the I/Q of the X polarization [dB].
            - paramFE.polX.skewI : delay of the I of the X polarization [s].
            - paramFE.polX.skewQ : delay of the Q of the X polarization [s].
            - paramFE.polY.phaseImb : phase imbalance of the I/Q of the Y polarization [rad].
            - paramFE.polY.ampImb : amplitude imbalance of the I/Q of the Y polarization [dB].
            - paramFE.polY.skewI : delay of the I of the Y polarization [s].
            - paramFE.polY.skewQ : delay of the Q of the Y polarization [s].

    paramPD : parameter object (struct), optional
        Parameters of the photodiodes (see photodiode model documentation)

    Returns
    -------
    S : np.array
        Downconverted signal after balanced detection.

    Notes
    -----
    A polarization-diversity coherent receiver detects the two orthogonal
    polarizations of the received field. The signal is split by a polarization beam
    splitter (PBS) into its components :math:`E_{s,x}` and :math:`E_{s,y}`, after a
    rotation of its state of polarization by the angle ``polRotation``, while the LO,
    launched at 45°, is split equally between the two polarizations (see
    :func:`pbs`). Each pair of signal and LO components is then detected by a
    single-polarization coherent receiver (see :func:`coherentReceiver`),

    .. math::
        :nowrap:

        \begin{equation}
            \mathbf{S}(t) = \begin{bmatrix} S_x(t) \\ S_y(t) \end{bmatrix} \propto
            \begin{bmatrix} E_{s,x}(t)E_{LO,x}^*(t) \\ E_{s,y}(t)E_{LO,y}^*(t) \end{bmatrix}. \tag{1}
        \end{equation}

    A polarization dependent loss of ``pdl`` dB is modeled by attenuating one
    polarization and amplifying the other by ``pdl/2`` dB, and a differential delay
    :math:`\tau` between the polarizations is modeled by advancing the :math:`x`
    component and delaying the :math:`y` component by :math:`\tau/2`.

    References
    ----------
    [1] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    [2] K. Kikuchi, “Fundamentals of Coherent Optical Fiber Communications”, J. Lightwave Technol., JLT, vol. 34, nº 1, p. 157–179, jan. 2016.
    """
    assert len(Es) == len(Elo), "Es and Elo need to have the same length"

    try:
        Fs = paramFE.Fs
    except AttributeError:
        logg.error("Simulation sampling frequency (Fs) not provided.")

    # define frontend parameters for the X polarization:
    paramX = parameters()
    paramX.Fs = Fs
    paramX.phaseImb = getattr(paramFE, "phaseImbX", 0)
    paramX.ampImb = getattr(paramFE, "ampImbX", 0)
    paramX.timeSkew = getattr(paramFE, "timeSkewX", 0)

    # define frontend parameters for the Y polarization:
    paramY = parameters()
    paramY.Fs = Fs
    paramY.phaseImb = getattr(paramFE, "phaseImbY", 0)
    paramY.ampImb = getattr(paramFE, "ampImbY", 0)
    paramY.timeSkew = getattr(paramFE, "timeSkewY", 0)

    if paramPD is None:
        paramPD = parameters()
        paramPD.Fs = Fs

    paramPDxPol = paramPD.copy()
    paramPDyPol = paramPD.copy()

    try:
        seed = paramPD.seed
        # make sure to use different seeds for each polarization
        paramPDxPol.seed = seed
        paramPDyPol.seed = seed + 3
    except AttributeError:
        pass

    polRotation = getattr(paramFE, "polRotation", 0)
    pdl = getattr(paramFE, "pdl", 0)
    polDelay = getattr(paramFE, "polDelay", 0)

    Elox, Eloy = pbs(Elo, θ=np.pi / 4)  # split LO into two orth. polarizations
    Esx, Esy = pbs(Es, θ=polRotation)  # split signal into two orth. polarizations

    if polDelay != 0:
        Esx = delaySignal(Esx, -polDelay / 2, Fs)  # apply delay to polarization X
        Esy = delaySignal(Esy, polDelay / 2, Fs)  # apply delay to polarization Y

    if pdl != 0:
        Esx = 10 ** (-(pdl / 2) / 20) * Esx  # apply PDL to pol.X
        Esy = 10 ** ((pdl / 2) / 20) * Esy  # apply PDL to pol.Y

    Sx = coherentReceiver(Esx, Elox, paramX, paramPDxPol)  # coherent detection of pol.X
    Sy = coherentReceiver(Esy, Eloy, paramY, paramPDyPol)  # coherent detection of pol.Y

    return np.array([Sx, Sy]).T


def edfa(Ei, param=None):
    r"""
    Implement simple EDFA model.

    Parameters
    ----------
    Ei : np.array
        Input signal field.
    param : optic.utils.parameters object, optional
        Parameters of the EDFA model.

        - param.G : amplifier gain [dB][default: 20 dB]
        - param.NF : EDFA noise figure [dB][default: 4.5 dB]
        - param.Fc : central optical frequency [Hz][default: 193.1 THz]
        - param.Fs : sampling frequency in [samples/s]
        - param.seed : random seed for noise generation [default: None]

    Returns
    -------
    Eo : np.array
        Amplified noisy optical signal.

    Notes
    -----
    The amplifier multiplies the optical field by :math:`\sqrt{G}`, where :math:`G` is
    the power gain, and adds the amplified spontaneous emission (ASE) noise, modeled
    as a complex circular Gaussian noise :math:`n(t)`:

    .. math::
        E_{out}(t) = \sqrt{G}\,E_{in}(t) + n(t). \tag{1}

    The power spectral density of the ASE noise, per polarization mode, is

    .. math::
        N_{ASE} = (G-1)\,n_{sp}\,h\nu, \tag{2}

    where :math:`h\nu` is the photon energy at the carrier frequency :math:`\nu = F_c`
    and :math:`n_{sp}` is the spontaneous emission factor, which is related to the
    noise figure :math:`NF` (in linear units) by

    .. math::
        n_{sp} = \frac{G\cdot NF - 1}{2(G-1)}. \tag{3}

    For a large gain, :math:`NF \approx 2n_{sp} \geq 2` (3 dB), which is the quantum
    limit of the noise figure of a phase-insensitive amplifier. The noise power within
    the simulation bandwidth is :math:`P_n = N_{ASE}F_s`, where :math:`F_s` is the
    sampling frequency.

    References
    ----------
    [1] R. -J. Essiambre,et al, "Capacity Limits of Optical Fiber Networks," in Journal of Lightwave Technology, vol. 28, no. 4, pp. 662-701, 2010, doi: 10.1109/JLT.2009.2039464.

    """
    try:
        Fs = param.Fs
    except AttributeError:
        logg.error("Simulation sampling frequency (Fs) not provided.")

    # check input parameters
    G = getattr(param, "G", 20)
    NF = getattr(param, "NF", 4.5)
    Fc = getattr(param, "Fc", 193.1e12)
    seed = getattr(param, "seed", None)

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

    noise = gaussianComplexNoise(Ei.shape, p_noise, seed)

    return Ei * np.sqrt(G_lin) + noise


def basicLaserModel(param=None):
    r"""
    Laser model with Maxwellian random walk phase noise and RIN.

    Parameters
    ----------
    param : optic.utils.parameters object, optional
        Parameters of the laser model.

        - param.P : laser power [dBm] [default: 10 dBm]
        - param.lw : laser linewidth [Hz] [default: 1 kHz]
        - param.RIN_var : variance of the RIN noise [default: 1e-20]
        - param.Fs : sampling rate [samples/s]
        - param.Ns : number of signal samples [default: 1e3]
        - param.seed : random seed for noise generation [default: None]
        - param.freqShift : frequency shift with respect to the central simulation frequency [Hz] [default: 0 Hz]

    Returns
    -------
    np.array
        Optical signal with phase noise and RIN.

    Notes
    -----
    The optical field at the laser output is modeled as

    .. math::
        E(t) = \sqrt{P + \delta P(t)}\;\exp\left\{j\left[2\pi\Delta f\,t
        + \phi(t)\right]\right\}, \tag{1}

    where :math:`P` is the average optical power, :math:`\Delta f` is the frequency
    shift with respect to the central frequency of the simulation, and
    :math:`\phi(t)` is the phase noise, modeled as a Wiener process whose increments
    over a sampling period :math:`T_s` have variance :math:`2\pi\Delta\nu T_s`, where
    :math:`\Delta\nu` is the laser linewidth (see :func:`optic.dsp.core.phaseNoise`).
    The relative intensity noise (RIN) is modeled by the random power fluctuation
    :math:`\delta P(t)`, a zero-mean Gaussian noise with variance ``RIN_var``
    (generated as a circularly-symmetric complex Gaussian sequence, see
    :func:`optic.dsp.core.gaussianComplexNoise`).

    References
    ----------
    [1] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    """
    try:
        Fs = param.Fs
    except AttributeError:
        logg.error("Simulation sampling frequency (Fs) not provided.")

    P = getattr(param, "P", 10)  # Laser power in dBm
    lw = getattr(param, "lw", 1e3)  # Linewidth in Hz
    RIN_var = getattr(param, "RIN_var", 1e-20)  # RIN variance
    Ns = getattr(param, "Ns", 1000)  # Number of samples of the signal
    seed = getattr(param, "seed", None)  # Seed for the random number generator
    freqShift = getattr(
        param, "freqShift", 0
    )  # Frequency shift with respect to the central simulation frequency

    if seed is None:
        seedPN = None
        seedRIN = None
    else:
        seedPN = seed
        seedRIN = seed + 73  # to ensure different seeds for phase noise and RIN

    # Simulate Maxwellian random walk phase noise
    pn = phaseNoise(lw, Ns, 1 / Fs, seedPN)

    # Simulate relative intensity noise  (RIN)[todo:check correct model]
    deltaP = gaussianComplexNoise(pn.shape, RIN_var, seedRIN)

    # Apply frequency shift if required
    if freqShift != 0:
        fo = 2 * np.pi * freqShift * np.arange(Ns) / Fs
    else:
        fo = 0

    # Return optical signal
    return np.sqrt(dBm2W(P) + deltaP) * np.exp(1j * (fo + pn))


def adc(sigIn, param):
    r"""
    Analog-to-digital converter (ADC) model.

    Parameters
    ----------
    sigIn : np.array
        Input signal.
    param : optic.utils.parameters object, optional
        Parameters of the ADC model.

        - param.inFs : sampling frequency of the input signal [samples/s][default: 1 sample/s]
        - param.outFs : sampling frequency of the output signal [samples/s][default: 1 sample/s]
        - param.jitter : jitter rms in seconds [s][default: 0 s]
        - param.nBits : number of bits used for quantization [default: 8 bits]
        - param.ENOB : effective number of bits of the ADC [default: 8 bits]
        - param.Vmax : maximum value for the ADC's full-scale range [V][default: 1V]
        - param.Vmin : minimum value for the ADC's full-scale range [V][default: -1V]
        - param.AAF : flag indicating whether to use anti-aliasing filters [default: True]
        - param.N : number of taps of the anti-aliasing filters [default: 201]

    Returns
    -------
    sigOut : np.array
        Resampled and quantized signal.

    Notes
    -----
    - The input signal will be clipped to the range [Vmin, Vmax] before quantization.
    - If the effective number of bits (ENOB) is less than nBits, additional noise will be added to the output signal to model the reduced resolution of the ADC. The noise power is calculated based on the difference between the ideal quantization noise power (corresponding to nBits) and the actual quantization noise power (corresponding to ENOB).
    - If AAF is enabled, anti-aliasing filters will be applied to the input signal before resampling and to the output signal after quantization to mitigate aliasing effects.

    The analog-to-digital conversion is modeled by the following sequence of
    operations:

    1. Anti-aliasing filtering: the input is lowpass filtered with cutoff frequency
       :math:`F_{out}/2`, the Nyquist frequency of the ADC.
    2. Sampling: the signal is resampled at the ADC sampling rate :math:`F_{out}`,
       with instants :math:`t_m = m/F_{out} + \epsilon_m` affected by a random
       timing jitter :math:`\epsilon_m \sim \mathcal{N}(0, \sigma_j^2)` (see
       :func:`optic.dsp.core.clockSamplingInterp`).
    3. Clipping and quantization: the samples are limited to the full-scale range
       :math:`[V_{min}, V_{max}]` and quantized with a uniform :math:`b`-bit quantizer
       (see :func:`optic.dsp.core.quantizer`).
    4. Output filtering: the quantized signal is filtered again by a lowpass filter
       with cutoff frequency :math:`F_{out}/2`.

    Steps 1 and 4 are skipped if the anti-aliasing filters are disabled (``AAF``).
    Real converters perform worse than an ideal quantizer with the same number of
    bits. This is described by the effective number of bits (ENOB), i.e., the
    resolution of an ideal quantizer with the same signal-to-noise-and-distortion
    ratio, :math:`\mathrm{SINAD} = 6.02\,\mathrm{ENOB} + 1.76` dB for a full-scale
    sinusoid. With the full-scale range :math:`V_{FS} = V_{max} - V_{min}`, the
    quantization noise power of a :math:`b`-bit quantizer is approximately
    :math:`V_{FS}^2 / (12 \cdot 2^{2b})`, and the degradation is modeled by adding a
    Gaussian noise with power

    .. math::
        P_{extra} = \frac{V_{FS}^2}{12}\left(2^{-2\,\mathrm{ENOB}} - 2^{-2b}\right), \tag{1}

    per real dimension (I and Q) of the signal, so that the total noise power
    corresponds to a quantizer with :math:`\mathrm{ENOB}` bits.
    """
    # Check and set default values for input parameters
    param.inFs = getattr(param, "inFs", 1)
    param.outFs = getattr(param, "outFs", 1)
    param.jitter = getattr(param, "jitter", 0)
    param.nBits = getattr(param, "nBits", 8)
    param.ENOB = getattr(param, "ENOB", 8)
    param.Vmax = getattr(param, "Vmax", 1)
    param.Vmin = getattr(param, "Vmin", -1)
    param.AAF = getattr(param, "AAF", True)
    param.N = getattr(param, "N", 201)

    # Extract individual parameters for ease of use
    inFs = param.inFs
    outFs = param.outFs
    jitter = param.jitter
    nBits = param.nBits
    Vmax = param.Vmax
    Vmin = param.Vmin
    AAF = param.AAF
    N = param.N
    ENOB = param.ENOB

    if ENOB > nBits:
        logg.warning(
            "ADC ENOB is greater than ADC nBits. The effective number of bits (ENOB) should be less than or equal to the number of bits (nBits) for a consistent ADC model."
        )

    # Reshape the input signal if needed to handle single-dimensional inputs
    try:
        sigIn.shape[1]
    except IndexError:
        sigIn = sigIn.reshape(len(sigIn), 1)

    # Apply anti-aliasing filters if AAF is enabled
    if AAF:
        # Anti-aliasing filters:
        Ntaps = min(sigIn.shape[0], N)
        hi = lowPassFIR(outFs / 2, inFs, Ntaps, typeF="rect")
        ho = lowPassFIR(outFs / 2, outFs, Ntaps, typeF="rect")
        sigIn = firFilter(hi, sigIn)

    if np.iscomplexobj(sigIn):
        # Signal interpolation to the ADC's sampling frequency
        sigOut = clockSamplingInterp(
            np.real(sigIn), inFs, outFs, jitter
        ) + 1j * clockSamplingInterp(np.imag(sigIn), inFs, outFs, jitter)

        # clipping to [Vmin, Vmax]
        sigOut = np.clip(sigOut, Vmin + 1j * Vmin, Vmax + 1j * Vmax)

        # Uniform quantization of the signal according to the number of bits of the ADC
        sigOut = quantizer(np.real(sigOut), nBits, Vmax, Vmin) + 1j * quantizer(
            np.imag(sigOut), nBits, Vmax, Vmin
        )
    else:
        # Signal interpolation to the ADC's sampling frequency
        sigOut = clockSamplingInterp(sigIn, inFs, outFs, jitter)

        # clipping to [Vmin, Vmax]
        sigOut = np.clip(sigOut, Vmin, Vmax)

        # Uniform quantization of the signal according to the number of bits of the ADC
        sigOut = quantizer(sigOut, nBits, Vmax, Vmin)

    # Apply anti-aliasing filters to the output if AAF is enabled
    if AAF:
        sigOut = firFilter(ho, sigOut)

    # Add noise corresponding to the effective number of bits (ENOB) of the ADC
    if nBits > ENOB:
        scale = Vmax - Vmin
        Pnq_ideal = scale**2 / (12 * (2 ** (2 * nBits)))
        Pnq_actual = scale**2 / (12 * (2 ** (2 * ENOB)))
        Pn_extra = Pnq_actual - Pnq_ideal

        if np.iscomplexobj(sigOut):
            sigOut += gaussianComplexNoise(sigOut.shape, 2 * Pn_extra)
        else:
            sigOut += gaussianNoise(sigOut.shape, Pn_extra)

    if sigOut.shape[1] == 1:
        # If the output is a single column, return it as a 1D array
        sigOut = sigOut.flatten()

    return sigOut


def dac(sigIn, param):
    r"""
    Digital-to-analog converter (DAC) model.

    Parameters
    ----------
    sigIn : np.array
        Input signal.
    param : optic.utils.parameters object, optional
        Parameters of the DAC model.

        - param.inFs : sampling frequency of the input signal [samples/s][default: 1 sample/s]
        - param.outFs : sampling frequency of the output signal [samples/s][default: 1 sample/s]
        - param.nBits : number of bits used for quantization [default: 8 bits]
        - param.ENOB : effective number of bits of the DAC [default: 8 bits]
        - param.jitter : jitter rms in seconds [s][default: 0 s]
        - param.Vpp : peak-to-peak voltage of the DAC's output signal [V][default: 2 V]
        - param.AIF : flag indicating whether to use anti-imaging filters [default: True]
        - param.N : number of taps of the anti-imaging filters [default: 201]

    Returns
    -------
    sigOut : np.array
        Resampled and quantized signal.

    Notes
    -----
    - The input signal will be clipped to the range [Vmin, Vmax] before quantization.
    - If AIF is enabled, anti-imaging filters will be applied to the output signal after quantization to mitigate imaging effects.

    The digital-to-analog conversion is modeled by the following sequence of
    operations:

    1. Quantization: the samples are quantized with a uniform :math:`b`-bit quantizer
       whose full-scale range :math:`[V_{min}, V_{max}]` is given by the minimum and
       maximum values of the input (see :func:`optic.dsp.core.quantizer`).
    2. Interpolation: the signal is converted from the input sampling rate
       :math:`F_{in}` to the output rate :math:`F_{out}` (see
       :func:`optic.dsp.core.clockSamplingInterp`), with an optional timing jitter.
    3. Anti-imaging filtering: the interpolated signal is filtered by a lowpass
       filter with cutoff frequency :math:`F_{out}/2` (if ``AIF`` is enabled).

    A finite effective number of bits (ENOB) is modeled by adding a Gaussian noise
    with power

    .. math::
        P_{extra} = \frac{V_{FS}^2}{12}\left(2^{-2\,\mathrm{ENOB}} - 2^{-2b}\right), \qquad
        V_{FS} = V_{max} - V_{min}, \tag{1}

    per real dimension of the signal (see :func:`adc`). Finally, the output is scaled
    so that the full-scale range corresponds to the peak-to-peak voltage
    :math:`V_{pp}` of the DAC,

    .. math::
        y(t) \leftarrow \frac{V_{pp}}{V_{max} - V_{min}}\,y(t). \tag{2}
    """
    # Check and set default values for input parameters
    param.inFs = getattr(param, "inFs", 1)
    param.outFs = getattr(param, "outFs", 1)
    param.nBits = getattr(param, "nBits", 8)
    param.ENOB = getattr(param, "ENOB", 8)
    param.jitter = getattr(param, "jitter", 0)
    param.Vpp = getattr(param, "Vpp", 2)
    param.AIF = getattr(param, "AIF", True)
    param.N = getattr(param, "N", 201)

    # Extract individual parameters for ease of use
    inFs = param.inFs
    outFs = param.outFs
    nBits = param.nBits
    ENOB = param.ENOB
    jitter = param.jitter
    Vpp = param.Vpp
    AIF = param.AIF
    N = param.N

    if ENOB > nBits:
        logg.warning(
            "DAC ENOB is greater than DAC nBits. The effective number of bits (ENOB) should be less than or equal to the number of bits (nBits) for a consistent ADC model."
        )

    # Reshape the input signal if needed to handle single-dimensional inputs
    try:
        sigIn.shape[1]
    except IndexError:
        sigIn = sigIn.reshape(len(sigIn), 1)

    if np.iscomplexobj(sigIn):
        # Uniform quantization of the signal according to the number of bits of the DAC
        Vmax = np.max([np.max(sigIn.real), np.max(sigIn.imag)])
        Vmin = np.min([np.min(sigIn.real), np.min(sigIn.imag)])

        sigOut = quantizer(np.real(sigIn), nBits, Vmax, Vmin) + 1j * quantizer(
            np.imag(sigIn), nBits, Vmax, Vmin
        )

        # Signal interpolation to the DAC's sampling frequency
        sigOut = clockSamplingInterp(
            np.real(sigOut), inFs, outFs, jitter
        ) + 1j * clockSamplingInterp(np.imag(sigOut), inFs, outFs, jitter)
    else:
        Vmax = np.max(sigIn)
        Vmin = np.min(sigIn)

        # Uniform quantization of the signal according to the number of bits of the DAC
        sigOut = quantizer(sigIn, nBits, Vmax, Vmin)

        # Signal interpolation to the DAC's sampling frequency
        sigOut = clockSamplingInterp(sigOut, inFs, outFs, jitter)

    # Apply anti-imaging filters to the output if AIF is enabled
    if AIF:
        ho = lowPassFIR(
            param.outFs / 2, param.outFs, min(sigOut.shape[0], N), typeF="rect"
        )
        sigOut = firFilter(ho, sigOut)

    # Add noise to the output signal based on the effective number of bits (ENOB)
    if nBits > ENOB:
        scale = Vmax - Vmin
        Pnq_ideal = scale**2 / (12 * (2 ** (2 * nBits)))
        Pnq_actual = scale**2 / (12 * (2 ** (2 * ENOB)))
        Pn_extra = Pnq_actual - Pnq_ideal

        if np.iscomplexobj(sigOut):
            sigOut += gaussianComplexNoise(sigOut.shape, 2 * Pn_extra)
        else:
            sigOut += gaussianNoise(sigOut.shape, Pn_extra)

    sigOut = sigOut * (
        Vpp / (Vmax - Vmin)
    )  # Scale the output signal to the DAC's specified peak-to-peak voltage

    if sigOut.shape[1] == 1:
        # If the output is a single column, return it as a 1D array
        sigOut = sigOut.flatten()
    return sigOut
