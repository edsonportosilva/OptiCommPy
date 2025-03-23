"""
=======================================================================
Models for optoelectronic devices (:mod:`optic.models.devices`)
=======================================================================

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
   basicLaserModel       -- Laser model with Maxwellian random walk phase noise and RIN
   adc                   -- Analog-to-digital converter (ADC) model
"""

"""Basic physical models for optical/electronic devices."""
import logging as logg

import numpy as np
import scipy.constants as const
from optic.utils import parameters, dBm2W
from optic.dsp.core import (
    lowPassFIR,
    gaussianComplexNoise,
    phaseNoise,
    clockSamplingInterp,
    quantizer,
)

try:
    from optic.dsp.coreGPU import checkGPU
    if checkGPU():
        from optic.dsp.coreGPU import firFilter
    else:
        from optic.dsp.core import firFilter
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

    References
    ----------
    [1] G. P. Agrawal, Fiber-Optic Communication Systems. Wiley, 2021.

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

class RingModulator:
    """
    A class representing an optical ring resonator with bus coupling. It correctly calculates both the time domain (transient)
    and frequency domain (steady state) response of the resonator to both input optical fields and voltage.


    References
    ----------
    [1] W. Sacher and J. Poon, Dynamics of microring resonator modulators. Optics Express, 2008.

    [2] W. Bogaerts et al, Silicon microring resonators. Laser & Photonics Reviews, 2012.
    """
    
    def __init__(
            self, 
            radius=10e-6, 
            resonant_wavelength=1550e-9,
            n_eff=2.4, 
            ng = 4.2,
            dn_dV = 2E-4,
            a=4000,  
            kappa=0.1, 
            buffer_size=1000000,
            # Add RC filter parameters
            rc_filter_enabled=False,
            rc_time_constant=1e-9  # Default 10 ps time constant
        ):
        """
        Initialize the ring resonator.
        
        Parameters:
        -----------
        radius : float
            Radius of the ring resonator in meters (default: 10e-6)
        resonant_wavelength : float
            Wavelength the ring will resonant at in meters (default: 1550 nm)
        n_eff : float
            Effective index measured at the rseonant wavelength of interest (default: 2.4)
        n_g : float
            Group index (default: 4.2)
        dn_dV : float
            Change in effective index per volt (default: 2E-4)
        a : float
            Round-trip propagation loss in dB per meter (default:4000 dB/m)
        kappa : float
            Fraction of power coupled from bus waveguide into ring: (default: 0.1)
        buffer_size : int
            Size of buffer for storing past field values (default: 1000000)
        rc_filter_enabled : bool
            Enable RC filter for voltage input (default: False)
        rc_time_constant : float
            Time constant for RC filter in seconds (default: 10 ps)
        """
        self.radius = radius
        self.resonant_wavelength = resonant_wavelength
        self.n_eff = n_eff
        self.ng = ng
        self.dn_dV = dn_dV

        #Calculate dn/dlambda from the neff and group index for use below
        self.dn_dlambda = (self.n_eff - self.ng)/self.resonant_wavelength
        
        # Calculate kappa from sigma if not provided (assuming lossless coupler)
        self.kappa = np.sqrt(kappa)
        self.sigma = np.sqrt(1 - kappa**2)
            
        # Calculate round-trip time
        self.Lrt = 2 * np.pi * radius  # Round-trip length
        self.junction_loss_dB_m = a  # Round-trip loss in dB/m
        self.a = np.sqrt(np.exp(-self.junction_loss_dB_m*self.Lrt/(10*np.log10(np.e))))
        self.tau = n_eff * self.Lrt / const.c  # Round-trip time

        #Calculate a phase offset that moves the resonance to the desired wavelength
        self.phase_offset = 0
        self.phase_offset = self.calculate_phase(self.resonant_wavelength)
        
        # Initialize buffer for delayed field values
        self.buffer_size = buffer_size
        self.buffer = np.zeros(buffer_size, dtype=complex)
        self.buffer_idx = 0
        self.buffer_initialized = False

        # Initialize RC filter parameters
        self.rc_filter_enabled = rc_filter_enabled
        self.rc_time_constant = rc_time_constant
        self.previous_filtered_voltage = 0.0  # Store last filtered voltage value
        
        print(f"Ring Resonator initialized with round-trip time: {self.tau:.3e} seconds")
        if self.rc_filter_enabled:
            print(f"RC filter enabled with time constant: {self.rc_time_constant:.3e} seconds")
        
    def reset(self):
        """Reset the resonator state."""
        self.buffer = np.zeros(self.buffer_size, dtype=complex)
        self.buffer_idx = 0
        self.buffer_initialized = False
        
    @property 
    def FSR(self):
        """
        Calculate the Free Spectral Range (FSR) of the resonator.
        
        Inputs:
        -------
        resonant_wavelength: float
            The resonance wavelength you want the FSR calculated at

        Returns:
        --------
        FSR : float
            Free Spectral Range in Hz
        """
        return self.resonant_wavelength**2/(self.ng * self.Lrt)
    
    @property
    def finesse(self):
        """
        Calculate the finesse of the resonator.
        
        Returns:
        --------
        finesse : float
            Resonator finesse
        """
        r = self.a * self.sigma
        finesse = np.pi * np.sqrt(r) / (1 - r)
        return finesse

    @property
    def FWHM(self):
        """
        The full width at half maximum of the loaded ring resonator
        """
        num = (1-self.sigma * self.a) * self.resonant_wavelength**2
        denom = np.pi * self.ng * self.Lrt * np.sqrt(self.sigma * self.a)
        return num/denom

    @property
    def quality_factor(self):
        """
        Loaded quality factor of the ring resonator
        """
        return self.resonant_wavelength/self.FWHM

    @property
    def photon_lifetime(self):
        """
        Photon lifetime in the ring resonator
        """
        return self.quality_factor/(2*np.pi * const.c/self.resonant_wavelength)
    
    def setup_sampling(self, dt):
        """
        Set up the sampling parameters based on time step.
        
        Parameters:
        -----------
        dt : float
            Time step in seconds
        
        Returns:
        --------
        buffer_samples : int
            Number of samples needed in buffer for delay
        """
        buffer_samples = int(np.ceil(self.tau / dt)) #This represents a buffer for the light currently propagating inside of the ring

        if buffer_samples > self.buffer_size:
            self.buffer = np.zeros(buffer_samples * 2, dtype=complex)
            self.buffer_size = buffer_samples * 2
            print(f"Warning: Buffer size increased to {self.buffer_size} to accommodate delay")
        return buffer_samples

    
    def process_waveform(self, input_waveform, dt, wavelength_offset=0.0, voltage_waveform=None):
        """
        Process an entire input waveform through the resonator with optimized performance.
        
        Parameters:
        -----------
        input_waveform : numpy.ndarray
            Array of complex input field samples
        dt : float
            Time step in seconds
        wavelength_offset : float
            Wavelength offset from resonance in meters for modulation
        voltage_waveform : numpy.ndarray or None
            The voltage waveform
            
        Returns:
        --------
        output_waveform : numpy.ndarray
            Array of complex output field samples
        """
        # Reset the resonator state
        self.reset()
        
        # Create output array
        n_samples = len(input_waveform)
        output_waveform = np.zeros(n_samples, dtype=complex)
        
        # Setup sampling parameters
        buffer_samples = self.setup_sampling(dt)
        
        # Pre-compute constant phase component (wavelength offset)
        base_phi = 0
        if wavelength_offset != 0.0:
            base_phi = 2*np.pi/(self.resonant_wavelength + wavelength_offset) * self.n_eff * self.Lrt - 2*np.pi/(self.resonant_wavelength) * self.n_eff * self.Lrt
        
        # Pre-compute voltage scaling factor
        voltage_factor = 2*np.pi/self.resonant_wavelength * self.dn_dV * self.Lrt
        
        # Pre-process voltage waveform if RC filter is enabled
        if voltage_waveform is not None and self.rc_filter_enabled:
            alpha = dt / (self.rc_time_constant + dt)
            filtered_voltage = np.zeros_like(voltage_waveform)
            
            # Vectorized first-order IIR filter implementation
            filtered_voltage[0] = alpha * voltage_waveform[0]  # Initial value
            for i in range(1, len(voltage_waveform)):
                filtered_voltage[i] = alpha * voltage_waveform[i] + (1 - alpha) * filtered_voltage[i-1]
        else:
            filtered_voltage = voltage_waveform
        
        # Allocate buffer for T values
        T_buffer = np.zeros(max(buffer_samples + n_samples, self.buffer_size), dtype=complex)
        
        # Initialize first buffer_samples values with sigma (initial condition)
        T_buffer[:buffer_samples] = self.sigma
        
        # Process each sample
        for i in range(n_samples):
            # Calculate phase for this sample
            phi = base_phi
            if voltage_waveform is not None:
                phi += voltage_factor * filtered_voltage[i]
            
            # Get delayed output from buffer
            T_delayed = T_buffer[i]
            
            # Calculate new transmission
            T_current = self.sigma + self.a * np.exp(-1j * phi) * (self.sigma * T_delayed - 1)
            
            # Store in buffer for later use
            T_buffer[i + buffer_samples] = T_current
            
            # Calculate output
            output_waveform[i] = input_waveform[i] * T_current
        
        return output_waveform

    def calculate_phase(self,wavelength, voltage=0):
        return 2*np.pi/(wavelength)*(self.n_eff + self.dn_dlambda * (wavelength - self.resonant_wavelength) + self.dn_dV * voltage) * self.Lrt - self.phase_offset
        
    def plot_frequency_response(self, lambda_start=1500e-9, lambda_end=1600e-9, points=1000, voltage = 0):
        """
        Plot the frequency response of the resonator.
        
        Parameters:
        -----------
        lambda_start : float
            Start wavelength in meters (default: 1500 nm)
        lambda_end : float
            End wavelength in meters (default: 1600 nm)
        points : int
            Number of frequency points (default: 1000)
        """
        wavelengths = np.linspace(lambda_start, lambda_end, points)
        phase_sweep = self.calculate_phase(wavelengths,voltage=voltage)
        num = self.sigma**2 + self.a**2 - 2*self.a*self.sigma*np.cos(phase_sweep)
        denom = 1+self.a**2 * self.sigma**2 - 2*self.a*self.sigma*np.cos(phase_sweep)
        power_transmission = np.real(num/denom * np.conj(num/denom))
        return wavelengths, power_transmission

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

    References
    ----------
    [1] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

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

    References
    ----------
    [1] G. P. Agrawal, Fiber-Optic Communication Systems. Wiley, 2021.

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

    References
    ----------
    [1] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    [2] K. Kikuchi, “Fundamentals of Coherent Optical Fiber Communications”, J. Lightwave Technol., JLT, vol. 34, nº 1, p. 157–179, jan. 2016.
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

    References
    ----------
    [1] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    [2] K. Kikuchi, “Fundamentals of Coherent Optical Fiber Communications”, J. Lightwave Technol., JLT, vol. 34, nº 1, p. 157–179, jan. 2016.
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

    References
    ----------
    [1] M. Seimetz, High-Order Modulation for Optical Fiber Transmission. em Springer Series in Optical Sciences. Springer Berlin Heidelberg, 2009.

    [2] K. Kikuchi, “Fundamentals of Coherent Optical Fiber Communications”, J. Lightwave Technol., JLT, vol. 34, nº 1, p. 157–179, jan. 2016.
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

        - param.G : amplifier gain [dB][default: 20 dB]
        - param.NF : EDFA noise figure [dB][default: 4.5 dB]
        - param.Fc : central optical frequency [Hz][default: 193.1 THz]
        - param.Fs : sampling frequency in [samples/s]

    Returns
    -------
    Eo : np.array
        Amplified noisy optical signal.

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


def basicLaserModel(param=None):
    """
    Laser model with Maxwellian random walk phase noise and RIN.

    Parameters
    ----------
    param : parameter object (struct), optional
        Parameters of the laser.

        - param.P: laser power [dBm] [default: 10 dBm]
        - param.lw: laser linewidth [Hz] [default: 1 kHz]
        - param.RIN_var: variance of the RIN noise [default: 1e-20]
        - param.Fs: sampling rate [samples/s]
        - param.Ns: number of signal samples [default: 1e3]

    Returns
    -------
    optical_signal : np.array
          Optical signal with phase noise and RIN.

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

    t = np.arange(0, Ns) * 1 / Fs

    # Simulate Maxwellian random walk phase noise
    pn = phaseNoise(lw, Ns, 1 / Fs)

    # Simulate relative intensity noise  (RIN)[todo:check correct model]
    deltaP = gaussianComplexNoise(pn.shape, RIN_var)

    # Return optical signal
    return np.sqrt(dBm2W(P) + deltaP) * np.exp(1j * pn)


def adc(Ei, param):
    """
    Analog-to-digital converter (ADC) model.

    Parameters
    ----------
    Ei : ndarray
        Input signal.
    param : core.parameter
        Resampling parameters:
            - param.Fs_in  : sampling frequency of the input signal [samples/s][default: 1 sample/s]
            - param.Fs_out : sampling frequency of the output signal [samples/s][default: 1 sample/s]
            - param.jitter_rms : root mean square (RMS) value of the jitter in seconds [s][default: 0 s]
            - param.nBits : number of bits used for quantization [default: 8 bits]
            - param.Vmax : maximum value for the ADC's full-scale range [V][default: 1V]
            - param.Vmin : minimum value for the ADC's full-scale range [V][default: -1V]
            - param.AAF : flag indicating whether to use anti-aliasing filters [default: True]
            - param.N : number of taps of the anti-aliasing filters [default: 201]

    Returns
    -------
    Eo : ndarray
        Resampled and quantized signal.

    """
    # Check and set default values for input parameters
    param.Fs_in = getattr(param, "Fs_in", 1)
    param.Fs_out = getattr(param, "Fs_out", 1)
    param.jitter_rms = getattr(param, "jitter_rms", 0)
    param.nBits = getattr(param, "nBits", 8)
    param.Vmax = getattr(param, "Vmax", 1)
    param.Vmin = getattr(param, "Vmin", -1)
    param.AAF = getattr(param, "AAF", True)
    param.N = getattr(param, "N", 201)

    # Extract individual parameters for ease of use
    Fs_in = param.Fs_in
    Fs_out = param.Fs_out
    jitter_rms = param.jitter_rms
    nBits = param.nBits
    Vmax = param.Vmax
    Vmin = param.Vmin
    AAF = param.AAF
    N = param.N

    # Reshape the input signal if needed to handle single-dimensional inputs
    try:
        Ei.shape[1]
    except IndexError:
        Ei = Ei.reshape(len(Ei), 1)

    # Get the number of modes (columns) in the input signal
    nModes = Ei.shape[1]

    # Apply anti-aliasing filters if AAF is enabled
    if AAF:
        # Anti-aliasing filters:
        Ntaps = min(Ei.shape[0], N)
        hi = lowPassFIR(param.Fs_out / 2, param.Fs_in, Ntaps, typeF="rect")
        ho = lowPassFIR(param.Fs_out / 2, param.Fs_out, Ntaps, typeF="rect")

        Ei = firFilter(hi, Ei)

    if np.iscomplexobj(Ei):
        # Signal interpolation to the ADC's sampling frequency
        Eo = clockSamplingInterp(
            Ei.reshape(-1, nModes).real, Fs_in, Fs_out, jitter_rms
        ) + 1j * clockSamplingInterp(
            Ei.reshape(-1, nModes).imag, Fs_in, Fs_out, jitter_rms
        )

        # Uniform quantization of the signal according to the number of bits of the ADC
        Eo = quantizer(Eo.real, nBits, Vmax, Vmin) + 1j * quantizer(
            Eo.imag, nBits, Vmax, Vmin
        )
    else:
        # Signal interpolation to the ADC's sampling frequency
        Eo = clockSamplingInterp(Ei.reshape(-1, nModes), Fs_in, Fs_out, jitter_rms)

        # Uniform quantization of the signal according to the number of bits of the ADC
        Eo = quantizer(Eo, nBits, Vmax, Vmin)

    # Apply anti-aliasing filters to the output if AAF is enabled
    if AAF:
        Eo = firFilter(ho, Eo)

    return Eo
