"""
========================================================
Models for DFB laser (:mod:`optic.models.laser`)
========================================================

.. autosummary::
   :toctree: generated/

   laser_dfb             -- Semiconductor DFB laser model.
"""


"""Basic physical models for DFB laser."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.integrate import solve_ivp,odeint
from scipy.constants import c, h, e
from optic.utils import parameters

def laser_dfb(param, current):
    """
    Simulate the behavior of a laser in a DFB (Distributed Feedback) structure.

    Parameters
    ----------
    param : Namespace
        An object containing parameters for the DFB simulation.
        - noise_terms : bool, optional
            Flag to include noise terms in the simulation. Default is False.
        - v : float, optional
            Active layer volume. Default is 1.5e-10 cm^3.
        - tau_n : float, optional
            Carrier lifetime. Default is 1.0e-9 s.
        - a0 : float, optional
            Active layer gain coefficient. Default is 2.5e-16 m^3.
        - vg : float, optional
            Group velocity. Default is 8.5e+9 m/s.
        - n_t : float, optional
            Carrier density at transparency. Default is 1.0e+18 cm^-3.
        - epsilon : float, optional
            Gain compression factor. Default is 1.0e-17 cm^3.
        - gamma : float, optional
            Mode confinement factor. Default is 0.4.
        - tau_p : float, optional
            Photon lifetime. Default is 3.0e-12 s.
        - beta : float, optional
            Fraction of spontaneous emission coupling. Default is 3.0e-5.
        - alpha : float, optional
            Linewidth enhancement factor. Default is 5.
        - sigma : float, optional
            Absorption cross-section. Default is 2e-20 m^2.
        - i_bias : float, optional
            Bias current. Default is 0.078 A.
        - i_max : float, optional
            Maximum current. Default is 0.188 A.
        - eta_0 : float, optional
            Quantum efficiency. Default is 0.4.
        - lmbd : float, optional
            Wavelength of the laser. Default is 1300e-9 m.
    
    current : Namespace
        An object containing information about the current distribution.
        - signal : ndarray
            Current signal distribution.
        - t : ndarray
            Time array representing the current signal over time.

    Returns
    -------
    tuple
        A tuple containing the updated `param`, the modified `current`, and the result of solving the laser DFB simulation.
    """
    param.noise_terms = getattr(param, "noise_terms", False)
    param.v = getattr(param, "v", 1.5e-10)
    param.tau_n = getattr(param, "tau_n", 1.0e-9)
    param.a0 = getattr(param, "a0", 2.5e-16)
    param.vg = getattr(param, "vg", 8.5e+9)
    param.n_t = getattr(param, "n_t", 1.0e+18)
    param.epsilon = getattr(param, "epsilon", 1.0e-17)
    param.gamma = getattr(param, "gamma", 0.4)
    param.tau_p = getattr(param, "tau_p", 3.0e-12)
    param.beta = getattr(param, "beta", 3.0e-5)
    param.alpha = getattr(param, "alpha", 5)
    param.sigma = getattr(param, "sigma", 2e-20)
    param.i_bias = getattr(param, 'i_bias', 0.078)
    param.i_max  = getattr(param, 'i_max', 0.188)
    param.eta_0 = getattr(param, 'eta_0', 0.4)
    param.lmbd = getattr(param, 'lmbd', 1300e-9)
    param.freq_0  = c/param.lmbd
    # Current Threshold
    param.ith = e/param.tau_n*param.v*(param.n_t + 1/(param.gamma * param.tau_p * param.vg * param.a0))
    # Electrical current
    current = set_current(param, current)

    return param, current, solve_laser_dfb(param, current)

def set_current(param, current):
    """
    Set the electrical current for a laser simulation.

    Parameters
    ----------
    param : Namespace
        An object containing parameters for the simulation.
        - i_max : float
            Maximum current.
        - i_bias : float
            Bias current.

    current : Namespace
        An object containing information about the current distribution.
        - signal : ndarray
            Current signal distribution.
        - t : ndarray
            Time array representing the current signal over time.
        - t_step : float
            Time step between consecutive elements in `t`.

    Returns
    -------
    Namespace
        An updated `current` object with the modified current signal and time step.
    """
    current.signal = param.i_max * current.signal + param.i_bias
    current.t_step = current.t[1]-current.t[0]
    return current

def laser_dynamic_response(y, param):
    """
    Calculate the dynamic response of a laser after simulation.

    Parameters
    ----------
    y : ndarray
        An array containing the state variables of the laser.
        - y[0, :] : float
            Carrier density.
        - y[1, :] : float
            Photon density.
        - y[2, :] : float
            Optical phase.

    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)
    
    Returns
    -------
    Namespace
        An object containing the calculated dynamic response parameters.
        - N : float
            Carrier density.
        - S : float
            Photon density.
        - phase : float
            Optical phase.
        - power : float
            Optical power.
        - chirp : float
            Chirp parameter.
        - e_out : complex
            Complex electric field.
    """
    out = parameters()
    # get physical parameters
    out.N = y[0,:]
    out.S = y[1,: ]
    out.phase = y[2,:]
    # get signal parameters
    out.power = (out.S/2) * (param.v * h * param.freq_0 * param.eta_0)/(param.gamma * param.tau_p)
    out.chirp = 1/(2*np.pi) * (param.alpha / 2) * (param.gamma * param.vg * param.a0 * (out.N - param.n_t) - 1/param.tau_p) # np.diff(y[2,:],prepend=y[2,0])/self.t_step
    out.e_out = np.sqrt(np.real(out.power)) * np.exp(1j * out.phase)
    return out

def get_current(current, t):
    """
    Get the electrical current at a specific time for a laser simulation.

    Parameters
    ----------
    current : Namespace
        An object containing information about the current distribution.
        (Refer to set_current function docstring for details.)
    
    t : float
        The specific time at which to retrieve the current.

    Returns
    -------
    float
        The electrical current at the specified time.
    """
    return np.real(current.signal[np.argwhere(t >= current.t)[-1]])

def solve_laser_dfb(param, current):
    """
    Solve the rate equations for a laser in a DFB (Distributed Feedback) structure.

    Parameters
    ----------
    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)

    current : Namespace
        An object containing information about the current distribution.
        (Refer to set_current function docstring for details.)

    Returns
    -------
    scipy.integrate.OdeResult
        The result of solving the rate equations for the laser DFB simulation.
    """
    # Initial conditions        
    return solve_ivp(
        laser_rate_equations,
        t_span=[current.t[0], current.t[-1]],
        y0=get_initial_conditions(param,current),
        args=(param,current,),
        method='RK45',
        t_eval = current.t,
        dense_output=True,
        #rtol = 1e-4,
        #atol = 1e-6,
    )

def get_initial_conditions(param,current):
    """
    Get the initial conditions for the rate equations of a laser in a DFB (Distributed Feedback) structure.

    Parameters
    ----------
    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)

    current : Namespace
        An object containing information about the current distribution.
        (Refer to laser_dfb function docstring for details.)

    Returns
    -------
    ndarray
        An array containing the initial conditions for the rate equations.
        - y0[0] : float
            Initial carrier density.
        - y0[1] : float
            Initial photon density.
        - y0[2] : float
            Initial phase.
    """
    y0 = np.zeros([3])        
    y0[1] = param.gamma * param.tau_p/(param.v * e) * (get_current(current,0)-param.ith)
    y0[0] = param.n_t + (1 + param.epsilon * y0[1]) / (param.vg * param.a0 * param.gamma * param.tau_p)
    y0[2] = (param.alpha / 2) * (param.gamma * param.vg * param.a0 * (y0[0] - param.n_t) - 1 / param.tau_p)
    return y0

def laser_rate_equations(t, y, param, current):
    """
    Calculate the rate equations for a laser in a DFB (Distributed Feedback) structure.

    Parameters
    ----------
    t : float
        Time.
    y : ndarray
        An array containing the state variables of the laser.
        - y[0] : float
            Carrier density.
        - y[1] : float
            Photon density.
        - y[2] : float
            Optical phase.
    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)
    current : Namespace
        An object containing information about the current distribution.
        (Refer to set_current function docstring for details.)

    Returns
    -------
    ndarray
        An array containing the calculated rate equations.
        - dy[0] : float
            Rate of change of carrier density.
        - dy[1] : float
            Rate of change of photon density.
        - dy[2] : float
            Rate of change of optical phase.
    """
    dy = np.zeros(3)        
    # Carrier density
    dy[0] = carrier_density(t,y,param,current)
    # Foton density
    dy[1] = photon_density(y,param)
    # Optical phase
    dy[2] = optical_phase(y,param)
    return add_noise_rate_equations(y,dy,param,current)

def carrier_density(t,y,param,current):
    """
    Calculate the rate of change of carrier density in a laser.

    Parameters
    ----------
    t : float
        Time.
    y : ndarray
        An array containing the state variables of the laser.
        - y[0] : float
            Carrier density.
        - y[1] : float
            Photon density.
        - y[2] : float
            Optical phase.
    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)
    current : Namespace
        An object containing information about the current distribution.
        (Refer to set_current function docstring for details.)

    Returns
    -------
    float
        The rate of change of carrier density.
    """
    current = get_current(current, t)
    return current/(e*param.v)-y[0]/param.tau_n-param.vg*param.a0*y[1]*(y[0]-param.n_t)/(1+param.epsilon*y[1])
  
def photon_density(y,param):
    """
    Calculate the rate of change of photon density in a laser.

    Parameters
    ----------
    y : ndarray
        An array containing the state variables of the laser.
        - y[0] : float
            Carrier density.
        - y[1] : float
            Photon density.
        - y[2] : float
            Optical phase.
    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)

    Returns
    -------
    float
        The rate of change of photon density.
    """
    return (param.gamma * param.a0 * param.vg * ((y[0]-param.n_t)/(1+param.epsilon*y[1]))-1/param.tau_p)*y[1]+param.beta*param.gamma*y[0]/param.tau_n
    
def optical_phase(y,param):
    """
    Calculate the rate of change of optical phase in a laser.

    Parameters
    ----------
    y : ndarray
        An array containing the state variables of the laser.
        - y[0] : float
            Carrier density.
        - y[1] : float
            Photon density.
        - y[2] : float
            Optical phase.
    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)

    Returns
    -------
    float
        The rate of change of optical phase.
    """
    return param.alpha/2 * (param.gamma*param.vg*param.a0*(y[0]-param.n_t)-1/param.tau_p)

def add_noise_rate_equations(y, dy, param, current):
    """
    Add noise terms to the rate equations for a laser.

    Parameters
    ----------
    y : ndarray
        An array containing the state variables of the laser.
        - y[0] : float
            Carrier density.
        - y[1] : float
            Photon density.
        - y[2] : float
            Optical phase.
    dy : ndarray
        An array containing the calculated rate equations.
        - dy[0] : float
            Rate of change of carrier density.
        - dy[1] : float
            Rate of change of photon density.
        - dy[2] : float
            Rate of change of optical phase.
    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)
    current : Namespace
        An object containing information about the current distribution.
        (Refer to laser_dfb function docstring for details.)

    Returns
    -------
    ndarray
        An array containing the rate equations with added noise terms.
    """
    # Calculate noise terms
    dn = laser_noise_sources(y,param,current) if param.noise_terms else np.zeros(3)
    # Add noise terms to dy
    dy += dn
    return dy

def laser_noise_sources(y,param,current):
    """
    Calculate noise sources for a laser.

    Parameters
    ----------
    y : ndarray
        An array containing the state variables of the laser.
        - y[0] : float
            Carrier density.
        - y[1] : float
            Photon density.
        - y[2] : float
            Optical phase.
    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)
    current : Namespace
        An object containing information about the current distribution.
        (Refer to laser_dfb function docstring for details.)

    Returns
    -------
    list
        A list containing noise sources.
        - fn : float
            Noise source related to the optical phase.
        - fs : float
            Noise source related to the carrier density and photon density.
        - fp : float
            Noise source related to the photon density.
    """
    # diffusion coefficient - see ref. [2]
    dss = (param.beta*y[0]*y[1]/param.tau_n)
    #dnn = (y[0]/self.tau_n)*(1+self.beta*y[1])
    dpp = (param.beta*y[0])/(4*param.tau_n*y[1])
    dzz = y[0]/param.tau_n
    # noise sources
    t_step_sqrt = np.sqrt(2 / current.t_step)
    fs = np.random.randn() * np.sqrt(dss) * t_step_sqrt
    fp = np.random.randn() * np.sqrt(dpp) * t_step_sqrt
    fz = np.random.randn() * np.sqrt(dzz) * t_step_sqrt
    fn = fz - fs
    return [fn, fs, fp]

def get_im_response(param, out, f, type='exact'):
    """
    Get the Y and Z parameters and the intensity modulation (IM) frequency response of a semiconductor laser.

    Parameters
    ----------
    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)

    out : Namespace
        An object containing laser simulation output.
        - N : ndarray
            Carrier density over time.
        - S : ndarray
            Photon density over time.
        - phase : ndarray
            Optical phase over time.
        - power : ndarray
            Optical power over time.
        - chirp : ndarray
            Optical chirp over time.
        - e_out : ndarray
            Optical output over time.

    f : float
        The frequency of interest for the IM frequency response.

    type : {'exact', 'approx.'}, optional
        The type of IM response to calculate. Default is 'exact'.
        - 'exact': Exact IM response.
        - 'approx.': Approximate IM response.

    Returns
    -------
    ndarray, ndarray, ndarray
        Arrays containing the IM response:
        - Y : ndarray
            Function of laser parameters and bias current.
        - Z : ndarray
            Function of laser parameters and bias current.
        - im_response : ndarray
            The calculated IM frequency response.
    """
    Y, Z = im_response_yz(param,out)
    if type=='exact':
        return Y, Z, im_response_hf(f,Y,Z)
    elif type=='approx.':
        return Y, Z, im_response_haf(f,Y,Z)
    else:
        print('Invalid type of IM response.')
        return -1
    
def im_response_yz(param,out):
    """
    Calculate the components Y and Z function of laser parameters and bias current.

    Parameters
    ----------
    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)

    out : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dynamic_response function docstring for details.)


    Returns
    -------
    ndarray, ndarray
        Arrays containing the components Y and Z of the IM response:
        - Y : ndarray
            Function of laser parameters and bias current.
        - Z : ndarray
            Function of laser parameters and bias current.
    """
    return im_response_y(param,out),im_response_z(param,out)

def im_response_y(param,out):
    """
    Calculate the function Y of laser parameters and bias current.

    Parameters
    ----------
    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)

    out : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dynamic_response function docstring for details.)

    Returns
    -------
    float
        The calculated component Y.
    """
    Y = param.vg*param.a0*out.S[-1]/(1+param.epsilon*out.S[-1]) + 1/param.tau_n + 1/param.tau_p
    Y -= param.gamma*param.a0*param.vg*(out.N[-1]-param.n_t)/(1+param.epsilon*out.S[-1])**2
    return Y

def im_response_z(param,out):
    """
    Calculate the function Z of laser parameters and bias current.

    Parameters
    ----------
    param : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dfb function docstring for details.)

    out : Namespace
        An object containing parameters for the simulation.
        (Refer to laser_dynamic_response function docstring for details.)

    Returns
    -------
    float
        The calculated component Z.
    """
    Z = param.vg*param.a0*out.S[-1]/(1+param.epsilon*out.S[-1]) * 1/param.tau_p + 1/(param.tau_p*param.tau_n)
    Z += (param.beta-1)*param.gamma*param.a0*param.vg/param.tau_n*(out.N[-1]-param.n_t)/(1+param.epsilon*out.S[-1])**2
    return Z

def im_response_hf(f,Y,Z):
    """
    Calculate the intensity modulation (IM) frequency response of a semiconductor laser.

    Parameters
    ----------
    f : float
        The frequency of interest for the IM response.

    Y : ndarray
        Function of laser parameters and bias current.

    Z : ndarray
        Function of laser parameters and bias current.

    Returns
    -------
    ndarray
        The calculated IM frequency response.
    """
    return Z/((1j*2*np.pi*f)**2+1j*2*np.pi*f*Y+Z)

def im_response_haf(f,Y,Z):
    """
    Calculate the approximated intensity modulation (IM) frequency response of a semiconductor laser.

    Parameters
    ----------
    f : float
        The frequency of interest for the IM response.

    Y : ndarray
        Function of laser parameters and bias current.

    Z : ndarray
        Function of laser parameters and bias current.

    Returns
    -------
    ndarray
        The calculated high-frequency approximate IM response.
    """
    fr = np.sqrt(Z)/(2*np.pi)
    return fr**2/((fr**2-f**2)+1j*f*Y/(2*np.pi))

def plot(t,y):
    """
    Plot various laser simulation outputs over time.

    Parameters
    ----------
    t : ndarray
        Time values.

    y : Namespace
        An object containing laser simulation output.
        - power : ndarray
            Optical power over time.
        - chirp : ndarray
            Optical chirp over time.
        - N : ndarray
            Carrier density over time.
        - S : ndarray
            Photon density over time.

    Returns
    -------
    ndarray
        Axes of the generated subplots.
    """
    fig,ax=plt.subplots(2,2,figsize=(12,6))
    ax[0,0].plot(1e9*t, 1e3*y.power)
    _extracted_from_plot_4(t, ax, 0, 0, 'Optical Power [mW]')
    ax[1,0].plot(1e9*t, 1e-9*np.real(y.chirp))
    _extracted_from_plot_4(t, ax, 1, 0, 'Chirp [GHz]')
    ax[0,1].plot(1e9*t, np.real(y.N))
    _extracted_from_plot_4(t, ax, 0, 1, r'Carrier density $N(t)$ [cm$^{-3}$]')
    ax[1,1].plot(1e9*t, np.real(y.S))
    _extracted_from_plot_4(t, ax, 1, 1, r'Photon Density $S(t)$ [cm$^{-3}$]')
    plt.tight_layout()
    return ax

def _extracted_from_plot_4(t, ax, arg1, arg2, arg3):
    ax[arg1, arg2].set_xlabel('Time [ns]')
    ax[arg1, arg2].set_ylabel(arg3)
    ax[arg1, arg2].set_xlim(1e9*np.array([t.min(),t.max()]))
    ax[arg1, arg2].grid(True)