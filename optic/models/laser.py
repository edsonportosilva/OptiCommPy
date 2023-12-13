import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.integrate import solve_ivp,odeint
from scipy.constants import c, h, e
from optic.utils import parameters

def laser_dfb(param, current):
    """
    Simulates the behavior of a distributed feedback (DFB) laser.

    Args:
        param: The parameters of the DFB laser.
        current: The current object.

    Returns:
        A tuple containing the updated parameters, current object, and the solution of the DFB laser simulation.
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

def laser_dynamic_response(y, param):
    """
    Calculates the dynamic response of the laser.

    Args:
        y: The state of the laser.
        param: The parameters of the laser.

    Returns:
        An object containing the calculated laser parameters including power, chirp, and output electric field.
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

def set_current(param, current):
    """
    Sets the current signal and time step for the current object.

    Args:
        param: The parameters for setting the current signal.
        current: The current object.

    Returns:
        The updated current object with the modified signal and time step.
    """
    current.signal = param.i_max * current.signal + param.i_bias
    current.t_step = current.t[1]-current.t[0]
    return current

def get_current(current, t):
    """
    Returns the current value at a given time.

    Args:
        current: The current object.
        t: The time value.

    Returns:
        The real part of the current signal at the specified time.
    """
    return np.real(current.signal[np.argwhere(t >= current.t)[-1]])

def solve_laser_dfb(param, current):
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
    Calculates the initial conditions for the laser rate equations.

    Args:
        param: The parameters of the rate equations.
        current: The current value.

    Returns:
        The array of initial conditions [y0[0], y0[1], y0[2]].
    """
    y0 = np.zeros([3])        
    y0[1] = param.gamma * param.tau_p/(param.v * e) * (get_current(current,0)-param.ith)
    y0[0] = param.n_t + (1 + param.epsilon * y0[1]) / (param.vg * param.a0 * param.gamma * param.tau_p)
    y0[2] = (param.alpha / 2) * (param.gamma * param.vg * param.a0 * (y0[0] - param.n_t) - 1 / param.tau_p)
    return y0

def laser_rate_equations(t, y, param, current):
    """
    Calculate the derivatives of the laser state variables based on the laser rate equations.

    Parameters
    ----------
    t : scalar
        Time value.
    y : np.array
        Array of laser state variables.
    param : parameter object (struct)
        Object with laser parameters.
    current : parameter object (struct)
        Object with current input signal.

    Returns
    -------
    np.array
        Array of derivatives of the laser state variables.
    """
    #param, current = args
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
    Calculate the rate of change of the carrier density.

    Parameters
    ----------
    t : scalar
        Time value.
    y : np.array
        Array of laser state variables.
    param : parameter object (struct)
        Object with laser parameters.
    current : parameter object (struct)
        Object with current input signal.

    Returns
    -------
    scalar
        Rate of change of the carrier density.
    """
    current = get_current(current, t)
    return current/(e*param.v)-y[0]/param.tau_n-param.vg*param.a0*y[1]*(y[0]-param.n_t)/(1+param.epsilon*y[1])
  
def photon_density(y,param):
    """
    Calculate the rate of change of the photon density.

    Parameters
    ----------
    y : np.array
        Array of laser state variables.
    param : parameter object (struct)
        Object with laser parameters.

    Returns
    -------
    scalar
        Rate of change of the photon density.
    """
    return (param.gamma * param.a0 * param.vg * ((y[0]-param.n_t)/(1+param.epsilon*y[1]))-1/param.tau_p)*y[1]+param.beta*param.gamma*y[0]/param.tau_n
    
def optical_phase(y,param):
    """
    Calculate the rate of change of the optical phase.

    Parameters
    ----------
    y : np.array
        Array of laser state variables.
    param : parameter object (struct)
        Object with laser parameters.

    Returns
    -------
    scalar
        Rate of change of the optical phase.
    """    
    return param.alpha/2 * (param.gamma*param.vg*param.a0*(y[0]-param.n_t)-1/param.tau_p)

def add_noise_rate_equations(y, dy, param, current):
    """
    Adds noise terms to the rate equations.

    Args:
        y: The current state of the rate equations.
        dy: The rate of change of the rate equations.
        param: The parameters of the rate equations.
        current: The current value.

    Returns:
        The updated rate of change of the rate equations.
    """
    # Calculate noise terms
    dn = laser_noise_sources(y,param,current) if param.noise_terms else np.zeros(3)
    # Add noise terms to dy
    dy += dn
    return dy

def laser_noise_sources(y,param,current):
    """
    Calculates the noise terms for the laser rate equations.

    Args:
        y: The current state of the rate equations.
        param: The parameters of the rate equations.
        current: The current value.

    Returns:
        A list of noise terms [fn, fs, fp].
    """
    # diffusion coefficient - see ref. [2]
    dss = (param.beta*y[0]*y[1]/param.tau_n)
    #dnn = (y[0]/self.tau_n)*(1+self.beta*y[1])
    dpp = (param.beta*y[0])/(4*param.tau_n*y[1])
    dzz = y[0]/param.tau_n
    # noise sources
    t_step_sqrt = np.sqrt(2 / current.t_step)
    fs = np.random.randn() * np.sqrt(dss) * t_step_sqrt
    fz = np.random.randn() * np.sqrt(dzz) * t_step_sqrt
    fp = np.random.randn() * np.sqrt(dpp) * t_step_sqrt
    fn = fz - fs
    return [fn, fs, fp]

def get_im_response(param, out, f, type='exact'):
    """
    Calculates the impulse response of an intensity modulator (IM).

    Args:
        param: The parameters of the IM.
        out: The output of the IM.
        f: The frequency values.
        type: The type of response to calculate ('exact' or 'approx.').

    Returns:
        The calculated Y, Z, and response values based on the specified type.
    """
    Y, Z = im_response_yz(param,out)
    if type=='exact':
        return Y, Z, im_response_hf(f,Y,Z)
    elif type=='approx.':
        return Y, Z, im_response_haf(f,Y,Z)
    else:
        print('Invalid type of IM response.')
        return -1

def im_response_hf(f,Y,Z):
    """
    Calculates the high-frequency response of an intensity modulator (IM) - Eq. [6] ref. [1].

    Args:
        f: The frequency values.
        Y: The Y parameter of the IM.
        Z: The Z parameter of the IM.

    Returns:
        The calculated high-frequency response of the IM.
    """
    return Z/((1j*2*np.pi*f)**2+1j*2*np.pi*f*Y+Z)

def im_response_haf(f,Y,Z):
    """
    Calculates the high-frequency approximation response of an intensity modulator (IM) - Eq. [12] ref. [1]

    Args:
        f: The frequency values.
        Y: The Y parameter of the IM.
        Z: The Z parameter of the IM.

    Returns:
        The calculated high-frequency approximation response of the IM.
    """
    fr = np.sqrt(Z)/(2*np.pi)
    return fr**2/((fr**2-f**2)+1j*f*Y/(2*np.pi))

def im_response_yz(param,out):
    """
    Calculates the Y and Z parameters for the intensity modulation frequency response - see ref. [1]

    Args:
        param: The parameters of the intensity modulator.
        out: The output of the intensity modulator.

    Returns:
        A tuple containing the calculated Y and Z parameters for the intensity modulation frequency response.
    """
    return im_response_y(param,out),im_response_z(param,out)

def im_response_y(param,out):
    """
    Calculates the Y parameter for the intensity modulation frequency response - Eq. [7] ref. [1].

    Args:
        param: The parameters of the intensity modulator.
        out: The output of the intensity modulator.

    Returns:
        The calculated Y parameter for the intensity modulation frequency response.
    """
    Y = param.vg*param.a0*out.S[-1]/(1+param.epsilon*out.S[-1]) + 1/param.tau_n + 1/param.tau_p
    Y -= param.gamma*param.a0*param.vg*(out.N[-1]-param.n_t)/(1+param.epsilon*out.S[-1])**2
    return Y

def im_response_z(param,out):
    """
    Calculates the Z parameter for the intensity modulation frequency response - Eq. [8] ref. [1].

    Args:
        param: The parameters of the intensity modulator.
        out: The output of the intensity modulator.

    Returns:
        The calculated Z parameter for the intensity modulation frequency response.
    """
    Z = param.vg*param.a0*out.S[-1]/(1+param.epsilon*out.S[-1]) * 1/param.tau_p + 1/(param.tau_p*param.tau_n)
    Z += (param.beta-1)*param.gamma*param.a0*param.vg/param.tau_n*(out.N[-1]-param.n_t)/(1+param.epsilon*out.S[-1])**2
    return Z

def plot(t,y):
    """
    Plots the optical power, chirp, carrier density, and photon density.

    Args:
        t: The time values.
        y: The values to be plotted.

    Returns:
        The matplotlib axes object containing the plot.
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