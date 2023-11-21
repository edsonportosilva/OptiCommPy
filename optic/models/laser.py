import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.integrate import solve_ivp,odeint
from scipy.constants import c, h, e

def laser_dfb(param):
## add noise terms
        param.noise_terms = getattr(param, "noise_terms", False)
        # active layer volume
        param.v = getattr(param, "v", 1.5e-10)
        # electron lifetime
        param.tau_n = getattr(param, "tau_n", 1.0e-9)
        # active layer gain coefficient
        param.a0 = getattr(param, "a0", 2.5e-16)
        # group velocity
        param.vg = getattr(param, "vg", 8.5e+9)
        # carrier density at transparency
        param.n_t = getattr(param, "n_t", 1.0e+18)
        # gain compression factor
        param.epsilon = getattr(param, "epsilon", 1.0e-17)
        # mode confinement factor
        param.gamma = getattr(param, "gamma", 0.4)
        # photon lifetime
        param.tau_p = getattr(param, "tau_p", 3.0e-12)
        # fraction of spontaneous emission coupling
        param.beta = getattr(param, "beta", 3.0e-5)
        # linewidth enchancement factor
        param.alpha = getattr(param, "alpha", 5)
        # gain cross section
        param.sigma = getattr(param, "sigma", 2e-20)
        # i_bias
        param.i_bias = getattr(param, 'i_bias', 0.078)
        # i_max
        param.i_max  = getattr(param, 'i_max', 0.188)
        # total differential quantum efficiency
        param.eta_0 = getattr(param, 'eta_0', 0.4)
        # wavelength
        param.lmbd = getattr(param, 'lmbd', 1300e-9)
        # frequency
        param.freq_0  = c/param.lmbd
        # threshold current
        param.ith = e/param.tau_n*param.v*(param.n_t + 1/(param.gamma * param.tau_p * param.vg * param.a0))