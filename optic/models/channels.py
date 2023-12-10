"""
==============================================================
Models for fiber optic channels (:mod:`optic.models.channels`)
==============================================================

.. autosummary::
   :toctree: generated/

   linearFiberChannel   -- Linear optical fiber channel model.
   ssfm                 -- Nonlinear fiber optic channel model based on the NLSE equation.
   manakovSSF           -- Nonlinear fiber optic channel model based on the Manakov equation.   
   awgn                 -- AWGN channel model.   
"""


"""Basic physical models for optical channels."""
import logging as logg

import numpy as np
import scipy.constants as const
from scipy.linalg import norm
from numpy.fft import fft, fftfreq, ifft
from tqdm.notebook import tqdm
from optic.utils import parameters
from optic.dsp.core import sigPow, gaussianComplexNoise, gaussianNoise
from optic.models.devices import edfa


def linearFiberChannel(Ei, param):
    """
    Simulate signal propagation through a linear fiber channel.

    Parameters
    ----------
    Ei : np.array
        Input optical field.
    param : parameter object  (struct)
        Object with physical/simulation parameters of the optical channel.

        - param.L: total fiber length [km][default: 50 km]

        - param.alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]

        - param.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]

        - param.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]

        - param.Fs: sampling frequency [Hz] [default: None]

        - param.returnParameters: bool, return channel parameters [default: False]

    Returns
    -------
    Eo : np.array
        Optical field at the output of the fiber.

    """
    try:
        Fs = param.Fs
    except AttributeError:
        logg.error("Simulation sampling frequency (Fs) not provided.")

    # check input parameters
    param.L = getattr(param, "L", 50)
    param.alpha = getattr(param, "alpha", 0.2)
    param.D = getattr(param, "D", 16)
    param.Fc = getattr(param, "Fc", 193.1e12)
    param.returnParameters = getattr(param, "returnParameters", False)

    L = param.L
    alpha = param.alpha
    D = param.D
    Fc = param.Fc
    returnParameters = param.returnParameters

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
        fft(Ei, axis=0) * np.exp(-α / 2 * L + 1j * (β2 / 2) * (ω**2) * L), axis=0
    )

    if Nmodes == 1:
        Eo = Eo.reshape(
            Eo.size,
        )

    return (Eo, param) if returnParameters else Eo


def ssfm(Ei, param=None):
    """
    Split-step Fourier method (symmetric, single-pol.).

    Parameters
    ----------
    Ei : np.array
        Input optical signal field.
    Fs : scalar
        Sampling frequency in Hz.
    param : parameter object  (struct)
        Object with physical/simulation parameters of the optical channel.

        - param.Ltotal: total fiber length [km][default: 400 km]

        - param.Lspan: span length [km][default: 80 km]

        - param.hz: step-size for the split-step Fourier method [km][default: 0.5 km]

        - param.alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]

        - param.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]

        - param.gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]

        - param.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]

        - param.Fs: simulation sampling frequency [samples/second][default: None]

        - param.prec: numerical precision [default: np.complex128]

        - param.amp: 'edfa', 'ideal', or 'None. [default:'edfa']

        - param.NF: edfa noise figure [dB] [default: 4.5 dB]

        - param.prgsBar: display progress bar? bolean variable [default:True]

        - param.returnParameters: bool, return channel parameters [default: False]

    Returns
    -------
    Ech : np.array
        Optical signal after nonlinear propagation.
    param : parameter object  (struct)
        Object with physical/simulation parameters used in the split-step alg.

    """
    try:
        Fs = param.Fs
    except AttributeError:
        logg.error("Simulation sampling frequency (Fs) not provided.")

    # check input parameters
    param.Ltotal = getattr(param, "Ltotal", 400)
    param.Lspan = getattr(param, "Lspan", 80)
    param.hz = getattr(param, "hz", 0.5)
    param.alpha = getattr(param, "alpha", 0.2)
    param.D = getattr(param, "D", 16)
    param.gamma = getattr(param, "gamma", 1.3)
    param.Fc = getattr(param, "Fc", 193.1e12)
    param.prec = getattr(param, "prec", np.complex128)
    param.amp = getattr(param, "amp", "edfa")
    param.NF = getattr(param, "NF", 4.5)
    param.prgsBar = getattr(param, "prgsBar", True)
    param.returnParameters = getattr(param, "returnParameters", False)

    Ltotal = param.Ltotal
    Lspan = param.Lspan
    hz = param.hz
    alpha = param.alpha
    D = param.D
    gamma = param.gamma
    Fc = param.Fc
    prec = param.prec
    amp = param.amp
    NF = param.NF
    prgsBar = param.prgsBar
    returnParameters = param.returnParameters

    # channel parameters
    c_kms = const.c / 1e3  # speed of light (vacuum) in km/s
    λ = c_kms / Fc
    α = alpha / (10 * np.log10(np.exp(1)))
    β2 = -(D * λ**2) / (2 * np.pi * c_kms)
    γ = gamma

    # edfa parameters
    paramAmp = parameters()
    paramAmp.G = alpha * Lspan
    paramAmp.NF = NF
    paramAmp.Fc = Fc
    paramAmp.Fs = Fs

    # generate frequency axis
    Nfft = len(Ei)
    ω = 2 * np.pi * Fs * fftfreq(Nfft)

    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(np.floor(Lspan / hz))

    Ech = Ei.reshape(
        len(Ei),
    )

    # define linear operator
    linOperator = np.exp(-(α / 2) * (hz / 2) + 1j * (β2 / 2) * (ω**2) * (hz / 2))

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
            Ech = edfa(Ech, paramAmp)
        elif amp == "ideal":
            Ech = Ech * np.exp(α / 2 * Nsteps * hz)
        elif amp is None:
            Ech = Ech * np.exp(0)

    return (
        (
            Ech.reshape(
                len(Ech),
            ),
            param,
        )
        if returnParameters
        else Ech
    )


def manakovSSF(Ei, param):
    """
    Run the Manakov split-step Fourier model (symmetric, dual-pol.).

    Parameters
    ----------
    Ei : np.array
        Input optical signal field.
    Fs : scalar
        Sampling frequency in Hz.
    param : parameter object  (struct)
        Object with physical/simulation parameters of the optical channel.

        - param.Ltotal: total fiber length [km][default: 400 km]

        - param.Lspan: span length [km][default: 80 km]

        - param.hz: step-size for the split-step Fourier method [km][default: 0.5 km]

        - param.alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]

        - param.D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]

        - param.gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]

        - param.Fc: carrier frequency [Hz] [default: 193.1e12 Hz]

        - param.Fs: simulation sampling frequency [samples/second][default: None]

        - param.prec: numerical precision [default: np.complex128]

        - param.amp: 'edfa', 'ideal', or 'None. [default:'edfa']

        - param.NF: edfa noise figure [dB] [default: 4.5 dB]

        - param.maxIter: max number of iter. in the trap. integration [default: 10]

        - param.tol: convergence tol. of the trap. integration.[default: 1e-5]

        - param.nlprMethod: adap step-size based on nonl. phase rot. [default: True]

        - param.maxNlinPhaseRot: max nonl. phase rot. tolerance [rad][default: 2e-2]

        - param.prgsBar: display progress bar? bolean variable [default:True]

        - param.saveSpanN: specify the span indexes to be outputted [default:[]]

        - param.returnParameters: bool, return channel parameters [default: False]

    Returns
    -------
    Ech : np.array
        Optical signal after nonlinear propagation.
    param : parameter object  (struct)
        Object with physical/simulation parameters used in the split-step alg.

    """
    try:
        Fs = param.Fs
    except AttributeError:
        logg.error("Simulation sampling frequency (Fs) not provided.")

    # check input parameters
    param.Ltotal = getattr(param, "Ltotal", 400)
    param.Lspan = getattr(param, "Lspan", 80)
    param.hz = getattr(param, "hz", 0.5)
    param.alpha = getattr(param, "alpha", 0.2)
    param.D = getattr(param, "D", 16)
    param.gamma = getattr(param, "gamma", 1.3)
    param.Fc = getattr(param, "Fc", 193.1e12)
    param.prec = getattr(param, "prec", np.complex128)
    param.amp = getattr(param, "amp", "edfa")
    param.NF = getattr(param, "NF", 4.5)
    param.maxIter = getattr(param, "maxIter", 10)
    param.tol = getattr(param, "tol", 1e-5)
    param.nlprMethod = getattr(param, "nlprMethod", True)
    param.maxNlinPhaseRot = getattr(param, "maxNlinPhaseRot", 2e-2)
    param.prgsBar = getattr(param, "prgsBar", True)
    param.saveSpanN = getattr(param, "saveSpanN", [param.Ltotal // param.Lspan])
    param.returnParameters = getattr(param, "returnParameters", False)

    Ltotal = param.Ltotal
    Lspan = param.Lspan
    hz = param.hz
    alpha = param.alpha
    D = param.D
    gamma = param.gamma
    Fc = param.Fc
    amp = param.amp
    NF = param.NF
    prec = param.prec
    maxIter = param.maxIter
    tol = param.tol
    prgsBar = param.prgsBar
    saveSpanN = param.saveSpanN
    nlprMethod = param.nlprMethod
    maxNlinPhaseRot = param.maxNlinPhaseRot
    returnParameters = param.returnParameters

    # channel parameters
    c_kms = const.c / 1e3  # speed of light (vacuum) in km/s
    λ = c_kms / Fc
    α = alpha / (10 * np.log10(np.exp(1)))
    β2 = -(D * λ**2) / (2 * np.pi * c_kms)
    γ = gamma

    # edfa parameters
    paramAmp = parameters()
    paramAmp.G = alpha * Lspan
    paramAmp.NF = NF
    paramAmp.Fc = Fc
    paramAmp.Fs = Fs

    # generate frequency axis
    Nfft = len(Ei)
    ω = 2 * np.pi * Fs * fftfreq(Nfft)

    Nspans = int(np.floor(Ltotal / Lspan))

    Ech_x = Ei[:, 0::2].T
    Ech_y = Ei[:, 1::2].T

    # define static part of the linear operator
    argLimOp = np.array(-(α / 2) + 1j * (β2 / 2) * (ω**2)).astype(prec)

    if Ech_x.shape[0] > 1:
        argLimOp = np.tile(argLimOp, (Ech_x.shape[0], 1))
    else:
        argLimOp = argLimOp.reshape(1, -1)

    if saveSpanN:
        Ech_spans = np.zeros((Ei.shape[0], Ei.shape[1] * len(saveSpanN))).astype(prec)
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
                hz_ = (
                    maxNlinPhaseRot / np.max(phiRot)
                    if Lspan - z_current >= maxNlinPhaseRot / np.max(phiRot)
                    else Lspan - z_current
                )
            elif Lspan - z_current < hz:
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
                lim = convergenceCondition(Ech_x_fd, Ech_y_fd, Ex_conv, Ey_conv)

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
            Ech_x = edfa(Ech_x, paramAmp)
            Ech_y = edfa(Ech_y, paramAmp)
        elif amp == "ideal":
            Ech_x = Ech_x * np.exp(α / 2 * Lspan)
            Ech_y = Ech_y * np.exp(α / 2 * Lspan)
        elif amp is None:
            Ech_x = Ech_x * np.exp(0)
            Ech_y = Ech_y * np.exp(0)

        if spanN in saveSpanN:
            Ech_spans[:, 2 * indRecSpan : 2 * indRecSpan + 1] = Ech_x.T
            Ech_spans[:, 2 * indRecSpan + 1 : 2 * indRecSpan + 2] = Ech_y.T
            indRecSpan += 1

    if saveSpanN:
        Ech = Ech_spans
    else:
        Ech = Ei.copy()
        Ech[:, 0::2] = Ech_x.T
        Ech[:, 1::2] = Ech_y.T

    if returnParameters:
        return Ech, param
    else:
        return Ech


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
    return np.sqrt(norm(Ex_fd - Ex_conv) ** 2 + norm(Ey_fd - Ey_conv) ** 2) / np.sqrt(
        norm(Ex_conv) ** 2 + norm(Ey_conv) ** 2
    )


def awgn(sig, snr, Fs=1, B=1, complexNoise=True):
    """
    Implement a basic AWGN channel model.

    Parameters
    ----------
    sig : np.array
        Input signal.
    snr : scalar
        Signal-to-noise ratio in dB.
    Fs : real scalar
        Sampling frequency. The default is 1.
    B : real scalar
        Signal bandwidth, defined as the length of the frequency interval [-B/2, B/2]. The default is 1.
    complexNoise : bool
        Generate complex-valued noise. The default is True.

    Returns
    -------
    np.array
        Input signal plus noise.

    """
    snr_lin = 10 ** (snr / 10)
    noiseVar = sigPow(sig) / snr_lin
    σ2 = (Fs / B) * noiseVar

    if complexNoise:
        noise = gaussianComplexNoise(sig.shape, σ2)
    else:
        noise = gaussianNoise(sig.shape, σ2)

    return sig + noise
