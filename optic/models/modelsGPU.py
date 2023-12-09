"""
====================================================================================
Functions adapted to run with GPU (CuPy) processing (:mod:`optic.models.modelsGPU`)
====================================================================================

.. autosummary::
   :toctree: generated/

   ssfm                 -- Nonlinear fiber optic channel model based on the NLSE equation [GPU].
   manakovSSF           -- Nonlinear fiber optic channel model based on the Manakov equation [GPU].  
   edfa                 -- Simple EDFA model (gain + AWGN noise)[GPU].
"""

"""Functions from models.py adapted to run with GPU (CuPy) processing."""
import logging as logg

import cupy as cp
import numpy as np
import scipy.constants as const
from cupy.linalg import norm
from cupy.random import normal
from cupyx.scipy.fft import fft, fftfreq, ifft
from tqdm.notebook import tqdm

from optic.utils import parameters
from optic.dsp.core import signal_power


def gaussianComplexNoise(shapeOut, σ2=1.0):
    """
    Generate complex circular Gaussian noise.

    Parameters
    ----------
    shapeOut : tuple of int
        Shape of ndarray to be generated.
    σ2 : float, optional
        Variance of the noise (default is 1).

    Returns
    -------
    noise : ndarray
        Generated complex circular Gaussian noise.
    """
    return normal(0, np.sqrt(σ2 / 2), shapeOut) + 1j * normal(
        0, np.sqrt(σ2 / 2), shapeOut
    )


def edfa(Ei, param):
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
    prec = getattr(param, "prec", cp.complex128)

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
    noise = cp.array(noise).astype(prec)

    return Ei * np.sqrt(G_lin) + noise


def ssfm(Ei, param):
    """
    Split-step Fourier method (symmetric, single-pol.).

    Parameters
    ----------
    Ei : np.array
        Input optical signal field.
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

        - param.prec: numerical precision [default: cp.complex128]

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
    param.prec = getattr(param, "prec", cp.complex128)
    param.amp = getattr(param, "amp", "edfa")
    param.NF = getattr(param, "NF", 4.5)
    param.prgsBar = getattr(param, "prgsBar", True)
    param.saveSpanN = getattr(param, "saveSpanN", [param.Ltotal // param.Lspan])
    param.returnParameters = getattr(param, "returnParameters", False)

    Ltotal = param.Ltotal
    Lspan = param.Lspan
    hz = param.hz
    alpha = param.alpha
    D = param.D
    gamma = param.gamma
    amp = param.amp
    NF = param.NF
    Fc = param.Fc
    prec = param.prec
    prgsBar = param.prgsBar
    saveSpanN = param.saveSpanN
    returnParameters = param.returnParameters

    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(np.floor(Lspan / hz))

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
    paramAmp.prec = prec

    c_kms = cp.asarray(c_kms, dtype=prec)  # speed of light (vacuum) in km/s
    λ = cp.asarray(λ, dtype=prec)
    α = cp.asarray(α, dtype=prec)
    β2 = cp.asarray(β2, dtype=prec)
    γ = cp.asarray(γ, dtype=prec)
    hz = cp.asarray(hz, dtype=prec)
    Ltotal = cp.asarray(Ltotal, dtype=prec)

    # generate frequency axis
    Nfft = len(Ei)
    ω = 2 * np.pi * Fs * fftfreq(Nfft).astype(prec)

    Ei_ = cp.asarray(Ei).astype(prec)

    Ech = Ei_.reshape(
        len(Ei),
    )
    # define linear operator
    linOperator = cp.array(
        cp.exp(-(α / 2) * (hz / 2) + 1j * (β2 / 2) * (ω**2) * (hz / 2))
    ).astype(prec)

    Ech_spans = cp.zeros((Ech.shape[0], len(saveSpanN))).astype(prec)

    indRecSpan = 0

    for spanN in tqdm(range(1, Nspans + 1), disable=not (prgsBar)):
        Ech = fft(Ech)  # single-polarization field

        # fiber propagation step
        for _ in range(1, Nsteps + 1):
            # First linear step (frequency domain)
            Ech = Ech * linOperator

            # Nonlinear step (time domain)
            Ech = ifft(Ech)
            Ech = Ech * cp.exp(1j * γ * (Ech * cp.conj(Ech)) * hz)

            # Second linear step (frequency domain)
            Ech = fft(Ech)
            Ech = Ech * linOperator

        # amplification step
        Ech = ifft(Ech)
        if amp == "edfa":
            Ech = edfa(Ech, paramAmp)
        elif amp == "ideal":
            Ech = Ech * cp.exp(α / 2 * Nsteps * hz)
        elif amp is None:
            Ech = Ech * cp.exp(0)

        if spanN in saveSpanN:
            Ech_spans[:, indRecSpan] = Ech
            indRecSpan += 1

    Ech = cp.asnumpy(Ech_spans)

    if Ech.shape[1] == 1:
        Ech = Ech.reshape(
            len(Ech),
        )

    return (Ech, param) if returnParameters else Ech


def manakovSSF(Ei, param):
    """
    Run the Manakov split-step Fourier model (symmetric, dual-pol.).

    Parameters
    ----------
    Ei : np.array
        Input optical signal field.

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

        - param.prec: numerical precision [default: cp.complex128]

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
    param.prec = getattr(param, "prec", cp.complex128)
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
    amp = param.amp
    NF = param.NF
    Fc = param.Fc
    prec = param.prec
    maxIter = param.maxIter
    tol = param.tol
    prgsBar = param.prgsBar
    saveSpanN = param.saveSpanN
    nlprMethod = param.nlprMethod
    maxNlinPhaseRot = param.maxNlinPhaseRot
    returnParameters = param.returnParameters

    Nspans = int(np.floor(Ltotal / Lspan))

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
    paramAmp.prec = prec

    c_kms = cp.asarray(c_kms, dtype=prec)  # speed of light (vacuum) in km/s
    λ = cp.asarray(λ, dtype=prec)
    α = cp.asarray(α, dtype=prec)
    β2 = cp.asarray(β2, dtype=prec)
    γ = cp.asarray(γ, dtype=prec)
    hz = cp.asarray(hz, dtype=prec)
    Ltotal = cp.asarray(Ltotal, dtype=prec)
    maxNlinPhaseRot = cp.asarray(maxNlinPhaseRot, dtype=prec)

    # generate frequency axis
    Nfft = len(Ei)
    ω = 2 * np.pi * Fs * fftfreq(Nfft).astype(prec)

    Ei_ = cp.asarray(Ei).astype(prec)

    Ech_x = Ei_[:, 0::2].T
    Ech_y = Ei_[:, 1::2].T

    # define static part of the linear operator
    argLimOp = cp.array(-(α / 2) + 1j * (β2 / 2) * (ω**2)).astype(prec)

    if Ech_x.shape[0] > 1:
        argLimOp = cp.tile(argLimOp, (Ech_x.shape[0], 1))
    else:
        argLimOp = argLimOp.reshape(1, -1)

    if saveSpanN:
        Ech_spans = cp.zeros((Ei_.shape[0], Ei_.shape[1] * len(saveSpanN))).astype(prec)
        indRecSpan = 0

    for spanN in tqdm(range(1, Nspans + 1), disable=not (prgsBar)):
        Ex_conv = Ech_x.copy()
        Ey_conv = Ech_y.copy()
        z_current = 0

        # fiber propagation steps
        while z_current < Lspan:
            Pch = Ech_x * cp.conj(Ech_x) + Ech_y * cp.conj(Ech_y)

            phiRot = nlinPhaseRot(Ex_conv, Ey_conv, Pch, γ)

            if nlprMethod:
                hz_ = (
                    maxNlinPhaseRot / cp.max(phiRot)
                    if Lspan - z_current >= maxNlinPhaseRot / cp.max(phiRot)
                    else Lspan - z_current
                )
            elif Lspan - z_current < hz:
                hz_ = Lspan - z_current  # check that the remaining
                # distance is not less than hz (due to non-integer
                # steps/span)
            else:
                hz_ = hz

            # define the linear operator
            linOperator = cp.exp(argLimOp * (hz_ / 2))

            # First linear step (frequency domain)
            Ex_hd = ifft(fft(Ech_x) * linOperator)
            Ey_hd = ifft(fft(Ech_y) * linOperator)

            # Nonlinear step (time domain)
            for nIter in range(maxIter):
                rotOperator = cp.exp(1j * phiRot * hz_)

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
            Ech_x = Ech_x * cp.exp(α / 2 * Lspan)
            Ech_y = Ech_y * cp.exp(α / 2 * Lspan)
        elif amp is None:
            Ech_x = Ech_x * cp.exp(0)
            Ech_y = Ech_y * cp.exp(0)
        if spanN in saveSpanN:
            Ech_spans[:, 2 * indRecSpan : 2 * indRecSpan + 1] = Ech_x.T
            Ech_spans[:, 2 * indRecSpan + 1 : 2 * indRecSpan + 2] = Ech_y.T
            indRecSpan += 1

    if saveSpanN:
        Ech = cp.asnumpy(Ech_spans)
    else:
        Ech_x = cp.asnumpy(Ech_x)
        Ech_y = cp.asnumpy(Ech_y)

        Ech = Ei.copy()
        Ech[:, 0::2] = Ech_x.T
        Ech[:, 1::2] = Ech_y.T

    return (Ech, param) if returnParameters else Ech


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
    return ((8 / 9) * γ * (Pch + Ex * cp.conj(Ex) + Ey * cp.conj(Ey)) / 2).real


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
    return cp.sqrt(norm(Ex_fd - Ex_conv) ** 2 + norm(Ey_fd - Ey_conv) ** 2) / cp.sqrt(
        norm(Ex_conv) ** 2 + norm(Ey_conv) ** 2
    )


def manakovDBP(Ei, param):
    """
    Run the Manakov SSF digital backpropagation (symmetric, dual-pol.).

    Parameters
    ----------
    Ei : np.array
        Input optical signal field.
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

        - param.prec: numerical precision [default: cp.complex128]

        - param.amp: 'edfa', 'ideal', or 'None. [default:'edfa']

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
        Optical signal after nonlinear backward propagation.
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
    param.prec = getattr(param, "prec", cp.complex128)
    param.amp = getattr(param, "amp", "edfa")
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
    amp = param.amp
    Fc = param.Fc
    prec = param.prec
    maxIter = param.maxIter
    tol = param.tol
    prgsBar = param.prgsBar
    saveSpanN = param.saveSpanN
    nlprMethod = param.nlprMethod
    maxNlinPhaseRot = param.maxNlinPhaseRot
    returnParameters = param.returnParameters

    Nspans = int(np.floor(Ltotal / Lspan))

    # channel parameters
    c_kms = const.c / 1e3  # speed of light (vacuum) in km/s
    λ = c_kms / Fc
    α = alpha / (10 * np.log10(np.exp(1)))
    β2 = -(D * λ**2) / (2 * np.pi * c_kms)
    γ = gamma

    c_kms = cp.asarray(c_kms, dtype=prec)  # speed of light (vacuum) in km/s
    λ = cp.asarray(λ, dtype=prec)
    α = cp.asarray(α, dtype=prec)
    β2 = cp.asarray(β2, dtype=prec)
    γ = cp.asarray(γ, dtype=prec)
    hz = cp.asarray(hz, dtype=prec)
    Ltotal = cp.asarray(Ltotal, dtype=prec)
    maxNlinPhaseRot = cp.asarray(maxNlinPhaseRot, dtype=prec)

    # generate frequency axis
    Nfft = len(Ei)
    ω = 2 * np.pi * Fs * fftfreq(Nfft).astype(prec)

    Ei_ = cp.asarray(Ei).astype(prec)

    Ech_x = Ei_[:, 0::2].T
    Ech_y = Ei_[:, 1::2].T

    # define static part of the linear operator
    argLimOp = cp.array((α / 2) - 1j * (β2 / 2) * (ω**2)).astype(prec)

    if Ech_x.shape[0] > 1:
        argLimOp = cp.tile(argLimOp, (Ech_x.shape[0], 1))
    else:
        argLimOp = argLimOp.reshape(1, -1)

    if saveSpanN:
        Ech_spans = cp.zeros((Ei_.shape[0], Ei_.shape[1] * len(saveSpanN))).astype(prec)
        indRecSpan = 0

    for spanN in tqdm(range(1, Nspans + 1), disable=not (prgsBar)):
        # reverse amplification step
        if amp in {"edfa", "ideal"}:
            Ech_x = Ech_x * cp.exp(-α / 2 * Lspan)
            Ech_y = Ech_y * cp.exp(-α / 2 * Lspan)
        elif amp is None:
            Ech_x = Ech_x * cp.exp(0)
            Ech_y = Ech_y * cp.exp(0)

        Ex_conv = Ech_x.copy()
        Ey_conv = Ech_y.copy()
        z_current = 0

        # reverse fiber propagation steps
        while z_current < Lspan:
            Pch = Ech_x * cp.conj(Ech_x) + Ech_y * cp.conj(Ech_y)

            phiRot = nlinPhaseRot(Ex_conv, Ey_conv, Pch, γ)

            if nlprMethod:
                hz_ = (
                    maxNlinPhaseRot / cp.max(phiRot)
                    if Lspan - z_current >= maxNlinPhaseRot / cp.max(phiRot)
                    else Lspan - z_current
                )
            elif Lspan - z_current < hz:
                hz_ = Lspan - z_current  # check that the remaining
                # distance is not less than hz (due to non-integer
                # steps/span)
            else:
                hz_ = hz

            # define the linear operator
            linOperator = cp.exp(argLimOp * (hz_ / 2))

            # First linear step (frequency domain)
            Ex_hd = ifft(fft(Ech_x) * linOperator)
            Ey_hd = ifft(fft(Ech_y) * linOperator)

            # Nonlinear step (time domain)
            for nIter in range(maxIter):
                rotOperator = cp.exp(-1j * phiRot * hz_)

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

        if spanN in saveSpanN:
            Ech_spans[:, 2 * indRecSpan : 2 * indRecSpan + 1] = Ech_x.T
            Ech_spans[:, 2 * indRecSpan + 1 : 2 * indRecSpan + 2] = Ech_y.T
            indRecSpan += 1

    if saveSpanN:
        Ech = cp.asnumpy(Ech_spans)
    else:
        Ech_x = cp.asnumpy(Ech_x)
        Ech_y = cp.asnumpy(Ech_y)

        Ech = Ei.copy()
        Ech[:, 0::2] = Ech_x.T
        Ech[:, 1::2] = Ech_y.T

    return (Ech, param) if returnParameters else Ech


def setPowerforParSSFM(sig, powers):
    powers_lin = 10 ** (powers / 10) * 1e-3
    powers_lin = powers_lin.repeat(2) / 2

    for i in np.arange(0, sig.shape[1], 2):
        for k in range(2):
            sig[:, i + k] = (
                np.sqrt(powers_lin[i] / signal_power(sig[:, i + k])) * sig[:, i + k]
            )
            print(
                "power mode %d: %.2f dBm"
                % (i + k, 10 * np.log10(signal_power(sig[:, i + k]) / 1e-3))
            )
    return sig
