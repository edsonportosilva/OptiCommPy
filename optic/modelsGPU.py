"""Functions from models.py adapted to run with GPU (CuPy) processing."""
import logging as logg

import cupy as cp
import numpy as np
import scipy.constants as const
from cupy.linalg import norm
from cupy.random import normal
from cupyx.scipy.fft import fft, fftfreq, ifft
from tqdm.notebook import tqdm

from optic.metrics import signal_power


def edfa(Ei, Fs, G=20, NF=4.5, Fc=193.1e12, prec=cp.complex128):
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
    noise = cp.array(noise).astype(prec)
    return Ei * cp.sqrt(G_lin) + noise


def ssfm(Ei, Fs, paramCh, prec=cp.complex128):
    """
    Run the split-step Fourier model (symmetric, single-pol.).

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
    paramCh.prgsBar = getattr(paramCh, "prgsBar", True)
    paramCh.saveSpanN = getattr(
        paramCh, "saveSpanN", [paramCh.Ltotal // paramCh.Lspan]
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
    prgsBar = paramCh.prgsBar
    saveSpanN = paramCh.saveSpanN

    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(np.floor(Lspan / hz))

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
            Ech = edfa(Ech, Fs, alpha * Lspan, NF, Fc)
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

    return Ech, paramCh


def manakovSSF(Ei, Fs, paramCh, prec=cp.complex128):
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
        paramCh, "saveSpanN", [paramCh.Ltotal // paramCh.Lspan]
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
    argLimOp = cp.array(-(α / 2) + 1j * (β2 / 2) * (ω**2)).astype(prec)

    if Ech_x.shape[0] > 1:
        argLimOp = cp.tile(argLimOp, (Ech_x.shape[0], 1))
    else:
        argLimOp = argLimOp.reshape(1, -1)

    if saveSpanN:
        Ech_spans = cp.zeros(
            (Ei_.shape[0], Ei_.shape[1] * len(saveSpanN))
        ).astype(prec)
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
            Ech_x = Ech_x * cp.exp(α / 2 * Lspan)
            Ech_y = Ech_y * cp.exp(α / 2 * Lspan)
        elif amp is None:
            Ech_x = Ech_x * cp.exp(0)
            Ech_y = Ech_y * cp.exp(0)
        if spanN in saveSpanN:
            Ech_spans[:, 2 * indRecSpan: 2 * indRecSpan + 1] = Ech_x.T
            Ech_spans[:, 2 * indRecSpan + 1: 2 * indRecSpan + 2] = Ech_y.T
            indRecSpan += 1

    if saveSpanN:
        Ech = cp.asnumpy(Ech_spans)
    else:
        Ech_x = cp.asnumpy(Ech_x)
        Ech_y = cp.asnumpy(Ech_y)

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
    return cp.sqrt(
        norm(Ex_fd - Ex_conv) ** 2 + norm(Ey_fd - Ey_conv) ** 2
    ) / cp.sqrt(norm(Ex_conv) ** 2 + norm(Ey_conv) ** 2)


def manakovDBP(Ei, Fs, paramCh, prec=cp.complex128):
    """
    Run the Manakov SSF digital backpropagation (symmetric, dual-pol.).

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
        Optical signal after nonlinear backward propagation.
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
    paramCh.maxIter = getattr(paramCh, "maxIter", 10)
    paramCh.tol = getattr(paramCh, "tol", 1e-5)
    paramCh.nlprMethod = getattr(paramCh, "nlprMethod", True)
    paramCh.maxNlinPhaseRot = getattr(paramCh, "maxNlinPhaseRot", 2e-2)
    paramCh.prgsBar = getattr(paramCh, "prgsBar", True)
    paramCh.saveSpanN = getattr(
        paramCh, "saveSpanN", [paramCh.Ltotal // paramCh.Lspan]
    )

    Ltotal = paramCh.Ltotal
    Lspan = paramCh.Lspan
    hz = paramCh.hz
    alpha = paramCh.alpha
    D = paramCh.D
    gamma = paramCh.gamma
    Fc = paramCh.Fc
    amp = paramCh.amp
    maxIter = paramCh.maxIter
    tol = paramCh.tol
    prgsBar = paramCh.prgsBar
    saveSpanN = paramCh.saveSpanN
    nlprMethod = paramCh.nlprMethod
    maxNlinPhaseRot = paramCh.maxNlinPhaseRot

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
        Ech_spans = cp.zeros(
            (Ei_.shape[0], Ei_.shape[1] * len(saveSpanN))
        ).astype(prec)
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

        if spanN in saveSpanN:
            Ech_spans[:, 2 * indRecSpan: 2 * indRecSpan + 1] = Ech_x.T
            Ech_spans[:, 2 * indRecSpan + 1: 2 * indRecSpan + 2] = Ech_y.T
            indRecSpan += 1

    if saveSpanN:
        Ech = cp.asnumpy(Ech_spans)
    else:
        Ech_x = cp.asnumpy(Ech_x)
        Ech_y = cp.asnumpy(Ech_y)

        Ech = Ei.copy()
        Ech[:, 0::2] = Ech_x.T
        Ech[:, 1::2] = Ech_y.T

    return Ech, paramCh


def setPowerforParSSFM(sig, powers):

    powers_lin = 10 ** (powers / 10) * 1e-3
    powers_lin = powers_lin.repeat(2) / 2

    for i in np.arange(0, sig.shape[1], 2):
        for k in range(2):
            sig[:, i + k] = (
                np.sqrt(powers_lin[i] / signal_power(sig[:, i + k]))
                * sig[:, i + k]
            )
            print(
                "power mode %d: %.2f dBm"
                % (i + k, 10 * np.log10(signal_power(sig[:, i + k]) / 1e-3))
            )
    return sig
