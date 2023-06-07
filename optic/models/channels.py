"""Basic physical models for optical channels."""
import logging as logg

import numpy as np
import scipy.constants as const
from scipy.linalg import norm
from numba import njit
from numpy.fft import fft, fftfreq, ifft
from numpy.random import normal
from tqdm.notebook import tqdm

from optic.dsp.core import lowPassFIR, signal_power, sigPow
from optic.models.devices import edfa

try:
    from optic.dsp.coreGPU import firFilter
except ImportError:
    from optic.dsp.core import firFilter


def linFiberCh(Ei, L, alpha, D, Fc, Fs):
    """
    Simulate signal propagation through a linear fiber channel.

    Parameters
    ----------
    Ei : np.array
        Input optical field.
    L : real scalar
        Length of the fiber.
    alpha : real scalar
        Fiber's attenuation coefficient in dB/km.
    D : real scalar
        Fiber's chromatic dispersion (2nd order) coefficient in ps/nm/km.
    Fc : real scalar
        Optical carrier frequency in Hz.
    Fs : real scalar
        Sampling rate of the simulation.

    Returns
    -------
    Eo : np.array
        Optical field at the output of the fiber.

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


def manakovSSF(Ei, Fs, paramCh, prec=np.complex128):
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
    paramCh.saveSpanN: specify the span indexes to be outputted [default:[]]

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

    Ech_x = Ei[:, 0::2].T
    Ech_y = Ei[:, 1::2].T

    # define static part of the linear operator
    argLimOp = np.array(-(α / 2) + 1j * (β2 / 2) * (ω**2)).astype(prec)

    if Ech_x.shape[0] > 1:
        argLimOp = np.tile(argLimOp, (Ech_x.shape[0], 1))
    else:
        argLimOp = argLimOp.reshape(1, -1)

    if saveSpanN:
        Ech_spans = np.zeros(
            (Ei.shape[0], Ei.shape[1] * len(saveSpanN))
        ).astype(prec)
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
            Ech_x = Ech_x * np.exp(α / 2 * Lspan)
            Ech_y = Ech_y * np.exp(α / 2 * Lspan)
        elif amp is None:
            Ech_x = Ech_x * np.exp(0)
            Ech_y = Ech_y * np.exp(0)

        if spanN in saveSpanN:
            Ech_spans[:, 2 * indRecSpan: 2 * indRecSpan + 1] = Ech_x.T
            Ech_spans[:, 2 * indRecSpan + 1: 2 * indRecSpan + 2] = Ech_y.T
            indRecSpan += 1

    if saveSpanN:
        Ech = Ech_spans
    else:
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
    return np.sqrt(
        norm(Ex_fd - Ex_conv) ** 2 + norm(Ey_fd - Ey_conv) ** 2
    ) / np.sqrt(norm(Ex_conv) ** 2 + norm(Ey_conv) ** 2)


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
        Input signal.
    snr : scalar
        Signal-to-noise ratio in dB.
    Fs : real scalar
        Sampling frequency. The default is 1.
    B : real scalar
        Signal bandwidth. The default is 1.

    Returns
    -------
    np.array
        Input signal plus noise.

    """
    snr_lin = 10 ** (snr / 10)
    noiseVar = sigPow(sig) / snr_lin
    σ = np.sqrt((Fs / B) * noiseVar)
    noise = normal(0, σ, sig.shape) + 1j * normal(0, σ, sig.shape)
    noise = 1 / np.sqrt(2) * noise

    return sig + noise
