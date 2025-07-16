"""
======================================================
Models perturbation (:mod:`optic.models.perturbation`)
======================================================

.. autosummary::
   :toctree: generated/

   calcPertCoeffMatrix                 -- Calculates the perturbation coefficients for intra-channel fiber nonlinear interference.
   calcNLINperturbation                -- Calculates the first-order perturbation AM NLIN model for dual-polarization signals.
   calcNLINperturbationSimplified      -- Calculates the first-order perturbation AM NLIN model with reduced number of coefficients.
   perturbationNLIN                    -- Main function to calculate perturbation NLIN for dual-polarization signals.

"""

"""Perturbation models for NLIN calculation."""
import numpy as np
from scipy.special import gammaincc, comb
from scipy.integrate import quad
from scipy.constants import c as c_light
from scipy.special import exp1
from numba import njit, prange
from optic.dsp.core import pnorm
from optic.utils import dBm2W, dotNumba
import logging


def calcPertCoeffMatrix(param):
    """
    Calculates the coefficient matrix for the first-order perturbation model of intra-channel fiber nonlinear interference.

    Parameters
    ----------
    param : parameter object  (struct)
        Object with physical/simulation parameters of the optical channel.

        - param.D : chromatic dispersion parameter [ps/nm/km] [default: 17]

        - param.alpha : fiber attenuation parameter [dB/km] [default: None]

        - param.lspan : span length [km] [default: None]

        - param.length : total fiber length [km] [default: None]

        - param.pulseWidth : pulse width (fraction of symbol period) [default: 0.5]

        - param.gamma : fiber nonlinear coefficient [1/W/km] [default: None]

        - param.Fc : carrier frequency [THz] [default: None]

        - param.powerWeighted : power-weighted coefficient calculation? Boolean variable [default: False]

        - param.Rs : symbol rate [baud] [default: None]

        - param.powerWeightN : power-weighting order [default: 10]

        - param.matrixOrder : nonlinear memory matrix order [default: 25]

    Returns
    -------
    C : ndarray of shape (2L+1, 2L+1)
        Matrix of perturbation coefficients for nonlinear impairments.
    C_ifwm : ndarray of shape (2L+1, 2L+1)
        Nonlinear coefficient matrix for intrachannel four-wave mixing (IFWM).
    C_ixpm : ndarray of shape (2L+1, 2L+1)
        Nonlinear coefficient matrix for intrachannel cross-phase modulation (IXPM).
    C_ispm : float
        Scalar nonlinear coefficient for intrachannel self-phase modulation (SPM).

    References
    ----------
    [1] Z. Tao, et al., "Analytical Intrachannel Nonlinear Models to Predict the Nonlinear Noise Waveform," Journal of Lightwave Technology, vol. 33, no. 10, pp. 2111-2119, 2015.
    [2] E. P. da Silva, et al., "Perturbation-Based FEC-Assisted Iterative Nonlinearity Compensation for WDM Systems," Journal of Lightwave Technology, vol. 37, no. 3, pp. 875-881, 2019.

    """
    D = getattr(param, "D", 17)  # Dispersion parameter (ps/nm/km)
    alpha = getattr(param, "alpha", 0.2)  # Attenuation (dB/km)
    lspan = getattr(param, "lspan", 50)  # Span length (km)
    length = getattr(param, "length", 800)  # Total length (km)
    pulseWidth = getattr(
        param, "pulseWidth", 0.5
    )  # Pulse width (fraction of symbol period)
    gamma = getattr(param, "gamma", 1.3)  # Nonlinear coefficient (1/W/km)
    Fc = getattr(param, "Fc", 193.2e12)  # Carrier frequency (Hz)
    powerWeighted = getattr(
        param, "powerWeighted", False
    )  # Power-weighted calculation (bool)
    Rs = getattr(param, "Rs", 32e9)  # Symbol rate (baud)
    powerWeightN = getattr(param, "powerWeightN", 10)  # Power-weighted order (int)
    matrixOrder = getattr(param, "matrixOrder", 25)  # Matrix order (int)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()
    c_kms = c_light / 1e3
    # signal parameters
    symbolPeriod = 1 / Rs  # Symbol period (s)
    pulseWidth = pulseWidth * symbolPeriod  # Pulse width (s)

    # Link parameters
    λ = c_kms / Fc
    alpha = alpha / (10 * np.log10(np.e))
    beta2 = -D * λ**2 / (2 * np.pi * c_kms)
    Leff = (1 - np.exp(-alpha * lspan)) / alpha
    nSpans = int(length / lspan)

    # Matrix indices
    m_vals = np.arange(-matrixOrder, matrixOrder + 1)
    M, N = np.meshgrid(m_vals, m_vals[::-1])

    # Calculate C_ispm
    constantIntegral = pulseWidth**4 / (3 * beta2**2)
    fun1 = lambda z, c: 1.0 / np.sqrt(c + z**2)
    C_ispm, _ = quad(lambda z: fun1(z, constantIntegral), 0, length)

    # Calculate C_ifwm
    if powerWeighted:
        Acoeff = M * N * symbolPeriod**2 / beta2
        sum1 = np.zeros_like(M, dtype=complex)
        Norder = powerWeightN

        log.info("Calculating matrix of perturbation coefficients (power-weighted)...")
        for indSpan in range(1, nSpans + 1):
            Bcoeff = -Norder / (alpha * Acoeff) + ((indSpan - 1) * lspan) / Acoeff

            sum2 = np.zeros_like(M, dtype=complex)
            for kk in range(1, Norder + 1):
                if indSpan != 1:
                    GammaPrevious = gammaincc(
                        1 - kk, 1j * (1 / Bcoeff - Acoeff / ((indSpan - 1) * lspan))
                    )
                else:
                    GammaPrevious = np.zeros_like(M, dtype=complex)
                GammaNext = gammaincc(
                    1 - kk, 1j * (1 / Bcoeff - Acoeff / (indSpan * lspan))
                )

                term = (
                    (-1) ** (kk + Norder)
                    * comb(Norder - 1, kk - 1)
                    * (1j / Bcoeff) ** kk
                    * (GammaPrevious - GammaNext)
                )

                if kk == 1:
                    sum2 = term
                else:
                    sum2 += term

            if indSpan == 1:
                sum1 = (np.exp(1j / Bcoeff) / Bcoeff ** (Norder - 1)) * sum2
            else:
                sum1 += (np.exp(1j / Bcoeff) / Bcoeff ** (Norder - 1)) * sum2

        C_ifwm = (Norder / alpha) ** Norder * (Acoeff**-Norder) * sum1
    else:
        log.info("Calculating matrix of perturbation coefficients (standard)...")
        C_ifwm = exp1(-1j * M * N * symbolPeriod**2 / (beta2 * length))

    # Calculate C_ixpm
    C_ixpm = 0.5 * exp1(
        (N - M) ** 2
        * symbolPeriod**2
        * pulseWidth**2
        / (3 * np.abs(beta2) ** 2 * length**2)
    )

    # Handle inf and nan values
    if powerWeighted:
        C_ifwm_mask = np.isnan(np.abs(C_ifwm)).astype(float)
        C_ifwm[np.isnan(np.abs(C_ifwm))] = 0
    else:
        C_ifwm_mask = np.isinf(np.abs(C_ifwm)).astype(float)
        C_ifwm[np.isinf(np.abs(C_ifwm))] = 0

    C_ixpm[np.isinf(np.abs(C_ixpm))] = 0
    C_ixpm = C_ifwm_mask * C_ixpm

    # Scale the matrices
    scale_factor = (
        1j
        * (8 / 9)
        * gamma
        * pulseWidth**2
        / (np.sqrt(3) * np.abs(beta2))
        * Leff
        / lspan
    )
    if powerWeighted:
        C_ifwm = -(8 / 9) * gamma * pulseWidth**2 / (np.sqrt(3) * beta2) * C_ifwm
        C_ixpm = scale_factor * C_ixpm
        C_ispm = scale_factor * C_ispm
    else:
        C_ifwm = scale_factor * C_ifwm
        C_ixpm = scale_factor * C_ixpm
        C_ispm = scale_factor * C_ispm

    # Combine results
    C = C_ifwm + C_ixpm
    C[matrixOrder, matrixOrder] = C_ispm

    log.info(
        "Matrix of perturbation coefficients calculated. Dimensions: %d x %d",
        2 * matrixOrder + 1,
        2 * matrixOrder + 1,
    )

    return C, C_ifwm, C_ixpm, C_ispm


@njit(parallel=True, fastmath=True)
def calcNLINperturbation(C_ifwm, C_ixpm, C_ispm, x, y, prec=np.complex64):
    """
    Calculates the perturbation-based additive and multiplicative NLIN for dual-polarization signals.

    Parameters
    ----------
    C_ifwm : ndarray of shape (2L+1, 2L+1)
        Nonlinear coefficient matrix for intrachannel four-wave mixing (IFWM).

    C_ixpm : ndarray of shape (2L+1, 2L+1)
        Nonlinear coefficient matrix for intrachannel cross-phase modulation (IXPM).

    C_ispm : float
        Scalar nonlinear coefficient for intrachannel self-phase modulation (SPM).

    x : ndarray of shape (N,)
        Input signal for polarization X (complex-valued).

    y : ndarray of shape (N,)
        Input signal for polarization Y (complex-valued).

    prec : data-type, optional
        Precision of the computation (`np.complex64` or `np.complex128`), by default `np.complex64`.

    Returns
    -------
    dx : ndarray of shape (N,)
        Nonlinear perturbation waveform for polarization X.

    dy : ndarray of shape (N,)
        Nonlinear perturbation waveform for polarization Y.

    phi_ixpm_x : ndarray of shape (N,)
        Phase rotation due to cross-phase modulation affecting polarization X.

    phi_ixpm_y : ndarray of shape (N,)
        Phase rotation due to cross-phase modulation affecting polarization Y.

    References
    ----------
    [1] Z. Tao, et al., "Analytical Intrachannel Nonlinear Models to Predict the Nonlinear Noise Waveform," Journal of Lightwave Technology, vol. 33, no. 10, pp. 2111-2119, 2015.
    [2] E. P. da Silva, et al., "Perturbation-Based FEC-Assisted Iterative Nonlinearity Compensation for WDM Systems," Journal of Lightwave Technology, vol. 37, no. 3, pp. 875-881, 2019.
    """
    L = (C_ifwm.shape[0] - 1) // 2
    D = C_ifwm.shape[0] - 1

    C = C_ifwm + C_ixpm
    C[D, D] = C_ispm
    C_m_non_equal_zero = C.copy()
    C_m_non_equal_zero[:, L] = np.inf

    mask_nonzero_T = np.isinf(C_m_non_equal_zero.T)
    mask_nonzero = np.isinf(C_m_non_equal_zero)

    C_ixpm_mask1 = (C_ixpm * mask_nonzero_T).flatten()
    C_ixpm_mask2 = (C_ixpm * mask_nonzero).flatten()
    C_ifwm = C_ifwm.flatten()

    # Normalize power
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    y = y / np.sqrt(np.mean(np.abs(y) ** 2))

    # Outputs
    Nsymb = len(x)
    dx = np.zeros(Nsymb, dtype=prec)
    dy = np.zeros(Nsymb, dtype=prec)
    phi_ixpm_x = np.zeros(Nsymb)
    phi_ixpm_y = np.zeros(Nsymb)

    # Prepad input
    symbX = np.zeros(Nsymb + 2 * D, dtype=prec)
    symbY = np.zeros(Nsymb + 2 * D, dtype=prec)
    symbX[D:-D] = x
    symbY[D:-D] = y

    # Precompute indexes
    indL = 2 * L + 1
    M = np.zeros((indL, indL), dtype=np.int64)
    for i in range(indL):
        M[i, :] = np.arange(indL)
    NplusM = -(M.T - L + M - L) + 2 * L
    NplusM = NplusM[:, ::-1]  # rotate and flip on axis 0

    for t in prange(D, len(symbX) - D):
        # Pre-allocate 2D arrays
        Xm = np.empty((indL, indL), dtype=prec)
        Ym = np.empty((indL, indL), dtype=prec)
        Xn = np.empty((indL, indL), dtype=prec)
        Yn = np.empty((indL, indL), dtype=prec)
        X_NplusM = np.empty((indL, indL), dtype=prec)
        Y_NplusM = np.empty((indL, indL), dtype=prec)

        windowX = symbX[t - D : t + D + 1]
        windowY = symbY[t - D : t + D + 1]

        X_center = windowX[L : L + indL]
        Y_center = windowY[L : L + indL]

        for i in range(indL):
            for j in range(indL):
                Xm[i, j] = X_center[j]
                Ym[i, j] = Y_center[j]
                Xn[i, j] = X_center[2 * L - i]  # flipud
                Yn[i, j] = Y_center[2 * L - i]  # flipud

        for i in range(indL):
            for j in range(indL):
                X_NplusM[i, j] = windowX[NplusM[i, j]]
                Y_NplusM[i, j] = windowY[NplusM[i, j]]

        Xm_flat = Xm.ravel()
        Ym_flat = Ym.ravel()
        Xn_flat = Xn.ravel()
        Yn_flat = Yn.ravel()
        X_NplusM_flat = X_NplusM.ravel()
        Y_NplusM_flat = Y_NplusM.ravel()

        A1 = np.abs(Xm_flat) ** 2
        A2 = np.abs(Ym_flat) ** 2
        M1 = Xn_flat * np.conj(X_NplusM_flat)
        M2 = Yn_flat * np.conj(Y_NplusM_flat)

        DX = (M1 + M2) * Xm_flat
        DY = (M2 + M1) * Ym_flat

        phi_ixpm_x[t - D] = np.imag(
            dotNumba(2 * A1 + A2, C_ixpm_mask1)
            + (np.abs(Xm_flat[0]) ** 2 + np.abs(Ym_flat[0]) ** 2) * C_ispm
        )
        phi_ixpm_y[t - D] = np.imag(
            dotNumba(2 * A2 + A1, C_ixpm_mask1)
            + (np.abs(Ym_flat[0]) ** 2 + np.abs(Xm_flat[0]) ** 2) * C_ispm
        )

        dx[t - D] = dotNumba(DX, C_ifwm) + dotNumba(M2 * Xm_flat, C_ixpm_mask2)
        dy[t - D] = dotNumba(DY, C_ifwm) + dotNumba(M1 * Ym_flat, C_ixpm_mask2)

    return dx, dy, phi_ixpm_x, phi_ixpm_y


@njit(parallel=True, fastmath=True)
def calcNLINperturbationSimplified(
    C_ifwm, C_ixpm, C_ispm, x, y, coeffTol=-20, prec=np.complex64
):
    """
    Calculates the perturbation-based additive and multiplicative NLIN with reduced
    number of coefficients.

    Parameters
    ----------
    C_ifwm : ndarray of shape (M,)
        Coefficient matrix for the Inverse Fourier-weighted filter model.

    C_ixpm : ndarray of shape (M,)
        Coefficient matrix for the Inverse XPM model.

    C_ispm : scalar
        Coefficient for the Inverse Single-Phase Modulation model.

    x : ndarray of shape (N,)
        Input signal for the X component (complex-valued).

    y : ndarray of shape (N,)
        Input signal for the Y component (complex-valued).

    coeffTol : float
        Coefficient magnitude tolerance in dB. Coefficients with a magnitude
        below this threshold (in dB) are excluded from the calculation to reduce
        computational complexity. Default is -20 dB.

    prec : dtype, optional
        The precision of the computation. Default is `np.complex64`.

    Returns
    -------
    dx : ndarray of shape (N,)
        The computed result for the X component after processing.

    dy : ndarray of shape (N,)
        The computed result for the Y component after processing.

    phi_ixpm_x : ndarray of shape (N,)
        Phase information related to the XPM effect on the X component.

    phi_ixpm_y : ndarray of shape (N,)
        Phase information related to the XPM effect on the Y component.

    References
    ----------
    [1] Z. Tao, et al., "Analytical Intrachannel Nonlinear Models to Predict the Nonlinear Noise Waveform," Journal of Lightwave Technology, vol. 33, no. 10, pp. 2111-2119, 2015.
    [2] E. P. da Silva, et al., "Perturbation-Based FEC-Assisted Iterative Nonlinearity Compensation for WDM Systems," Journal of Lightwave Technology, vol. 37, no. 3, pp. 875-881, 2019.
    """
    # Definitions
    L = (len(C_ifwm) - 1) // 2
    D = len(C_ifwm) - 1

    C = C_ifwm + C_ixpm
    C[D, D] = C_ispm
    C_m_non_equal_zero = C.copy()
    C_m_non_equal_zero[:, L] = np.inf
    C_ixpm_mask1 = C_ixpm * np.isinf(C_m_non_equal_zero.T)
    C_ixpm_mask2 = C_ixpm * np.isinf(C_m_non_equal_zero)

    # Normalize power
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    y = y / np.sqrt(np.mean(np.abs(y) ** 2))

    # Pre-allocations
    Nsymb = len(x)
    dx = np.zeros(Nsymb, dtype=prec)
    dy = np.zeros(Nsymb, dtype=prec)
    phi_ixpm_x = np.zeros(Nsymb)
    phi_ixpm_y = np.zeros(Nsymb)

    symbX = np.zeros(Nsymb + 2 * D, dtype=prec)
    symbY = np.zeros(Nsymb + 2 * D, dtype=prec)
    symbX[D:-D] = x
    symbY[D:-D] = y

    # Indexing matrices
    indL = 2 * L + 1
    M = np.zeros((indL, indL), dtype=np.int64)
    for i in range(indL):
        M[i, :] = np.arange(indL)
    NplusM = -(M.T - L + M - L) + 2 * L
    NplusM = NplusM[:, ::-1]  # rotate and flip on axis 0

    # Flatten
    C_ixpm_mask1 = C_ixpm_mask1.flatten()
    C_ixpm_mask2 = C_ixpm_mask2.flatten()
    C_ifwm = C_ifwm.flatten()
    C = C.flatten()

    # Coefficient reduction
    absC = np.abs(C)
    maxC = np.max(absC)
    nReducedCoeffs = np.sum(20 * np.log10(absC / maxC) > coeffTol)
    ind_sort = np.argsort(-absC)[:nReducedCoeffs]

    C_ixpm_mask1 = C_ixpm_mask1[ind_sort]
    C_ixpm_mask2 = C_ixpm_mask2[ind_sort]
    C_ifwm = C_ifwm[ind_sort]

    reductionFactor = np.round(100 * (1 - nReducedCoeffs / len(C)), 2)

    for t in prange(D, len(symbX) - D):
        # Pre-allocate 2D arrays
        Xm = np.zeros((indL, indL), dtype=prec)
        Ym = np.zeros((indL, indL), dtype=prec)
        Xn = np.zeros((indL, indL), dtype=prec)
        Yn = np.zeros((indL, indL), dtype=prec)
        X_NplusM = np.empty((indL, indL), dtype=prec)
        Y_NplusM = np.empty((indL, indL), dtype=prec)

        windowX = symbX[t - D : t + D + 1]
        windowY = symbY[t - D : t + D + 1]

        X_center = windowX[L : L + indL]
        Y_center = windowY[L : L + indL]

        for i in range(indL):
            for j in range(indL):
                Xm[i, j] = X_center[j]
                Ym[i, j] = Y_center[j]
                Xn[i, j] = X_center[2 * L - i]  # flipud
                Yn[i, j] = Y_center[2 * L - i]  # flipud

        for i in range(indL):
            for j in range(indL):
                X_NplusM[i, j] = windowX[NplusM[i, j]]
                Y_NplusM[i, j] = windowY[NplusM[i, j]]

        # Flatten and select only significant terms
        Xm_flat = Xm.ravel()[ind_sort]
        Ym_flat = Ym.ravel()[ind_sort]
        Xn_flat = Xn.ravel()[ind_sort]
        Yn_flat = Yn.ravel()[ind_sort]
        X_NplusM_flat = X_NplusM.ravel()[ind_sort]
        Y_NplusM_flat = Y_NplusM.ravel()[ind_sort]

        A1 = np.abs(Xm_flat) ** 2
        A2 = np.abs(Ym_flat) ** 2
        M1 = Xn_flat * np.conj(X_NplusM_flat)
        M2 = Yn_flat * np.conj(Y_NplusM_flat)

        DX = (M1 + M2) * Xm_flat
        DY = (M2 + M1) * Ym_flat

        phi_ixpm_x[t - D] = np.imag(
            dotNumba(2 * A1 + A2, C_ixpm_mask1)
            + (np.abs(Xm_flat[0]) ** 2 + np.abs(Ym_flat[0]) ** 2) * C_ispm
        )
        phi_ixpm_y[t - D] = np.imag(
            dotNumba(2 * A2 + A1, C_ixpm_mask1)
            + (np.abs(Ym_flat[0]) ** 2 + np.abs(Xm_flat[0]) ** 2) * C_ispm
        )

        dx[t - D] = dotNumba(DX, C_ifwm) + dotNumba(M2 * Xm_flat, C_ixpm_mask2)
        dy[t - D] = dotNumba(DY, C_ifwm) + dotNumba(M1 * Ym_flat, C_ixpm_mask2)

    return dx, dy, phi_ixpm_x, phi_ixpm_y, nReducedCoeffs, reductionFactor


def perturbationNLIN(Ein, param):
    """
    Calculates the perturbation-based additive and multiplicative NLIN for dual-polarization signals.

    Parameters
    ----------
    Ein : ndarray of shape (N, 2)
        Input signal for dual-polarization (complex-valued).
        The first column represents the X polarization, and the second column represents the Y polarization.

    param : parameter object  (struct)
        Object with physical/simulation parameters of the optical channel.

        - param.D : chromatic dispersion parameter [ps/nm/km] [default: 17]

        - param.alpha : fiber attenuation parameter [dB/km] [default: None]

        - param.lspan : span length [km] [default: None]

        - param.length : total fiber length [km] [default: None]

        - param.pulseWidth : pulse width (fraction of symbol period) [default: 0.5]

        - param.gamma : fiber nonlinear coefficient [1/W/km] [default: None]

        - param.Fc : carrier frequency [THz] [default: None]

        - param.powerWeighted : power-weighted coefficient calculation? Boolean variable [default: False]

        - param.Rs : symbol rate [baud] [default: None]

        - param.powerWeightN : power-weighting order [default: 10]

        - param.matrixOrder : nonlinear memory matrix order [default: 25]

        - mode : 'AM' for standard perturbation calculation or 'AMR' for reduced complexity calculation [default: 'AM']

    Returns
    -------
    nlin : ndarray of shape (N, 2)
        Nonlinear perturbation for dual-polarization signals.
        The first column represents the X polarization, and the second column represents the Y polarization.

    References
    ----------
    [1] Z. Tao, et al., "Analytical Intrachannel Nonlinear Models to Predict the Nonlinear Noise Waveform," Journal of Lightwave Technology, vol. 33, no. 10, pp. 2111-2119, 2015.
    [2] E. P. da Silva, et al., "Perturbation-Based FEC-Assisted Iterative Nonlinearity Compensation for WDM Systems," Journal of Lightwave Technology, vol. 37, no. 3, pp. 875-881, 2019.
    """
    param.D = getattr(param, "D", 17)  # Dispersion parameter (ps/nm/km)
    param.alpha = getattr(param, "alpha", 0.2)  # Attenuation (dB/km)
    param.lspan = getattr(param, "lspan", 50)  # Span length (km)
    param.length = getattr(param, "length", 800)  # Total length (km)
    param.pulseWidth = getattr(
        param, "pulseWidth", 0.5
    )  # Pulse width (fraction of symbol period)
    param.gamma = getattr(param, "gamma", 1.3)  # Nonlinear coefficient (1/W/km)
    param.Fc = getattr(param, "Fc", 193.1e12)  # Carrier frequency (Hz)
    param.powerWeighted = getattr(
        param, "powerWeighted", False
    )  # Power-weighted calculation (bool)
    param.Rs = getattr(param, "Rs", 32e9)  # Symbol rate (baud)
    param.powerWeightN = getattr(
        param, "powerWeightN", 10
    )  # Power-weighted order (int)
    param.matrixOrder = getattr(param, "matrixOrder", 25)  # Matrix order (int)
    mode = getattr(param, "mode", "AM")  # Dispersion parameter (ps/nm/km)
    prec = getattr(
        param, "prec", np.complex128
    )  # Precision of the computation (complex64 or complex128)

    coeffTol = getattr(param, "coeffTol", -20)
    Pin = getattr(param, "Pin", 0)  # Power (dBm)

    Plaunch = dBm2W(Pin)  # Launch power (W)
    PeakPower = 0.5 * Plaunch  # Peak power (W)

    Ein[:, 0] = pnorm(Ein[:, 0])
    Ein[:, 1] = pnorm(Ein[:, 1])

    # Calculate the perturbation coefficients matrix
    C, C_ifwm, C_ixpm, C_ispm = calcPertCoeffMatrix(param)

    nlin = np.zeros((len(Ein), 2), dtype=Ein.dtype)
    if mode == "AM":
        # Calculate the perturbation-based additive and multiplicative NLIN
        dx, dy, phi_ixpm_x, phi_ixpm_y = calcNLINperturbation(
            C_ifwm, C_ixpm, C_ispm, Ein[:, 0], Ein[:, 1], prec
        )
    elif mode == "AMR":
        # Calculate the perturbation-based additive and multiplicative NLIN with reduced complexity
        dx, dy, phi_ixpm_x, phi_ixpm_y, nReducedCoeffs, reductionFactor = (
            calcNLINperturbationSimplified(
                C_ifwm, C_ixpm, C_ispm, Ein[:, 0], Ein[:, 1], coeffTol, prec
            )
        )
        logging.info(
            f"Reduced complexity perturbation calculation: {nReducedCoeffs} coefficients used, reduction factor: {reductionFactor:.2f}%"
        )

    # Scale the perturbation results according to the peak power
    deltaX = PeakPower ** (3 / 2) * dx
    deltaY = PeakPower ** (3 / 2) * dy
    phiX = PeakPower * phi_ixpm_x
    phiY = PeakPower * phi_ixpm_y

    # Calculate the nonlinear perturbation for each polarization
    nlin[:, 0] = np.sqrt(PeakPower) * Ein[:, 0] * (
        np.exp(1j * phiX) - 1
    ) + deltaX * np.exp(1j * phiX)
    nlin[:, 1] = np.sqrt(PeakPower) * Ein[:, 1] * (
        np.exp(1j * phiY) - 1
    ) + deltaY * np.exp(1j * phiY)

    return nlin
