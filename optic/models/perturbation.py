"""
=======================================================================================
Perturbation models for fiber nonlinear interference (:mod:`optic.models.perturbation`)
=======================================================================================

.. autosummary::
   :toctree: generated/

   calcPertCoeffMatrix                 -- Calculates the coefficients for the intrachannel nonlinear first-order perturbation model.
   calcNLINperturbation                -- Fast calculation of the first-order perturbation model.
   calcNLINperturbationSimplified      -- Fast calculation of the first-order perturbation model with reduced number of coefficients.
   perturbationNLIN                    -- Main function to calculate intrachannel NLIN via first-order perturbation models.

"""

"""Perturbation models for NLIN calculation."""
import logging

import numpy as np
from numba import njit, prange
from scipy.constants import c as c_light
from scipy.integrate import quad
from scipy.special import comb, exp1, gammaincc

from optic.dsp.core import pnorm
from optic.utils import dBm2W


def calcPertCoeffMatrix(param):
    """
    Calculates the coefficients for the intrachannel nonlinear first-order perturbation model.

    Parameters
    ----------
    param : optic.utils.parameters object
        Object with physical/simulation parameters of the optical channel.

        - param.D : chromatic dispersion parameter [ps/nm/km] [default: 17 ps/nm/km]
        - param.alpha : fiber attenuation parameter [dB/km] [default: 0.2 dB/km]
        - param.lspan : span length [km] [default: 50 km]
        - param.length : total fiber length [km] [default: 800 km]
        - param.pulseWidth : pulse width (fraction of symbol period) [default: 0.5]
        - param.gamma : fiber nonlinear coefficient [1/W/km] [default: 1.3 1/W/km]
        - param.Fc : carrier frequency [THz] [default: 193.2e12 Hz]
        - param.powerWeighted : power-weighted coefficient calculation? Boolean variable [default: False]
        - param.Rs : symbol rate [baud] [default: 32e9 baud]
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


def _pairTerms(Cf, ii, jj, L):
    """
    Split the FWM coefficients at the positions (ii, jj) into symmetric and ordered terms.

    The coefficient of the four-wave mixing term :math:`x_{t+m}x_{t+n}x^*_{t+m+n}`
    is stored at the row :math:`i = L - n` and column :math:`j = m + L`. When the
    coefficients of :math:`(m, n)` and :math:`(n, m)` are equal (which is the case of
    the complete model, where they depend on the product :math:`mn`), the two terms
    are evaluated together, with half of the memory accesses.

    Returns
    -------
    sym : tuple of arrays
        (m, n, coefficient) of the symmetric terms, with :math:`m \\le n` (the
        coefficient of the diagonal terms is halved, since they are counted twice).
    ordered : tuple of arrays
        (m, n, coefficient) of the terms whose partner is absent or different.
    """
    n_ = 2 * L + 1
    W = np.zeros((n_, n_), dtype=Cf.dtype)
    W[ii, jj] = Cf[ii, jj]

    partner = W[::-1, ::-1].T  # coefficient of (n, m) at the position of (m, n)
    paired = (W != 0) & (W == partner)
    ordered = (W != 0) & ~paired

    iGrid, jGrid = np.indices((n_, n_))
    canonical = paired & (iGrid + jGrid <= 2 * L)  # m <= n

    def terms(mask, weight=None):
        i, j = np.nonzero(mask)
        coef = W[i, j] if weight is None else W[i, j] * weight[i, j]
        return j - L, L - i, coef

    weight = np.where(iGrid + jGrid == 2 * L, 0.5, 1.0)  # diagonal: m == n
    return terms(canonical, weight), terms(ordered)


@njit(parallel=True, fastmath=True, cache=True)
def _fwmKernel(xr, xi, yr, yi, symM, symN, symR, symI, ordM, ordN, ordR, ordI, D, N, block):
    """
    Four-wave mixing sums of the first-order perturbation model.

    The signals are given as real and imaginary parts, zero-padded with D samples at
    each end. For each term (m, n, c), accumulates in dx and dy

        c * (x[t+n] x*[t+m+n] + y[t+n] y*[t+m+n]) * x[t+m]   and   (... ) * y[t+m],

    or, for the symmetric terms, the sum of the terms (m, n) and (n, m). The loops
    over the time samples are innermost, and are performed on blocks that fit in the
    L1 cache.
    """
    dxr = np.zeros(N, dtype=xr.dtype)
    dxi = np.zeros(N, dtype=xr.dtype)
    dyr = np.zeros(N, dtype=xr.dtype)
    dyi = np.zeros(N, dtype=xr.dtype)
    nBlocks = (N + block - 1) // block

    for b in prange(nBlocks):
        t0 = b * block
        nt = min(N - t0, block)
        axr = np.zeros(nt, dtype=xr.dtype)
        axi = np.zeros(nt, dtype=xr.dtype)
        ayr = np.zeros(nt, dtype=xr.dtype)
        ayi = np.zeros(nt, dtype=xr.dtype)

        # symmetric terms: (m, n) + (n, m)
        for k in range(len(symM)):
            om = t0 + D + symM[k]
            on = t0 + D + symN[k]
            os_ = t0 + D + symM[k] + symN[k]
            cr = symR[k]
            ci = symI[k]
            for t in range(nt):
                xmr = xr[om + t]
                xmi = xi[om + t]
                xnr = xr[on + t]
                xni = xi[on + t]
                ymr = yr[om + t]
                ymi = yi[om + t]
                ynr = yr[on + t]
                yni = yi[on + t]
                xsr = xr[os_ + t]
                xsi = -xi[os_ + t]  # conjugate
                ysr = yr[os_ + t]
                ysi = -yi[os_ + t]

                pr = xmr * xnr - xmi * xni  # p = xm * xn
                pi = xmr * xni + xmi * xnr
                qr = ymr * ynr - ymi * yni  # q = ym * yn
                qi = ymr * yni + ymi * ynr
                wr = (xmr * ynr - xmi * yni) + (xnr * ymr - xni * ymi)  # w = xm*yn + xn*ym
                wi = (xmr * yni + xmi * ynr) + (xnr * ymi + xni * ymr)

                # A = 2 p conj(xs) + w conj(ys),  B = w conj(xs) + 2 q conj(ys)
                # (the factors 2 are additions: an integer literal would promote
                # single-precision arrays to double precision)
                p2r = pr + pr
                p2i = pi + pi
                q2r = qr + qr
                q2i = qi + qi
                ar = (p2r * xsr - p2i * xsi) + (wr * ysr - wi * ysi)
                ai = (p2r * xsi + p2i * xsr) + (wr * ysi + wi * ysr)
                br = (wr * xsr - wi * xsi) + (q2r * ysr - q2i * ysi)
                bi = (wr * xsi + wi * xsr) + (q2r * ysi + q2i * ysr)

                axr[t] += cr * ar - ci * ai
                axi[t] += cr * ai + ci * ar
                ayr[t] += cr * br - ci * bi
                ayi[t] += cr * bi + ci * br

        # ordered terms
        for k in range(len(ordM)):
            om = t0 + D + ordM[k]
            on = t0 + D + ordN[k]
            os_ = t0 + D + ordM[k] + ordN[k]
            cr = ordR[k]
            ci = ordI[k]
            for t in range(nt):
                xnr = xr[on + t]
                xni = xi[on + t]
                ynr = yr[on + t]
                yni = yi[on + t]
                xsr = xr[os_ + t]
                xsi = -xi[os_ + t]
                ysr = yr[os_ + t]
                ysi = -yi[os_ + t]

                sr = (xnr * xsr - xni * xsi) + (ynr * ysr - yni * ysi)
                si = (xnr * xsi + xni * xsr) + (ynr * ysi + yni * ysr)
                tr = cr * sr - ci * si
                ti = cr * si + ci * sr

                xmr = xr[om + t]
                xmi = xi[om + t]
                ymr = yr[om + t]
                ymi = yi[om + t]
                axr[t] += tr * xmr - ti * xmi
                axi[t] += tr * xmi + ti * xmr
                ayr[t] += tr * ymr - ti * ymi
                ayi[t] += tr * ymi + ti * ymr

        for t in range(nt):
            dxr[t0 + t] = axr[t]
            dxi[t0 + t] = axi[t]
            dyr[t0 + t] = ayr[t]
            dyi[t0 + t] = ayi[t]

    return dxr, dxi, dyr, dyi


@njit(parallel=True, fastmath=True, cache=True)
def _xpmKernel(xr, xi, yr, yi, m1, c1i, n2, c2r, c2i, mFirst, spmI, D, N):
    """
    Cross-phase and self-phase modulation terms of the first-order perturbation model.

    The signals are given as real and imaginary parts, zero-padded with D samples at
    each end. Returns the phase rotations of the two polarizations and the additive
    terms (real and imaginary parts), with O(L) operations per symbol.
    """
    phiX = np.zeros(N)
    phiY = np.zeros(N)
    dxr = np.zeros(N)
    dxi = np.zeros(N)
    dyr = np.zeros(N)
    dyi = np.zeros(N)

    for t in prange(N):
        p = t + D

        # phase rotations: row L of C_ixpm (n = 0) and self-phase modulation
        sx = 0.0
        sy = 0.0
        for k in range(len(m1)):
            q = p + m1[k]
            px = xr[q] * xr[q] + xi[q] * xi[q]
            py = yr[q] * yr[q] + yi[q] * yi[q]
            sx += c1i[k] * (px + px + py)
            sy += c1i[k] * (py + py + px)
        q = p + mFirst
        pf = xr[q] * xr[q] + xi[q] * xi[q] + yr[q] * yr[q] + yi[q] * yi[q]
        phiX[t] = sx + pf * spmI
        phiY[t] = sy + pf * spmI

        # additive terms: column L of C_ixpm (m = 0)
        axr = 0.0
        axi = 0.0
        ayr = 0.0
        ayi = 0.0
        for k in range(len(n2)):
            q = p + n2[k]
            px = xr[q] * xr[q] + xi[q] * xi[q]
            py = yr[q] * yr[q] + yi[q] * yi[q]
            axr += c2r[k] * py
            axi += c2i[k] * py
            ayr += c2r[k] * px
            ayi += c2i[k] * px
        dxr[t] = xr[p] * axr - xi[p] * axi
        dxi[t] = xr[p] * axi + xi[p] * axr
        dyr[t] = yr[p] * ayr - yi[p] * ayi
        dyi[t] = yr[p] * ayi + yi[p] * ayr

    return phiX, phiY, dxr, dxi, dyr, dyi


def _nlinPerturbation(C_ifwm, C_ixpm, C_ispm, x, y, ind, prec, block=256):
    """
    First-order perturbation model, using the coefficients at the flat indexes `ind`.

    The coefficient at the (row-major) position :math:`(i, j)` of the matrices
    multiplies the symbols at the offsets :math:`m = j - L` and :math:`n = L - i`.
    """
    L = (C_ifwm.shape[0] - 1) // 2
    D = 2 * L
    n_ = 2 * L + 1
    N = len(x)
    dtype = np.finfo(prec).dtype
    ii, jj = np.divmod(np.asarray(ind), n_)

    # normalize power, split in real and imaginary parts, and zero-pad
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    y = y / np.sqrt(np.mean(np.abs(y) ** 2))
    xr, xi, yr, yi = (np.zeros(N + 2 * D, dtype=dtype) for _ in range(4))
    xr[D : D + N], xi[D : D + N] = x.real, x.imag
    yr[D : D + N], yi[D : D + N] = y.real, y.imag

    # four-wave mixing terms
    (sM, sN, sC), (oM, oN, oC) = _pairTerms(C_ifwm, ii, jj, L)
    dxr, dxi, dyr, dyi = _fwmKernel(
        xr, xi, yr, yi,
        sM.astype(np.int64), sN.astype(np.int64), sC.real.astype(dtype), sC.imag.astype(dtype),
        oM.astype(np.int64), oN.astype(np.int64), oC.real.astype(dtype), oC.imag.astype(dtype),
        D, N, block,
    )  # fmt: skip
    # cross-phase and self-phase modulation terms (only the row L and the column L of
    # C_ixpm are non-zero; the self-phase term uses the field at the offset of the
    # first coefficient)
    cx = C_ixpm[ii, jj]
    row, col = ii == L, jj == L
    x64, xi64, y64, yi64 = (a.astype(np.float64) for a in (xr, xi, yr, yi))
    phi_x, phi_y, pxr, pxi, pyr, pyi = _xpmKernel(
        x64, xi64, y64, yi64,
        (jj - L)[row].astype(np.int64), cx[row].imag.copy(),
        (L - ii)[col].astype(np.int64), cx[col].real.copy(), cx[col].imag.copy(),
        int(jj[0]) - L, float(np.imag(C_ispm)), D, N,
    )  # fmt: skip

    dx = (dxr + pxr) + 1j * (dxi + pxi)
    dy = (dyr + pyr) + 1j * (dyi + pyi)

    return dx.astype(prec), dy.astype(prec), phi_x, phi_y


def calcNLINperturbation(C_ifwm, C_ixpm, C_ispm, x, y, prec=np.complex64):
    """
    Fast calculation of the first-order perturbation model.

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
    ind = np.arange((2 * L + 1) ** 2)

    return _nlinPerturbation(C_ifwm, C_ixpm, C_ispm, x, y, ind, prec)


def calcNLINperturbationSimplified(
    C_ifwm, C_ixpm, C_ispm, x, y, coeffTol=-20, prec=np.complex64
):
    """
    Fast calculation of the first-order perturbation model with reduced number of coefficients.

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
    L = (C_ifwm.shape[0] - 1) // 2
    D = 2 * L

    # coefficient reduction
    C = C_ifwm + C_ixpm
    C[D, D] = C_ispm
    absC = np.abs(C).flatten()
    with np.errstate(divide="ignore"):
        nReducedCoeffs = int(np.sum(20 * np.log10(absC / np.max(absC)) > coeffTol))
    indSort = np.argsort(-absC)[:nReducedCoeffs]
    reductionFactor = np.round(100 * (1 - nReducedCoeffs / len(absC)), 2)

    dx, dy, phi_ixpm_x, phi_ixpm_y = _nlinPerturbation(
        C_ifwm, C_ixpm, C_ispm, x, y, indSort, prec
    )

    return dx, dy, phi_ixpm_x, phi_ixpm_y, nReducedCoeffs, reductionFactor


def perturbationNLIN(Ein, param):
    """
    Calculates the intrachannel NLIN via first-order perturbation models.

    Parameters
    ----------
    Ein : ndarray of shape (N, 2)
        Input signal for dual-polarization (complex-valued).
        The first column represents the X polarization, and the second column represents the Y polarization.

    param : optic.utils.parameters object
        Object with physical/simulation parameters of the optical channel.

        - param.D : chromatic dispersion parameter [ps/nm/km] [default: 17 ps/nm/km]
        - param.alpha : fiber attenuation parameter [dB/km] [default: 0.2 dB/km]
        - param.lspan : span length [km] [default: 50 km]
        - param.length : total fiber length [km] [default: 800 km]
        - param.pulseWidth : pulse width (fraction of symbol period) [default: 0.5]
        - param.gamma : fiber nonlinear coefficient [1/W/km] [default: 1.3 1/W/km]
        - param.Fc : carrier frequency [THz] [default: 193.2e12 Hz]
        - param.powerWeighted : power-weighted coefficient calculation? Boolean variable [default: False]
        - param.Rs : symbol rate [baud] [default: 32e9 baud]
        - param.powerWeightN : power-weighting order [default: 10]
        - param.matrixOrder : nonlinear memory matrix order [default: 25]
        - param.mode : 'AM' for standard perturbation calculation or 'AMR' for reduced complexity calculation [default: 'AM']
        - param.Pin : launch power per channel [dBm] [default: 0 dBm]
        - param.coeffTol : threshold for ignoring small perturbation coefficients [dB] [default: -20 dB]
        - param.prec : numerical precision [default: np.complex128]

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
