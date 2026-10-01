# -*- coding: utf-8 -*-
"""
Test functions in the optic.models.perturbation module.

The first-order perturbation model is compared with a literal (loop by loop)
implementation of its equations, which is slow and used here as the reference.

"""

import numpy as np
import pytest

from optic.models import perturbation as pert
from optic.utils import parameters


def makeParam(**kwargs):
    base = dict(length=100, lspan=50, matrixOrder=3, Pin=2.0, Rs=32e9)
    base.update(kwargs)
    param = parameters()
    for key, value in base.items():
        setattr(param, key, value)
    return param


def qamSymbols(N, seed=0):
    rng = np.random.default_rng(seed)
    levels = np.array([-3, -1, 1, 3])
    return (rng.choice(levels, N) + 1j * rng.choice(levels, N)) / np.sqrt(10)


def referencePerturbation(C_ifwm, C_ixpm, C_ispm, x, y, ind=None):
    """
    Literal implementation of the first-order perturbation model.

    For each symbol, the fields at the offsets of every coefficient are gathered in
    matrices (Xm = x[t+m], Xn = x[t+n], X_NplusM = x[t+m+n]), flattened, and the
    coefficients at the flat positions `ind` (all, by default) are applied. The XPM
    terms use the row L (n = 0) and the column L (m = 0) of C_ixpm.
    """
    L = (C_ifwm.shape[0] - 1) // 2
    D = 2 * L
    n_ = 2 * L + 1
    ind = np.arange(n_**2) if ind is None else ind

    rows, cols = np.indices((n_, n_))
    mask1 = (rows == L).astype(float)  # row L
    mask2 = (cols == L).astype(float)  # column L
    cf = C_ifwm.flatten()[ind]
    c1 = (C_ixpm * mask1).flatten()[ind]
    c2 = (C_ixpm * mask2).flatten()[ind]

    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    y = y / np.sqrt(np.mean(np.abs(y) ** 2))
    N = len(x)
    px = np.zeros(N + 2 * D, dtype=complex)
    py = np.zeros(N + 2 * D, dtype=complex)
    px[D : D + N], py[D : D + N] = x, y

    # window index of the field x[t+m] (matrix Xm), x[t+n] (Xn) and x[t+m+n]
    idxM = (L + cols).flatten()[ind]
    idxN = (3 * L - rows).flatten()[ind]
    idxNM = (2 * L - rows + cols).flatten()[ind]

    dx, dy = np.zeros(N, complex), np.zeros(N, complex)
    phix, phiy = np.zeros(N), np.zeros(N)
    for t in range(N):
        wx, wy = px[t : t + 2 * D + 1], py[t : t + 2 * D + 1]
        Xm, Xn, Xnm = wx[idxM], wx[idxN], wx[idxNM]
        Ym, Yn, Ynm = wy[idxM], wy[idxN], wy[idxNM]
        A1, A2 = np.abs(Xm) ** 2, np.abs(Ym) ** 2
        M1, M2 = Xn * np.conj(Xnm), Yn * np.conj(Ynm)
        first = A1[0] + A2[0]
        phix[t] = np.imag(np.dot(2 * A1 + A2, c1) + first * C_ispm)
        phiy[t] = np.imag(np.dot(2 * A2 + A1, c1) + first * C_ispm)
        dx[t] = np.dot((M1 + M2) * Xm, cf) + np.dot(M2 * Xm, c2)
        dy[t] = np.dot((M1 + M2) * Ym, cf) + np.dot(M1 * Ym, c2)

    return dx, dy, phix, phiy


@pytest.fixture(scope="module")
def coefficients():
    _, C_ifwm, C_ixpm, C_ispm = pert.calcPertCoeffMatrix(makeParam())
    return C_ifwm, C_ixpm, C_ispm


def reducedIndexes(coefficients, coeffTol):
    """Flat indexes of the coefficients kept by the reduced-complexity model."""
    C_ifwm, C_ixpm, C_ispm = coefficients
    L = (C_ifwm.shape[0] - 1) // 2
    C = C_ifwm + C_ixpm
    C[2 * L, 2 * L] = C_ispm  # as in the model: the self-phase coefficient replaces a corner
    absC = np.abs(C).flatten()
    with np.errstate(divide="ignore"):  # the central coefficient is zero
        nKept = int(np.sum(20 * np.log10(absC / absC.max()) > coeffTol))

    return np.argsort(-absC)[:nKept], absC.size


def relError(out, ref):
    return max(np.abs(o - r).max() / np.abs(r).max() for o, r in zip(out, ref))


class TestFirstOrderPerturbation:
    @pytest.mark.parametrize("N", [5, 40, 300])
    def test_full_model_matches_the_reference(self, coefficients, N):
        x, y = qamSymbols(N, 0), qamSymbols(N, 1)

        out = pert.calcNLINperturbation(*coefficients, x, y, np.complex128)
        ref = referencePerturbation(*coefficients, x, y)

        assert [o.shape for o in out] == [(N,)] * 4
        assert relError(out, ref) < 1e-12

    @pytest.mark.parametrize("coeffTol", [-5, -15, -30])
    def test_reduced_model_matches_the_reference(self, coefficients, coeffTol):
        x, y = qamSymbols(60, 0), qamSymbols(60, 1)
        out = pert.calcNLINperturbationSimplified(*coefficients, x, y, coeffTol, np.complex128)

        # the retained coefficients: those within coeffTol dB of the largest one
        ind, size = reducedIndexes(coefficients, coeffTol)
        nKept = len(ind)
        ref = referencePerturbation(*coefficients, x, y, ind)

        assert out[4] == nKept
        assert out[5] == pytest.approx(round(100 * (1 - nKept / size), 2))
        assert relError(out[:4], ref) < 1e-12

    def test_reduced_model_with_all_coefficients_has_the_additive_terms_of_the_full_model(
        self, coefficients
    ):
        x, y = qamSymbols(80, 0), qamSymbols(80, 1)

        full = pert.calcNLINperturbation(*coefficients, x, y, np.complex128)
        reduced = pert.calcNLINperturbationSimplified(*coefficients, x, y, -1000, np.complex128)

        # the central coefficient of C is zero, so it is never kept
        assert reduced[4] == 48 and reduced[5] == pytest.approx(round(100 / 49, 2))
        # the additive terms are the same; the phase terms differ only in the field used
        # by the self-phase term (the one at the first retained coefficient)
        np.testing.assert_allclose(reduced[0], full[0], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(reduced[1], full[1], rtol=1e-12, atol=1e-14)

    def test_single_precision(self, coefficients):
        x, y = qamSymbols(200, 0), qamSymbols(200, 1)

        out = pert.calcNLINperturbation(*coefficients, x, y, np.complex64)
        ref = referencePerturbation(*coefficients, x, y)

        assert out[0].dtype == np.complex64 and out[1].dtype == np.complex64
        assert out[2].dtype == np.float64
        assert relError(out, ref) < 2e-5

    @pytest.mark.parametrize("block", [1, 7, 64, 256, 4096])
    def test_result_does_not_depend_on_the_block_size(self, coefficients, block):
        x, y = qamSymbols(300, 0), qamSymbols(300, 1)
        L = 3
        ind = np.arange((2 * L + 1) ** 2)
        args = (*coefficients, x, y, ind, np.complex128)

        ref = pert._nlinPerturbation(*args)
        out = pert._nlinPerturbation(*args, block=block)

        for o, r in zip(out, ref):
            np.testing.assert_allclose(o, r, rtol=1e-12, atol=1e-13)

    def test_xpm_terms_are_included(self, coefficients):
        """The XPM terms (row/column L of C_ixpm) must contribute to every output."""
        C_ifwm, C_ixpm, C_ispm = coefficients
        x, y = qamSymbols(100, 0), qamSymbols(100, 1)

        with_xpm = pert.calcNLINperturbation(C_ifwm, C_ixpm, C_ispm, x, y, np.complex128)
        without = pert.calcNLINperturbation(C_ifwm, np.zeros_like(C_ixpm), C_ispm, x, y, np.complex128)

        for a, b in zip(with_xpm, without):
            assert np.abs(a - b).max() > 1e-3 * np.abs(a).max()

    def test_inputs_are_not_modified(self, coefficients):
        C_ifwm, C_ixpm, C_ispm = (np.copy(c) for c in coefficients)
        x, y = qamSymbols(50, 0) * 3, qamSymbols(50, 1) * 3
        x0, y0 = x.copy(), y.copy()

        pert.calcNLINperturbation(C_ifwm, C_ixpm, C_ispm, x, y)
        pert.calcNLINperturbationSimplified(C_ifwm, C_ixpm, C_ispm, x, y, -10)

        np.testing.assert_array_equal(x, x0)
        np.testing.assert_array_equal(y, y0)
        for a, b in zip((C_ifwm, C_ixpm), coefficients):
            np.testing.assert_array_equal(a, b)

    def test_symbols_are_power_normalized(self, coefficients):
        x, y = qamSymbols(100, 0), qamSymbols(100, 1)

        a = pert.calcNLINperturbation(*coefficients, x, y, np.complex128)
        b = pert.calcNLINperturbation(*coefficients, 5 * x, 0.2 * y, np.complex128)

        for u, v in zip(a, b):
            np.testing.assert_allclose(u, v, rtol=1e-12, atol=1e-14)


class TestCoefficientPairing:
    def test_complete_model_uses_only_symmetric_terms(self):
        """C_ifwm depends only on the product m*n: (m, n) and (n, m) share a coefficient."""
        L = 6
        _, C_ifwm, _, _ = pert.calcPertCoeffMatrix(makeParam(matrixOrder=L))
        n_ = 2 * L + 1
        ii, jj = np.divmod(np.arange(n_**2), n_)

        (sM, sN, sC), (oM, oN, oC) = pert._pairTerms(C_ifwm, ii, jj, L)

        assert len(oM) == 0
        assert np.all(sM <= sN)
        # every ordered pair with a non-zero coefficient is covered exactly once
        nonzero = np.count_nonzero(C_ifwm)
        diagonal = np.count_nonzero(sM == sN)
        assert 2 * (len(sM) - diagonal) + diagonal == nonzero
        # the diagonal terms have half of the coefficient (they are counted twice)
        m, n = 2, 2
        assert sC[(sM == m) & (sN == n)][0] == pytest.approx(0.5 * C_ifwm[L - n, m + L])

    def test_reduced_models_may_have_unpaired_terms(self, coefficients):
        """The self-phase coefficient replaces one corner of C, which breaks the symmetry."""
        x, y = qamSymbols(60, 0), qamSymbols(60, 1)
        L = 3
        ind, _ = reducedIndexes(coefficients, -20)
        ii, jj = np.divmod(ind, 2 * L + 1)
        C_ifwm = coefficients[0]

        (sM, _, _), (oM, _, _) = pert._pairTerms(C_ifwm, ii, jj, L)

        # whichever the split, the result is the one of the reference (see above)
        assert len(sM) + len(oM) > 0
        out = pert.calcNLINperturbationSimplified(*coefficients, x, y, -20, np.complex128)
        assert relError(out[:4], referencePerturbation(*coefficients, x, y, ind)) < 1e-12


class TestPerturbationNLIN:
    @pytest.mark.parametrize("mode", ["AM", "AMR"])
    def test_matches_the_reference(self, mode):
        Ein = np.stack([qamSymbols(120, 0), qamSymbols(120, 1)], axis=1)
        param = makeParam(mode=mode, coeffTol=-15, prec=np.complex128, Pin=3.0)

        out = pert.perturbationNLIN(Ein.copy(), param)

        # reference: the equations applied to the coefficients of the model
        # (calcPertCoeffMatrix defaults to Fc = 193.2 THz, perturbationNLIN to 193.1 THz)
        _, C_ifwm, C_ixpm, C_ispm = pert.calcPertCoeffMatrix(makeParam(Fc=193.1e12))
        ind = reducedIndexes((C_ifwm, C_ixpm, C_ispm), -15)[0] if mode == "AMR" else None
        dx, dy, phx, phy = referencePerturbation(C_ifwm, C_ixpm, C_ispm, Ein[:, 0], Ein[:, 1], ind)
        P = 0.5 * 1e-3 * 10 ** (3.0 / 10)
        xn = Ein[:, 0] / np.sqrt(np.mean(np.abs(Ein[:, 0]) ** 2))
        yn = Ein[:, 1] / np.sqrt(np.mean(np.abs(Ein[:, 1]) ** 2))
        ref = np.stack(
            [
                np.sqrt(P) * xn * (np.exp(1j * P * phx) - 1) + P**1.5 * dx * np.exp(1j * P * phx),
                np.sqrt(P) * yn * (np.exp(1j * P * phy) - 1) + P**1.5 * dy * np.exp(1j * P * phy),
            ],
            axis=1,
        )

        assert out.shape == (120, 2)
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-15)

    def test_nonlinear_perturbation_vanishes_without_nonlinearity(self):
        Ein = np.stack([qamSymbols(100, 0), qamSymbols(100, 1)], axis=1)

        out = pert.perturbationNLIN(Ein, makeParam(gamma=0.0, prec=np.complex128))

        assert np.abs(out).max() < 1e-30
