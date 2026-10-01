# -*- coding: utf-8 -*-
"""
Test functions in the optic.comm.modulation module.

"""

import numpy as np
import pytest

from optic.comm.modulation import (
    demap,
    demodulateGray,
    detector,
    grayCode,
    grayMapping,
    minEuclid,
    modulateGray,
    pamConst,
    pskConst,
    qamConst,
)
from optic.utils import bitarray2dec, dec2bitarray


class TestGrayCode:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_grayCode_returns_all_codewords_of_the_requested_width(self, n):
        code = grayCode(n)

        assert len(code) == 2**n
        assert all(len(word) == n for word in code)
        assert len(set(code)) == 2**n

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_consecutive_codewords_differ_in_a_single_bit(self, n):
        code = grayCode(n)

        for current, following in zip(code, code[1:]):
            assert sum(a != b for a, b in zip(current, following)) == 1


class TestConstellations:
    @pytest.mark.parametrize("M", [4, 16, 64])
    @pytest.mark.parametrize("constType", ["qam", "psk", "pam"])
    def test_grayMapping_returns_M_symbols(self, M, constType):
        assert grayMapping(M, constType).size == M

    def test_grayMapping_rejects_qam_orders_that_are_not_square(self):
        with pytest.raises(ValueError):
            grayMapping(2, "qam")

    @pytest.mark.parametrize("M", [4, 16, 64])
    def test_pskConst_symbols_lie_on_the_unit_circle(self, M):
        np.testing.assert_allclose(np.abs(pskConst(M)), 1.0)

    @pytest.mark.parametrize("M", [2, 4, 16])
    def test_pamConst_symbols_are_real_and_equally_spaced(self, M):
        const = np.sort(np.real(pamConst(M)).flatten())

        np.testing.assert_allclose(np.imag(pamConst(M)), 0.0)
        np.testing.assert_allclose(np.diff(const), np.diff(const)[0])

    @pytest.mark.parametrize("M", [4, 16, 64])
    def test_qamConst_is_a_square_grid_symmetric_about_the_origin(self, M):
        const = qamConst(M).flatten()

        assert const.size == M
        assert np.unique(np.round(np.real(const), 9)).size == int(np.sqrt(M))
        assert np.unique(np.round(np.imag(const), 9)).size == int(np.sqrt(M))
        assert np.sum(const) == pytest.approx(0.0)

    def test_ook_constellation_always_has_two_levels(self):
        const = grayMapping(2, "ook").flatten()

        assert const.size == 2
        np.testing.assert_allclose(np.sort(np.real(const)), [0.0, 1.0])


class TestSymbolMapping:
    def test_minEuclid_returns_the_index_of_the_closest_symbol(self):
        symb = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        const = np.array([1 + 1j, 3 + 3j, 2 + 2j])

        np.testing.assert_array_equal(minEuclid(symb, const), np.array([0, 2, 1]))

    def test_minEuclid_is_robust_to_small_perturbations(self):
        rng = np.random.default_rng(0)
        const = grayMapping(16, "qam").flatten()
        indices = rng.integers(0, 16, 200)
        symb = const[indices] + 1e-3 * rng.normal(size=200)

        np.testing.assert_array_equal(minEuclid(symb, const), indices)

    def test_demap_converts_symbol_indices_into_bit_sequences(self):
        indSymb = np.array([0, 2, 1])
        bitMap = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        np.testing.assert_array_equal(
            demap(indSymb, bitMap), np.array([0, 0, 1, 0, 0, 1])
        )

    def test_demap_is_consistent_with_the_bit_to_decimal_helpers(self):
        bitMap = dec2bitarray(np.arange(8), 3)
        indSymb = np.array([5, 2, 7])

        bits = demap(indSymb, bitMap).reshape(-1, 3)

        assert [bitarray2dec(row) for row in bits] == [5, 2, 7]


class TestModulateDemodulate:
    @pytest.mark.parametrize("M", [4, 16, 64])
    @pytest.mark.parametrize("constType", ["qam", "psk", "pam"])
    def test_modulation_and_demodulation_are_lossless_without_noise(
        self, M, constType
    ):
        bitsPerSymbol = int(np.log2(M))
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, 500 * bitsPerSymbol)

        symb = modulateGray(bits, M, constType)
        recovered = demodulateGray(symb, M, constType)

        assert symb.size == bits.size // bitsPerSymbol
        np.testing.assert_array_equal(recovered, bits)

    @pytest.mark.parametrize("M", [4, 16])
    def test_modulateGray_only_produces_constellation_symbols(self, M):
        rng = np.random.default_rng(2)
        bits = rng.integers(0, 2, 400 * int(np.log2(M)))

        symb = modulateGray(bits, M, "qam")

        assert np.all(np.isin(symb, grayMapping(M, "qam").flatten()))

    @pytest.mark.parametrize("constType", ["qam", "psk"])
    def test_nearest_neighbour_symbols_differ_in_a_single_bit(self, constType):
        # defining property of Gray mapping: the closest constellation symbols
        # differ in exactly one bit, so most symbol errors cost a single bit
        M = 16
        bitsPerSymbol = int(np.log2(M))
        const = grayMapping(M, constType).flatten()
        bitMap = dec2bitarray(np.arange(M), bitsPerSymbol)

        distances = np.abs(const[:, None] - const[None, :])
        np.fill_diagonal(distances, np.inf)
        minDistance = np.min(distances)

        for i in range(M):
            neighbours = np.flatnonzero(
                np.isclose(distances[i], minDistance, rtol=1e-6)
            )
            for j in neighbours:
                assert np.sum(bitMap[i] != bitMap[j]) == 1


class TestDetector:
    @pytest.mark.parametrize("rule", ["MAP", "ML"])
    def test_detector_recovers_the_transmitted_symbols_at_high_snr(self, rule):
        rng = np.random.default_rng(3)
        const = grayMapping(16, "qam").flatten()
        indices = rng.integers(0, 16, 500)
        r = const[indices] + 1e-3 * (
            rng.normal(size=500) + 1j * rng.normal(size=500)
        )

        decided, indDec = detector(r, 1e-6, const, rule=rule)

        np.testing.assert_array_equal(indDec, indices)
        np.testing.assert_allclose(decided, const[indices])

    def test_map_and_ml_agree_for_uniform_priors(self):
        rng = np.random.default_rng(4)
        const = grayMapping(4, "qam").flatten()
        r = const[rng.integers(0, 4, 200)] + 0.1 * rng.normal(size=200)

        decidedMAP, _ = detector(r, 0.1, const, rule="MAP")
        decidedML, _ = detector(r, 0.1, const, rule="ML")

        np.testing.assert_allclose(decidedMAP, decidedML)

    def test_map_detection_favours_symbols_with_a_higher_prior(self):
        const = np.array([-1.0 + 0j, 1.0 + 0j])
        r = np.array([0.05 + 0j])  # slightly closer to +1

        decided, _ = detector(r, 1.0, const, px=np.array([0.99, 0.01]), rule="MAP")

        assert decided[0].real == pytest.approx(-1.0)
