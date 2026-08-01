# -*- coding: utf-8 -*-
"""
Test functions in the optic.utils module.

"""

import numpy as np
import pytest

from optic.utils import (
    bitarray2dec,
    ber2Qfactor,
    dB2lin,
    dBm2W,
    dec2bitarray,
    dotNumba,
    lin2dB,
    llr2bitProb,
    parameters,
)


class TestParameters:
    def test_attributes_can_be_set_and_read(self):
        param = parameters()
        param.Fs = 32e9
        param.M = 16

        assert param.Fs == 32e9
        assert param.M == 16

    def test_copy_returns_an_independent_object(self):
        param = parameters()
        param.values = np.array([1.0, 2.0, 3.0])
        param.M = 16

        paramCopy = param.copy()
        paramCopy.M = 64
        paramCopy.values[0] = 99.0

        assert param.M == 16
        assert param.values[0] == 1.0


class TestUnitConversions:
    @pytest.mark.parametrize("x", [1e-3, 0.5, 1.0, 2.0, 1e3])
    def test_lin2dB_and_dB2lin_are_inverse_operations(self, x):
        assert dB2lin(lin2dB(x)) == pytest.approx(x)

    @pytest.mark.parametrize(
        "dBm, watts", [(-30, 1e-6), (0, 1e-3), (10, 1e-2), (30, 1.0)]
    )
    def test_dBm2W_converts_reference_values(self, dBm, watts):
        assert dBm2W(dBm) == pytest.approx(watts)

    def test_ber2Qfactor_is_consistent_with_the_gaussian_approximation(self):
        # BER = 0.5*erfc(Q/sqrt(2)), with Q returned in dB by ber2Qfactor
        from scipy.special import erfc

        for ber in [1e-2, 1e-3, 1e-5, 1e-9]:
            Q = dB2lin(ber2Qfactor(ber))
            assert 0.5 * erfc(Q / np.sqrt(2)) == pytest.approx(ber, rel=1e-6)

    def test_ber2Qfactor_decreases_with_increasing_ber(self):
        Q = [ber2Qfactor(ber) for ber in [1e-9, 1e-6, 1e-3, 1e-2]]
        assert np.all(np.diff(Q) < 0)


class TestBitArrayConversions:
    @pytest.mark.parametrize("bit_width", [1, 4, 8, 16])
    def test_dec2bitarray_and_bitarray2dec_are_inverse_operations(self, bit_width):
        for x in range(2**bit_width):
            bits = dec2bitarray(x, bit_width)

            assert len(bits) == bit_width
            assert bitarray2dec(bits) == x

    def test_dec2bitarray_handles_arrays_of_decimals(self):
        bits = dec2bitarray(np.array([0, 1, 2, 3]), 2)

        np.testing.assert_array_equal(
            bits, np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        )

    def test_dec2bitarray_uses_msb_first_ordering(self):
        np.testing.assert_array_equal(dec2bitarray(1, 4), np.array([0, 0, 0, 1]))
        np.testing.assert_array_equal(dec2bitarray(8, 4), np.array([1, 0, 0, 0]))


class TestDotNumba:
    def test_matches_numpy_dot_for_real_arrays(self):
        a = np.arange(1.0, 6.0)
        b = np.arange(5.0, 0.0, -1.0)

        assert dotNumba(a, b) == pytest.approx(np.dot(a, b))

    def test_matches_numpy_dot_for_complex_arrays(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=16) + 1j * rng.normal(size=16)
        b = rng.normal(size=16) + 1j * rng.normal(size=16)

        assert dotNumba(a, b) == pytest.approx(np.dot(a, b))


class TestLLR2BitProb:
    def test_matches_the_logistic_function(self):
        llr = np.linspace(-20, 20, 41).reshape(-1, 1)

        np.testing.assert_allclose(
            llr2bitProb(llr, np.float64), 1 / (1 + np.exp(llr)), rtol=1e-12
        )

    def test_zero_llr_maps_to_equiprobable_bits(self):
        np.testing.assert_allclose(llr2bitProb(np.zeros((4, 2))), 0.5)

    def test_probabilities_decrease_with_increasing_llr(self):
        # LLRs follow the log(P(b=0)/P(b=1)) convention, so P(b=1) decreases
        probs = llr2bitProb(np.linspace(-10, 10, 21).reshape(-1, 1)).flatten()

        assert np.all(np.diff(probs) < 0)

    def test_is_numerically_stable_for_large_magnitude_llrs(self):
        # a naive 1/(1 + exp(llr)) implementation overflows here
        llr = np.array([[-1e4, -800.0, 0.0, 800.0, 1e4]])
        probs = llr2bitProb(llr)

        assert np.all(np.isfinite(probs))
        assert np.all((probs >= 0) & (probs <= 1))
        np.testing.assert_allclose(probs[0, :2], 1.0)
        np.testing.assert_allclose(probs[0, 3:], 0.0)

    def test_preserves_the_shape_of_the_input(self):
        assert llr2bitProb(np.zeros((32, 4))).shape == (32, 4)
