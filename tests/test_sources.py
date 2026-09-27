# -*- coding: utf-8 -*-
"""
Test functions in the optic.comm.sources module.

"""

import numpy as np
import pytest

from optic.comm.sources import cazacSequence, prbsGenerator


class TestPrbsGenerator:
    @pytest.mark.parametrize("order", [7, 9, 11, 13, 15])
    def test_prbs_is_a_maximal_length_sequence(self, order):
        period = 2**order - 1
        bits = prbsGenerator(order, period, seed=1)

        # an m-sequence has 2^(n-1) ones per period...
        assert bits.sum() == 2 ** (order - 1)

        # ...and a two-valued periodic autocorrelation (period at zero lag, -1 elsewhere)
        s = 1 - 2 * bits
        S = np.fft.fft(s)
        autocorr = np.real(np.fft.ifft(S * np.conj(S)))
        np.testing.assert_allclose(autocorr[0], period)
        np.testing.assert_allclose(autocorr[1:], -1, atol=1e-6)

    def test_prbs7_satisfies_its_generator_polynomial_recursion(self):
        # g(x) = x^7 + x^6 + 1  ->  b[k] = b[k-7] xor b[k-6]
        bits = prbsGenerator(7, 500, seed=1)

        np.testing.assert_array_equal(bits[7:], bits[:-7] ^ bits[1:-6])


class TestCazacSequence:
    @pytest.mark.parametrize("N, M", [(63, 1), (139, 5), (64, 1), (64, 3), (256, 7)])
    def test_cazac_has_constant_amplitude_and_zero_autocorrelation(self, N, M):
        x = cazacSequence(N, M)

        np.testing.assert_allclose(np.abs(x), 1.0)

        # ideal periodic autocorrelation: N at zero lag, zero elsewhere
        autocorr = np.fft.ifft(np.abs(np.fft.fft(x)) ** 2)
        np.testing.assert_allclose(autocorr[0], N)
        np.testing.assert_allclose(autocorr[1:], 0, atol=1e-8)
