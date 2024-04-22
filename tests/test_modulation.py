import unittest
import numpy as np
from optic.utils import bitarray2dec, dec2bitarray
from optic.comm.modulation import grayCode, grayMapping, minEuclid, demap, modulateGray, demodulateGray

class TestModulationFunctions(unittest.TestCase):
    def test_GrayCode(self):
        # Test GrayCode function for different values of n
        for n in range(1, 6):
            with self.subTest(n=n):
                result = grayCode(n)
                self.assertEqual(len(result), 2**n)
                self.assertEqual(len(result[0]), n)

    def test_GrayMapping(self):
        # Test GrayMapping function for different values of M and constType
        M_values = [4, 16, 64]
        constTypes = ['qam', 'psk', 'pam']
        for M in M_values:
            for constType in constTypes:
                with self.subTest(M=M, constType=constType):
                    result = grayMapping(M, constType)
                    self.assertEqual(len(result), M)

    def test_minEuclid(self):
        # Test minEuclid function with some example inputs
        symb = np.array([1+1j, 2+2j, 3+3j])
        const = np.array([1+1j, 3+3j, 2+2j])
        result = minEuclid(symb, const)
        np.testing.assert_array_equal(result, np.array([0, 2, 1]))

    def test_demap(self):
        # Test demap function with some example inputs
        indSymb = np.array([0, 2, 1])
        bitMap = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        result = demap(indSymb, bitMap)
        np.testing.assert_array_equal(result, np.array([0, 0, 1, 0, 0, 1]))

    def test_modulateGray(self):
        # Test modulateGray function with some example inputs
        bits = np.array([0, 1, 0, 0, 1, 0])
        M = 4
        constType = 'qam'
        result = modulateGray(bits, M, constType)
        self.assertEqual(len(result), len(bits) // int(np.log2(M)))

    def test_demodulateGray(self):
        # Test demodulateGray function with some example inputs
        symb = np.array([1+1j, 3+3j, 2+2j])
        M = 4
        constType = 'qam'
        result = demodulateGray(symb, M, constType)
        self.assertEqual(len(result), len(symb) * int(np.log2(M)))

if __name__ == '__main__':
    unittest.main()
