# """
# ======================================
# General utilities (:mod:`optic.utils`)
# ======================================

# .. autosummary::
#    :toctree: generated/

#    parameters             -- Class to be used as a struct of parameters
#    lin2dB                 -- Convert linear value to dB (decibels)
#    dB2lin                 -- Convert dB (decibels) to a linear value
#    dBm2W                  -- Convert dBm to Watts
#    dec2bitarray           -- Convert decimals to arrays of bits
#    decimal2bitarray       -- Convert decimal to array of bits
#    bitarray2dec           -- Convert array of bits to decimal
# """

"""General utilities."""
import numpy as np


class parameters:
    """
    Basic class to be used as a struct of parameters

    """

    pass

    def view(self):
        """
        Prints the attributes and their values in either standard or scientific notation.

        """
        for attr, value in self.__dict__.items():
            if isinstance(value, (int, float)) and value > 1000:
                print(f"{attr}: {value:.2e}")
            else:
                print(f"{attr}: {value}")


def lin2dB(x):
    """
    Convert linear value to dB (decibels).

    Parameters
    ----------
    x : float
        The linear value to be converted to dB.

    Returns
    -------
    float
        The value converted to dB, i.e 10log10(x).
    """
    return 10 * np.log10(x)


def dB2lin(x):
    """
    Convert dB (decibels) to a linear value.

    Parameters
    ----------
    x : float
        The value in dB to be converted to a linear value.

    Returns
    -------
    float
        The linear value.
    """
    return 10 ** (x / 10)


def dBm2W(x):
    """
    Convert dBm to Watts.

    Parameters
    ----------
    x : float
        The power value in dBm to be converted to Watts.

    Returns
    -------
    float
        The power value in Watts.
    """
    return 1e-3 * 10 ** (x / 10)


def dec2bitarray(x, bit_width):
    """
    Converts a positive integer or an array-like of positive integers to a NumPy array of the specified size containing
    bits (0 and 1).

    Parameters
    ----------
    x : int or array-like of int
        Positive integer(s) to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 2D NumPy array of int
        Array containing the binary representation of all the input decimal(s).

    """

    if isinstance(x, int):
        return decimal2bitarray(x, bit_width)
    result = np.zeros((len(x), bit_width), dtype=np.int64)
    for pox, number in enumerate(x):
        result[pox] = decimal2bitarray(number, bit_width)
    return result


def decimal2bitarray(x, bit_width):
    """
    Converts a positive integer to a NumPy array of the specified size containing bits (0 and 1). This version is slightly
    quicker but only works for one integer.

    Parameters
    ----------
    x : int
        Positive integer to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D NumPy array of int
        Array containing the binary representation of the input decimal.

    """
    result = np.zeros(bit_width, dtype=np.int64)
    i = 1
    pox = 0
    while i <= x:
        if i & x:
            result[bit_width - pox - 1] = 1
        i <<= 1
        pox += 1
    return result


def bitarray2dec(x_bitarray):
    """
    Converts an input NumPy array of bits (0 and 1) to a decimal integer.

    Parameters
    ----------
    x_bitarray : 1D array of int
        Input NumPy array of bits.

    Returns
    -------
    number : int or array of int
        Integer representation(s) of the input bit array(s).
    """
    number = 0

    for i in range(len(x_bitarray)):
        number = number + x_bitarray[i] * pow(2, len(x_bitarray) - 1 - i)

    return number
