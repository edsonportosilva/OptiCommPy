import numpy as np


class parameters:
    """
    Basic class to be used as a struct of parameters
    """

    pass


def lin2dB(x):
    """
    Convert linear value to dB (decibels).

    Parameters:
    x (float): The linear value to be converted to dB.

    Returns:
    float: The value in dB.
    """
    return 10 * np.log10(x)


def dB2lin(x):
    """
    Convert dB (decibels) to a linear value.

    Parameters:
    x (float): The value in dB to be converted to a linear value.

    Returns:
    float: The linear value.
    """
    return 10 ** (x / 10)


def dBm2W(x):
    """
    Convert dBm to Watts.

    Parameters:
    x (float): The power value in dBm to be converted to Watts.

    Returns:
    float: The power value in Watts.
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
    result = np.zeros((len(x), bit_width), dtype=np.int)
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
    result = np.zeros(bit_width, dtype=np.int)
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
    x_bitarray : 1D or 2D NumPy array of int
        Input NumPy array of bits.

    Returns
    -------
    number : int or array of int
        Integer representation(s) of the input bit array(s).
    """

    if x_bitarray.ndim == 1:
        number = 0
        for i in range(len(x_bitarray)):
            number += x_bitarray[i] * 2 ** (len(x_bitarray) - 1 - i)
        return number
    elif x_bitarray.ndim == 2:
        numbers = np.zeros(x_bitarray.shape[0], dtype=np.int)
        for j in range(x_bitarray.shape[0]):
            numbers[j] = bitarray2dec(x_bitarray[j])
        return numbers
    else:
        raise ValueError("Input array must be 1D or 2D.")
