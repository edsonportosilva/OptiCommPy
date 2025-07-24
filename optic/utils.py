"""
======================================
General utilities (:mod:`optic.utils`)
======================================

.. autosummary::
   :toctree: generated/

   parameters             -- Class to be used as a struct of parameters.
   lin2dB                 -- Convert linear value to dB (decibels).
   dB2lin                 -- Convert dB (decibels) to a linear value.
   dBm2W                  -- Convert dBm to Watts.
   dec2bitarray           -- Convert decimals to arrays of bits.
   decimal2bitarray       -- Convert decimal to array of bits.
   dotNumba              -- Compute dot product using Numba.
   bitarray2dec           -- Convert array of bits to decimal.
   ber2Qfactor            -- Convert bit error rate (BER) to Q factor in dB.
"""

"""General utilities."""
import numpy as np
from numba import njit
from scipy.special import erfcinv


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
            if isinstance(value, (int, float)) and value > 10000:
                print(f"{attr}: {value:.2e}")
            else:
                print(f"{attr}: {value}")

    def to_engineering_notation(self, value):
        prefixes = {
            -12: 'p',  # pico
            -9:  'n',  # nano
            -6:  'µ',  # micro
            -3:  'm',  # milli
             0:  '',   # base unit (no prefix)
             3:  'k',  # kilo
             6:  'M',  # mega
             9:  'G',  # giga
            12:  'T',  # tera
            15:  'P',  # peta
        }

        if isinstance(value, (int, float)) and (abs(value) >= 10000 or (0 < abs(value) < 0.0001)):
            exponent = int(np.floor(np.log10(abs(value))))  # Calculate the exponent
            exponent = (exponent // 3) * 3  # Round to nearest multiple of 3

            scaled_value = value / (10 ** exponent)  # Scale the value
            prefix = prefixes.get(exponent, '')  # Get the corresponding prefix

            return f"{scaled_value:.1f} {prefix}"
        return value

    def table(self):
        attributes = vars(self)
        markdown = "| Parameter Name | Value |\n"
        markdown += "|----------------|-----------------|\n"
        
        for name, value in attributes.items():
            if isinstance(value, (list, np.ndarray, tuple)):
                markdown += f"| {name} | Array |\n"
            else:
                # Apply engineering notation where necessary
                formatted_value = self.to_engineering_notation(value)
                markdown += f"| {name} | {formatted_value} |\n"
        
        return print(markdown)

    def latex_table(self):
        attributes = vars(self)
        latex = "\\begin{tabular}{|c|c|}\n"
        latex += "\\hline\n"
        latex += "Parameter Name & Value \\\\\n"
        latex += "\\hline\n"
        
        for name, value in attributes.items():
            if isinstance(value, (list, np.ndarray, tuple)):
                latex += f"{name} & Array \\\\\n"
            else:
                # Apply engineering notation where necessary
                formatted_value = self.to_engineering_notation(value)
                latex += f"{name} & {formatted_value} \\\\\n"
            latex += "\\hline\n"
        
        latex += "\\end{tabular}"
        return print(latex)


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


# @njit
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


# @njit
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


@njit
def dotNumba(a, b):
    """
    Computes the dot product of two 1D arrays in a Numba-compatible way.

    This function is equivalent to `np.dot` for 1D arrays but can be
    JIT-compiled with Numba for accelerated execution.

    Parameters
    ----------
    a : ndarray of shape (N,)
        First input array (complex-valued).

    b : ndarray of shape (N,)
        Second input array (complex-valued).

    Returns
    -------
    result : complex
        The dot product of `a` and `b`, computed as the sum of element-wise products.

    Notes
    -----
    - Both input arrays must have the same length.
    - This function initializes the result as a complex number to support
      complex-valued operations.
    """
    return np.sum(a * b)


def ber2Qfactor(ber):
    """
    Converts a bit error rate (BER) to a Q factor in dB.

    Parameters
    ----------
    ber : float
        The bit error rate to be converted.

    Returns
    -------
    float
        The Q factor corresponding to the input BER.
    """
    return 10 * np.log10(np.sqrt(2) * erfcinv(2 * ber))
