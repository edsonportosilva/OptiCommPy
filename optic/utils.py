
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
    return 10**(x / 10)

def dBm2W(x):
    """
    Convert dBm to Watts.

    Parameters:
    x (float): The power value in dBm to be converted to Watts.

    Returns:
    float: The power value in Watts.
    """
    return 1e-3 * 10**(x / 10)