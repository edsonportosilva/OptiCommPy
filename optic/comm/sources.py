"""
================================================================================
Sources of communication signals (:mod:`optic.comm.sources`)
================================================================================

.. autosummary::
   :toctree: generated/

   bert                     -- Calculate BER and Q-factor for optical communication using On-Off Keying (OOK).   
"""
import numpy as np
from numba import njit

def bitSource(nbits, mode='random', order=None, seed=None):
    """
    Generate a random bit sequence of length nbits.
    
    Parameters
    ----------
    nbits : int
        Number of bits in the sequence.
        
    Returns
    -------
    bits : ndarray
        An array of random bits.
    """
    if seed is not None:
        np.random.seed(seed)
    if mode == 'random':
        if seed is not None:
            np.random.seed(seed) # Seed the random number generator      
        bits = np.random.randint(0, 2, nbits)
    elif mode == 'prbs':
        if order is None:
            Warning("PRBS order not specified. Using the default order 23.")
            bits = prbsSequence(nbits)
        else:
            bits = prbsSequence(nbits, order)

    return bits

@njit
def prbsSequence(length, order=23):
    """
    Generate a Pseudo-Random Binary Sequence (PRBS) based on the specified order.

    Parameters
    ----------
    length : int
        The length of the PRBS sequence to generate.
    order : int
        The order of the PRBS sequence. Valid orders are 7, 9, 11, 13, 15, 23, 31.

    Returns
    -------
    np.ndarray
        An array of binary values (0 and 1) representing the PRBS sequence.

    Raises
    ------
    ValueError
        If the specified order is not supported.

    Notes
    -----
    The function uses Linear Feedback Shift Register (LFSR) technique to generate
    the PRBS. Feedback taps for each order are defined based on known generator
    polynomials.

    """
    # Polynomial taps for orders 7, 9, 11, 13, 15, 23, 31 (can be extended as needed)
    taps = {
        7: np.array([7, 6]),     # PRBS-7: x^7 + x^6 + 1
        9: np.array([9, 5]),     # PRBS-9: x^9 + x^5 + 1
        11: np.array([11, 9]),   # PRBS-11: x^11 + x^9 + 1
        13: np.array([13, 12]),  # PRBS-13: x^13 + x^12 + 1
        15: np.array([15, 14]),  # PRBS-15: x^15 + x^14 + 1
        23: np.array([23, 18]),  # PRBS-23: x^23 + x^18 + 1
        31: np.array([31, 28])   # PRBS-31: x^31 + x^28 + 1
    }
    
    if order not in taps:
        raise ValueError("Order not supported. Available orders: 7, 9, 11, 13, 15, 23, 31.")
        
    # Initialize the LFSR with all ones (standard approach)
    lfsr = np.ones(order, dtype=np.int64)  # Array with all ones
    seq = np.zeros(length, dtype=np.int64)
    
    for i in range(length):
        # Output bit is the last bit of the LFSR
        seq[i] = lfsr[-1]
        
        # Calculate feedback bit using manual XOR of tap positions
        feedback = 0
        for tap in taps[order]:
            feedback ^= lfsr[tap - 1]  # Adjust for zero-based index
        
        # Update LFSR: shift left and insert feedback bit into the MSB
        lfsr = np.roll(lfsr, shift=-1)
        lfsr[-1] = feedback
    
    return seq

