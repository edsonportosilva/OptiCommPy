"""
=========================================================
Sources of discrete sequences (:mod:`optic.comm.sources`)
=========================================================

.. autosummary::
   :toctree: generated/

   bitSource          -- Generate a random bit sequence of length nbits.
   prbsGenerator      -- Generate a Pseudo-Random Binary Sequence (PRBS) of the given order.
   symbolSource       -- Generate a random symbol sequence from a given modulation scheme.  
"""
import logging as logg

import numpy as np
from numba import njit
from optic.comm.modulation import qamConst, pamConst, pskConst, apskConst


def bitSource(nbits, mode="random", order=None, seed=None):
    """
    Generate a sequence of bits of length `nbits` either as a random bit sequence 
    or a pseudo-random binary sequence (PRBS).

    Parameters
    ----------
    nbits : int
        The number of bits in the sequence.
    mode : {'random', 'prbs'}, optional
        The mode of the bit generation. If 'random', a sequence of random bits is generated.
        If 'prbs', a pseudo-random binary sequence (PRBS) is generated. Default is 'random'.
    order : int, optional
        The order of the PRBS generator. Only used if `mode` is 'prbs'. If not specified, a 
        default order of 23 is used.
    seed : int, optional
        The seed for the random number generator. Only applicable when `mode` is 'random'.
        If not provided, a random seed will be used.

    Returns
    -------
    bits : np.array
        An array of bits of length `nbits`, either randomly generated or from a PRBS.
    References
    ----------
    [1] Wikipedia, "Pseudorandom binary sequence," https://en.wikipedia.org/wiki/Pseudorandom_binary_sequence
    [2] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """
    if seed is not None:
        np.random.seed(seed)
    if mode == "random":
        if seed is not None:
            np.random.seed(seed)  # Seed the random number generator
        bits = np.random.randint(0, 2, nbits)
    elif mode == "prbs":
        if order is None:
            logg.warn("PRBS order not specified. Using the default order 23.")
            prbs = prbsGenerator()
        else:
            prbs = prbsGenerator(order)

        if len(prbs) < nbits:
            prbs = np.tile(prbs, nbits // len(prbs) + 1)

        bits = prbs[:nbits]

    return bits


@njit
def prbsGenerator(order=23):
    """
    Generate a Pseudo-Random Binary Sequence (PRBS) of the given order.

    Parameters
    ----------
    order : int
        The order of the PRBS sequence. Supported orders are 7, 9, 11, 13, 15, 23, 31.

    Returns
    -------
    bits : np.array
        A NumPy array of bits representing the PRBS sequence.

    References
    ----------
    [1] Wikipedia, "Pseudorandom binary sequence," https://en.wikipedia.org/wiki/Pseudorandom_binary_sequence
    """
    # Predefined taps for each PRBS order
    taps = {
        7: (6, 5),  # PRBS-7: x^7 + x^6 + 1
        9: (8, 4),  # PRBS-9: x^9 + x^5 + 1
        11: (10, 8),  # PRBS-11: x^11 + x^9 + 1
        13: (12, 11),  # PRBS-13: x^13 + x^12 + 1
        15: (14, 13),  # PRBS-15: x^15 + x^14 + 1
        23: (22, 17),  # PRBS-23: x^23 + x^18 + 1
        31: (30, 27),  # PRBS-31: x^31 + x^28 + 1
    }

    if order not in taps:
        raise ValueError(
            f"PRBS order {order} is not supported. Supported orders are 7, 9, 11, 13, 15, 23, 31."
        )

    # Initialize parameters
    lenPRBS = 2**order - 1
    tap_a, tap_b = taps[order]

    bits = np.zeros(lenPRBS, dtype=np.int64)

    lfsr = 1
    for i in range(1, lenPRBS):
        fb = (lfsr >> tap_a) ^ (lfsr >> tap_b) & 1
        lfsr = ((lfsr << 1) + fb) & (lenPRBS)
        bits[i - 1] = fb

    return bits


def symbolSource(
    nSymbols,
    M=4,
    constType="qam",
    dist="uniform",
    shapingFactor=0.0,
    px=None,
    seed=None,
):
    """
    Generate a random symbol sequence of length nSymbols based on the specified modulation scheme and order.

    Parameters
    ----------
    nSymbols : int
        Number of symbols to generate.
    M : int
        Modulation order. This defines the size of the constellation. Default is 4.
    constType : str, optional
        The type of modulation scheme. Supported types are 'qam', 'pam', 'psk', and
        'apsk'. Default is 'qam'.
    dist : str, optional
        The probability distribution to use for generating symbols. Can be 'uniform'
        or 'Maxwell-Boltzmann'. Default is 'uniform'.
    shapingFactor : float, optional
        The shaping factor used when the distribution is 'Maxwell-Boltzmann'. This
        controls the Gaussian shaping of the constellation points. Default is 0.0.
    px : array-like, optional
        Probability distribution of the constellation points. If `None`, the distribution
        is determined by `dist`. Default is `None`.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility. Default is `None`.

    Returns
    -------
    symbols : np.array
        A NumPy array of symbols randomly drawn from the specified constellation with the given probability distribution.

    Notes
    -----
    The function generates symbols from a specified modulation scheme and applies either a uniform or Maxwell-Boltzmann
    distribution to the constellation points. The Maxwell-Boltzmann distribution has its shape parametrized shaped by the 
    `shapingFactor`. Instead of using the default distributions, a custom probability distribution `px` can be provided.

    References
    ----------
    [1] Junho Cho and Peter J. Winzer, "Probabilistic Constellation Shaping for Optical Fiber Communications," J. Lightwave Technol. 37, 1590-1607 (2019)
    """
    if seed is not None:
        np.random.seed(seed)

    if constType == "qam":
        constellation = qamConst(M)
    elif constType == "pam":
        constellation = pamConst(M)
    elif constType == "psk":
        constellation = pskConst(M)
    elif constType == "apsk":
        constellation = apskConst(M)
    else:
        logg.error(
            "Invalid constellation type. Supported types are 'qam', 'pam', 'psk', and 'apsk'."
        )

    if px is None:
        if dist == "uniform":
            px = np.ones(M) / M
        elif dist == "maxwell-boltzmann":
            px = np.exp(-shapingFactor * np.abs(constellation) ** 2)
            px = px / np.sum(px)
            px = px.flatten()

    constellation = constellation / np.sqrt(
        np.sum(px * np.abs(constellation.flatten()) ** 2)
    )

    symbols = np.random.choice(constellation.flatten(), nSymbols, p=px)

    return symbols
