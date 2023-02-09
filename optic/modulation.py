"""Digital modulation utilities."""
import logging as logg

import numpy as np
from commpy.utilities import bitarray2dec, dec2bitarray
from numba import njit, prange
from numpy.matlib import repmat


def GrayCode(n):
    """
    Gray code generator.

    Parameters
    ----------
    n : int
        length of the codeword in bits.

    Returns
    -------
    code : list
           list of binary strings of the gray code.

    """
    code = []

    for i in range(1 << n):
        # Generating the decimal
        # values of gray code then using
        # bitset to convert them to binary form
        val = i ^ (i >> 1)

        # Converting to binary string
        s = bin(val)[2::]
        code.append(s.zfill(n))
    return code


def GrayMapping(M, constType):
    """
    Gray Mapping for digital modulations.

    Parameters
    ----------
    M : int
        modulation order
    constType : 'qam', 'psk', 'pam' or 'ook'.
        type of constellation.

    Returns
    -------
    const : list
        list with constellation symbols (sorted according their corresponding
        Gray bit sequence as integer decimal).

    """
    if M != 2 and constType == "ook":
        logg.warn("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2
    L = int(M - 1) if constType in ["pam", "ook"] else int(np.sqrt(M) - 1)
    bitsSymb = int(np.log2(M))

    code = GrayCode(bitsSymb)
    if constType in ["pam", "ook"]:
        const = np.arange(-L, L + 1, 2)
    elif constType == "qam":
        PAM = np.arange(-L, L + 1, 2)
        PAM = np.array([PAM])

        # generate complex square M-QAM constellation
        const = repmat(PAM, L + 1, 1) + 1j * repmat(np.flip(PAM.T, 0), 1, L + 1)
        const = const.T

        for ind in np.arange(1, L + 1, 2):
            const[ind] = np.flip(const[ind], 0)
    elif constType == "psk":
        pskPhases = np.arange(0, 2 * np.pi, 2 * np.pi / M)

        # generate complex M-PSK constellation
        const = np.exp(1j * pskPhases)
    const = const.reshape(M, 1)
    const_ = np.zeros((M, 2), dtype=complex)

    for ind in range(M):
        const_[ind, 0] = const[ind, 0]  # complex constellation symbol
        const_[ind, 1] = int(code[ind], 2)  # mapped bit sequence (as integer decimal)
    # sort complex symbols column according to their mapped bit sequence (as integer decimal)
    const = const_[const_[:, 1].real.argsort()]
    const = const[:, 0]

    if constType in ["pam", "ook"]:
        const = const.real
    return const


@njit(parallel=True)
def minEuclid(symb, const):
    """
    Find minimum Euclidean distance.

    Find closest constellation symbol w.r.t the Euclidean distance in the
    complex plane.

    Parameters
    ----------
    symb : np.array
        Received constellation symbols.
    const : np.array
        Reference constellation.

    Returns
    -------
    array of int
        indexes of the closest constellation symbols.

    """
    ind = np.zeros(symb.shape, dtype=np.int64)
    for ii in prange(len(symb)):
        ind[ii] = np.abs(symb[ii] - const).argmin()
    return ind


@njit(parallel=True)
def demap(indSymb, bitMap):
    """
    Contellation symbol index to bit sequence demapping.

    Parameters
    ----------
    indSymb : np.array of ints
        Indexes of received symbol sequence.
    bitMap : (M, log2(M)) np.array
        bit-to-symbol mapping.

    Returns
    -------
    decBits : np.array
        Sequence of demapped bits.

    """
    M = bitMap.shape[0]
    b = int(np.log2(M))

    decBits = np.zeros(len(indSymb) * b, dtype='int')

    for i in prange(len(indSymb)):
        decBits[i * b : i * b + b] = bitMap[indSymb[i], :]
    return decBits


def modulateGray(bits, M, constType):
    """
    Modulate bit sequences to constellation symbol sequences (w/ Gray mapping).

    Parameters
    ----------
    bits : array of ints
        sequence of data bits.
    M : int
        order of the modulation format.
    constType : string
        'qam', 'psk', 'pam' or 'ook'.

    Returns
    -------
    array of complex constellation symbols
        bits modulated to complex constellation symbols.

    """
    if M != 2 and constType == "ook":
        logg.warn("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2
    bitsSymb = int(np.log2(M))
    const = GrayMapping(M, constType)

    symb = bits.reshape(-1, bitsSymb).T
    symbInd = bitarray2dec(symb)

    return const[symbInd]


def demodulateGray(symb, M, constType):
    """
    Demodulate symbol sequences to bit sequences (w/ Gray mapping).

    Hard demodulation is based on minimum Euclidean distance.

    Parameters
    ----------
    symb : array of complex constellation symbols
        sequence of constellation symbols to be demodulated.
    M : int
        order of the modulation format.
    constType : string
        'qam', 'psk', 'pam' or 'ook'.

    Returns
    -------
    array of ints
        sequence of demodulated bits.

    """
    if M != 2 and constType == "ook":
        logg.warn("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2
    const = GrayMapping(M, constType)

    # get bit to symbol mapping
    indMap = minEuclid(const, const)
    bitMap = dec2bitarray(indMap, int(np.log2(M)))
    b = int(np.log2(M))
    bitMap = bitMap.reshape(-1, b)

    # demodulate received symbol sequence
    indrx = minEuclid(symb, const)

    return demap(indrx, bitMap)
