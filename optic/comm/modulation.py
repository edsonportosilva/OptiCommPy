"""
===========================================================
Digital modulation utilities (:mod:`optic.comm.modulation`)
===========================================================

.. autosummary::
   :toctree: generated/

   grayCode                 -- Gray code generator
   grayMapping              -- Gray Mapping for digital modulations
   pamConst                 -- Generate a Pulse Amplitude Modulation (PAM) constellation.
   qamConst                 -- Generate a Quadrature Amplitude Modulation (QAM) constellation.
   pskConst                 -- Generate a Phase Shift Keying (PSK) constellation.
   apskConst                -- Generate an Amplitude-Phase Shift Keying (APSK) constellation.
   minEuclid                -- Find minimum Euclidean distance
   demap                    -- Contellation symbol index to bit sequence demapping
   modulateGray             -- Modulate bit sequences to constellation symbol sequences (w/ Gray mapping)
   demodulateGray           -- Demodulate symbol sequences (minEuclid + hard decisions) to bit sequences (assuming Gray mapping)
"""


"""Digital modulation utilities."""
import logging as logg

import numpy as np
from optic.utils import bitarray2dec, dec2bitarray
from numba import njit, prange


def grayCode(n):
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


def grayMapping(M, constType):
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
    const : np.array
        constellation symbols (sorted according their corresponding
        Gray bit sequence as integer decimal).
    
    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """
    if M != 2 and constType == "ook":
        logg.warn("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2

    bitsSymb = int(np.log2(M))

    code = grayCode(bitsSymb)
    if constType == "ook":
        const = np.arange(0, 2)
    elif constType == "pam":
        const = pamConst(M)
    elif constType == "qam":
        const = qamConst(M)
    elif constType == "psk":
        const = pskConst(M)
    elif constType == "apsk":
        const = apskConst(M)

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


def pamConst(M):
    """
    Generate a Pulse Amplitude Modulation (PAM) constellation.

    Parameters
    ----------
    M : int
        Number of symbols in the constellation. It must be an integer.

    Returns
    -------
    np.array
        1D PAM constellation.
    
    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """
    L = int(M - 1)
    return np.arange(-L, L + 1, 2)


def qamConst(M):
    """
    Generate a Quadrature Amplitude Modulation (QAM) constellation.

    Parameters
    ----------
    M : int
        Number of symbols in the constellation. It must be a perfect square.

    Returns
    -------
    const : np.array
        Complex square M-QAM constellation.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """
    L = int(np.sqrt(M) - 1)

    # generate 1D PAM constellation
    PAM = np.arange(-L, L + 1, 2)
    PAM = np.array([PAM])

    # generate complex square M-QAM constellation
    const = np.tile(PAM, (L + 1, 1))
    const = const + 1j * np.flipud(const.T)

    for ind in np.arange(1, L + 1, 2):
        const[ind] = np.flip(const[ind], 0)

    return const


def pskConst(M):
    """
    Generate a Phase Shift Keying (PSK) constellation.

    Parameters
    ----------
    M : int
        Number of symbols in the constellation. It must be a power of 2 positive integer.

    Returns
    -------
    np.array
        Complex M-PSK constellation.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """
    # generate complex M-PSK constellation
    pskPhases = np.arange(0, 2 * np.pi, 2 * np.pi / M)
    return np.exp(1j * pskPhases)


def apskConst(M, m1=None, phaseOffset=None):
    """
    Generate an Amplitude-Phase Shift Keying (APSK) constellation.

    Parameters
    ----------
    M : int
        Constellation order.
    m1 : int
        Number of bits used to index the radii of the constellation.

    Returns
    -------
    const : np.array
        APSK constellation

    References
    ----------
    [1] Z. Liu, et al "APSK Constellation with Gray Mapping," IEEE Communications Letters, vol. 15, no. 12, pp. 1271-1273, 2011.

    [2] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """
    if m1 is None:
        if M == 16:
            m1 = 1
        elif M == 32:
            m1 = 2
        elif M == 64:
            m1 = 2
        elif M == 128:
            m1 = 3
        elif M == 256:
            m1 = 3
        elif M == 512:
            m1 = 4
        elif M == 1024:
            m1 = 4

    nRings = int(2**m1)  # bits that index the rings
    m2 = int(np.log2(M) - m1)  # bits that index the symbols per ring

    symbolsPerRing = int(2**m2)

    const = np.zeros((M,), dtype=np.complex64)

    if phaseOffset is None:
        phaseOffset = np.pi / symbolsPerRing

    for idx in range(nRings):
        radius = np.sqrt(-np.log(1 - ((idx + 1) - 0.5) * symbolsPerRing / M))

        if (idx + 1) % 2 == 1:
            const[idx * symbolsPerRing : (idx + 1) * symbolsPerRing] = radius * np.flip(
                pskConst(symbolsPerRing)
            )
        else:
            const[
                idx * symbolsPerRing : (idx + 1) * symbolsPerRing
            ] = radius * pskConst(symbolsPerRing)

    return const * np.exp(1j * phaseOffset)


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
    np.array of int
        indexes of the closest constellation symbols.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.

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
    
    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.

    """
    M = bitMap.shape[0]
    b = int(np.log2(M))

    decBits = np.zeros(len(indSymb) * b, dtype="int")

    for i in prange(len(indSymb)):
        decBits[i * b : i * b + b] = bitMap[indSymb[i], :]
    return decBits

@njit
def detector(r, σ2, constSymb, px=None, rule='MAP'):
    """
    Perform symbol detection using either the MAP (Maximum A Posteriori) or ML (Maximum Likelihood) rule.

    Parameters
    ----------
    r : np.array
        The received signal.
    σ2 : float
        The noise variance.
    constSymb : np.array
        The constellation symbols.
    px : np.array, optional
        The prior probabilities of each symbol. If None, uniform priors are assumed.
    rule : str, optional
        The detection rule to use. Either 'MAP' (default) or 'ML'.

    Returns
    -------
    tuple
        A tuple containing:
            - np.array: The detected symbols.
            - np.array: The indices of the detected symbols in the constellation.

    Notes:
    ------
    If `px` is None or `rule` is 'ML', uniform priors are assumed.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.
    """
    if px is None or rule == 'ML':
        px = 1 / constSymb.size * np.ones(constSymb.size)
           
    decided = np.zeros(r.size, dtype=np.complex64) 
    indDec = np.zeros(r.size, dtype=np.int64) 
    π = np.pi  
    
    if rule == 'MAP':
        for ii, ri in enumerate(r): # for each received symbol        
            log_probMetric = np.zeros(constSymb.size)

            # calculate MAP probability metric        
            # calculate log(P(sm|r)) = log(p(r|sm)*P(sm)) for m= 1,2,...,M
            log_probMetric = - np.abs(ri - constSymb)**2 / σ2 + np.log(px)

            # find the constellation symbol with the largest P(sm|r)       
            indDec[ii] = np.argmax(log_probMetric)

            # make the decision in favor of the symbol with the largest metric
            decided[ii] = constSymb[indDec[ii]]
            
    elif rule == 'ML':      
        for ii, ri in enumerate(r): # for each received symbol        
            distMetric = np.zeros(constSymb.size)        
            # calculate distance metric   

            # calculate |r-sm|**2, for m= 1,2,...,M
            distMetric = np.abs(ri - constSymb)**2

            # find the constellation symbol with the smallest distance metric       
            indDec[ii] = np.argmin(distMetric)

            # make the decision in favor of the symbol with the smallest metric
            decided[ii] = constSymb[indDec[ii]]
    else:
        print('Detection rule should be either MAP or ML')
        
    
    return decided, indDec
    
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
        'qam', 'psk', 'apsk', 'pam' or 'ook'.

    Returns
    -------
    array of complex constellation symbols
        bits modulated to complex constellation symbols.
    
    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.

    """
    if M != 2 and constType == "ook":
        logg.warn("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2
    bitsSymb = int(np.log2(M))
    const = grayMapping(M, constType)

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
        'qam', 'psk', 'apsk', 'pam' or 'ook'.

    Returns
    -------
    array of ints
        sequence of demodulated bits.

    References
    ----------
    [1] Proakis, J. G., & Salehi, M. Digital Communications (5th Edition). McGraw-Hill Education, 2008.

    """
    if M != 2 and constType == "ook":
        logg.warn("OOK has only 2 symbols, but M != 2. Changing M to 2.")
        M = 2
    const = grayMapping(M, constType)

    # get bit to symbol mapping
    indMap = minEuclid(const, const)
    bitMap = dec2bitarray(indMap, int(np.log2(M)))
    b = int(np.log2(M))
    bitMap = bitMap.reshape(-1, b)

    # demodulate received symbol sequence
    indrx = minEuclid(symb, const)

    return demap(indrx, bitMap)