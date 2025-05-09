# -*- coding: utf-8 -*-

# import numpy as np
# from commpy.channelcoding.ldpc import triang_ldpc_systematic_encode as enc
# from commpy.channelcoding.ldpc import ldpc_bp_decode as dec
# import re

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from numba import njit

def par2gen(H):
    """
    Convert a binary parity-check matrix H into a systematic generator matrix G over GF(2).

    Parameters
    ----------
    H : ndarray of shape (n - k, n)
        Parity-check matrix with entries in {0, 1}. Typically used in linear block codes such as LDPC codes.

    Returns
    -------
    G : ndarray of shape (k, n)
        Systematic generator matrix corresponding to the input parity-check matrix H.
        The form of G is `[I_k | P]`, where `I_k` is the identity matrix and `P` is a binary matrix.

    colSwaps : ndarray of shape (n,)
        Indices representing the column permutations applied to H to obtain Hnew.
        These permutations are required to match the systematic form used in G.

    Hnew : ndarray of shape (n - k, n)
        The modified parity-check matrix after Gaussian elimination and column reordering.
        This matrix has the identity portion on the right-hand side, corresponding to
        a standard systematic form `[P^T | I_{n-k}]`.

    Notes
    -----
    The function first applies Gaussian elimination over GF(2) to bring H to row echelon form.
    Then, it identifies and reorders columns to isolate an identity submatrix (I) and its
    complement (P). Using the transposed left portion of H, it constructs the corresponding
    systematic generator matrix.

    This method assumes that the parity-check matrix H is full rank and that the identity
    portion can be isolated on the right via column permutations.
    """
    n = H.shape[1]
    k = n - H.shape[0]
    
    Hs = gaussElim(H) # Reduce matrix to row echelon form
    
    # do the necessary column swaps
    cols = np.arange(Hs.shape[1])    
    indP = cols[np.sum(Hs[:, 0:],axis=0) > 1]  # indexes of cols belonging to matrix P
    indI = cols[np.sum(Hs[:, 0:],axis=0) == 1] # indexes of cols belonging to the indentity
    
    Hnew = np.hstack((Hs[:,cols[indP]], 
                     Hs[:,cols[indI]]))
    
    colSwaps = np.hstack((indP, indI))
        
    # systematic generator matrix G
    G = np.hstack((np.eye(k), Hnew[:,0:k].T))
    
    return G, colSwaps, Hnew

def gaussElim(H):
    """
    Perform Gaussian elimination over GF(2) to reduce a binary matrix to row echelon form.

    Parameters
    ----------
    H : ndarray or scipy.sparse.csr_matrix or scipy.sparse.csc_matrix
        Input binary matrix to be reduced. The matrix can be dense or sparse, but must contain only 0s and 1s.
        Typically used with LDPC parity-check matrices.

    Returns
    -------
    matrix : ndarray of uint8
        The row echelon form of the input matrix over GF(2). The returned matrix has the same shape
        as `H`, and all arithmetic is done modulo 2 (i.e., using XOR logic).

    Notes
    -----
    This function performs Gaussian elimination in binary arithmetic (mod 2), where:
    - Addition is done via XOR,
    - Multiplication is done via AND.

    The algorithm includes row swaps to bring pivots (1s) to the diagonal positions, and eliminates
    all 1s below and above the pivot in each column.

    The function automatically handles SciPy CSR and CSC sparse matrices by converting them to dense format.
    
    """
    if type(H) == csr_matrix:
        matrix = csr_matrix.todense(H).astype(np.uint8) 
    elif type(H) == csc_matrix:
        matrix = csc_matrix.todense(H).astype(np.uint8) 
    elif type(H) == coo_matrix:
        matrix = coo_matrix.todense(H).astype(np.uint8)
    else:
        matrix = H

    #print("matrix type", type(matrix))
    # Reduce matrix to row echelon form
    matrix = np.array(matrix, dtype=np.uint8)
          
    rowCount, columnCount = matrix.shape
    lead = 0

    for r in range(rowCount):
        if lead >= columnCount:
            break
        i = r
        while matrix[i, lead] == 0:
            i += 1
            if i == rowCount:
                i = r
                lead += 1
                if lead == columnCount:
                    break
        if lead == columnCount:
            break
        
        # Troca as linhas
        matrix[[r, i]] = matrix[[i, r]]
        
        # Elimina os 1s nas outras linhas
        for j in range(rowCount):
            if j != r and matrix[j, lead] == 1:
                matrix[j] = np.mod(matrix[j] + matrix[r], 2)
        
        lead += 1

    return matrix

def encodeLDPC(H, bits):
    """
    Encode binary messages using a parity-check matrix of a linear block code (e.g., LDPC).

    Parameters
    ----------
    H : ndarray of shape (n - k, n)
        Binary parity-check matrix with entries in {0, 1}. It defines the linear constraints
        of the code and is used to derive a corresponding systematic generator matrix.

    bits : ndarray of shape (k, N)
        Binary input messages to be encoded. Each column represents a message of length `k`,
        and there are `N` such messages.

    Returns
    -------
    codewords : ndarray of shape (n, N)
        Binary encoded codewords. Each column is a codeword of length `n` corresponding
        to the respective input message in `bits`. The codewords satisfy `H @ c = 0 (mod 2)`.

    Notes
    -----
    This function performs the following steps:

    1. Uses `par2gen(H)` to compute the corresponding systematic generator matrix `G`
       of the form `[I_k | P]`, and stores the column permutation used to achieve it.
    2. Encodes the input messages using `encoder(G, bits)` to get systematic codewords.
    3. Applies the inverse column permutation to return the codewords in the original bit ordering.

    The arithmetic is performed over GF(2). The result satisfies the parity-check condition
    `H @ codeword = 0 mod 2` for each column.

    """
    G, colSwaps, _ = par2gen(H) # get systematic generator matrix G    
    G = G.astype(np.int32)
    return encoder(G, bits), colSwaps

@njit
def encoder(G, bits):
    """
    Encode binary messages using a generator matrix over GF(2).

    Parameters
    ----------
    G : ndarray of shape (k, n)
        Generator matrix with entries in {0, 1}. Each row corresponds to a basis vector
        of the code, and the matrix defines the linear transformation from input bits to codewords.

    bits : ndarray of shape (k, N)
        Binary input messages to encode. Each column is a message of length `k`, and there are `N` messages
        in total.

    Returns
    -------
    codewords : ndarray of shape (n, N)
        Encoded binary codewords. Each column corresponds to a codeword of length `n` resulting
        from applying the generator matrix to the respective input message.

    Notes
    -----
    This function performs matrix multiplication over GF(2), i.e., using XOR for addition
    and AND for multiplication.

    The function transposes the generator matrix internally so that each codeword is computed as:

        codeword = G.T @ bits[:, i] mod 2

    which is equivalent to bits[i] @ G mod 2 in standard linear coding.

    """
    G = G.T
    n, k = G.shape
    _, N = bits.shape
    codewords = np.zeros((n, N), dtype=np.uint8)
    for col in range(N):  # for each input word
        for i in range(n):  # for each codeword bit
            acc = 0
            for j in range(k):
                acc ^= G[i, j] & bits[j, col]  # binary dot product
            codewords[i, col] = acc
    return codewords


# def ldpcDecode(llr, interlv, LDPCparams, nIter, alg="SPA"):
#     """
#     Decode binary LDPC encoded data bits
#     b = np.random.randint(2, size=(K, Nwords))

#     """
    
#     fecID = LDPCparams['filename'][12:]
    
#     num = [float(s) for s in re.findall(r'-?\d+\.?\d*', fecID)]    
    
#     N = LDPCparams["n_vnodes"]
#     n = int(num[0])     
#     dep = int(N-n)
    
#     # generate deinterleaver
#     deinterlv = interlv.argsort()

#     # deinterleave received LLRs
#     llr = llr.reshape(-1, n)
#     llr = llr[:, deinterlv]
    
#     # depuncturing
#     if dep > 0:
#         llr = np.concatenate((llr, np.zeros((llr.shape[0], dep))), axis=1)
                
#     # deinterlv = interlv.argsort()
    
#     # reshape received LLRs
#     llr = llr.reshape(-1, n)    
        
#     # depuncturing
#     if dep > 0:
#         llr = np.concatenate((llr, np.zeros((llr.shape[0], dep))), axis=1)
    
#     decodedBits_hd = (-np.sign(llr)+1)//2 
#     decodedBits_hd = decodedBits_hd.reshape(-1, N).T
    
#     print(llr.shape)
    
#     llr = llr[:, deinterlv]
#     llr = llr.ravel()
        
#     # decode received code words
#     decodedBits, llr_out = dec(llr, LDPCparams, alg, nIter)
        
#     return decodedBits, decodedBits_hd, llr_out

