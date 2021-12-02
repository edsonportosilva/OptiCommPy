# -*- coding: utf-8 -*-

import numpy as np
from commpy.channelcoding.ldpc import triang_ldpc_systematic_encode as enc
from commpy.channelcoding.ldpc import ldpc_bp_decode as dec
import re

def ldpcEncode(b, LDPCparams):
    """
    Encode data bits with binary LDPC code
    b = np.random.randint(2, size=(K, Nwords))

    """
    fecFamily = LDPCparams['filename'][6:11]
    fecID = LDPCparams['filename'][12:]
    
    num = [float(s) for s in re.findall(r'-?\d+\.?\d*', fecID)]    
    n = int(num[0])
       
    if fecFamily == 'AR4JA':
        N = n
    else:
        N = LDPCparams["n_vnodes"]

    # generate random interleaver
    interlv = np.random.permutation(N)

    # encode bits
    codedBits = enc(b, LDPCparams)
    interCodedBits = (codedBits[interlv, :].T).reshape(1, -1).T

    return interCodedBits, codedBits, interlv


def ldpcDecode(llr, interlv, LDPCparams, nIter, alg="SPA"):
    """
    Decode binary LDPC encoded data bits
    b = np.random.randint(2, size=(K, Nwords))

    """
    
    fecID = LDPCparams['filename'][12:]
    
    num = [float(s) for s in re.findall(r'-?\d+\.?\d*', fecID)]    
    
    N = LDPCparams["n_vnodes"]
    n = int(num[0])
    
    dep = int(N-n)
    
    # generate deinterleaver
    deinterlv = interlv.argsort()

    # deinterleave received LLRs
    llr = llr.reshape(-1, n)
    llr = llr[:, deinterlv]
    
    # depuncturing
    if dep > 0:
        llr = np.concatenate((llr, np.zeros((llr.shape[0], dep))), axis=1)
                
    llr = llr.ravel()
        
    # decode received code words
    decodedBits, llr_out = dec(llr, LDPCparams, alg, nIter)

    return decodedBits, llr_out


# @njit
# def loggaldecode(H, r, Nloop, Lc, Nl, Ml):
#     # function x = loggaldecode(A,r,Nloop,Lc)
#     #
#     # Do log-likelihood decoding on a low-density parity check code
#     # H = parity check matrix
#     # r = received signal vector
#     # Nloop = number of iterations
#     # Lc = channel reliability
#     # Nl = Nl set of sparse representation
#     # Ml = Ml set of sparse representation

#     # Copyright 2004 by Todd K. Moon
#     # Permission is granted to use this program/data
#     # for educational/research only

#     M, N = H.shape
#     H = H.astype(np.float64)

#     # Nl = []
#     # Ml = []
#     # for m in range(M):
#     #     Nl.append([])

#     # for n in range(N):
#     #     Ml.append([])

#     # # Build the sparse representation of A using the M and N sets

#     # for m in range(M):
#     #     for n in range(N):
#     #         if H[m, n]:
#     #             Nl[m].append(n)
#     #             Ml[n].append(m)

#     # Initialize the probabilities
#     eta = np.zeros((M, N), dtype=np.float64)
#     lasteta = np.zeros((M, N), dtype=np.float64)
#     pr = np.zeros(1, dtype=np.float64)
#     lamb = r

#     for loop in range(Nloop):

#         for m in range(M):  # for each row (check)
#             for n in Nl[m]:  # work across the columns ("horizontally")
#                 pr[:] = 1.0
#                 for pn in Nl[m]:
#                     if pn == n:
#                         continue
#                     pr[:] *= np.tanh(
#                         (-lamb[pn] + lasteta[m, pn]) / 2
#                     )  # accumulate the product

#                 eta[m, n] = -2 * np.arctanh(pr[0])
#                 eta[m, n] = max([eta[m, n], -500.0])
#                 eta[m, n] = min([eta[m, n],  500.0])


#         lasteta = eta  # save to subtract to obtain extrinsic for next time around
#         # fprintf(1,'eta:'); splatexform(1,eta,1);
#         lamb = r
#         for n in range(N):  # for each column (bit)
#             #lamb[n] = Lc * r[n]

#             for m in Ml[n]:  # work down the rows ("vertically")
#                 lamb[n] = lamb[n] + eta[m, n]

#                 if lamb[n] >= 500:
#                     lamb[n] = 500.0
#                 elif lamb[n] <= -500:
#                     lamb[n] = -500.0

#         x = ((np.sign(lamb) + 1) / 2 ).astype(np.float64)  # compute decoded bits for stopping criterion
#         z1 = ((H @ x) % 2).T

#         if np.all(z1.astype(np.int32) == 0):
#             break
#     # end for loop

#     # if ~np.all(z1 == 0):
#     #     print("Decoding failure after %d iterations" % Nloop)
#     # else:
#     #     print("Successfull Decoding after %d iterations" % (loop + 1))

#     return lamb, x
