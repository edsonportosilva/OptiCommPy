# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 19:23:45 2021

@author: edson
"""
import numpy as np
from numba import njit

@njit
def loggaldecode(H, r, Nloop, Lc, Nl, Ml):
    # function x = loggaldecode(A,r,Nloop,Lc)
    #
    # Do log-likelihood decoding on a low-density parity check code
    # H = parity check matrix
    # r = received signal vector
    # Nloop = number of iterations
    # Lc = channel reliability
    # Nl = Nl set of sparse representation
    # Ml = Ml set of sparse representation

    # Copyright 2004 by Todd K. Moon
    # Permission is granted to use this program/data
    # for educational/research only

    M, N = H.shape
    H = H.astype(np.float64)

    # Nl = []
    # Ml = []
    # for m in range(M):
    #     Nl.append([])

    # for n in range(N):
    #     Ml.append([])

    # # Build the sparse representation of A using the M and N sets

    # for m in range(M):
    #     for n in range(N):
    #         if H[m, n]:
    #             Nl[m].append(n)
    #             Ml[n].append(m)

    # Initialize the probabilities
    eta = np.zeros((M, N), dtype=np.float64)
    lasteta = np.zeros((M, N), dtype=np.float64)
    pr = np.zeros(1, dtype=np.float64)
    lamb = Lc * r   

    for loop in range(Nloop):
        
        for m in range(M):  # for each row (check)
            for n in Nl[m]:  # work across the columns ("horizontally")
                pr[:] = 1.0
                for pn in Nl[m]:
                    if pn == n:
                        continue
                    pr[:] *= np.tanh(
                        (-lamb[pn] + lasteta[m, pn]) / 2
                    )  # accumulate the product

                eta[m, n] = -2 * np.arctanh(pr[0])
                eta[m, n] = max([eta[m, n], -100.0])
                eta[m, n] = min([eta[m, n],  100.0])
         
                    
        lasteta = eta  # save to subtract to obtain extrinsic for next time around
        # fprintf(1,'eta:'); splatexform(1,eta,1);

        for n in range(N):  # for each column (bit)
            lamb[n] = Lc * r[n]

            for m in Ml[n]:  # work down the rows ("vertically")
                lamb[n] = lamb[n] + eta[m, n]
            
            if lamb[n] > 100:
                lamb[n] = 100.0
            elif lamb[n] < -100:
                lamb[n] = -100.0
   
        x = ((np.sign(lamb) + 1) / 2 ).astype(np.float64)  # compute decoded bits for stopping criterion
        z1 = ((H @ x) % 2).T

        if np.all(z1.astype(np.int32) == 0):
            break
    # end for loop

    # if ~np.all(z1 == 0):
    #     print("Decoding failure after %d iterations" % Nloop)
    # else:
    #     print("Successfull Decoding after %d iterations" % (loop + 1))

    return lamb, x
