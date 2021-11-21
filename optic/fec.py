# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 19:23:45 2021

@author: edson
"""
import numpy as np


def loggaldecode(A, r, Nloop, Lc):
    # function x = loggaldecode(A,r,Nloop,Lc)
    #
    # Do log-likelihood decoding on a low-density parity check code
    # A = parity check matrix
    # r = received signal vector
    # Nloop = number of iterations
    # Lc = channel reliability

    # Copyright 2004 by Todd K. Moon
    # Permission is granted to use this program/data
    # for educational/research only

    M, N = A.shape

    Nl = []
    Ml = []
    for m in range(M):
        Nl.append([])

    for n in range(N):
        Ml.append([])

    # Build the sparse representation of A using the M and N sets

    for m in range(M):
        for n in range(N):
            if A[m, n]:
                Nl[m].append(n)
                Ml[n].append(m)

    # idx = find(A != 0) # identify the "sparse" locations of A
    # The index vector idx is used to emulate the sparse operations
    # of the decoding algorithm.  In practice, true sparse operations
    # and spare storage should be used.

    # Initialize the probabilities
    eta = np.zeros((M, N))
    lasteta = np.zeros((M, N))
    lamb = Lc * r
    # fprintf(1,'lamb[0]:'); splatexform(1,lamb,1);

    for loop in range(Nloop):
        # print('lamb:', np.round(lamb,2).T)
        # fprintf(1,'loop=#d\n',loop);

        for m in range(M):  # for each row (check)
            for n in Nl[m]:  # work across the columns ("horizontally")
                pr = 1
                for pn in Nl[m]:
                    if pn == n:
                        continue
                    pr = pr * np.tanh(
                        (-lamb[pn] + lasteta[m, pn]) / 2
                    )  # accumulate the product

                eta[m, n] = -2 * np.arctanh(pr)
                
                eta[eta < -100] = -100
                eta[eta > 100] = 100
                
        lasteta = eta  # save to subtract to obtain extrinsic for next time around
        # fprintf(1,'eta:'); splatexform(1,eta,1);

        for n in range(N):  # for each column (bit)
            lamb[n] = Lc * r[n]

            for m in Ml[n]:  # work down the rows ("vertically")
                lamb[n] = lamb[n] + eta[m, n]

        lamb[lamb < -200] = -200
        lamb[lamb > 200] = 200

        #   fprintf(1,'lamb:'); splatexform(1,lamb,1);
        #   p = exp(lamb) ./ (1+exp(lamb));  # needed only for comparison purposes!
        #   fprintf(1,'p:'); splatexform(1,p,1);

        x = ((np.sign(lamb) + 1) / 2 ).astype('int')  # compute decoded bits for stopping criterion
        z1 = ((A @ x) % 2).T

        #  fprintf(1,'x: ');latexform(1,x,1);
        #  fprintf(1,'z: ');latexform(1,z1,1);
        if np.all(z1 == 0):
            break
    # end for loop

    # if ~np.all(z1 == 0):
    #     print("Decoding failure after %d iterations" % Nloop)
    # else:
    #     print("Successfull Decoding after %d iterations" % (loop + 1))

    # print('z1:', z1)
    # print('x:', x.T)
    #print("r:", np.round(r, 2).T)
    #print("lamb:", np.round(lamb, 2).T)
    # lamb = np.array([lamb]).T

    return lamb, x
