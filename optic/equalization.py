import numpy as np
import scipy.constants as const
from commpy.modulation import QAMModem
from numba import njit
from numpy.fft import fft, fftfreq, ifft
from tqdm.notebook import tqdm

from optic.models import linFiberCh
import logging as logg

def edc(Ei, L, D, Fc, Fs):
    """
    Electronic chromatic dispersion compensation (EDC)

    :param Ei: dispersed signal
    :param L: fiber length [km]
    :param D: chromatic dispersion parameter [ps/nm/km]
    :param Fc: carrier frequency [Hz]
    :param Fs: sampling frequency [Hz]

    :return Eo: CD compensated signal
    """
    return linFiberCh(Ei, L, 0, -D, Fc, Fs)

def mimoAdaptEqualizer(x, dx=[], paramEq=[]):
    """
    N-by-N MIMO adaptive equalizer
    
    """
   
    # check input parameters
    numIter    = getattr(paramEq, 'numIter', 1)
    nTaps      = getattr(paramEq, 'nTaps', 15)
    mu         = getattr(paramEq, 'mu', [1e-3])
    lambdaRLS  = getattr(paramEq, 'lambdaRLS', 0.99)
    SpS        = getattr(paramEq, 'SpS', 2)
    H          = getattr(paramEq, 'H', [])
    L          = getattr(paramEq, 'L', [])
    Hiter      = getattr(paramEq, 'Hiter', [])
    storeCoeff = getattr(paramEq, 'storeCoeff', False)
    alg        = getattr(paramEq, 'alg', ['nlms'])
    M          = getattr(paramEq, 'M', 4)
    prgsBar    = getattr(paramEq, "prgsBar", True)

    # We want all the signal sequences to be disposed in columns:
    if not len(dx):
        dx = x.copy()

    try:
        if x.shape[1] > x.shape[0]:
            x = x.T       
    except IndexError:
        x  = x.reshape(len(x),1)       

    try:        
        if dx.shape[1] > dx.shape[0]:
            dx = dx.T
    except IndexError:        
        dx = dx.reshape(len(dx),1)

    nModes = int(x.shape[1]) # number of sinal modes (order of the MIMO equalizer)

    Lpad    = int(np.floor(nTaps/2))
    zeroPad = np.zeros((Lpad, nModes), dtype='complex')
    x = np.concatenate((zeroPad, x, zeroPad)) # pad start and end of the signal with zeros

    # Defining training parameters:
    mod = QAMModem(m=M) # commpy QAM constellation modem object
    constSymb = mod.constellation/np.sqrt(mod.Es) # complex-valued constellation symbols

    totalNumSymb = int(np.fix((len(x)-nTaps)/SpS+1))

    if not L: # if L is not defined
        L = [totalNumSymb] # Length of the output (1 sample/symbol) of the training section       

    if not H: # if H is not defined
        H  = np.zeros((nModes**2, nTaps), dtype='complex')

        for initH in range(nModes): # initialize filters' taps
            H[initH + initH*nModes, int(np.floor(H.shape[1]/2))] = 1 # Central spike initialization

    # Equalizer training:
    if type(alg) == list: 

        yEq   = np.zeros((totalNumSymb,x.shape[1]), dtype='complex')
        errSq = np.zeros((totalNumSymb,x.shape[1])).T        

        nStart = 0
        for indstage, runAlg in enumerate(alg):    
            logg.info(f'{runAlg} - training stage #%d', indstage)

            nEnd = nStart+L[indstage]

            if indstage == 0:
                for indIter in tqdm(range(numIter), disable=not(prgsBar)):
                    logg.info(f'{runAlg} pre-convergence training iteration #%d', indIter)
                    yEq[nStart:nEnd,:], H, errSq[:,nStart:nEnd], Hiter = coreAdaptEq(x[nStart*SpS:nEnd*SpS,:], dx[nStart:nEnd,:],
                                                                                     SpS, H, L[indstage], mu[indstage], lambdaRLS, nTaps,
                                                                                     storeCoeff, runAlg, constSymb)
                    logg.info(f'{runAlg} MSE = %.6f.', np.nanmean(errSq[:,nStart:nEnd]))
            else:
                yEq[nStart:nEnd,:], H, errSq[:,nStart:nEnd], Hiter = coreAdaptEq(x[nStart*SpS:nEnd*SpS,:], dx[nStart:nEnd,:],
                                                                             SpS, H, L[indstage], mu[indstage], lambdaRLS, nTaps,
                                                                             storeCoeff, runAlg, constSymb)
                logg.info(f'{runAlg} MSE = %.6f.', np.nanmean(errSq[:,nStart:nEnd]))

            nStart = nEnd
    else:    
        for indIter in tqdm(range(numIter), disable=not(prgsBar)):
            logg.info(f'{alg}training iteration #%d', indIter)
            yEq, H, errSq, Hiter = coreAdaptEq(x, dx, SpS, H, L, mu, nTaps, storeCoeff, alg, constSymb)
            logg.info(f'{alg}MSE = %.6f.', np.nanmean(errSq))

    return  yEq, H, errSq, Hiter

@njit
def coreAdaptEq(x, dx, SpS, H, L, mu, lambdaRLS, nTaps, storeCoeff, alg, constSymb):
    """
    Adaptive equalizer core processing function
    
    """
    
    # allocate variables
    nModes  = int(x.shape[1])
    indTaps = np.arange(0, nTaps)
    indMode = np.arange(0, nModes)

    errSq   = np.empty((nModes, L))
    yEq = x[:L].copy()
    yEq[:]  = np.nan
    outEq   = np.array([[0+1j*0]]).repeat(nModes).reshape(nModes, 1)

    if storeCoeff:
        Hiter = np.array([[0+1j*0]]).repeat((nModes**2)*nTaps*L).reshape(nModes**2, nTaps, L)
    else:
        Hiter = np.array([[0+1j*0]]).repeat((nModes**2)*nTaps).reshape(nModes**2, nTaps, 1)

    if alg == 'rls':   
        Sd = np.eye(nTaps, dtype=np.complex128)
        a = Sd.copy()
        for _ in range(nTaps-1):
            Sd = np.concatenate((Sd, a))        

    # Radii cma, rde
    Rcma = (np.mean(np.abs(constSymb)**4)/np.mean(np.abs(constSymb)**2))*np.ones((1, nModes))+1j*0
    Rrde = np.unique(np.abs(constSymb))

    for ind in range(L):   
        outEq[:] = 0

        indIn = indTaps + ind*SpS # simplify indexing and improve speed

        # pass signal sequence through the equalizer:
        for N in range(nModes):
            inEq   = x[indIn, N].reshape(len(indIn), 1) # slice input coming from the Nth mode            
            outEq += H[indMode+N*nModes,:]@inEq         # add contribution from the Nth mode to the equalizer's output                 

        yEq[ind,:] = outEq.T        

        # update equalizer taps acording to the specified 
        # algorithm and save squared error:        
        if alg == 'nlms':
            H, errSq[:,ind] = nlmsUp(x[indIn, :], dx[ind,:], outEq, mu, H, nModes)
        elif alg == 'cma':
            H, errSq[:,ind] = cmaUp(x[indIn, :], Rcma, outEq, mu, H, nModes) 
        elif alg == 'dd-lms':
            H, errSq[:,ind] = ddlmsUp(x[indIn, :], constSymb, outEq, mu, H, nModes)
        elif alg == 'rde':
            H, errSq[:,ind] = rdeUp(x[indIn, :], Rrde, outEq, mu, H, nModes)
        elif alg == 'da-rde':
            H, errSq[:,ind] = dardeUp(x[indIn, :], dx[ind,:], outEq, mu, H, nModes)
        elif alg == 'rls':
            H, Sd, errSq[:,ind] = rlsUp(x[indIn, :], dx[ind,:], outEq, lambdaRLS, H, Sd, nModes)
        elif alg == 'dd-rls':
            H, Sd, errSq[:,ind] = ddrlsUp(x[indIn, :], constSymb, outEq, lambdaRLS, H, Sd, nModes)
        elif alg == 'static':
            errSq[:,ind] = errSq[:,ind-1]
        else:
            raise ValueError('Equalization algorithm not specified (or incorrectly specified).')

        if storeCoeff:
            Hiter[:,:, ind] = H  
        else:
            Hiter[:,:, 1] = H

    return yEq, H, errSq, Hiter

@njit
def nlmsUp(x, dx, outEq, mu, H, nModes):
    """
    coefficient update with the NLMS algorithm    
    """          
    indMode = np.arange(0, nModes)
    err = dx - outEq.T # calculate output error for the NLMS algorithm 

    errDiag = np.diag(err[0]) # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode+N*nModes # simplify indexing and improve speed
        inAdapt = x[:, N].T/np.linalg.norm(x[:,N])**2 # NLMS normalization
        inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
        H[indUpdTaps,:] += mu*errDiag@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2

@njit
def rlsUp(x, dx, outEq, λ, H, Sd, nModes):
    """
    coefficient update with the RLS algorithm    
    """
    nTaps   = H.shape[1]
    indMode = np.arange(0, nModes)
    indTaps = np.arange(0, nTaps)

    err = dx - outEq.T # calculate output error for the NLMS algorithm 

    errDiag = np.diag(err[0]) # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdModes = indMode+N*nModes
        indUpdTaps  = indTaps+N*nTaps

        Sd_ = Sd[indUpdTaps,:]

        inAdapt = np.conj(x[:, N]).reshape(-1,1) # input samples
        inAdaptPar = (inAdapt.T).repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation

        Sd_ = (1/λ)*(Sd_ - (Sd_@(inAdapt@(np.conj(inAdapt).T))@Sd_)/( λ + (np.conj(inAdapt).T)@Sd_@inAdapt ) )

        H[indUpdModes,:] += errDiag@(Sd_@inAdaptPar.T).T;

        Sd[indUpdTaps,:]  = Sd_;

    return H, Sd, np.abs(err)**2

@njit
def ddlmsUp(x, constSymb, outEq, mu, H, nModes):
    """
    coefficient update with the DD-LMS algorithm    
    """      
    indMode    = np.arange(0, nModes)
    outEq      = outEq.T
    decided    = np.zeros(outEq.shape, dtype=np.complex128)

    for k in range(nModes):
        indSymb = np.argmin(np.abs(outEq[0,k] - constSymb))
        decided[0,k] = constSymb[indSymb]

    err = decided - outEq # calculate output error for the DDLMS algorithm   

    errDiag = np.diag(err[0]) # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode+N*nModes # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
        H[indUpdTaps,:] += mu*errDiag@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2

@njit
def ddrlsUp(x, constSymb, outEq, λ, H, Sd, nModes):
    """
    coefficient update with the DD-RLS algorithm    
    """
    nTaps   = H.shape[1]
    indMode = np.arange(0, nModes)
    indTaps = np.arange(0, nTaps)

    outEq      = outEq.T
    decided    = np.zeros(outEq.shape, dtype=np.complex128)

    for k in range(nModes):
        indSymb = np.argmin(np.abs(outEq[0,k] - constSymb))
        decided[0,k] = constSymb[indSymb]

    err = decided - outEq # calculate output error for the DDLMS algorithm 

    errDiag = np.diag(err[0]) # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdModes = indMode+N*nModes
        indUpdTaps  = indTaps+N*nTaps

        Sd_ = Sd[indUpdTaps,:]

        inAdapt = np.conj(x[:, N]).reshape(-1,1) # input samples
        inAdaptPar = (inAdapt.T).repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation

        Sd_ = (1/λ)*(Sd_ - (Sd_@(inAdapt@(np.conj(inAdapt).T))@Sd_)/( λ + (np.conj(inAdapt).T)@Sd_@inAdapt ) )

        H[indUpdModes,:] += errDiag@(Sd_@inAdaptPar.T).T;

        Sd[indUpdTaps,:]  = Sd_;

    return H, Sd, np.abs(err)**2

@njit
def cmaUp(x, R, outEq, mu, H, nModes):
    """
    coefficient update with the CMA algorithm    
    """      
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    err   = R - np.abs(outEq)**2 # calculate output error for the CMA algorithm 

    prodErrOut = np.diag(err[0])@np.diag(outEq[0]) # define diagonal matrix 

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode+N*nModes # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
        H[indUpdTaps,:] += mu*prodErrOut@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2

@njit
def rdeUp(x, R, outEq, mu, H, nModes):
    """
    coefficient update with the RDE algorithm    
    """      
    indMode    = np.arange(0, nModes)
    outEq      = outEq.T
    decidedR   = np.zeros(outEq.shape,dtype=np.complex128)

    # find closest constellation radius
    for k in range(nModes):
        indR = np.argmin(np.abs(R - np.abs(outEq[0,k])))
        decidedR[0,k] = R[indR]

    err  = decidedR**2 - np.abs(outEq)**2 # calculate output error for the RDE algorithm 

    prodErrOut = np.diag(err[0])@np.diag(outEq[0]) # define diagonal matrix 

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode+N*nModes # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
        H[indUpdTaps,:] += mu*prodErrOut@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2

@njit
def dardeUp(x, dx, outEq, mu, H, nModes):
    """
    coefficient update with the data-aided RDE algorithm    
    """      
    indMode    = np.arange(0, nModes)
    outEq      = outEq.T
    decidedR   = np.zeros(outEq.shape,dtype=np.complex128)

    # find exact constellation radius
    for k in range(nModes):        
        decidedR[0,k] = np.abs(dx[k])

    err  = decidedR**2 - np.abs(outEq)**2 # calculate output error for the RDE algorithm 

    prodErrOut = np.diag(err[0])@np.diag(outEq[0]) # define diagonal matrix 

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode+N*nModes # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
        H[indUpdTaps,:] += mu*prodErrOut@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2


def dbp(Ei, Fs, Ltotal, Lspan, hz=0.5, alpha=0.2, gamma=1.3, D=16, Fc=193.1e12):
    """
    Digital backpropagation (symmetric, single-pol.)

    :param Ei: input signal
    :param Ltotal: total fiber length [km]
    :param Lspan: span length [km]
    :param hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :param alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :param D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :param gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :param Fc: carrier frequency [Hz][default: 193.1e12 Hz]
    :param Fs: sampling frequency [Hz]

    :return Ech: backpropagated signal
    """
    # c = 299792458   # speed of light (vacuum)
    c_kms = const.c / 1e3
    λ = c_kms / Fc
    α = -alpha / (10 * np.log10(np.exp(1)))
    β2 = (D * λ ** 2) / (2 * np.pi * c_kms)
    γ = -gamma

    Nfft = len(Ei)

    ω = 2 * np.pi * Fs * fftfreq(Nfft)

    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(np.floor(Lspan / hz))

    Ech = Ei.reshape(len(Ei),)
    Ech = fft(Ech)  # single-polarization field

    linOperator = np.exp(-(α / 2) * (hz / 2) + 1j * (β2 / 2) * (ω ** 2) * (hz / 2))

    for _ in tqdm(range(Nspans)):
        Ech = Ech * np.exp((α / 2) * Nsteps * hz)

        for _ in range(Nsteps):
            # First linear step (frequency domain)
            Ech = Ech * linOperator

            # Nonlinear step (time domain)
            Ech = ifft(Ech)
            Ech = Ech * np.exp(1j * γ * (Ech * np.conj(Ech)) * hz)

            # Second linear step (frequency domain)
            Ech = fft(Ech)
            Ech = Ech * linOperator

    Ech = ifft(Ech)

    return Ech.reshape(len(Ech),)
