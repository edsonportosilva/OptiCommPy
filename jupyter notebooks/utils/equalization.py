import numpy as np
from commpy.utilities  import signal_power
from commpy.modulation import QAMModem
from tqdm.notebook import tqdm
from utils.core import parameters

from numba import njit, jit

def mimoAdaptEqualizer(x, dx=[], paramEq=[]):              
    """
    N-by-N MIMO adaptive equalizer
    
    """
   
    # check input parameters
    numIter    = getattr(paramEq, 'numIter', 1)
    nTaps      = getattr(paramEq, 'nTaps', 15)
    mu         = getattr(paramEq, 'mu', 1e-3)
    SpS        = getattr(paramEq, 'SpS', 2)
    H          = getattr(paramEq, 'H', [])
    L          = getattr(paramEq, 'L', [])
    Hiter      = getattr(paramEq, 'Hiter', [])
    storeCoeff = getattr(paramEq, 'storeCoeff', False)
    alg        = getattr(paramEq, 'alg', ['nlms'])
    M          = getattr(paramEq, 'M', 4)    
    
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

    if not L: # if L is not defined
        L = [int(np.fix((len(x)-nTaps)/SpS+1))] # Length of the output (1 sample/symbol) of the training section       
    
    if len(H) == 0: # if H is not defined
        H  = np.zeros((nModes**2, nTaps), dtype='complex')
        
        for initH in range(0, nModes): # initialize filters' taps
            H[initH + initH*nModes, int(np.floor(H.shape[1]/2))] = 1 # Central spike initialization
           
    # Equalizer training:
    for indIter in tqdm(range(0, numIter)):
        print(alg[0],'training iteration #%d'%indIter)        
        yEq, H, errSq, Hiter = coreAdaptEq(x, dx, SpS, H, L[0], mu, nTaps, storeCoeff, alg[0], constSymb)               
        print(alg[0],'MSE = %.6f.'%np.nanmean(errSq))
        
    return  yEq, H, errSq, Hiter

@jit(nopython=True)
def coreAdaptEq(x, dx, SpS, H, L, mu, nTaps, storeCoeff, alg, constSymb):
    """
    Adaptive equalizer core processing function
    
    """
    
    # allocate variables
    nModes  = int(x.shape[1])
    indTaps = np.arange(0, nTaps) 
    indMode = np.arange(0, nModes)
    
    errSq   = np.empty((nModes, L))
    yEq     = x[0:L].copy()    
    yEq[:]  = np.nan    
    outEq   = np.array([[0+1j*0]]).repeat(nModes).reshape(nModes, 1)
    
    if storeCoeff:
        Hiter = np.array([[0+1j*0]]).repeat((nModes**2)*nTaps*L).reshape(nModes**2, nTaps, L)
    else:
        Hiter = np.array([[0+1j*0]]).repeat((nModes**2)*nTaps).reshape(nModes**2, nTaps, 1)
        
    # Radii cma, rde
    Rcma = (np.mean(np.abs(constSymb)**4)/np.mean(np.abs(constSymb)**2))*np.ones((1, nModes))+1j*0          
    Rrde = np.unique(np.abs(constSymb))
        
    for ind in range(0, L):       
        outEq[:] = 0
            
        indIn = indTaps + ind*SpS # simplify indexing and improve speed

        # pass signal sequence through the equalizer:
        for N in range(0, nModes):
            inEq   = x[indIn, N].reshape(len(indIn), 1) # slice input coming from the Nth mode            
            outEq += H[indMode+N*nModes,:]@inEq         # add contribution from the Nth mode to the equalizer's output                 
                        
        yEq[ind,:] = outEq.T        
                    
        # update equalizer taps acording to adaptive specified 
        # training algorithm and save squared error:        
        if alg == 'nlms':
            H, errSq[:,ind] = nlmsUp(x[indIn, :], dx[ind,:], outEq, mu, H, nModes)
        elif alg == 'cma':
            H, errSq[:,ind] = cmaUp(x[indIn, :], Rcma, outEq, mu, H, nModes) 
        elif alg == 'ddlms':
            H, errSq[:,ind] = ddlmsUp(x[indIn, :], constSymb, outEq, mu, H, nModes)
        elif alg == 'rde':
            H, errSq[:,ind] = rdeUp(x[indIn, :], Rrde, outEq, mu, H, nModes)
        
        if storeCoeff:
            Hiter[:,:, ind] = H  
        else:
            Hiter[:,:, 1] = H
            
    return yEq, H, errSq, Hiter

@jit(nopython=True)
def nlmsUp(x, dx, outEq, mu, H, nModes):
    """
    coefficient update with the NLMS algorithm    
    """          
    indMode = np.arange(0, nModes)    
    err = dx - outEq.T # calculate output error for the NLMS algorithm 
    
    errDiag = np.diag(err[0]) # define diagonal matrix from error array
    
    # update equalizer taps 
    for N in range(0, nModes):
            indUpdTaps = indMode+N*nModes # simplify indexing and improve speed
            inAdapt = x[:, N].T/np.linalg.norm(x[:,N])**2 # NLMS normalization
            inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
            H[indUpdTaps,:] = H[indUpdTaps,:] + mu*errDiag@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2

@jit(nopython=True)
def ddlmsUp(x, constSymb, outEq, mu, H, nModes):
    """
    coefficient update with the DDLMS algorithm    
    """      
    indMode    = np.arange(0, nModes)
    outEq      = outEq.T
    decided    = outEq.copy()
    decided[:] = np.nan
       
    for k in range(0, outEq.shape[1]):
        indSymb = np.argmin(np.abs(outEq[0,k] - constSymb))
        decided[0,k] = constSymb[indSymb]
                
    err = decided - outEq # calculate output error for the DDLMS algorithm   

    errDiag = np.diag(err[0]) # define diagonal matrix from error array
   
    # update equalizer taps 
    for N in range(0, nModes):
            indUpdTaps = indMode+N*nModes # simplify indexing
            inAdapt = x[:, N].T
            inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
            H[indUpdTaps,:] = H[indUpdTaps,:] + mu*errDiag@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2


@jit(nopython=True)
def cmaUp(x, R, outEq, mu, H, nModes):
    """
    coefficient update with the CMA algorithm    
    """      
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    err   = R - np.abs(outEq)**2 # calculate output error for the CMA algorithm 

    prodErrOut = np.diag(err[0])@np.diag(outEq[0]) # define diagonal matrix 
    
    # update equalizer taps  
    for N in range(0, nModes):
            indUpdTaps = indMode+N*nModes # simplify indexing
            inAdapt = x[:, N].T
            inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
            H[indUpdTaps,:] = H[indUpdTaps,:] + mu*prodErrOut@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2

@jit(nopython=True)
def rdeUp(x, R, outEq, mu, H, nModes):
    """
    coefficient update with the RDE algorithm    
    """      
    indMode    = np.arange(0, nModes)
    outEq      = outEq.T    
    decidedR    = outEq.copy()
    decidedR[:] = np.nan
    
    # find closest constellation radius
    for k in range(0, outEq.shape[1]):
        indR = np.argmin(np.abs(R - np.abs(outEq[0,k])))
        decidedR[0,k] = R[indR]
        
    err  = decidedR**2 - np.abs(outEq)**2 # calculate output error for the RDE algorithm 
    
    prodErrOut = np.diag(err[0])@np.diag(outEq[0]) # define diagonal matrix 
    
    # update equalizer taps 
    for N in range(0, nModes):
            indUpdTaps = indMode+N*nModes # simplify indexing
            inAdapt = x[:, N].T
            inAdaptPar = inAdapt.repeat(nModes).reshape(len(x), -1).T # expand input to parallelize tap adaptation
            H[indUpdTaps,:] = H[indUpdTaps,:] + mu*prodErrOut@np.conj(inAdaptPar) # gradient descent update   

    return H, np.abs(err)**2
