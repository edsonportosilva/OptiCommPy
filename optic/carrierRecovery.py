import numpy as np
from commpy.modulation import QAMModem
from commpy.utilities import signal_power
from numba import njit
from tqdm.notebook import tqdm

from optic.core import parameters


def cpr(x, dx=[], paramCPR=[]):              
    """
    Carrier phase recovery (CPR)
    
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
    
    
    # Defining training parameters:
    mod = QAMModem(m=M) # commpy QAM constellation modem object
    constSymb = mod.constellation/np.sqrt(mod.Es) # complex-valued constellation symbols
    for indMode in range(nModes):       
        for indstage, runAlg in enumerate(alg):

        
    return  yCPR

@njit
def coreCPR(x, dx, constSymb):
    """
    Carrier phase recovery core processing function
    
    """
    
    # allocate variables
        
    for ind in range(0, L):       
                    
        # update equalizer taps acording to the specified 
        # algorithm and save squared error:        
        if alg == 'bps':
            H, errSq[:,ind] = bps(x, constSymb, B, N)        
        else:
            raise ValueError('CPR algorithm not specified (or incorrectly specified).')
        
            
    return yCPR

@njit
def bps(x, constSymb, B, N):
    """
    blind phase search (BPS) algorithm  
    """          
    phi_test = np.arange(0, B)*(np.pi/2)/B
    dist = np.zeros((constSymb.shape[0], B), dtype=np.float)
    
    for indPhase, phi in enumerate(phi_test):        
        dist[:,indPhase] = np.abs(x-constSymb*np.exp(1j*phi))
        
    phi_dec = np.argwhere( dist == np.min(dist) )[1]


