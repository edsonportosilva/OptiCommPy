from scipy.signal import lfilter
import numpy as np
from commpy.filters import rrcosfilter, rcosfilter
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt

def firFilter(h, x):
    """
    :param h: impulse response (symmetric)
    :param x: input signal 
    :return y: output filtered signal    
    """   
    N = h.size
    x = np.pad(x, (0, int(N/2)),'constant')
    y = lfilter(h,1,x)
    
    return y[int(N/2):y.size]

def pulseShape(pulseType, SpS=2, N=1024, alpha=0.1, Ts=1):
    """
    :param pulseType: 'rect','nrz','rrc'
    :param SpS: samples per symbol
    :param N: number of filter coefficients
    :param alpha: RRC rolloff factor
    :param Ts: symbol period
    :return filterCoeffs: normalized filter coefficients   
    """  
    Fa = (1/Ts)*SpS
    
    t = np.linspace(-2, 2, SpS)
    Te = 1       
    
    if pulseType == 'rect':
        filterCoeffs = np.ones(SpS)        
    elif pulseType == 'nrz':
        filterCoeffs = np.convolve(np.ones(SpS), 2/(np.sqrt(np.pi)*Te)*np.exp(-t**2/Te), mode='full')        
    elif pulseType == 'rrc':
        tindex, filterCoeffs = rrcosfilter(N, alpha, Ts, Fa)
    elif pulseType == 'rc':
        tindex, filterCoeffs = rcosfilter(N, alpha, Ts, Fa)
        
    return filterCoeffs/np.sqrt(np.sum(filterCoeffs**2))


def eyediagram(sig, Nsamples, SpS, n=3):

    y = sig[0:Nsamples].real
    x = np.arange(0,y.size,1) % (n*SpS)

    k = gaussian_kde(np.vstack([x, y]))
    k.set_bandwidth(bw_method=k.factor/4)

    xi, yi = 1.1*np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=1, shading='auto');
    plt.show();

    
