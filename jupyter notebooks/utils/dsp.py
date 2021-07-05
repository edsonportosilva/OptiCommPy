from scipy.signal import lfilter
import numpy as np
from commpy.filters import rrcosfilter, rcosfilter
from commpy.utilities  import upsample
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
        filterCoeffs = np.concatenate((np.zeros(int(SpS/2)), np.ones(SpS), np.zeros(int(SpS/2))))       
    elif pulseType == 'nrz':
        filterCoeffs = np.convolve(np.ones(SpS), 2/(np.sqrt(np.pi)*Te)*np.exp(-t**2/Te), mode='full')        
    elif pulseType == 'rrc':
        tindex, filterCoeffs = rrcosfilter(N, alpha, Ts, Fa)
    elif pulseType == 'rc':
        tindex, filterCoeffs = rcosfilter(N, alpha, Ts, Fa)
        
    return filterCoeffs/np.sqrt(np.sum(filterCoeffs**2))


def eyediagram(sig, Nsamples, SpS, n=3, ptype='fast', plotlabel=None):
    """
    Plots the eye diagram of a modulated signal waveform

    :param Nsamples: number os samples to be plotted
    :param SpS: samples per symbol
    :param n: number of symbol periods
    :param type: 'fast' or 'fancy'
    :param plotlabel: label for the plot legend
    """
    
    if np.iscomplex(sig).any():
        d = 1
        plotlabel_ = plotlabel+' [real]'
    else:
        d = 0
        plotlabel_ = plotlabel
        
    for ind in range(0, d+1):
        if ind == 0:
            y = sig[0:Nsamples].real
            x = np.arange(0,y.size,1) % (n*SpS)            
        else:
            y = sig[0:Nsamples].imag
            plotlabel_ = plotlabel+' [imag]'       
     
        plt.figure();
        if ptype == 'fancy':            
            k = gaussian_kde(np.vstack([x, y]))
            k.set_bandwidth(bw_method=k.factor/5)

            xi, yi = 1.1*np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=1, shading='auto');
            plt.show();
        elif ptype == 'fast':
            y[x == n*SpS] = np.nan;
            y[x == 0] = np.nan;
            
            plt.plot(x/SpS, y, color='blue', alpha=0.8, label=plotlabel_);
            plt.xlim(min(x/SpS), max(x/SpS))
            plt.xlabel('symbol period (Ts)')
            plt.ylabel('amplitude')
            plt.title('eye diagram')
            
            if plotlabel != None:
                plt.legend(loc='upper left')
                
            plt.grid()
            plt.show();
    
def sincInterp(x, Fa):
    
    Fa_sinc = 32*Fa
    Ta_sinc = 1/Fa_sinc
    Ta = 1/Fa
    t = np.arange(0, x.size*32)*Ta_sinc
    
    plt.figure()  
    y = upsample(x,32)
    y[y==0] = np.nan
    plt.plot(t,y.real,'ko', label='x[k]')
    
    x_sum = 0
    for k in range(0, x.size):
        xk_interp = x[k]*np.sinc((t-k*Ta)/Ta)
        x_sum += xk_interp
        plt.plot(t, xk_interp)           
    
    #plt.plot(t,x_sum,'k--',label ='$\hat{x}(t) =\sum_{k}\;x_{k}\;sinc[(t-kT_a)/T_a]$')
    plt.legend(loc="upper right")
    plt.xlim(min(t), max(t))
    plt.grid()
    
    return x_sum, t
