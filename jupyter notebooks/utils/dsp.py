from scipy.signal import lfilter
import numpy as np
from commpy.filters import rrcosfilter, rcosfilter
from commpy.utilities  import upsample
from scipy.stats.kde import gaussian_kde
import scipy.constants as const
from utils.models import linFiberCh
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq, fftshift
from tqdm.notebook import tqdm

def firFilter(h, x):
    """
    Implements FIR filtering and compensates filter delay
    (assuming the impulse response is symmetric)
    
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
    Generate pulse shaping filters
    
    :param pulseType: 'rect','nrz','rrc'
    :param SpS: samples per symbol
    :param N: number of filter coefficients
    :param alpha: RRC rolloff factor
    :param Ts: symbol period
    
    :return filterCoeffs: normalized filter coefficients   
    """  
    fa = (1/Ts)*SpS
    
    t = np.linspace(-2, 2, SpS)
    Te = 1       
    
    if pulseType == 'rect':
        filterCoeffs = np.concatenate((np.zeros(int(SpS/2)), np.ones(SpS), np.zeros(int(SpS/2))))       
    elif pulseType == 'nrz':
        filterCoeffs = np.convolve(np.ones(SpS), 2/(np.sqrt(np.pi)*Te)*np.exp(-t**2/Te), mode='full')        
    elif pulseType == 'rrc':
        tindex, filterCoeffs = rrcosfilter(N, alpha, Ts, fa)
    elif pulseType == 'rc':
        tindex, filterCoeffs = rcosfilter(N, alpha, Ts, fa)
        
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
    return None
    
def sincInterp(x, fa):
    
    fa_sinc = 32*fa
    Ta_sinc = 1/fa_sinc
    Ta = 1/fa
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
    
    plt.legend(loc="upper right")
    plt.xlim(min(t), max(t))
    plt.grid()
    
    return x_sum, t

def lowPassFIR(fc, fa, N, typeF = 'rect'):
    """
    Calculate FIR coeffs for a lowpass filter
    
    :param fc : cutoff frequency
    :param fa : sampling frequency
    :param N  : number of coefficients
    :param typeF : 'rect' or 'gauss'
    
    :return h : FIR filter coefficients
    """
    fu = fc/fa    
    d  = (N-1)/2    
    n  = np.arange(0, N)    
  
    # calculate filter coefficients
    if typeF == 'rect':
        h = (2*fu)*np.sinc(2*fu*(n-d))
    elif typeF == 'gauss':    
        h = np.sqrt(2*np.pi/np.log(2))*fu*np.exp(-(2/np.log(2))*(np.pi*fu*(n-d))**2)
    
    return h

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
    Eo = linFiberCh(Ei, L, 0, -D, Fc, Fs)
    
    return Eo

def cpr(Ei, N, constSymb, symbTx):    
    """
    Carrier phase recovery (CPR)
    
    """    
    ϕ  = np.zeros(Ei.shape)    
    θ  = np.zeros(Ei.shape)
    
    for k in range(0,len(Ei)):
        
        decided = np.argmin(np.abs(Ei[k]*np.exp(1j*θ[k-1]) - constSymb)) # find closest constellation symbol
        
        if k % 50 == 0:
            ϕ[k] = np.angle(symbTx[k]/(Ei[k])) # phase estimation with pilot symbol
        else:
            ϕ[k] = np.angle(constSymb[decided]/(Ei[k])) # phase estimation after symbol decision
                
        if k > N:
            θ[k]  = np.mean(ϕ[k-N:k]) # moving average filter
        else:           
            θ[k] = np.angle(symbTx[k]/(Ei[k]))
            
    Eo = Ei*np.exp(1j*θ) # compensate phase rotation
        
    return Eo, ϕ, θ

def fourthPowerFOE(Ei, Ts, plotSpec=False):
    """
    4th power frequency offset estimator (FOE)    
    """
        
    Fs = 1/Ts
    Nfft = len(Ei)
    
    f = Fs*fftfreq(Nfft)
    f = fftshift(f)
    
    f4 = 10*np.log10(np.abs(fftshift(fft(Ei**4))))    
    indFO = np.argmax(f4)

    if plotSpec:
        plt.figure()
        plt.plot(f, f4, label = '$|FFT(s[k]^4)|[dB]$')
        plt.plot(f[indFO], f4[indFO],'x',label='$4f_o$')
        plt.legend()
        plt.xlim(min(f), max(f))
        plt.grid()
    
    return f[indFO]/4

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
    #c = 299792458   # speed of light (vacuum)
    c_kms = const.c/1e3
    λ  = c_kms/Fc
    α  = -alpha/(10*np.log10(np.exp(1)))
    β2 = (D*λ**2)/(2*np.pi*c_kms)
    γ  = -gamma
            
    Nfft = len(Ei)

    ω = 2*np.pi*Fs*fftfreq(Nfft)
    
    Nspans = int(np.floor(Ltotal/Lspan))
    Nsteps = int(np.floor(Lspan/hz))   
        
    Ech = Ei.reshape(len(Ei),)    
    Ech = fft(Ech) #single-polarization field    
    
    linOperator = np.exp(-(α/2)*(hz/2) + 1j*(β2/2)*(ω**2)*(hz/2))
        
    for spanN in tqdm(range(0, Nspans)):
        
        Ech = Ech*np.exp((α/2)*Nsteps*hz)
                
        for stepN in range(0, Nsteps):            
            # First linear step (frequency domain)
            Ech = Ech*linOperator            
                      
            # Nonlinear step (time domain)
            Ech = ifft(Ech)
            Ech = Ech*np.exp(1j*γ*(Ech*np.conj(Ech))*hz)
            
            # Second linear step (frequency domain)
            Ech = fft(Ech)       
            Ech = Ech*linOperator             
                
    Ech = ifft(Ech) 
       
    return Ech.reshape(len(Ech),)
