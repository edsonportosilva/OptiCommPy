import numpy as np
from numpy.fft import fft, ifft, fftfreq
import scipy.constants as const

def mzm(Ai, Vπ, u, Vb):
    """
    MZM modulator 
    
    :param Vπ: Vπ-voltage
    :param Vb: bias voltage
    :param u:  modulator's driving signal (real-valued)
    :param Ai: amplitude of the input CW carrier
    
    :return Ao: output optical signal
    """
    π  = np.pi
    Ao = Ai*np.cos(0.5/Vπ*(u+Vb)*π)
    
    return Ao

def iqm(Ai, u, Vπ, VbI, VbQ):
    """
    IQ modulator 
    
    :param Vπ: MZM Vπ-voltage
    :param VbI: in-phase MZM bias voltage
    :param VbQ: quadrature MZM bias voltage    
    :param u:  modulator's driving signal (complex-valued baseband)
    :param Ai: amplitude of the input CW carrier
    
    :return Ao: output optical signal
    """
    Ao = mzm(Ai/np.sqrt(2), Vπ, u.real, VbI) + 1j*mzm(Ai/np.sqrt(2), Vπ, u.imag, VbQ)
    
    return Ao

def linFiberCh(Ei, L, alpha, D, Fc, Fs):
    """
    Linear fiber channel w/ loss and chromatic dispersion

    :param Ei: optical signal at the input of the fiber
    :param L: fiber length [km]
    :param alpha: loss coeficient [dB/km]
    :param D: chromatic dispersion parameter [ps/nm/km]   
    :param Fc: carrier frequency [Hz]
    :param Fs: sampling frequency [Hz]
    
    :return Eo: optical signal at the output of the fiber
    """
    #c  = 299792458   # speed of light [m/s](vacuum)    
    c_kms = const.c/1e3
    λ  = c_kms/Fc
    α  = alpha/(10*np.log10(np.exp(1)))
    β2 = -(D*λ**2)/(2*np.pi*c_kms)
    
    Nfft = len(Ei)

    ω = 2*np.pi*Fs*fftfreq(Nfft)
    ω = ω.reshape(ω.size,1)
    
    try:
        Nmodes = Ei.shape[1]
    except IndexError:
        Nmodes = 1
        Ei = Ei.reshape(Ei.size,Nmodes)

    ω = np.tile(ω,(1, Nmodes))
    Eo = ifft(fft(Ei,axis=0)*np.exp(-α*L + 1j*(β2/2)*(ω**2)*L), axis=0)
    
    if Nmodes == 1:
        Eo = Eo.reshape(Eo.size,)
        
    return Eo

def balancedPD(E1, E2, R=1):
    '''
    Balanced photodetector (BPD)
    
    :param E1: input field [nparray]
    :param E2: input field [nparray]
    :param R: photodiode responsivity [scalar]
    
    :return: balanced photocurrent
    '''
    assert R > 0, 'PD responsivity should be a positive scalar'
    assert E1.size == E2.size, 'E1 and E2 need to have the same size'
    
    i1 = R*E1*np.conj(E1)
    i2 = R*E2*np.conj(E2)    

    return i1-i2

def hybrid_2x4_90deg(E1, E2):
    '''
    Optical 2 x 4 90° hybrid
    
    :param E1: input signal field [nparray]
    :param E2: input LO field [nparray]
        
    :return: hybrid outputs
    '''
    assert E1.size == E2.size, 'E1 and E2 need to have the same size'
    
    # optical hybrid transfer matrix    
    T = np.array([[ 1/2,  1j/2,  1j/2, -1/2],
                  [ 1j/2, -1/2,  1/2,  1j/2],
                  [ 1j/2,  1/2, -1j/2, -1/2],
                  [-1/2,  1j/2, -1/2,  1j/2]])
    
    Ei = np.array([E1, np.zeros((E1.size,)), 
                   np.zeros((E1.size,)), E2])    
    
    Eo = T@Ei
    
    return Eo
    
def coherentReceiver(Es, Elo, Rd=1):
    '''
    Single polarization coherent optical front-end
    
    :param Es: input signal field [nparray]
    :param Elo: input LO field [nparray]
    :param Rd: photodiode resposivity [scalar]
    
    :return: downconverted signal after balanced detection    
    '''
    assert Rd > 0, 'PD responsivity should be a positive scalar'
    assert Es.size == Elo.size, 'Es and Elo need to have the same size'
    
    # optical 2 x 4 90° hybrid 
    Eo = hybrid_2x4_90deg(Es, Elo)
        
    # balanced photodetection
    sI = balancedPD(Eo[1,:], Eo[0,:], Rd)
    sQ = balancedPD(Eo[2,:], Eo[3,:], Rd)
    
    return sI + 1j*sQ

def edfa(Ei, G, nf, Fc, Fs):
    '''
    Simple EDFA model
    
    '''
    nf_lin   = 10**(nf/10)
    G_lin    = 10**(G/10)
    nsp      = (G_lin*nf_lin - 1)/(2*(G_lin - 1))
    N_ase    = (G_lin - 1)*nsp*const.h*Fc
    p_noise  = N_ase*Fs    
    noise    = np.random.normal(0, np.sqrt(p_noise), Ei.shape) + 1j*np.random.normal(0, np.sqrt(p_noise), Ei.shape)
    return Ei*np.sqrt(G_lin) + noise

def ssfm(Ei, Fs, Ltotal, Lspan, hz=0.5, alpha=0.2, gamma=1.3, D=16, Fc=193.1e12, amp='edfa', NF=4.5):      
    '''
    Split-step Fourier method (symmetric, single-pol.)

    :param Ei: input signal
    :param Ltotal: total fiber length [km]
    :param Lspan: span length [km]
    :param hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :param alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :param D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :param gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :param amp: 'edfa', 'ideal', or 'None. [default:'edfa']
    :param NF: edfa noise figure [dB] [default: 4.5 dB]
    :param Fc: carrier frequency [Hz] [default: 193.1e12 Hz]
    :param Fs: sampling frequency [Hz]

    :return Ech: propagated signal
    '''             
    #c = 299792458   # speed of light (vacuum)
    c_kms = const.c/1e3
    λ  = c_kms/Fc
    α  = alpha/(10*np.log10(np.exp(1)))
    β2 = -(D*λ**2)/(2*np.pi*c_kms)
    γ  = gamma
            
    Nfft = len(Ein)

    ω = 2*np.pi*Fs*fftfreq(Nfft)
    
    Nspans = int(np.floor(Ltotal/Lspan))
    Nsteps = int(np.floor(Lspan/hz))
    
    Ech = Ei.reshape(len(Ei),)  
      
    linOperator = np.exp(-(α/2)*(hz/2) + 1j*(β2/2)*(ω**2)*(hz/2))
    
    for spanN in tqdm(range(1, Nspans+1)):   
        Ech = fft(Ech) #single-polarization field
        
        # fiber propagation step
        for stepN in range(1, Nsteps+1):            
            # First linear step (frequency domain)
            Ech = Ech*linOperator            

            # Nonlinear step (time domain)
            Ech = ifft(Ech)
            Ech = Ech*np.exp(1j*γ*(Ech*np.conj(Ech))*hz)

            # Second linear step (frequency domain)
            Ech = fft(Ech)       
            Ech = Ech*linOperator           

        # amplification step
        Ech = ifft(Ech)
        if amp =='edfa':
            Ech = edfa(Ech, alpha*Lspan, NF, Fc, Fs)
        elif amp =='ideal':
            Ech = Ech*np.exp(α/2*Nsteps*hz)
        elif amp == None:
            Ech = Ech*np.exp(0);         
          
    return Ech.reshape(len(Ech), 1)
