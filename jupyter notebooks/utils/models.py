import numpy as np
from numpy.fft import fft, ifft, fftfreq

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
    c  = 299792458   # speed of light [m/s](vacuum)    
    c_kms = c/1e3
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
