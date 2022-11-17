import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from tqdm.notebook import tqdm
import matplotlib.mlab as mlab
import scipy as sp

from commpy.utilities  import upsample
from optic.models import mzm, photodiode
from optic.metrics import signal_power
from optic.dsp import firFilter, pulseShape, lowPassFIR
from optic.core import parameters
from optic.plot import eyediagram
from optic.amplification import edfaSM, OSA

from optic.tx import simpleWDMTx
from scipy.constants import c, Planck
from numpy.fft import fft,fftfreq

import timeit

# EDFA parameters
param_edfa = parameters()
param_edfa.type  = "AGC"
param_edfa.value = 20
param_edfa.nf    = 5
param_edfa.forPump = {'pump_signal': np.array([200e-3]), 'pump_lambda': np.array([980e-9])}
param_edfa.bckPump = {'pump_signal': np.array([000e-3]), 'pump_lambda': np.array([980e-9])}
param_edfa.type = 'AGC'
param_edfa.file = 'C:\\Users\\Adolfo\\Documents\\GitHub\\OptiCommPy\\jupyter\\giles_MP980.dat'
param_edfa.fileunit = 'nm'
param_edfa.gmtc = 'Bessel'

# Transmitter parameters:
paramTx = parameters()
paramTx.M   = 4             # order of the modulation format
paramTx.Rs  = 40e9          # symbol rate [baud]
paramTx.SpS = 256           # samples per symbol
paramTx.Nbits = 2**12       # total number of bits per polarization
paramTx.pulse = 'rrc'       # pulse shaping filter
paramTx.Ntaps = 1024        # number of pulse shaping filter coefficients
paramTx.alphaRRC = 0.01     # RRC rolloff
paramTx.Pch_dBm = -15       # power per WDM channel [dBm]
paramTx.Nch     = 40        # number of WDM channels
paramTx.Fc      = c/1550e-9 # central optical frequency of the WDM spectrum
paramTx.freqSpac = 200e9    # WDM grid spacing
paramTx.Nmodes = 2          # number of signal modes [2 for polarization multiplexed signals]

# generate WDM signal
sigWDM_Tx, symbTx_, paramTx = simpleWDMTx(paramTx)

Fs = paramTx.Rs*paramTx.SpS
Tw = 1/Fs * (paramTx.Nbits / np.log2(paramTx.M)) * paramTx.SpS
simOpticalBand = (Fs*(c/paramTx.Fc)**2)/c

print('Sample rate [THz]: %5.3f' %(1e-12*Fs))
print('Time window [ns]:  %5.3f' %(1e9*Tw))

print('Central wavelength [nm]: %6.2f' %(1e9*c/paramTx.Fc))
print('Simulation window  [nm]: %f - [%6.2f nm - %6.2f nm]' 
      %(1e9*simOpticalBand, 1e9*(c/paramTx.Fc-simOpticalBand/2), 1e9*(c/paramTx.Fc+simOpticalBand/2)))
print('Distância entre pontos [GHz]: %f' %(1e-9/Tw))
print('Number of points: %d' %(Fs*Tw))
print('Number of modes: %d' %(paramTx.Nmodes))
print('Average power - TX [mW] : %.3f mW' %(1000*np.sum(np.mean(sigWDM_Tx * np.conj(sigWDM_Tx), axis = 0).real)))
print('Average power - TX [dBm] : %.3f dBm' %(10*np.log10(np.sum(1000*np.mean(sigWDM_Tx * np.conj(sigWDM_Tx), axis = 0).real))))

start = timeit.timeit()
Eout  = edfaSM(sigWDM_Tx, Fs, paramTx.Fc, param_edfa)
end = timeit.timeit()
print(end - start)

#lenFrqSg, isy = np.shape(Eout)
#freqSgn = Fs * fftfreq(lenFrqSg) + paramTx.Fc
#EinFFT  = fft(sigWDM_Tx, axis = 0)/lenFrqSg
#EoutFFT = fft(Eout, axis = 0)/lenFrqSg
#plt.plot(1e6*c/freqSgn, 10*np.log10(1000*np.abs(EoutFFT)**2))
#plt.plot(1e6*c/freqSgn, 10*np.log10(1000*np.abs(EinFFT)**2))