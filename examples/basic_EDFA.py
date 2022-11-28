import numpy as np
import matplotlib.pyplot as plt

from optic.core import parameters
from optic.tx import simpleWDMTx
from optic.amplification import edfaSM, OSA

from scipy.constants import c
from numpy.fft import fft,fftfreq

import logging as logg
logg.getLogger().setLevel(logg.INFO)
logg.basicConfig(format='%(message)s')

# EDFA parameters
param_edfa = parameters()
param_edfa.type     = "none"
param_edfa.value    = 20 #dB
param_edfa.forPump  = {'pump_signal': np.array([100e-3]), 'pump_lambda': np.array([980e-9])}
param_edfa.bckPump  = {'pump_signal': np.array([000e-3]), 'pump_lambda': np.array([980e-9])}
param_edfa.file     = '..\\OptiCommPy\\optic\\ampParams\\giles_MP980.dat'
param_edfa.fileunit = 'nm'
param_edfa.gmtc     = 'Bessel'
param_edfa.algo     = 'Giles_spatial'
param_edfa.tol      = 0.05
param_edfa.tolCtrl  = 0.5

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

lenFrqSg,isy = np.shape(sigWDM_Tx)
Fs = paramTx.Rs*paramTx.SpS
#Tw = 1/Fs * (paramTx.Nbits / np.log2(paramTx.M)) * paramTx.SpS
simOpticalBand = (Fs*(c/paramTx.Fc)**2)/c

# information TX
print('Sample rate [THz]: %5.3f' %(1e-12*Fs))
print('Time window [ns]:  %5.3f' %(1e9*lenFrqSg/Fs))
print('Central wavelength [nm]: %6.2f' %(1e9*c/paramTx.Fc))
print('Simulation window  [nm]: %f - [%6.2f nm - %6.2f nm]' 
      %(1e9*simOpticalBand, 1e9*(c/paramTx.Fc-simOpticalBand/2), 1e9*(c/paramTx.Fc+simOpticalBand/2)))
print('Distância entre pontos [GHz]: %f' %(1e-9*Fs/lenFrqSg))
print('Number of points: %d' %(lenFrqSg))
print('Number of modes: %d' %(paramTx.Nmodes))
print('Average power - TX [mW] : %.3f mW' %(1000*np.sum(np.mean(sigWDM_Tx * np.conj(sigWDM_Tx), axis = 0).real)))
print('Average power - TX [dBm] : %.3f dBm' %(10*np.log10(np.sum(1000*np.mean(sigWDM_Tx * np.conj(sigWDM_Tx), axis = 0).real))))

# amplification
Eout, PumpF, PumpB = edfaSM(sigWDM_Tx, Fs, paramTx.Fc, param_edfa)

# information amp
print('Average power - RX amp [mW] : %.3f mW' %(1000*np.sum(np.mean(sigWDM_Tx * np.conj(Eout), axis = 0).real)))
print('Average power - RX amp [dBm] : %.3f dBm' %(10*np.log10(np.sum(1000*np.mean(Eout * np.conj(Eout), axis = 0).real))))

# plot signal
lenFrqSg, isy = np.shape(Eout)
freqSgn = Fs * fftfreq(lenFrqSg) + paramTx.Fc

EinFFT  = fft(sigWDM_Tx, axis = 0)/lenFrqSg
EoutFFT = fft(Eout, axis = 0)/lenFrqSg

plt.plot(1e9*c/freqSgn, 10*np.log10(1000*np.abs(EinFFT)**2))
plt.xlabel('Wavelength [nm]')
plt.ylabel('Optical power [dBm]')
plt.xlim([1515,1585])
plt.ylim([-100,0])
plt.grid()

plt.plot(1e9*c/freqSgn, 10*np.log10(1000*np.abs(EoutFFT)**2))
plt.xlabel('Wavelength [nm]')
plt.ylabel('Optical power [dBm]')
plt.xlim([1515,1585])
plt.ylim([-100,0])
plt.grid()