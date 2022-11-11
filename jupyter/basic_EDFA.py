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

# simulation parameters
SpS = 16

Rs = 10e9 # Symbol rate (for OOK case Rs = Rb)
Tsymb = 1/Rs # Symbol period in seconds
Fs = 1/(Tsymb/SpS) # Signal sampling frequency (samples/second)
Ts = 1/Fs # Sampling period
Fc = 193.1e12

Pi_dBm = 0 # optical signal power at modulator input in dBm

# MZM parameters
Vπ = 2
Vb = -Vπ/2
Pi = 10**(Pi_dBm/10)*1e-3 # optical signal power in W at the MZM input

# generate pseudo-random bit sequence
bitsTx = np.random.randint(2, size=100000)
n = np.arange(0, bitsTx.size)

# map bits to electrical pulses
symbTx = 2*bitsTx-1
symbTx = symbTx/np.sqrt(signal_power(symbTx))

# upsampling
symbolsUp = upsample(symbTx, SpS)

# typical NRZ pulse
pulse = pulseShape('nrz', SpS)
pulse = pulse/max(abs(pulse))

# pulse formatting
sigTx = firFilter(pulse, symbolsUp)

# optical modulation
Ai = np.sqrt(Pi)*np.ones(sigTx.size)
sigTxo = mzm(Ai, sigTx, Vπ, Vb)
sigTxo = np.reshape(sigTxo, (np.size(sigTxo), 1))

param_edfa = parameters()
param_edfa.type  = "AGC"
param_edfa.value = 20
param_edfa.nf    = 5
param_edfa.forPump = {'pump_signal': np.array([100e-3]), 'pump_lambda': np.array([980e-9])}
param_edfa.type = 'AGC'
param_edfa.file = 'C:\\Users\\Adolfo\\Documents\\GitHub\\OptiCommPy\\jupyter\\giles_MP980.dat'
param_edfa.fileunit = 'nm'
param_edfa.gmtc = 'Bessel'

Ei, param_edfa = edfaSM(sigTxo, Fs, Fc, param_edfa)