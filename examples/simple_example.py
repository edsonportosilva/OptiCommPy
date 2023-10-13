
import numpy as np
from commpy.utilities  import upsample
from optic.models.devices import mzm, photodiode
from optic.models.channels import linearFiberChannel
from optic.comm.modulation import modulateGray
from optic.comm.metrics import ook_BERT
from optic.dsp.core import firFilter, pulseShape
from optic.core import parameters
from scipy.special import erfc

## Intensity modulation (IM) with On-Off Keying (OOK)

# simulation parameters
SpS = 16    # samples per symbol
M = 2       # order of the modulation format
Rs = 10e9   # Symbol rate
Fs = SpS*Rs # Signal sampling frequency (samples/second)
np.random.seed(seed=256) # fixing the seed to get reproducible results

# typical NRZ pulse
pulse = pulseShape('nrz', SpS)
pulse = pulse/max(abs(pulse)) # normalize to 1 Vpp

# MZM parameters
Vpi = 2
Vb = -Vpi/2
Pi_dBm = 0 # laser optical power at the input of the MZM in dBm
Pi = 10**(Pi_dBm/10)*1e-3 # convert from dBm to W

# linear fiber optical channel parameters
paramCh = parameters()
paramCh.L = 90         # total link distance [km]
paramCh.alpha = 0.2    # fiber loss parameter [dB/km]
paramCh.D = 16         # fiber dispersion parameter [ps/nm/km]
paramCh.Fc = 193.1e12  # central optical frequency [Hz]
paramCh.Fs = Fs

# photodiode parameters
paramPD = parameters()
paramPD.ideal = False
paramPD.B = Rs
paramPD.Fs = Fs

## Simulation
print('\nStarting simulation...')

# generate pseudo-random bit sequence
bitsTx = np.random.randint(2, size=100000)

# generate 2-PAM modulated symbol sequence
symbTx = modulateGray(bitsTx, M, 'pam')    

# upsampling
symbolsUp = upsample(symbTx, SpS)

# pulse shaping
sigTx = firFilter(pulse, symbolsUp)

# optical modulation
Ai = np.sqrt(Pi)*np.ones(sigTx.size) # ideal cw laser envelope
sigTxo = mzm(Ai, sigTx, Vpi, Vb)

# linear fiber channel model
sigCh = linearFiberChannel(sigTxo, paramCh)

# noisy photodiode (thermal noise + shot noise + bandwidth limitation)
I_Rx = photodiode(sigCh, paramPD)
I_Rx = I_Rx/np.std(I_Rx)

# capture samples in the middle of signaling intervals
I_Rx = I_Rx[0::SpS]

# calculate the BER and Q-factor
BER, Q = ook_BERT(I_Rx,seed=256)

print('\nReceived signal parameters:')
print(f'Q-factor = {Q:.2f} ')
print(f'BER = {BER:.2e}')

# theoretical error probability from Q-factor
Pb = 0.5 * erfc(Q / np.sqrt(2))  
print(f'Pb = {Pb:.2e}\n')
