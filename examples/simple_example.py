
import numpy as np
from commpy.utilities  import upsample
from optic.models.devices import mzm, photodiode
from optic.models.channels import linearFiberChannel
from optic.comm.modulation import modulateGray
from optic.comm.metrics import ook_BERT
from optic.dsp.core import firFilter, pulseShape
from optic.utils import parameters, dBm2W
from scipy.special import erfc

## Intensity modulation (IM) with On-Off Keying (OOK)

# simulation parameters
SpS = 16    # samples per symbol
M = 2       # order of the modulation format
Rs = 10e9   # Symbol rate
Fs = SpS*Rs # Signal sampling frequency (samples/second)
Pi_dBm = 0  # laser optical power at the input of the MZM in dBm
Pi = dBm2W(Pi_dBm) # convert from dBm to W

# typical NRZ pulse
pulse = pulseShape('nrz', SpS)
pulse = pulse/max(abs(pulse)) # normalize to 1 Vpp

# MZM parameters
paramMZM = parameters()
paramMZM.Vpi = 2
paramMZM.Vb = -paramMZM.Vpi/2

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
print('\nStarting simulation...', end="")

# generate pseudo-random bit sequence
np.random.seed(seed=123) # fixing the seed to get reproducible results
bitsTx = np.random.randint(2, size=100000)

# generate 2-PAM modulated symbol sequence
symbTx = modulateGray(bitsTx, M, 'pam')    

# upsampling
symbolsUp = upsample(symbTx, SpS)

# pulse shaping
sigTx = firFilter(pulse, symbolsUp)

# optical modulation
Ai = np.sqrt(Pi) # ideal cw laser constant envelope
sigTxo = mzm(Ai, sigTx, paramMZM)

# linear fiber channel model
sigCh = linearFiberChannel(sigTxo, paramCh)

# noisy PD (thermal noise + shot noise + bandwidth limit)
I_Rx = photodiode(sigCh, paramPD)

# capture samples in the middle of signaling intervals
I_Rx = I_Rx[0::SpS]

# calculate the BER and Q-factor
BER, Q = ook_BERT(I_Rx)
print('simulation completed.')

print('\nReceived signal parameters:')
print(f'Q-factor = {Q:.2f} ')
print(f'BER = {BER:.2e}')

# theoretical error probability from Q-factor
Pb = 0.5 * erfc(Q / np.sqrt(2))  
print(f'Pb = {Pb:.2e}\n')
