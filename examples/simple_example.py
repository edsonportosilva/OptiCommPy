import numpy as np
from optic.models.devices import mzm, photodiode
from optic.models.channels import linearFiberChannel
from optic.comm.modulation import modulateGray
from optic.comm.metrics import ook_BERT
from optic.dsp.core import firFilter, pulseShape, upsample, pnorm
from optic.utils import parameters, dBm2W
from scipy.special import erfc
import matplotlib.pyplot as plt

## Intensity modulation (IM) with On-Off Keying (OOK)

# simulation parameters
SpS = 16  # samples per symbol
M = 2  # order of the modulation format
Rs = 10e9  # Symbol rate
Fs = SpS * Rs  # Signal sampling frequency (samples/second)
Pi_dBm = 0  # laser optical power at the input of the MZM in dBm
Pi = dBm2W(Pi_dBm)  # convert from dBm to W

# typical NRZ pulse
pulse = pulseShape("nrz", SpS)
pulse = pulse / max(abs(pulse))  # normalize to 1 Vpp

# MZM parameters
paramMZM = parameters()
paramMZM.Vpi = 2
paramMZM.Vb = -paramMZM.Vpi / 2

# linear fiber optical channel parameters
paramCh = parameters()
paramCh.L = 90  # total link distance [km]
paramCh.alpha = 0.2  # fiber loss parameter [dB/km]
paramCh.D = 16  # fiber dispersion parameter [ps/nm/km]
paramCh.Fc = 193.1e12  # central optical frequency [Hz]
paramCh.Fs = Fs

# photodiode parameters
paramPD = parameters()
paramPD.ideal = False
paramPD.B = Rs
paramPD.Fs = Fs

## Simulation
print("\nStarting simulation...", end="")

# generate pseudo-random bit sequence
np.random.seed(seed=123)  # fixing the seed to get reproducible results
bitsTx = np.random.randint(2, size=100000)

# generate 2-PAM modulated symbol sequence
symbTx = modulateGray(bitsTx, M, "pam")

# upsampling
symbolsUp = upsample(symbTx, SpS)

# pulse shaping
sigTx = firFilter(pulse, symbolsUp)

# optical modulation
Ai = np.sqrt(Pi)  # ideal cw laser constant envelope
sigTxo = mzm(Ai, sigTx, paramMZM)

# linear fiber channel model
sigCh = linearFiberChannel(sigTxo, paramCh)

# noisy PD (thermal noise + shot noise + bandwidth limit)
I_Rx = photodiode(sigCh, paramPD)

# capture samples in the middle of signaling intervals
I_Rx = I_Rx[0::SpS]

# calculate the BER and Q-factor
BER, Q = ook_BERT(I_Rx)
print("simulation completed.")

print("\nTransmission performance metrics:")
print(f"Q-factor = {Q:.2f} ")
print(f"BER = {BER:.2e}")

# theoretical error probability from Q-factor
Pb = 0.5 * erfc(Q / np.sqrt(2))
print(f"Pb = {Pb:.2e}\n")

# Generate BER vs Pin curve

# simulation parameters
SpS = 16        # Samples per symbol
M = 2           # order of the modulation format
Rs = 10e9       # Symbol rate (for the OOK case, Rs = Rb)
Fs = SpS*Rs     # Signal sampling frequency (samples/second)
Ts = 1/Fs       # Sampling period

# MZM parameters
paramMZM = parameters()
paramMZM.Vpi = 2
paramMZM.Vb = -paramMZM.Vpi/2

# typical NRZ pulse
pulse = pulseShape('nrz', SpS)
pulse = pulse/max(abs(pulse))

# photodiode parameters
paramPD = parameters()
paramPD.ideal = False
paramPD.B = Rs
paramPD.Fs = Fs
   
powerValues = np.arange(-30,-14) # power values at the input of the pin receiver
BER = np.zeros(powerValues.shape)
Q = np.zeros(powerValues.shape)

for indPi, Pi_dBm in enumerate(powerValues):
    
    Pi = dBm2W(Pi_dBm+3) # optical signal power in W at the MZM input

    # generate pseudo-random bit sequence
    np.random.seed(seed=123)  # fixing the seed to get reproducible results
    bitsTx = np.random.randint(2, size=10**5)
    n = np.arange(0, bitsTx.size)

    # generate ook modulated symbol sequence
    symbTx = modulateGray(bitsTx, M, 'pam')    
    symbTx = pnorm(symbTx) # power normalization

    # upsampling
    symbolsUp = upsample(symbTx, SpS)

    # pulse formatting
    sigTx = firFilter(pulse, symbolsUp)

    # optical modulation
    Ai = np.sqrt(Pi)
    sigTxo = mzm(Ai, sigTx, paramMZM)

    # pin receiver
    I_Rx = photodiode(sigTxo.real, paramPD)
    I_Rx = I_Rx/np.std(I_Rx)

    # capture samples in the middle of signaling intervals
    I_Rx = I_Rx[0::SpS]

    # calculate the BER and Q-factor
    BER[indPi], Q[indPi] = ook_BERT(I_Rx)

    print(f'\nPin = {Pi_dBm:.2f} dBm')
    print('\nReceived signal parameters:')
    print(f'Q-factor = {Q[indPi]:.2f} ')
    print(f'BER = {BER[indPi]:.2e}')


plt.figure()
plt.plot(powerValues, np.log10(BER),'-o',label='BER')
plt.grid()
plt.ylabel('log10(BER)')
plt.xlabel('Pin (dBm)')
plt.title('Bit-error performance vs input power at the pin receiver')
plt.legend()
plt.ylim(-10,0)
plt.xlim(min(powerValues), max(powerValues))
plt.show()

plt.figure()
plt.plot(powerValues, 10*np.log10(Q),'-o',label='Q-factor')
plt.grid()
plt.ylabel('Q-factor (dB)')
plt.xlabel('Pin (dBm)')
plt.title('Q-factor performance vs input power at the pin receiver')
plt.legend()
#plt.ylim(-10,0)
plt.xlim(min(powerValues), max(powerValues))
plt.show()