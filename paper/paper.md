---
title: 'OptiCommPy: Open-source Simulation of Fiber Optic Communications with Python'
tags:
  - Python
  - Optical Communications
  - Digital Signal Processing
  - Photonics  
authors:
  - name: Edson Porto da Silva
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Adolfo Fernandes Herbster
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1"  
affiliations:
 - name: Electrical Engineering Department, Federal University of Campina Grande (UFCG), Brazil
   index: 1
date: 01 February 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

# Statement of need

Optical fiber communication dominates the transmission of high-speed data traffic, owing to various physical, engineering, and economic factors. Worldwide efforts are continuously being made to research and develop optical communication technologies that can support both current and future Internet infrastructure. The expansion of optical networks necessitates a swift transition from scientific breakthroughs in research labs to telecommunications industry products and solutions. Furthermore, the ever-increasing demand for bandwidth and connectivity places constant pressure on the development of faster and more efficient optical fiber communications [@Winzer2017].

Today, optical communication systems engineering is a multidisciplinary field encompassing various areas of science and technology, including laser science, photonic devices, fiber optics modeling and engineering, digital signal processing, and communications theory. As we approach the limits of information transmission through optical fibers, more sophisticated engineering is required for the construction of optical transmitters and receivers, involving advanced digital signal processing (DSP) [@Essiambre2010], [@Savory2010]. The emergence of high-speed application-specific integrated circuits (ASICs) and advanced DSP algorithms has propelled coherent optical transmission systems to the forefront of high-capacity transmission via optical fibers [@Sun2020].

Whether in the research or development stages, the study of optical communication systems typically necessitates the use of robust computational models to simulate various aspects of the system. For instance, it may be essential to comprehend how information-carrying signals transmitted over fibers will be affected by propagation phenomena such as chromatic dispersion (CD), polarization mode dispersion (PMD), nonlinear effects, and noise [@Agrawal2002]. This information ultimately determines the performance metrics of the transmission system, which play a crucial role in selecting the most suitable technology to become an industrial standard.

Presently, a variety of optical communication simulation toolboxes are accessible. While the majority of these are proprietary software packages [@OptiSystem], [@vpi], [@optsim], a few are open-source but are designed to operate within proprietary software environments such as Matlab\textsuperscript{\textregistered} [@robochameleon], [@optilux].

In this paper, we present OptiCommPy, an open-source Python package designed for simulating optical communication systems and subsystems. OptiCommPy is freely accessible, providing researchers, students, and engineers with the option to simulate various optical communication systems at the physical layer. Additionally, the toolbox incorporates numerous digital signal processing (DSP) algorithms, particularly essential for coherent optical systems.

# Software description

The module structure of the OptiCommPy package is illustrated in Fig.~\ref{fig:pckg-struct}. At the top level, the package is named `optic`, containing five sub-packages: `comm, models, dsp, utils`, and `plot`.

![Software description](OptiCommPy.png)

The `comm` sub-package comprises three modules designed for implementing various digital modulation and demodulation schemes [@Proakis2001], including pulse amplitude modulation (PAM), quadrature amplitude modulation (QAM), phase-shift keying (PSK), and on-off keying (OOK). Evaluating the performance of these diverse digital communication schemes is made possible through different metrics, such as bit-error-rate (BER), symbol-error-rate (SER), error vector magnitude (EVM), mutual information (MI), and generalized mutual information (GMI) [@Alvarado2018], all available within the `comm.metrics` module.

The `models` sub-package contains the majority of the mathematical/physical models used to build OptiCommPy simulations. Within the `models.devices` module, one can access models for a range of optical devices, encompassing optical Mach-Zehnder modulators, photodiodes, optical hybrids, optical coherent receivers, and more. These functions serve as fundamental building blocks for constructing simulations of optical transmitters and receivers. In the `models.channels` module, a collection of mathematical models for the fiber optic channel is provided, spanning from basic additive white Gaussian noise (AWGN) and linear propagation models to more sophisticated non-linear propagation models rooted in variants of the split-step Fourier method (SSFM)[@Agrawal2002]. In particular, it includes an implementation of the Manakov model to simulate nonlinear transmission over a fiber optic channel with polarization-multiplexing [@Marcuse1997a]. Certain computationally intensive models, such as the Manakov SSFM, have a CuPy-based version [@cupy_learningsys2017] accessible via `models.modelsGPU`, designed specifically for execution with CUDA GPU acceleration [@cuda].

The `dsp` sub-package is a collection of DSP algorithms ranging from basic signal processing operations, such as linear finite impulse response filtering, to more advanced algorithms, e.g. blind adaptive equalization. Currently, the `dsp` sub-package contains three specialized modules: `dsp.clockRecovery` provides algorithms for clock and timing synchronization; `dsp.equalization` contains implementations of common digital equalization algorithms used in optical communications; `dsp.carrierRecovery` provides algorithms for carrier frequency and phase recovery. These sub-packages covers all the basic DSP functionalities required in most of the modern coherent optical transceivers.

Finally, the `utils` and the `plot` sub-packages provide functions that implement a few general utilities and custom plotting functions to visualize signals (eyediagram plots, constellation plots, etc).

# Basic example of usage
In this section, a basic illustrative example of the usage of the OptiCommPy package is presented. A schematic for the transmission system being simulated is shown in Fig.~\ref{fig:basic-IMDD}. It consists of a 10~Gb/s NRZ OOK transmission over 100~km of optical fiber with direct detection at the receiver assuming a photodiode with 10 GHz of bandwidth. The fiber channel is modeled as a linear propagation medium exhibiting loss and chromatic dispersion. All the simulation parameters are listed in Table~\ref{tab:simparam}.


| Parameter                                 | Value          |
|-------------------------------------------|----------------|
| Order of the modulation format ($M$)      | 2              |
| Symbol rate ($R_s$)                       | 10 Gb/s        |
| Pulse shape                               | NRZ            |
| Laser output power ($P_i$)                | $3$ dBm        |
| MZM $V_\pi$ voltage                       | 2.0 V          |
| MZM bias voltage ($V_b$)                  | -1.0 V         |
| Total link distance ($L$)                 | 100 km         |
| Fiber loss parameter ($\alpha$)           | 0.2 dB/km      |
| Fiber dispersion parameter ($D$)          | 16 ps/nm/km    |
| Central optical frequency ($F_c$)         | $193.1$ THz    |
| Photodiode bandwidth ($B$)                | $10$ GHz       |
| Simulation sampling rate ($F_s$)          | 160 GSamples/s |


\subsection{Building the simulation setup}
To build the corresponding simulation setup with OptiCommPy, first the necessary functions need to be imported as shown in listing ~\ref{lis:setup1}. 


```python
import numpy as np    
from optic.models.devices import mzm, photodiode
from optic.models.channels import linearFiberChannel
from optic.comm.modulation import modulateGray
from optic.comm.metrics import bert
from optic.dsp.core import firFilter, pulseShape, upsample
from optic.utils import parameters, dBm2W
from scipy.special import erfc
```

Given that each element in the simulation requires a specific set of parameters, the subsequent step involves defining all necessary parameters to establish the desired simulation setup, as illustrated in listing~\ref{lis:setup2}. In this particular example, we assume that the electrical signal driving the optical modulator is a binary non-return-to-zero (NRZ) signal. The optical modulator is modeled as an ideal Mach-Zehnder Modulator (MZM). The fiber channel is considered linear and affected only by loss and chromatic dispersion. Lastly, at the receiver, a photodiode with a frequency response limited to 10~GHz is configured, subject to the influence of thermal and shot noise.

```python
# Intensity modulation direct-detection 
# (IM/DD) with On-Off Keying (OOK)

# simulation parameters
SpS = 16    # samples per symbol
M = 2       # order of the modulation format
Rs = 10e9   # Symbol rate
Fs = SpS*Rs # Signal sampling frequency (samples/second)
Pi_dBm = 3  # laser optical power at the input of the MZM in dBm
Pi = dBm2W(Pi_dBm) # convert from dBm to W

# NRZ pulse
pulse = pulseShape('nrz', SpS) # typical NRZ pulse shape
pulse = pulse/max(abs(pulse))  # normalize to 1 Vpp

# MZM parameters
paramMZM = parameters()
paramMZM.Vpi = 2              # modulator's Vpi voltage
paramMZM.Vb = -paramMZM.Vpi/2 # bias at the quadrature point

# linear fiber optical channel parameters
paramCh = parameters()
paramCh.L = 100        # total link distance [km]
paramCh.alpha = 0.2    # fiber loss parameter [dB/km]
paramCh.D = 16         # fiber dispersion parameter [ps/nm/km]
paramCh.Fc = 193.1e12  # central optical frequency [Hz]
paramCh.Fs = Fs        # sampling frequency [samples/s]

# photodiode parameters
paramPD = parameters()
paramPD.ideal = False  # w/o noise, w/o bandwidth limitation?
paramPD.B = Rs         # photodiode bandwidth limitation [Hz]
paramPD.Fs = Fs        # sampling frequency [samples/s]
```

After defining all the required parameters, the core simulation code, as depicted in listing.~\ref{lis:setup3}, establishes the signal flow through each OptiCommPy model, extending from the initial bit source to the direct-detection optical receiver. Starting with a pseudorandom bit sequence, the signal undergoes upsampling and pulse shaping, resulting in a series of binary non-return-to-zero (NRZ) pulses. The corresponding eyediagram, located in the bottom-left corner of Fig.~\ref{fig:basic-IMDD}, visually portrays this signal. The electrical NRZ signal then drives the MZM, biased at the quadrature point. An ideal laser generates the optical carrier, free from phase and intensity noise. This optical signal, upon exiting the MZM, traverses the linear fiber channel, where it is subjected to losses and chromatic dispersion. Finally, the signal is received by the photodiode. An illustrative eyediagram of the resulting received signal can be observed in the bottom-right corner of Fig.~\ref{fig:basic-IMDD}.


```python
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
```

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References