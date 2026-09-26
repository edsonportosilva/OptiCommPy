
.. image:: ../../figures/logo_OptiCommPy.jpg
   :alt: OptiCommPy logo
   :width: 75%  
   :align: center

**Open-source simulation of fiber optic communication systems with Python**

**OptiCommPy** is a Python framework to simulate fiber optic communication systems, from
the bits at the transmitter to the performance metrics at the receiver. It brings together
physical models of optical and optoelectronic devices, linear and nonlinear fiber propagation,
optical amplification, receiver digital signal processing (DSP), forward error correction (FEC)
and performance metrics, so that complete **IM-DD** and **coherent** optical links can be built
and studied in a few lines of code.

It is designed for **students** learning optical communications, **researchers** prototyping and
benchmarking new DSP algorithms or system concepts, and **engineers** who need a transparent,
scriptable simulation environment.

.. image:: ../../figures/eyeDisp.gif
     :alt: Animated density eye diagram of a two-level signal
     :width: 45%      
.. image:: ../../figures/40GOOK_spectrum.jpg
     :alt: Simulated optical spectrum (power spectral density versus frequency) of a 40G OOK signal centered at 193.1 THz
     :width: 45% 
     

.. image:: ../../figures/DSP.jpg
     :alt: 16-QAM constellations of the received signal: as detected, after dispersion compensation, after adaptive equalization, and after carrier frequency and phase recovery
     :width: 600px  
     :align: center

|PyPI| |PyPI - Downloads| |Documentation Status| |DOI| |JOSS| |PyPI - All Downloads|

Why OptiCommPy?
---------------

- **End-to-end simulation**: transmitter, fiber channel, amplifiers, receiver front-end, DSP,
  FEC and metrics in a single package, with a consistent API.
- **Physically meaningful models**: devices and channels are described by their physical
  parameters (Vπ, extinction ratio, responsivity, noise figure, fiber loss, dispersion,
  nonlinearity, laser linewidth, ...).
- **Fast**: performance-critical routines are compiled with `Numba <https://numba.pydata.org/>`__,
  and the most demanding ones (split-step fiber propagation, digital backpropagation, blind
  phase search) also run on NVIDIA GPUs via `CuPy <https://cupy.dev/>`__.
- **Transparent and hackable**: plain NumPy/SciPy code that is easy to read, modify and extend,
  ideal for teaching and for prototyping new algorithms.
- **Peer reviewed and tested**: published in the
  `Journal of Open Source Software <https://doi.org/10.21105/joss.06600>`__, with an automated
  test suite.

Features
--------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Area
     - What is included
   * - **Modulation and sources**
     - M-PAM, OOK, square M-QAM, M-PSK and APSK constellations with Gray mapping; random bits,
       PRBS, CAZAC (Zadoff-Chu) sequences; probabilistic constellation shaping
       (Maxwell-Boltzmann); NRZ, RC and RRC pulse shaping; OFDM modulation and demodulation
   * - **Transmitters**
     - WDM transmitter with multiple channels and polarization modes, optical PAM transmitter,
       laser model with phase noise and relative intensity noise (RIN), DAC with quantization
       and ENOB
   * - **Optical and optoelectronic devices**
     - Phase modulator, Mach-Zehnder modulator (MZM), IQ modulator, polarization beam splitter,
       90° optical hybrid, variable optical attenuator, PIN photodiode (shot and thermal noise,
       bandwidth limitation), balanced photodetector, single- and dual-polarization coherent
       receivers with IQ imbalance and skew, ADC with jitter, quantization and ENOB
   * - **Fiber channel**
     - Linear fiber channel (loss and chromatic dispersion), nonlinear Schrödinger equation
       (NLSE) and Manakov models solved with the split-step Fourier method with adaptive step
       size, first-order perturbation models of nonlinear interference, AWGN channel
   * - **Optical amplification**
     - Simple EDFA model (gain and ASE noise) and an advanced EDFA model solving the erbium rate
       and propagation equations
   * - **Receiver DSP**
     - Resampling, matched filtering, Gardner clock recovery, chromatic dispersion compensation,
       N×N MIMO adaptive equalization (CMA, RDE, NLMS, DD-LMS, DA-RDE, RLS, DD-RLS), Manakov
       digital backpropagation, frequency offset estimation, carrier phase recovery (blind phase
       search, DD-PLL, Viterbi & Viterbi), sequence synchronization
   * - **IM-DD DSP**
     - Feedforward (FFE), decision feedback (DFE) and Volterra equalizers, maximum likelihood
       sequence estimation (MLSE)
   * - **Forward error correction**
     - LDPC encoding and decoding (sum-product and min-sum belief propagation, DVB-S2 codes,
       ALIST files) and Hamming codes
   * - **Performance metrics**
     - BER, SER, SNR, Q-factor, EVM, mutual information (MI), generalized mutual information
       (GMI) and normalized GMI (NGMI), log-likelihood ratios, theoretical BER/MI/GMI curves,
       OSNR evolution in multi-span links
   * - **Visualization**
     - Density constellation plots, eye diagrams, power spectral density, decision boundaries,
       animated constellations

Installation
------------

OptiCommPy requires Python 3.10 or newer. Install the latest release from
`PyPI <https://pypi.org/project/OptiCommPy/>`__:

.. code-block:: bash

   pip install OptiCommPy

or install the development version from GitHub:

.. code-block:: bash

   git clone https://github.com/edsonportosilva/OptiCommPy.git
   cd OptiCommPy
   pip install .

**GPU support (optional):** to run the GPU implementations, install
`CuPy <https://docs.cupy.dev/en/stable/install.html>`__ for your CUDA version, for example:

.. code-block:: bash

   pip install cupy-cuda12x

Dependencies: numpy>=1.24.4, scipy>=1.13.0, matplotlib>=3.7.0, numba>=0.54.0, tqdm>=4.64.1,
simple-pid>=1.0.1, mpl-scatter-density>=0.8, prettytable>=3.16.0, and optionally
cupy-cuda12x>=13.1.0 for GPU processing.

Quick start
-----------

Simulate a 10 Gb/s NRZ-OOK transmission over 100 km of optical fiber with a direct-detection
receiver, and measure its bit error rate:

.. code-block:: python

   import numpy as np
   from optic.comm.sources import bitSource
   from optic.comm.modulation import modulateGray
   from optic.comm.metrics import bert
   from optic.dsp.core import firFilter, pulseShape, upsample, anorm
   from optic.models.devices import mzm, photodiode
   from optic.models.channels import linearFiberChannel
   from optic.utils import parameters, dBm2W

   # 10 Gb/s NRZ-OOK over 100 km of fiber with direct detection
   SpS, Rs = 16, 10e9  # samples per symbol, symbol rate
   Fs = SpS * Rs       # sampling frequency

   paramBits = parameters()
   paramBits.nBits, paramBits.seed = 100_000, 123

   paramPulse = parameters()
   paramPulse.pulseType, paramPulse.SpS = "nrz", SpS

   paramMZM = parameters()
   paramMZM.Vpi, paramMZM.Vb = 2, -1

   paramCh = parameters()
   paramCh.L, paramCh.alpha, paramCh.D = 100, 0.2, 16  # km, dB/km, ps/nm/km
   paramCh.Fc, paramCh.Fs = 193.1e12, Fs

   paramPD = parameters()
   paramPD.ideal, paramPD.B, paramPD.Fs, paramPD.seed = False, Rs, Fs, 456

   # transmitter: bits -> 2-PAM symbols -> NRZ pulses -> MZM
   bitsTx = bitSource(paramBits)
   symbTx = modulateGray(bitsTx, 2, "pam")
   sigTx = anorm(firFilter(pulseShape(paramPulse), upsample(symbTx, SpS)))
   sigTxo = mzm(np.sqrt(dBm2W(3)), sigTx, paramMZM)  # 3 dBm laser

   # fiber channel (loss + chromatic dispersion) and noisy photodiode
   sigRx = photodiode(linearFiberChannel(sigTxo, paramCh), paramPD)

   # BER and Q-factor from the samples at the center of each symbol
   BER, Q = bert(sigRx[0::SpS])
   print(f"BER = {BER:.2e}, Q-factor = {Q:.2f}")

The *Getting started* example in this documentation walks through this example step by step
and extends it to BER versus received power curves.

Examples
--------

The `examples <https://github.com/edsonportosilva/OptiCommPy/tree/main/examples>`__ folder of the
repository contains Jupyter notebooks covering the main features of the package. Most of them
can be run directly in the browser with Google Colab, through the *Open in Colab* button at
their top.

- **First steps**: `Getting started <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/getting_started.ipynb>`__
- **IM-DD systems**: `Basic OOK transmission <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/basic_OOK_transmission.ipynb>`__,
  `PAM transmission <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/basic_IMDD_PAM_transmission.ipynb>`__,
  `Equalizers for IM-DD <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_equalizers_for_IMDD_transmission.ipynb>`__,
  `Photodiode model <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_photodiode_model.ipynb>`__
- **Coherent WDM systems**: `WDM transmission <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_WDM_transmission.ipynb>`__,
  `WDM transmission with amplification <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_WDM_amp_transmission.ipynb>`__,
  `Nonlinearity compensation with DBP <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_NLC_withDBP_WDM_transmission.ipynb>`__,
  `Perturbation models <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_perturbation_models.ipynb>`__
- **Optical amplification**: `Basic EDFA <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/basic_EDFA.ipynb>`__,
  `OOK transmission with advanced EDFA model <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/OOK_transmission_with_advanced_EDFA_model.ipynb>`__
- **DSP**: `Core DSP functions <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_dsp_core_functions.ipynb>`__,
  `Clock recovery <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_clockRecovery.ipynb>`__,
  `Carrier phase recovery <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_carrierPhaseRecovery.ipynb>`__,
  `Sequence synchronization <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_sequence_synchronizer.ipynb>`__
- **Communication theory**: `Modulation <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_modulation.ipynb>`__,
  `Sources <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_sources.ipynb>`__, `OFDM <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_ofdm.ipynb>`__,
  `FEC <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_fec.ipynb>`__, `Metrics <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/test_metrics.ipynb>`__
- **GPU processing**: `GPU benchmark <https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/benchmarck_GPU_processing.ipynb>`__

Documentation
-------------

The full documentation, with the API reference of every module and rendered example notebooks,
is available at `opticommpy.readthedocs.io <https://opticommpy.readthedocs.io/en/latest/index.html>`__.

To build the documentation locally, install OptiCommPy with the documentation dependencies and
run Sphinx from the root of the repository:

.. code-block:: bash

   pip install .[docs]
   sphinx-build -b html docs/source docs/build/html

Contributing
------------

Contributions are welcome, from bug reports and documentation improvements to new models and
algorithms.

#. Open an `issue <https://github.com/edsonportosilva/OptiCommPy/issues>`__ to report a bug or to
   discuss the feature you want to implement.
#. Fork the repository and create a new branch from the latest version of ``main``.
#. Follow the conventions adopted in the code (naming, NumPy-style docstrings, etc.).
#. Add tests for your changes and make sure the test suite passes (``pip install pytest``, then
   ``pytest tests``).
#. Include an example of usage for new features, ideally as a notebook in the ``examples`` folder.
#. Open a pull request.

For suggestions or questions about OptiCommPy, get in touch by e-mail (edsonporto88@gmail.com).

Citing OptiCommPy
-----------------

If you use OptiCommPy in your research, please cite the paper:

   Edson Porto da Silva, Adolfo Fernandes Herbster. "OptiCommPy: Open-source Simulation of Fiber
   Optic Communications with Python", *Journal of Open Source Software*, 9(98), 6600, (2024).
   https://doi.org/10.21105/joss.06600

.. code-block:: bibtex

   @article{daSilva2024OptiCommPy,
     author  = {da Silva, Edson Porto and Herbster, Adolfo Fernandes},
     title   = {{OptiCommPy}: Open-source Simulation of Fiber Optic Communications with {Python}},
     journal = {Journal of Open Source Software},
     year    = {2024},
     volume  = {9},
     number  = {98},
     pages   = {6600},
     doi     = {10.21105/joss.06600}
   }

License
-------

OptiCommPy is distributed under the
`GNU General Public License v3.0 <https://github.com/edsonportosilva/OptiCommPy/blob/main/LICENSE>`__.

.. |PyPI| image:: https://img.shields.io/pypi/v/OptiCommPy?label=pypi%20package
   :alt: PyPI package version
.. |PyPI - Downloads| image:: https://img.shields.io/pypi/dm/OptiCommPy
   :alt: PyPI monthly downloads
.. |Documentation Status| image:: https://readthedocs.org/projects/opticommpy/badge/?version=latest
   :alt: Documentation status
   :target: https://opticommpy.readthedocs.io/en/latest/?badge=latest
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11450597.svg
   :alt: Zenodo DOI
   :target: https://doi.org/10.5281/zenodo.11450597
.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.06600/status.svg
   :alt: JOSS paper
   :target: https://doi.org/10.21105/joss.06600
.. |PyPI - All Downloads| image:: https://static.pepy.tech/personalized-badge/opticommpy?period=total&units=NONE&left_color=GRAY&right_color=GREEN&left_text=PyPI+Downloads
   :alt: Total PyPI downloads

   
