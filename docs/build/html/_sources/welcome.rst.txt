With OptiCommPy you can design and run simulations of optical communications systems at the physical layer level.
A broad range of configurations can be assembled with several different features.

Available features
===================

* Several digital modulations available (M-PAM, square M-QAM, M-PSK, OOK) to simulate IM-DD and coherent optical systems.

* Numerical models to simulate optical transmitters, optical amplification, nonlinear propagation over optical fibers, and optical receivers.

* CPU and GPU-based implementations of the `split-step Fourier Method <https://en.wikipedia.org/wiki/Split-step_method>`_ to simulate polarization multiplexed WDM transmission.

* Standard digital signal processing (DSP) blocks employed in coherent optical receivers, such as:
  
  * Signal resampling.

  * Clock recovery.
  
  * Matched filtering.
  
  * Electronic chromatic dispersion compensation (EDC).
  
  * Several NxN MIMO adaptive equalization algorithms.
  
  * Carrier phase recovery algorithms.


* For most of the cases, `Numba <https://numba.pydata.org/>`_ is used to speed up the core DSP functions.

* Evaluate transmission performance with metrics such as:

  * Bit-error-rate (BER).
  
  * Symbol-error-rate (SER).
  
  * Error vector magnitude (EVM).
  
  * Mutual information (MI).
  
  * Generalized mutual information (GMI).
  
  * Normalized generalized mutual information (NGMI).

* Visualization of the spectrum of electrical/optical signals, signal constellations, and eyediagrams.
