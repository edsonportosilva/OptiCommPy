<p align="center">
<img src="https://github.com/edsonportosilva/OptiCommPy/blob/main/figures/logo_OptiCommPy.jpg" width="500">
</p>

This repository is a Python-based framework to simulate systems, subsystems, and components of fiber optic communication systems, for educational and research purposes.

<p align="center">
<img class="center" src="https://github.com/edsonportosilva/OptiCommPy/blob/main/figures/eyeDisp.gif" width="400">  <img class="center" src="https://github.com/edsonportosilva/OptiCommPy/blob/main/figures/40GOOK_spectrum.jpg" width="400">

<img src="https://github.com/edsonportosilva/OptiCommPy/blob/main/figures/DSP.jpg" width="800">

</p>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edsonportosilva/OptiCommPy/HEAD?urlpath=lab) [![Documentation Status](https://readthedocs.org/projects/opticommpy/badge/?version=latest)](https://opticommpy.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10304945.svg)](https://doi.org/10.5281/zenodo.10304945)


## Available features

* Several digital modulations available (M-PAM, square M-QAM, M-PSK, OOK) to simulate IM-DD and coherent optical systems.
* Numerical models to simulate optical transmitters, optical amplification, nonlinear propagation over optical fibers, and optical receivers.
* CPU and GPU-based implementations of the [*split-step Fourier Method*](https://en.wikipedia.org/wiki/Split-step_method) to simulate polarization multiplexed WDM transmission.
* Standard digital signal processing (DSP) blocks employed in coherent optical receivers, such as:
  - *Signal resampling.* 
  - *Matched filtering.*
  - *Clock recovery.*
  - *Electronic chromatic dispersion compensation (EDC)*.
  - *Several NxN MIMO adaptive equalization algorithms*.
  - *Carrier phase recovery algorithms.* 
* For most of the cases, [Numba](https://numba.pydata.org/) is used to speed up the core DSP functions.  
* Evaluate transmission performance with metrics such as:
  - *Bit-error-rate* (BER).
  - *Symbol-error-rate* (SER).
  - *Error vector magnitude* (EVM).
  - *Mutual information* (MI).
  - *Generalized mutual information* (GMI).  
  - *Normalized generalized mutual information* (NGMI). 
* Visualization of the spectrum of electrical/optical signals, signal constellations, and eyediagrams.

## How can I contribute?

If you want to contribute to this project, implement the feature you want and send me a pull request. If you want to suggest new features or discuss anything related to OptiCommPy, please get in touch with me (edsonporto88@gmail.com).

## Requirements/Dependencies

- python>=3.2
- numpy>=1.9.2
- scipy>=0.15.0
- matplotlib>=1.4.3
- scikit-commpy>=0.7.0
- numba>=0.54.1
- tqdm>=4.64.1
- simple-pid>=1.0.1
- mpl-scatter-density>=0.7.0

## Installation

Using pip:

```
pip install OptiCommPy
```

Cloning the repository from GitHub:

```
$ git clone https://github.com/edsonportosilva/OptiCommPy.git
$ cd OptiCommPy
$ python setup.py install
```
or 

```
$ git clone https://github.com/edsonportosilva/OptiCommPy.git
$ cd OptiCommPy
$ pip install .
```
## Documentation
We are continuously making efforts to improve the code documentation. You can find the latest documentation in [opticommpy.readthedocs.io](https://opticommpy.readthedocs.io/en/latest/index.html).

## Citing this repository

Edson Porto da Silva, Adolfo Herbster, Carlos Daniel Fontes da Silva, & Joaquin Matres. (2023). edsonportosilva/OptiCommPy: v0.7.0-alpha (v0.7.0-alpha). Zenodo. https://doi.org/10.5281/zenodo.10304945
