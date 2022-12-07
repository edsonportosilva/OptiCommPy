# OptiCommPy: Fiber Optic Communications with Python

Simulate optical communications systems with Python. This repository is a Python-based framework to simulate systems, subsystems and components of fiber optic communication systems, for educational and research purposes.
<p align="center">
<img class="center" src="https://github.com/edsonportosilva/OptiCommPy/blob/main/figures/eyeDisp.gif" width="400">  <img class="center" src="https://github.com/edsonportosilva/OptiCommPy/blob/main/figures/40GOOK_spectrum.jpg" width="400">

<img src="https://github.com/edsonportosilva/OptiCommPy/blob/main/figures/DSP.jpg" width="800">

</p>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edsonportosilva/OptiCommpy-public/HEAD?urlpath=lab)

## Available features

* QAM/PSK signal generation with [CommPy](https://github.com/veeresht/CommPy)
* Numerical models to simulate optical transmitters, nonlinear propagation over optical fibers, and optical receivers.
* CPU and GPU-based implementations of the [*split-step Fourier Method*](https://en.wikipedia.org/wiki/Split-step_method) to simulate polarization multiplexed WDM transmission.
* Standard digital signal processing (DSP) blocks employed in coherent optical receivers. For most of the cases, [Numba](https://numba.pydata.org/) is used to speed up the core functions.
* Tools to evaluate transmission performance metrics such as *bit-error-rate* (BER), *symbol-error-rate* (SER), *error vector magnitude* (EVM), *mutual information* (MI), *generalized mutual information* (GMI).


## How can I contribute?

If you want to contribute to this project, just implement the feature you want and send me a pull request. If you want to suggest new features or discuss anything related to OptiCommPy, please get in touch with me (edsonporto88@gmail.com).

## Requirements/Dependencies

- python 3.2 or above
- numpy 1.10 or above
- scipy 0.15 or above
- matplotlib 1.4 or above
- Commpy 0.7.0 or above
- numba 0.54.1 or above
- tqdm

## Installation

Clone from github and install as follows::

```
$ git clone https://github.com/edsonportosilva/OptiCommPy.git
$ cd OptiCommPy
$ python setup.py install
```

## Citing this repository

Edson Porto da Silva, & Adolfo Herbster. (2021). edsonportosilva/OptiCommpy: First release (alpha version) (v0.1.0-alpha).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5668218.svg)](https://doi.org/10.5281/zenodo.5668218)
