# Fiber Optic Communications with Python

Simulate optical communications systems with Python.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edsonportosilva/OpticCommpy-public/HEAD?urlpath=lab)

## Available features

* QAM/PSK signal generation with [Commpy](https://github.com/veeresht/CommPy)
* Numerical models to simulate optical transmitters, nonlinear propagation over optical fibers, and optical receivers.
* CPU and GPU-based implementations of the [*split-step Fourier Method*](https://en.wikipedia.org/wiki/Split-step_method) to simulate polarization multiplexed WDM transmission.
* Standard digital signal processing (DSP) blocks employed in coherent optical receivers.
* Tools to evaluate transmission performance metrics such as *bit-error-rate* (BER), *symbol-error-rate* (SER), *mutual information* (MI), *generalized mutual information* (GMI).


How can I contribute?
---------------------
If you want to contribute to this project, just implement the feature you want and send me a pull request. If you want to suggest new features or discuss anything related to OpticCommPy, please get in touch with me (edsonporto88@gmail.com).

Requirements/Dependencies
-------------------------
- python 3.2 or above
- numpy 1.10 or above
- scipy 0.15 or above
- matplotlib 1.4 or above
- nose 1.3 or above
- sympy 1.7 or above
- numba 0.54.1 or above

Installation
------------

Clone from github and install as follows::

```
$ git clone https://github.com/edsonportosilva/OpticCommPy.git
$ cd OpticCommPy
$ python setup.py install
