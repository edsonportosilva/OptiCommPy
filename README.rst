
.. image:: ../../figures/logo_OptiCommPy.jpg
   :width: 75%  
   :align: center

**OptiCommPy** is a Python-based framework to simulate systems, subsystems, and 
components of fiber optic communication systems, for educational and research purposes.

.. image:: ../../figures/eyeDisp.gif
     :width: 45%      
.. image:: ../../figures/40GOOK_spectrum.jpg
     :width: 45% 
     

.. image:: ../../figures/DSP.jpg
     :width: 600px  
     :align: center

|PyPI| |PyPI - Downloads| |Documentation Status| |DOI|

Available features
------------------

-  Several digital modulations available (M-PAM, square M-QAM, M-PSK,
   OOK) to simulate IM-DD and coherent optical systems.
-  Numerical models to simulate optical transmitters, optical
   amplification, nonlinear propagation over optical fibers, and optical
   receivers.
-  CPU and GPU-based implementations of the `split-step Fourier
   Method <https://en.wikipedia.org/wiki/Split-step_method>`__ to
   simulate polarization multiplexed WDM transmission.
-  Standard digital signal processing (DSP) blocks employed in coherent
   optical receivers, such as:

   -  *Signal resampling.*
   -  *Matched filtering.*
   -  *Clock recovery.*
   -  *Electronic chromatic dispersion compensation (EDC)*.
   -  *Several NxN MIMO adaptive equalization algorithms*.
   -  *Carrier phase recovery algorithms.*

-  For most of the cases, `Numba <https://numba.pydata.org/>`__ is used
   to speed up the core DSP functions.
-  Evaluate transmission performance with metrics such as:

   -  *Bit-error-rate* (BER).
   -  *Symbol-error-rate* (SER).
   -  *Error vector magnitude* (EVM).
   -  *Mutual information* (MI).
   -  *Generalized mutual information* (GMI).
   -  *Normalized generalized mutual information* (NGMI).

-  Visualization of the spectrum of electrical/optical signals, signal
   constellations, and eyediagrams.

How can I contribute?
---------------------

If you want to contribute to this project, implement the feature you
want and send me a pull request. If you want to suggest new features or
discuss anything related to OptiCommPy, please get in touch with me
(edsonporto88@gmail.com).

Requirements/Dependencies
-------------------------

-  python>=3.2
-  numpy>=1.24.4
-  scipy>=1.13.0
-  matplotlib>=3.7.0
-  numba>=0.54.1,<=0.57.0
-  tqdm>=4.64.1
-  simple-pid>=1.0.1
-  mpl-scatter-density>=0.7.0
-  sphinx-rtd-theme>=1.2.2
-  nbsphinx>=0.9.3
-  nbsphinx-link>=1.3.0
-  cupy-cuda12x >= 13.1.0 (optional, in case GPU processing is desired)

Installation
------------

Using pip:

::

   pip install OptiCommPy

Cloning the repository from GitHub:

::

   $ git clone https://github.com/edsonportosilva/OptiCommPy.git
   $ cd OptiCommPy
   $ pip install .

Documentation
-------------

We are continuously making efforts to improve the code documentation.
You can find the latest documentation in
`opticommpy.readthedocs.io <https://opticommpy.readthedocs.io/en/latest/index.html>`__.

Citing this repository
----------------------

Edson Porto da Silva, Adolfo Herbster, Carlos Daniel Fontes da Silva, &
Joaquin Matres. (2024). edsonportosilva/OptiCommPy: v0.9.0-alpha
(v0.9.0-alpha). Zenodo. https://doi.org/10.5281/zenodo.11450597

.. |PyPI| image:: https://img.shields.io/pypi/v/OptiCommPy?label=pypi%20package
.. |PyPI - Downloads| image:: https://img.shields.io/pypi/dm/OptiCommPy
.. |Documentation Status| image:: https://readthedocs.org/projects/opticommpy/badge/?version=latest
   :target: https://opticommpy.readthedocs.io/en/latest/?badge=latest
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11450597.svg
   :target: https://doi.org/10.5281/zenodo.11450597

   

