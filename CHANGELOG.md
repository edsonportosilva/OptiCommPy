# Changelog

All notable changes to OptiCommPy are documented in this file.

## [v0.11.0-alpha] - 2026-10-15

Changes since v0.10.0-alpha.

### Added
- **Receiver DSP:** feedforward (FFE), decision feedback (DFE) and Volterra equalizers (up to cubic order) for IM-DD links, with samples-per-symbol support, training mode and custom initial coefficients; adaptive filtering cores (NLMS, DD-LMS, CMA, RDE, DA-RDE, RLS, DD-RLS) with time- and frequency-domain MIMO equalization; Levinson recursion, autocorrelation and whitening filter; `optic.dsp.synchronization` module with `syncDataSequences`; frequency shifting (`freqShift`) and M-th power frequency offset estimation.
- **Forward error correction:** LDPC encoding and decoding (edge-based message passing, DVB-S2 codes, ALIST files), Hamming codes, extrinsic LLR metric and MLSE (Viterbi) routine.
- **Devices and models:** DAC model with quantization, ENOB and jitter; ENOB, clipping and noise model in the ADC; extinction ratio support in the MZM and IQ modulator; IQ imbalance and skew (`iqMixing`) and frontend parameters in the coherent receiver; NLIN perturbation models rewritten with Numba kernels.
- **Metrics:** theoretical GMI and mutual information (MI) support.
- **Plotting:** customizable eye diagram and power spectral density plots; `figsize` option in constellation plots; comprehensive plotting tests.
- **Tests and CI:** pytest-based test suite and GitHub Actions workflow running it on Python 3.10-3.13.
- **Packaging:** GitHub Actions workflow that publishes the package to PyPI (Trusted Publishing) when a release is published; the package version is now derived from the Git tag (setuptools-scm).
- `CITATION.cff` metadata and a revamped `README` with full project overview.

### Changed
- Python 3.10 or newer is required.
- Numba caching and `fastmath` enabled in performance-critical DSP, FEC and model routines; FIR filtering uses `fftconvolve`.
- Renamed parameters for consistency (`Ei`/`Eo` to `sigIn`/`sigOut`, `outEq` to `yEq`, `Ipd_sat` to `IpdSat`, `signalRotation` to `polRotation`); `awgn` now takes a `parameters` object.
- Docstrings expanded with equations and theory notes across the `optic` package, now with a Returns section for all documented public functions; Sphinx documentation refreshed and `optic.dsp.coreGPU` added to it.
- Modern NumPy/SciPy/Matplotlib APIs replace deprecated ones.

### Fixed
- AWGN noise variance for real-valued signals.
- Sign handling of LLRs for bit probabilities.
- `ber2Qfactor` dB conversion formula, extinction ratio dB-to-linear conversion in the MZM, and ADC anti-alias filter frequency inputs.
- PRBS-13 taps, CAZAC sequence formula and RRC filter tap scaling with the symbol period.
- Viterbi & Viterbi carrier phase recovery for PSK/APSK constellations.
- Photodiode noise seeds are now independent across receiver channels; PSD labels in single-mode plots.

[0.11.0]: https://github.com/edsonportosilva/OptiCommPy/releases/tag/v0.11.0
