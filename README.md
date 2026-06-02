# LQG Echo Delay Analyzer — v2.1

**Author:** Mário Sérgio Guilherme Junior  
**ORCID:** [0009-0002-5509-0683](https://orcid.org/0009-0002-5509-0683)  
**Year:** 2026

---

## Overview

This repository provides the full analysis code for the paper:

> *"Gravitational Wave Echo Timing in Loop Quantum Gravity: Kerr Correction Factor and Predictions for GWTC Events"*  
> Guilherme Junior, M. S. (2026)

The code computes and compares gravitational-wave echo search windows for
**LQG** (with the exact Kerr correction factor η) versus the
**Schwarzschild approximation** (η = 1), covering all 58 events in GWTC-1/2/3.

### Key result

The Schwarzschild approximation systematically **underestimates** echo delays
for spinning black holes. The correction factor is:

```
η(a★) = (r₊² + a²) / [(r₊ − r₋) · r₊]
       = [(1+s)² + a★²] / [2s(1+s)],   s = √(1 − a★²)
```

η(0) = 1 (recovers Schwarzschild); η → ∞ as a★ → 1.

---

## Bundled GWOSC Posterior Samples

The `posteriors/` directory includes official PE data releases from LIGO/Virgo/KAGRA:

| File | Event | Catalog | Zenodo |
|------|-------|---------|--------|
| `IGWN-GWTC2p1-v2-GW150914_...h5` | GW150914 | GWTC-2.1 | [6513631](https://zenodo.org/records/6513631) |
| `IGWN-GWTC2p1-v2-GW170729_...h5` | GW170729 | GWTC-2.1 | [6513631](https://zenodo.org/records/6513631) |
| `IGWN-GWTC2p1-v2-GW190521_...h5` | GW190521 | GWTC-2.1 | [5117702](https://zenodo.org/records/5117702) |
| `IGWN-GWTC3p0-v1-GW191109_...h5` | GW191109 | GWTC-3   | [5546662](https://zenodo.org/records/5546662) |

These files are redistributed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license
of the original GWOSC data release. Credit: LIGO Scientific Collaboration, Virgo Collaboration,
KAGRA Collaboration.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/lqg-echo-analyzer.git
cd lqg-echo-analyzer
pip install numpy matplotlib h5py
```

---

## Usage

### Quick start (uses bundled posteriors)

```bash
python lqg_echo_analyzer.py
```

This automatically uses the 4 bundled HDF5 files in `posteriors/` (Mode A)
and uniform Monte Carlo for all remaining events (Mode B).

### All options

```bash
# Specify a different posteriors directory
python lqg_echo_analyzer.py --posteriors-dir /path/to/more/h5files/

# External event catalog (CSV)
python lqg_echo_analyzer.py my_catalog.csv --posteriors-dir ./posteriors/

# Reproduce v1.0 figures (old tau_damp formula)
python lqg_echo_analyzer.py --v1-tau

# Set Monte Carlo sample count
python lqg_echo_analyzer.py --n-mc 10000
```

### Environment variable

```bash
LQG_POSTERIORS_DIR=./posteriors/ python lqg_echo_analyzer.py
```

---

## Uncertainty propagation modes

| Mode | Label | Description |
|------|-------|-------------|
| A | `*` | Real GWOSC posterior samples — joint (M_f, a_f) distribution |
| B | `~` | Uniform Monte Carlo over published 90% CL intervals |

Mode A automatically activates when a matching HDF5 file is found.
Mode B is the conservative fallback.

---

## Outputs

| File | Description |
|------|-------------|
| `echo_results.csv` | Full numerical results for all events |
| `echo_delays_table.tex` | LaTeX table with 90% CL uncertainties |
| `echo_comparison.pdf/png` | LQG vs Schwarzschild comparison plot |
| `eta_curve.pdf/png` | η(a★) correction factor curve |
| `bias_vs_spin.pdf/png` | Schwarzschild bias as a function of spin |
| `pipeline_mismatch.pdf/png` | Mismatch parameter M = δt/τ_damp |
| `posterior_pdf.pdf/png` | PDFs of echo delay (Mode A events) |

---

## Obtaining additional GWOSC files

```bash
pip install zenodo_get
zenodo-get 6513631   # GWTC-2.1 (O1+O2 re-analysis)
zenodo-get 5117702   # GWTC-2.1 (O3a)
zenodo-get 5546662   # GWTC-3 (O3b)
```

Place the downloaded `.h5` files in the `posteriors/` directory.

---

## CSV format for custom catalogs

```
event,Mf,Mf_lo,Mf_hi,af,af_lo,af_hi
GW150914,62.0,58.0,65.0,0.67,0.60,0.72
```

---

## Physical constants used

| Constant | Value |
|----------|-------|
| G | 6.674 × 10⁻¹¹ m³ kg⁻¹ s⁻² |
| c | 3 × 10⁸ m/s |
| M☉ | 1.989 × 10³⁰ kg |
| ρ_Planck | c⁵/(ħG²) |
| ρ_c (LQG) | 0.41 ρ_Planck |

---

## Citation

If you use this code, please cite:

```bibtex
@software{guilherme_junior_2026_lqg_echo,
  author       = {Guilherme Junior, Mário Sérgio},
  title        = {{LQG Echo Delay Analyzer v2.1}},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX},
  orcid        = {0009-0002-5509-0683}
}
```

*(Replace XXXXXXX with the actual Zenodo DOI after upload.)*

---

## License

The analysis code (`lqg_echo_analyzer.py`) is released under the
[MIT License](LICENSE).

The bundled GWOSC posterior samples (`posteriors/*.h5`) are redistributed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
per the original LIGO/Virgo/KAGRA data release terms.
