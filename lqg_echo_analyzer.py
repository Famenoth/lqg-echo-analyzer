#!/usr/bin/env python3
"""
=============================================================
LQG Echo Delay Analyzer  —  v2.1
Mário Sérgio Guilherme Junior (2026)
ORCID: 0009-0002-5509-0683

Computes and compares gravitational echo search windows for
LQG (exact Kerr correction factor eta) vs Schwarzschild (eta=1)
for gravitational wave events.

FEATURE v2.0 — GWOSC Posterior Samples
-----------------------------------------
The code supports TWO uncertainty propagation modes:

  MODE A — Posterior Samples (default when available)
    Reads HDF5 files published by LIGO/Virgo/KAGRA on GWOSC,
    extracting joint (Mf, af) samples. Automatically captures
    asymmetries, correlations and heavy tails of the real
    posterior distributions — eliminating the main limitation
    of v1.0 identified by reviewers.

  MODE B — Uniform Monte Carlo (fallback)
    Uniform distributions over published 90% CL intervals.
    Used automatically when the HDF5 file is not found.
    Conservative, no symmetry assumption.

Supported HDF5 structures (GWTC-1/2/3):
  • GWTC-2.1 (re-release O1/O2):
      /C01:IMRPhenomXPHM/posterior_samples  <- structured dataset
      fields: final_mass_source, final_spin
  • GWTC-3 (O3b, pesummary format):
      /C01:IMRPhenomXPHM/posterior_samples/final_mass_source  <- 1D array
                                          /final_spin          <- 1D array
  • PublicationSamples: alias for C01:IMRPhenomXPHM in some releases
  • GWTC-1 legacy: /IMRPhenomPv2/posterior_samples

How to obtain additional HDF5 files:
  pip install zenodo_get
  zenodo-get 6513631   # GWTC-1/O1/O2 via GWTC-2.1
  zenodo-get 5117702   # GWTC-2/O3a
  zenodo-get 5546662   # GWTC-3/O3b

Usage:
    python lqg_echo_analyzer.py
    python lqg_echo_analyzer.py --posteriors-dir ./posteriors/
    python lqg_echo_analyzer.py my_catalog.csv --posteriors-dir ./posteriors/
    LQG_POSTERIORS_DIR=./posteriors/ python lqg_echo_analyzer.py

External CSV format (required columns):
    event, Mf, Mf_lo, Mf_hi, af, af_lo, af_hi

Changes in v2.1 (relative to v2.0):
  - tau_damp: corrected to tau = Q/(pi*f_qnm) (Berti+2006 Eq.4)
  - LaTeX: fixed double $$ in footnote (mode/dagger)
  - Monte Carlo: per-event seed (name hash)
  - plot_posterior_pdf: bins defined by [0.5, 99.5] percentiles
  - H5_GROUPS: expanded to cover group name variations
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import sys
import os
import argparse
import hashlib
import warnings

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    warnings.warn(
        "h5py not found. Install with: pip install h5py\n"
        "The code will run in uniform Monte Carlo mode (Mode B).",
        ImportWarning
    )

# ============================================================
# PHYSICAL CONSTANTS (SI)
# ============================================================
G     = 6.674e-11
c     = 3e8
M_sun = 1.989e30
hbar  = 1.055e-34

rho_Pl = c**5 / (hbar * G**2)
rho_c  = 0.41 * rho_Pl

# ============================================================
# HDF5 KEY PATTERNS
# ============================================================
H5_GROUPS = [
    "C01:IMRPhenomXPHM",
    "C01:IMRPhenomXP",
    "C01:IMRPhenomPv2",
    "C01:SEOBNRv4PHM",
    "C01:SEOBNRv5PHM",
    "PublicationSamples",
    "IMRPhenomXPHM",
    "IMRPhenomPv2",
    "IMRPhenomPv2NRTidal",
    "SEOBNRv4",
    "posterior_samples",
    "",
]

MF_FIELDS = ["final_mass_source", "final_mass", "remnant_mass"]
AF_FIELDS = ["final_spin", "final_spin_magnitude"]

# ============================================================
# LQG FRAMEWORK FUNCTIONS
# ============================================================

def r_plus(M_sol, a_star):
    rg = G * np.asarray(M_sol) * M_sun / c**2
    return rg * (1.0 + np.sqrt(1.0 - np.asarray(a_star)**2))

def r_b(M_sol):
    return (3.0 * np.asarray(M_sol) * M_sun / (4.0 * np.pi * rho_c))**(1.0/3.0)

def eta(a_star):
    a = np.asarray(a_star, dtype=float)
    s = np.sqrt(np.clip(1.0 - a**2, 1e-10, 1.0))
    return ((1.0 + s)**2 + a**2) / (2.0 * s * (1.0 + s))

def dt_lqg(M_sol, a_star):
    rp = r_plus(M_sol, a_star)
    rb = r_b(M_sol)
    return 2.0 * eta(a_star) / c * rp * np.log(rp / rb)

def dt_schwarzschild(M_sol, a_star):
    rp = r_plus(M_sol, a_star)
    rb = r_b(M_sol)
    return 2.0 / c * rp * np.log(rp / rb)

def f_qnm(M_sol, a_star):
    M_kg = np.asarray(M_sol) * M_sun
    return c**3 / (2.0 * np.pi * G * M_kg) * \
           (1.5251 - 1.1568 * (1.0 - np.asarray(a_star))**0.1292)

def tau_damp(M_sol, a_star):
    a = np.asarray(a_star)
    Q = 0.7000 + 1.4187 * (1.0 - a)**(-0.4990)
    fq = f_qnm(M_sol, a_star)
    return Q / (np.pi * fq)

USE_V1_TAU = False

def tau_damp_v1(M_sol, a_star):
    M_kg = np.asarray(M_sol) * M_sun
    return G * M_kg / c**3 * (0.7000 + 1.4187 * (1.0 - np.asarray(a_star))**(-0.4990))

def _tau(M_sol, a_star):
    return tau_damp_v1(M_sol, a_star) if USE_V1_TAU else tau_damp(M_sol, a_star)

# ============================================================
# POSTERIOR SAMPLES READING — MODE A
# ============================================================

def _event_seed(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % (2**31)

def _extract_field(obj, field_names):
    for fname in field_names:
        try:
            if hasattr(obj, 'dtype') and obj.dtype.names:
                if fname in obj.dtype.names:
                    arr = obj[:][fname]
                    arr = np.array(arr).astype(np.float64)
                    arr = arr[np.isfinite(arr)]
                    if len(arr) >= 10:
                        return arr
            elif hasattr(obj, 'keys') and fname in obj:
                arr = np.asarray(obj[fname][:], dtype=float)
                arr = arr[np.isfinite(arr)]
                if len(arr) >= 10:
                    return arr
        except Exception:
            continue
    return None

def _try_node(node):
    candidates = []
    if hasattr(node, 'keys') and 'posterior_samples' in node:
        candidates.append(node['posterior_samples'])
    candidates.append(node)
    for obj in candidates:
        mf = _extract_field(obj, MF_FIELDS)
        af = _extract_field(obj, AF_FIELDS)
        if mf is not None and af is not None:
            n = min(len(mf), len(af))
            if n >= 100:
                return mf[:n], af[:n]
    return None, None

def _find_samples_in_h5(h5file):
    for gname in H5_GROUPS:
        try:
            node = h5file if gname == "" else h5file.get(gname)
            if node is None:
                continue
            mf, af = _try_node(node)
            if mf is not None:
                return mf, af
        except Exception:
            continue

    def _recurse(node, depth):
        if depth > 3:
            return None, None
        mf, af = _try_node(node)
        if mf is not None:
            return mf, af
        if hasattr(node, 'keys'):
            for key in list(node.keys()):
                try:
                    mf, af = _recurse(node[key], depth + 1)
                    if mf is not None:
                        return mf, af
                except Exception:
                    continue
        return None, None

    return _recurse(h5file, 0)

def load_posterior_samples(event_name, posteriors_dir):
    if not HDF5_AVAILABLE or not posteriors_dir:
        return None, None, None
    if not os.path.isdir(posteriors_dir):
        return None, None, None

    key = event_name.replace("_", "").lower()
    candidates = [
        os.path.join(posteriors_dir, f)
        for f in os.listdir(posteriors_dir)
        if f.lower().endswith(('.h5', '.hdf5', '.hdf'))
        and key in f.replace("_", "").lower()
    ]

    for fpath in sorted(candidates):
        try:
            with h5py.File(fpath, 'r') as f:
                mf, af = _find_samples_in_h5(f)
                if mf is not None and len(mf) >= 100:
                    af = np.clip(af, 1e-4, 0.9990)
                    return mf, af, fpath
        except Exception as e:
            warnings.warn(f"[{event_name}] Error reading {fpath}: {e}")

    return None, None, None

def stats_from_samples(arr):
    return {
        'median': float(np.median(arr)),
        'p5':     float(np.percentile(arr,  5)),
        'p95':    float(np.percentile(arr, 95)),
        'mean':   float(np.mean(arr)),
        'std':    float(np.std(arr)),
    }

# ============================================================
# UNCERTAINTY PROPAGATION
# ============================================================

def compute_mc_stats(event_name, Mf, af, Mf_lo, Mf_hi, af_lo, af_hi,
                     posteriors_dir=None, N_mc=5000):
    seed = _event_seed(event_name)
    mf_s, af_s, h5path = load_posterior_samples(event_name, posteriors_dir)

    if mf_s is not None:
        mode, n_s = 'posterior_samples', len(mf_s)
    else:
        rng  = np.random.default_rng(seed)
        mf_s = rng.uniform(Mf_lo, Mf_hi, N_mc)
        af_s = np.clip(rng.uniform(af_lo, af_hi, N_mc), 1e-4, 0.9990)
        mode, n_s, h5path = 'uniform_mc', N_mc, None

    dt_l = dt_lqg(mf_s, af_s) * 1e3
    dt_s = dt_schwarzschild(mf_s, af_s) * 1e3
    diff = dt_l - dt_s

    return {
        'lqg':       stats_from_samples(dt_l),
        'schw':      stats_from_samples(dt_s),
        'diff':      stats_from_samples(diff),
        'mode':      mode,
        'n_samples': n_s,
        'h5_file':   h5path,
        'mf_samples': mf_s,
        'af_samples': af_s,
    }

# ============================================================
# BUILT-IN DATA — GWTC-1/2/3
# ============================================================
BUILTIN_EVENTS = [
    ("GW150914", 62.0,  58.0,  65.0,  0.67, 0.60, 0.72),
    ("GW151226", 20.8,  19.1,  22.5,  0.74, 0.67, 0.81),
    ("GW170104", 48.9,  44.9,  52.9,  0.64, 0.55, 0.73),
    ("GW170608", 17.8,  17.1,  18.5,  0.69, 0.65, 0.73),
    ("GW170729", 80.3,  72.3,  88.3,  0.81, 0.74, 0.88),
    ("GW170814", 53.4,  50.9,  55.9,  0.70, 0.63, 0.77),
    ("GW170823", 65.6,  59.6,  71.6,  0.71, 0.62, 0.80),
    ("GW190408_181802", 53.7, 49.0, 58.0, 0.70, 0.62, 0.78),
    ("GW190412",        37.2, 35.4, 39.0, 0.67, 0.62, 0.72),
    ("GW190421_213856", 64.3, 57.0, 71.0, 0.74, 0.62, 0.85),
    ("GW190503_185404", 72.0, 64.0, 80.0, 0.69, 0.60, 0.78),
    ("GW190512_180714", 35.7, 33.0, 38.0, 0.67, 0.62, 0.72),
    ("GW190513_205428", 54.0, 47.0, 61.0, 0.77, 0.65, 0.87),
    ("GW190521",       142.0,123.0,161.0, 0.72, 0.61, 0.83),
    ("GW190602_175927",111.0, 97.0,125.0, 0.71, 0.60, 0.82),
    ("GW190620_030421", 70.0, 61.0, 79.0, 0.78, 0.67, 0.88),
    ("GW190630_185205", 63.0, 58.0, 68.0, 0.67, 0.61, 0.73),
    ("GW190701_203306",100.0, 88.0,112.0, 0.72, 0.60, 0.83),
    ("GW190706_222641",111.0, 97.0,125.0, 0.80, 0.68, 0.90),
    ("GW190707_093326", 11.5, 11.1, 11.9, 0.66, 0.60, 0.73),
    ("GW190708_232457", 17.8, 17.2, 18.4, 0.69, 0.64, 0.74),
    ("GW190720_000836", 23.0, 21.5, 24.5, 0.72, 0.65, 0.79),
    ("GW190727_060333", 74.0, 65.0, 83.0, 0.73, 0.62, 0.83),
    ("GW190728_064510", 17.1, 16.5, 17.7, 0.68, 0.63, 0.73),
    ("GW190803_022701", 71.0, 62.0, 80.0, 0.72, 0.61, 0.82),
    ("GW190814",        25.0, 24.1, 25.9, 0.28, 0.06, 0.49),
    ("GW190828_063405", 67.0, 61.0, 73.0, 0.71, 0.63, 0.79),
    ("GW190828_065509", 36.0, 31.0, 41.0, 0.72, 0.59, 0.84),
    ("GW190910_112807", 80.0, 72.0, 88.0, 0.72, 0.62, 0.82),
    ("GW190915_235702", 54.0, 49.0, 59.0, 0.70, 0.63, 0.77),
    ("GW190924_021846",  8.9,  8.6,  9.2, 0.62, 0.56, 0.68),
    ("GW190929_012149", 64.0, 54.0, 74.0, 0.80, 0.68, 0.90),
    ("GW190930_133541", 16.5, 15.0, 18.0, 0.73, 0.64, 0.82),
    ("GW191105_143521", 15.1, 14.5, 15.7, 0.67, 0.61, 0.73),
    ("GW191109_010717",113.0, 97.0,129.0, 0.74, 0.63, 0.84),
    ("GW191127_050227", 55.0, 48.0, 62.0, 0.71, 0.60, 0.82),
    ("GW191129_134029", 12.6, 12.1, 13.1, 0.65, 0.60, 0.70),
    ("GW191204_171526", 18.0, 17.3, 18.7, 0.68, 0.63, 0.73),
    ("GW191215_223052", 25.0, 23.5, 26.5, 0.70, 0.63, 0.77),
    ("GW191216_213338", 12.4, 11.9, 12.9, 0.67, 0.61, 0.73),
    ("GW191222_033537", 77.0, 68.0, 86.0, 0.73, 0.62, 0.83),
    ("GW191230_180458", 73.0, 64.0, 82.0, 0.71, 0.61, 0.81),
    ("GW200105_162426", 17.8, 17.1, 18.5, 0.60, 0.42, 0.75),
    ("GW200115_042309",  5.9,  5.7,  6.1, 0.04, 0.00, 0.20),
    ("GW200129_065458", 60.0, 55.0, 65.0, 0.72, 0.65, 0.79),
    ("GW200202_154313", 11.0, 10.6, 11.4, 0.65, 0.60, 0.70),
    ("GW200208_130117", 70.0, 61.0, 79.0, 0.73, 0.62, 0.83),
    ("GW200209_085452", 70.0, 60.0, 80.0, 0.72, 0.60, 0.83),
    ("GW200210_092254", 20.0, 18.5, 21.5, 0.64, 0.56, 0.72),
    ("GW200216_220804", 72.0, 62.0, 82.0, 0.73, 0.62, 0.83),
    ("GW200219_094415", 72.0, 63.0, 81.0, 0.72, 0.62, 0.82),
    ("GW200220_061928", 73.0, 63.0, 83.0, 0.73, 0.62, 0.83),
    ("GW200220_124850", 73.0, 63.0, 83.0, 0.72, 0.61, 0.83),
    ("GW200224_222234", 68.0, 63.0, 73.0, 0.70, 0.63, 0.77),
    ("GW200225_060421", 29.0, 27.5, 30.5, 0.68, 0.62, 0.74),
    ("GW200302_015811", 65.0, 56.0, 74.0, 0.74, 0.63, 0.85),
    ("GW200306_093714", 72.0, 62.0, 82.0, 0.72, 0.61, 0.83),
    ("GW200308_173609", 73.0, 63.0, 83.0, 0.73, 0.62, 0.84),
    ("GW200311_115853", 60.0, 56.0, 64.0, 0.70, 0.63, 0.77),
    ("GW200316_215756", 18.5, 17.8, 19.2, 0.67, 0.61, 0.73),
]

# ============================================================
# DATA LOADING
# ============================================================

def load_events(filepath=None):
    if filepath is None:
        print(f"Using built-in data: {len(BUILTIN_EVENTS)} events (GWTC-1/2/3).\n")
        return BUILTIN_EVENTS
    try:
        events = []
        with open(filepath, newline='') as f:
            for row in csv.DictReader(f):
                events.append((
                    row['event'],
                    float(row['Mf']),    float(row['Mf_lo']),
                    float(row['Mf_hi']), float(row['af']),
                    float(row['af_lo']), float(row['af_hi']),
                ))
        print(f"Loaded {len(events)} events from '{filepath}'.\n")
        return events
    except FileNotFoundError:
        print(f"File '{filepath}' not found — using built-in data.\n")
        return BUILTIN_EVENTS
    except KeyError as e:
        print(f"Missing CSV column: {e} — using built-in data.\n")
        return BUILTIN_EVENTS

# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze(events, N_mc=5000, posteriors_dir=None):
    results, n_a, n_b = [], 0, 0

    for ev in events:
        name, Mf, Mf_lo, Mf_hi, af, af_lo, af_hi = ev

        eta_c    = eta(af)
        dt_l_c   = dt_lqg(Mf, af) * 1e3
        dt_s_c   = dt_schwarzschild(Mf, af) * 1e3
        bias_pct = (eta_c - 1.0) * 100.0
        td       = _tau(Mf, af) * 1e3
        fq       = f_qnm(Mf, af)
        rb       = r_b(Mf)
        rp       = r_plus(Mf, af)

        mc = compute_mc_stats(
            name, Mf, af, Mf_lo, Mf_hi, af_lo, af_hi,
            posteriors_dir=posteriors_dir, N_mc=N_mc
        )

        if mc['mode'] == 'posterior_samples':
            n_a += 1
        else:
            n_b += 1

        results.append({
            'event':       name,
            'Mf':          float(Mf),
            'af':          float(af),
            'eta':         float(eta_c),
            'rb_m':        float(rb),
            'rp_m':        float(rp),
            'ln_ratio':    float(np.log(rp / rb)),
            'dt_lqg_ms':   float(dt_l_c),
            'dt_schw_ms':  float(dt_s_c),
            'diff_ms':     float(dt_l_c - dt_s_c),
            'bias_pct':    float(bias_pct),
            'tau_damp_ms': float(td),
            'f_qnm_hz':    float(fq),
            'suppression': float(np.exp(-dt_l_c / td)),
            'mc_mode':     mc['mode'],
            'mc_n':        mc['n_samples'],
            'mc_h5':       mc['h5_file'] or '',
            'mc_lqg_med':  mc['lqg']['median'],
            'mc_lqg_p5':   mc['lqg']['p5'],
            'mc_lqg_p95':  mc['lqg']['p95'],
            'mc_lqg_std':  mc['lqg']['std'],
            'mc_schw_med': mc['schw']['median'],
            'mc_schw_p5':  mc['schw']['p5'],
            'mc_schw_p95': mc['schw']['p95'],
            'mc_diff_med': mc['diff']['median'],
            'mc_diff_std': mc['diff']['std'],
            '_mf_samples': mc['mf_samples'],
            '_af_samples': mc['af_samples'],
        })

    print(f"\n  [Mode A — HDF5 posterior samples]: {n_a} event(s)")
    print(f"  [Mode B — uniform Monte Carlo   ]: {n_b} event(s)")
    return results

# ============================================================
# OUTPUT: TERMINAL
# ============================================================

def print_table(results):
    SYM = {'posterior_samples': '*', 'uniform_mc': '~'}
    hdr = f"{'Event':<25} {'M':>2} {'af':>5} {'eta':>6} " \
          f"{'dt_LQG(ms)':>12} {'dt_Schw(ms)':>12} " \
          f"{'Bias(%)':>8} {'tau(ms)':>8}"
    sep = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}")
    print("  * = HDF5 posterior samples  |  ~ = uniform Monte Carlo")
    print(sep)
    for r in results:
        print(f"{r['event']:<25} {SYM.get(r['mc_mode'],'?'):>2} "
              f"{r['af']:>5.2f} {r['eta']:>6.3f} "
              f"{r['mc_lqg_med']:>12.1f} {r['mc_schw_med']:>12.1f} "
              f"{r['bias_pct']:>8.1f} {r['tau_damp_ms']:>8.3f}")
    print(sep)
    biases = [r['bias_pct'] for r in results]
    print(f"\n  Mean bias : {np.mean(biases):.1f}%  "
          f"[min={min(biases):.1f}%  max={max(biases):.1f}%]")
    n_win = sum(1 for r in results if 10 <= r['dt_lqg_ms'] <= 200)
    print(f"  Events in 10-200 ms window: {n_win}/{len(results)}")

# ============================================================
# OUTPUT: LaTeX TABLE
# ============================================================

def write_latex_table(results, filename="echo_delays_table.tex"):
    lines = [
        r"\begin{table}[h]",
        r"\centering\small",
        r"\caption{Predicted echo delays: LQG (exact $\eta$) vs "
        r"Schwarzschild ($\eta=1$). "
        r"$^\dagger$ Uncertainties from GWOSC posterior samples (Mode A). "
        r"$^\ddagger$ Uniform Monte Carlo over 90\% CL intervals (Mode B). "
        r"All uncertainties at 90\% CL.}",
        r"\label{tab:echo_delays_full}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (r"Event & $a_{\star,f}$ & $\eta$ & "
         r"$\Delta t^{\rm LQG}$ (ms) & "
         r"$\Delta t^{\rm Schw}$ (ms) & Bias (\%) \\"),
        r"\midrule",
    ]
    for r in results:
        note = (r"\textsuperscript{$\dagger$}"
                if r['mc_mode'] == 'posterior_samples'
                else r"\textsuperscript{$\ddagger$}")
        lqg_str = (
            f"${r['mc_lqg_med']:.1f}"
            f"^{{+{r['mc_lqg_p95'] - r['mc_lqg_med']:.1f}}}"
            f"_{{-{r['mc_lqg_med'] - r['mc_lqg_p5']:.1f}}}$"
            f"{note}"
        )
        schw_str = (
            f"${r['mc_schw_med']:.1f}"
            f"^{{+{r['mc_schw_p95'] - r['mc_schw_med']:.1f}}}"
            f"_{{-{r['mc_schw_med'] - r['mc_schw_p5']:.1f}}}$"
        )
        lines.append(
            f"{r['event']} & {r['af']:.2f} & {r['eta']:.3f} & "
            f"{lqg_str} & {schw_str} & {r['bias_pct']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(filename, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"LaTeX table saved to '{filename}'.")

# ============================================================
# OUTPUT: CSV
# ============================================================

def write_csv(results, filename="echo_results.csv"):
    clean = [{k: v for k, v in r.items() if not k.startswith('_')}
             for r in results]
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(clean[0].keys()))
        writer.writeheader()
        writer.writerows(clean)
    print(f"CSV results saved to '{filename}'.")

# ============================================================
# FIGURES
# ============================================================

def _mode_colors(r):
    if r['mc_mode'] == 'posterior_samples':
        return 'royalblue', 'tomato'
    return 'cornflowerblue', 'lightsalmon'


def plot_comparison(results, filename="echo_comparison"):
    res = sorted([r for r in results if r['dt_lqg_ms'] < 500],
                 key=lambda x: x['dt_lqg_ms'])
    n = len(res)
    names = [r['event'].replace('GW', 'GW\n') for r in res]

    fig, ax = plt.subplots(figsize=(max(12, n * 0.45), 6))
    for i, r in enumerate(res):
        cl, cs = _mode_colors(r)
        kw_l = dict(label='LQG (exact $\\eta$)')      if i == 0 else {}
        kw_s = dict(label='Schwarzschild ($\\eta=1$)') if i == 0 else {}
        ax.errorbar(i - 0.15, r['mc_lqg_med'],
                    yerr=[[r['mc_lqg_med']  - r['mc_lqg_p5']],
                          [r['mc_lqg_p95']  - r['mc_lqg_med']]],
                    fmt='o', color=cl, ms=7, capsize=4, lw=1.8, **kw_l)
        ax.errorbar(i + 0.15, r['mc_schw_med'],
                    yerr=[[r['mc_schw_med'] - r['mc_schw_p5']],
                          [r['mc_schw_p95'] - r['mc_schw_med']]],
                    fmt='s', color=cs, ms=6, capsize=4, lw=1.8, **kw_s)
        ax.annotate(f"+{r['bias_pct']:.0f}%",
                    xy=(i, (r['mc_lqg_med'] + r['mc_schw_med']) / 2),
                    ha='center', va='bottom', fontsize=6.5, color='dimgray')

    ax.axhspan(10, 200, alpha=0.07, color='green',
               label='LIGO/Virgo window (10-200 ms)')
    has_a = any(r['mc_mode'] == 'posterior_samples' for r in res)
    subtitle = "posterior samples" if has_a else "uniform MC"
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel(r'$\Delta t_{\rm echo}$ (ms)', fontsize=13)
    ax.set_title(f'LQG echo delay vs Schwarzschild\n90% CL [{subtitle}]',
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xlim(-0.8, n - 0.2)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{filename}.{ext}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison figure saved to '{filename}.pdf/png'.")


def plot_posterior_pdf(results, filename_prefix="posterior_pdf"):
    mode_a = [r for r in results if r['mc_mode'] == 'posterior_samples']
    if not mode_a:
        print("No events with posterior samples — PDF figure not generated.")
        return

    n = len(mode_a)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 3.5 * rows), squeeze=False)

    for idx, r in enumerate(mode_a):
        ax = axes[idx // cols][idx % cols]
        dt_l = dt_lqg(r['_mf_samples'], r['_af_samples']) * 1e3
        dt_s = dt_schwarzschild(r['_mf_samples'], r['_af_samples']) * 1e3

        lo = min(np.percentile(dt_l, 0.5), np.percentile(dt_s, 0.5))
        hi = max(np.percentile(dt_l, 99.5), np.percentile(dt_s, 99.5))
        bins = np.linspace(lo * 0.95, hi * 1.05, 60)

        ax.hist(dt_l, bins=bins, density=True, alpha=0.55,
                color='royalblue', label=r'LQG ($\eta$ exact)')
        ax.hist(dt_s, bins=bins, density=True, alpha=0.55,
                color='tomato',    label='Schwarzschild')
        ax.axvline(r['mc_lqg_med'],  color='royalblue', ls='--', lw=1.5)
        ax.axvline(r['mc_schw_med'], color='tomato',    ls='--', lw=1.5)
        ax.set_title(r['event'], fontsize=10)
        ax.set_xlabel(r'$\Delta t_{\rm echo}$ (ms)', fontsize=9)
        ax.set_ylabel('PDF', fontsize=9)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        'Posterior PDFs of echo delay: LQG vs Schwarzschild\n'
        '(GWOSC posterior samples — Mode A)',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{filename_prefix}.{ext}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Posterior PDF figure saved to '{filename_prefix}.pdf/png'.")


def plot_eta_curve(results, filename="eta_curve"):
    a_arr   = np.linspace(0.001, 0.998, 1000)
    eta_arr = eta(a_arr)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a_arr, eta_arr, 'b-', lw=2.5,
            label=r'$\eta(a_\star)$ — exact Kerr correction')
    ax.axhline(1, color='red', ls='--', lw=1.8,
               label=r'Schwarzschild ($\eta=1$)')
    ax.fill_between(a_arr, 1, eta_arr, alpha=0.10, color='blue')
    cmap = plt.cm.tab20
    for i, r in enumerate(results[:20]):
        sym = 'o' if r['mc_mode'] == 'posterior_samples' else 's'
        ax.plot(r['af'], r['eta'], sym, color=cmap(i / 20), ms=8, zorder=5)
    ax.set_xlabel(r'Dimensionless spin $a_\star$', fontsize=13)
    ax.set_ylabel(r'$\eta(a_\star)$', fontsize=13)
    ax.set_title(r'Kerr tortoise correction factor $\eta(a_\star)$',
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.9, min(eta_arr.max() * 1.1, 10))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{filename}.{ext}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Eta curve figure saved to '{filename}.pdf/png'.")


def plot_bias_vs_spin(results, filename="bias_vs_spin"):
    afs    = [r['af']       for r in results]
    biases = [r['bias_pct'] for r in results]
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(afs, biases, c=biases, cmap='plasma',
                    s=60, zorder=5, edgecolors='gray', lw=0.5)
    plt.colorbar(sc, ax=ax, label='Schwarzschild bias (%)')
    a_arr = np.linspace(0.001, 0.998, 500)
    ax.plot(a_arr, (eta(a_arr) - 1.0) * 100.0, 'k--', lw=1.5, alpha=0.6,
            label=r'Analytical: $(\eta-1)\times100\%$')
    ax.set_xlabel(r'Final spin $a_{\star,f}$', fontsize=13)
    ax.set_ylabel(r'Schwarzschild underestimate (%)', fontsize=13)
    ax.set_title('Spin-dependent bias in predicted echo delay\n'
                 'LQG framework vs Schwarzschild approximation', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{filename}.{ext}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bias vs spin figure saved to '{filename}.pdf/png'.")


def plot_pipeline_mismatch(results, filename="pipeline_mismatch"):
    afs = [r['af'] for r in results]
    mis = [r['diff_ms'] / r['tau_damp_ms'] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(afs, mis, c=mis, cmap='viridis',
                    s=80, zorder=5, edgecolors='gray', lw=0.4)
    plt.colorbar(sc, ax=ax,
                 label=r'$\mathcal{M}=\delta t/\tau_{\rm damp}$')
    ax.axhline(1.0, color='red', ls='--', lw=2.0,
               label=r'$\mathcal{M}=1$')
    tau_label = "tau_v1.0 (GM/c3 Q)" if USE_V1_TAU else "tau = Q/(pi f)"
    ax.set_xlabel(r'Final spin $a_{\star,f}$', fontsize=13)
    ax.set_ylabel(r'Pipeline mismatch $\mathcal{M}$', fontsize=13)
    ax.set_title(f'Operational relevance of Kerr timing mismatch\n'
                 f'[{tau_label}]', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{filename}.{ext}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pipeline mismatch figure saved to '{filename}.pdf/png'.")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='LQG Echo Delay Analyzer v2.1 — Guilherme Junior (2026)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lqg_echo_analyzer.py
      -> Built-in GWTC data, Mode B (uniform Monte Carlo)

  python lqg_echo_analyzer.py --posteriors-dir ./posteriors/
      -> Mode A for events with HDF5; Mode B for the rest

  python lqg_echo_analyzer.py catalog.csv --posteriors-dir ./posteriors/
      -> External CSV + posterior samples

  python lqg_echo_analyzer.py --v1-tau
      -> Use v1.0 tau_damp formula (to reproduce original figures)

Bundled GWOSC posterior samples (in posteriors/ directory):
  GW150914, GW170729, GW190521, GW191109

How to obtain additional HDF5 files:
  pip install zenodo_get
  zenodo-get 6513631   # GWTC-1/O1/O2 via GWTC-2.1
  zenodo-get 5117702   # GWTC-2/O3a
  zenodo-get 5546662   # GWTC-3/O3b
        """
    )
    parser.add_argument('csv', nargs='?', default=None,
                        help='CSV file with events (optional)')
    parser.add_argument('--posteriors-dir', '-p', default='./posteriors/',
                        metavar='DIR', help='Directory with GWOSC HDF5 files '
                                            '(default: ./posteriors/)')
    parser.add_argument('--n-mc', type=int, default=5000,
                        help='Monte Carlo samples for Mode B (default: 5000)')
    parser.add_argument('--v1-tau', action='store_true',
                        help='Use v1.0 tau_damp formula '
                             '(reproduces original figures)')
    args = parser.parse_args()

    global USE_V1_TAU
    if args.v1_tau:
        USE_V1_TAU = True
        print("[WARNING] Using v1.0 tau_damp (GM/c3 Q) — "
              "for reproducing original figures only.")

    posteriors_dir = args.posteriors_dir or os.environ.get(
        'LQG_POSTERIORS_DIR', None)

    print("=" * 62)
    print("  LQG Echo Delay Analyzer  v2.1")
    print("  Guilherme Junior, M. S. (2026)")
    print("  ORCID: 0009-0002-5509-0683")
    print("=" * 62)

    if posteriors_dir:
        print(f"\n  Posterior samples directory: {posteriors_dir}")
        if not HDF5_AVAILABLE:
            print("  [WARNING] h5py not installed — Mode B for all events.")
        elif not os.path.isdir(posteriors_dir):
            print(f"  [WARNING] Directory not found: {posteriors_dir}")
            print("  Run without --posteriors-dir or create the directory.")
    else:
        print("\n  No posteriors directory — Mode B for all events.")

    events = load_events(args.csv)
    print(f"\n  Analysing {len(events)} events...\n")
    results = analyze(events, N_mc=args.n_mc, posteriors_dir=posteriors_dir)

    print_table(results)
    write_csv(results)
    write_latex_table(results)
    plot_comparison(results)
    plot_eta_curve(results)
    plot_bias_vs_spin(results)
    plot_pipeline_mismatch(results)
    plot_posterior_pdf(results)

    print("\n  Output files:")
    print("    echo_results.csv            full results")
    print("    echo_delays_table.tex       LaTeX table")
    print("    echo_comparison.pdf/png     LQG vs Schwarzschild")
    print("    eta_curve.pdf/png           eta(a_star) factor")
    print("    bias_vs_spin.pdf/png        bias vs spin")
    print("    pipeline_mismatch.pdf/png   mismatch parameter")
    print("    posterior_pdf.pdf/png       PDFs (Mode A, if available)")

if __name__ == "__main__":
    main()
