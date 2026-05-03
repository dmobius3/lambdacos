# Λcos: Apparent Phantom Crossing as Template Bias

Code and data for the paper:

**B. Shatto, "Apparent Phantom Crossing as Template Bias: A Non-Phantom Test Case with Λcos" (2026).**

This repository contains the analysis pipeline and figure-generation scripts needed to reproduce all results in the paper.

---

## Overview

The Λcos model is a one-parameter deformation of the fiducial flat ΛCDM expansion history using a bounded auxiliary variable. It yields

$$\frac{H^2(z)}{H_0^2} = \alpha\,(1+z)^3 \;-\; \beta\,(1+z) \;+\; \Omega_\Lambda$$

with $\alpha$, $\beta$ determined by $s_0$ and a fixed reference $\Omega_\Lambda$. Under the fiducial-matter diagnostic split, the effective residual satisfies $w_\mathrm{eff}(z) > -1$.

Reproducible results in this repository:

- **Joint Pantheon+ + DESI DR2 BAO fit** for ΛCDM, Λcos, and wCDM (§5.2, §5.7)
- **Template-bias mock fits** across CPL, BA, JBP, and a three-parameter polynomial (§4.2)
- **CPL threshold scan** across $s_0 \in [0.01, 0.40]$ (§4.3)
- **Clock exponent comparison** for Models A, B, C, D (Appendix A)
- **Figure generation** for Figs. 1–4

---

## Citation

If you use this code or data, please cite the paper and the Zenodo archive of this repository.

```bibtex
@article{Shatto2026Lambdacos,
  title  = {Apparent Phantom Crossing as Template Bias: A Non-Phantom
            Test Case with Λcos},
  author = {Shatto, B.},
  year   = {2026}
}

@misc{ShattoLambdacosCode2026,
  author = {Shatto, B.},
  title  = {Λcos: Code and Data for "Apparent Phantom Crossing as Template Bias"},
  year   = {2026},
  doi    = {10.5281/zenodo.19798852}
}
```

---

## Repository contents

```
.
├── README.md                                  This file
├── LICENSE                                    MIT
├── requirements.txt                           Python dependencies
├── data/
│   ├── pantheon_plus.csv                      Pantheon+ SNe Ia magnitudes
│   ├── pantheon_plus_cov.npy                  1701 × 1701 statistical + systematic covariance
│   ├── desi_dr2_bao.csv                       DESI DR2 BAO observables
│   └── desi_dr2_bao_cov.npy                   Inter-observable covariance for the 13 BAO points
├── scripts/
│   ├── fit_lcdm.py                            Flat ΛCDM MCMC fit (§5.2)
│   ├── fit_lcos.py                            Λcos MCMC fit, Ω_Λ = 0.685 fixed (§5.2)
│   ├── fit_wcdm.py                            wCDM MCMC fit (§5.7)
│   ├── fit_clock_exponents.py                 Clock exponent comparison (Appendix A)
│   ├── template_bias.py                       Template-bias mocks + Fig. 1 (§4.2)
│   ├── threshold_scan.py                      CPL threshold scan + Fig. 2 (§4.3)
│   └── make_plots.py                          Λcos corner (Fig. 3) and residuals (Fig. 4)
├── results/                                   MCMC chains, post-burn samples, summaries, generated figures
│   ├── lcdm_chain.npy, lcdm_post.csv, lcdm_summary.json, lcdm_corner.png
│   ├── lcos_chain.npy, lcos_post.csv, lcos_summary.json, lcos_corner.{png,pdf}
│   ├── wcdm_chain.npy, wcdm_post.csv, wcdm_summary.json, wcdm_corner.png
│   ├── clock_exponent_{A,B,C,D}_chain.npy
│   ├── clock_exponent_{A,B,C,D}_postburn.csv
│   ├── clock_exponent_results.csv             Appendix A summary across all four models
│   ├── template_bias.csv, template_bias.{png,pdf}     §4.2 fits and Fig. 1
│   ├── threshold_scan.csv, threshold_scan.{png,pdf}   §4.3 scan and Fig. 2
│   └── residuals.{png,pdf}                    Fig. 4
├── tables/
│   └── clock_exponent_appendix_A_fits.csv     Curated Appendix A reference values
└── figures/
    ├── fig1_template_bias_overlay.pdf         §4.2: w(z) overlays for CPL/BA/JBP/Polynomial
    ├── fig2_threshold_scan.pdf                §4.3: recovered (w₀, w_a) vs s₀
    ├── fig3_lcos_corner.pdf                   §5.2: Λcos posterior in (s₀, H₀r_d, M_B)
    └── fig4_hubble_residuals.pdf              §5.2: Pantheon+ binned residuals for ΛCDM and Λcos
```

`figures/` holds the paper-facing PDFs at their published filenames. They are stable copies of the corresponding script outputs in `results/`.

---

## Installation

Python 3.10 or later recommended.

```bash
git clone https://github.com/dmobius3/lambda-cos.git
cd lambda-cos
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):

```
numpy>=1.24
scipy>=1.10
emcee>=3.1
corner>=2.2
matplotlib>=3.7
h5py>=3.8
pandas>=2.0
```

Total install footprint about 200 MB. The primary SN+BAO MCMC fit takes roughly 5 minutes on a recent laptop (32 walkers, 5000 steps).

---

## Quickstart: reproducing the primary fit

The headline result is the joint Pantheon+ + DESI DR2 BAO Λcos fit (§5.2). All scripts are written to be run from the `scripts/` directory and resolve paths via `../data/` and `../results/`.

```bash
cd scripts
python fit_lcos.py
```

Outputs to `results/`: `lcos_chain.npy`, `lcos_post.csv`, `lcos_summary.json`, `lcos_corner.png`.

For the ΛCDM baseline:

```bash
python fit_lcdm.py
```

For wCDM (§5.7):

```bash
python fit_wcdm.py
```

Reference summary values from the deposited posteriors:

```
Λcos:    s0        ≈ 0.081  (mean), 0.072 (median)
         s0 95% UL ≈ 0.181  (flat prior)
         H0 r_d    ≈ 10007  km/s
         M_B       ≈ -19.353
         tau_max   ≈ 45.9
ΛCDM:    Ω_m       ≈ 0.312
         H0 r_d    ≈ 10045  km/s
         M_B       ≈ -19.355
         tau_max   ≈ 35.8
wCDM:    Ω_m       ≈ 0.296
         w         ≈ -0.853
         Δχ²       ≈ -13.05 vs flat ΛCDM
         ΔAIC      ≈ -11.05
         ΔBIC      ≈  -5.60
         tau_max   ≈ 44.9
```

---

## Reproducing each figure

```bash
cd scripts

# Figure 1 — w(z) recoveries from CPL, BA, JBP, Polynomial
python template_bias.py
# -> results/template_bias.{png,pdf}

# Figure 2 — CPL threshold scan
python threshold_scan.py
# -> results/threshold_scan.{png,pdf}

# Figures 3 and 4 — Λcos corner plot and Pantheon+ residuals
# Requires lcdm_post.csv and lcos_post.csv (run fit_lcdm.py and fit_lcos.py first)
python make_plots.py
# -> results/lcos_corner.{png,pdf}   (Fig. 3)
# -> results/residuals.{png,pdf}     (Fig. 4)
```

The paper-facing PDFs in `figures/` (`fig1_template_bias_overlay.pdf`, `fig2_threshold_scan.pdf`, `fig3_lcos_corner.pdf`, `fig4_hubble_residuals.pdf`) are stable copies of the corresponding `results/` outputs renamed to match the in-paper figure numbers.

---

## Reproducing the template-bias scan (§4.3)

```bash
cd scripts
python threshold_scan.py
```

Iterates CPL fits across $s_0 \in [0.01, 0.40]$ in steps of $0.01$, recording $(w_0, w_a, \chi^2)$ at each value to `results/threshold_scan.csv` and producing Fig. 2.

The single-$s_0$ mock comparison across all four parameterizations (Table in §4.2) is produced separately by `template_bias.py`, which writes `results/template_bias.csv` and Fig. 1.

---

## Reproducing Appendix A (clock exponent selection)

```bash
cd scripts
python fit_clock_exponents.py
```

Models A (n = 0), B (n = −1), C (n = +1), D (n = −1/2) are fit with the same MCMC setup as the primary Λcos run. Outputs to `results/`:

- `clock_exponent_{A,B,C,D}_chain.npy` — full chains
- `clock_exponent_{A,B,C,D}_postburn.csv` — post-burn samples
- `clock_exponent_results.csv` — summary with one row per model: best-fit parameters, χ² split (SN, BAO, total), Δχ² vs ΛCDM, acceptance fraction

The curated paper-facing values are also deposited at `tables/clock_exponent_appendix_A_fits.csv`.

---

## Data sources and provenance

| Dataset | Source | Reference |
|---|---|---|
| Pantheon+ SNe Ia | [pantheonplussh0es.github.io](https://pantheonplussh0es.github.io/) | Brout et al., *Astrophys. J.* **938**, 110 (2022) |
| DESI DR2 BAO | [data.desi.lbl.gov](https://data.desi.lbl.gov/) | DESI Collaboration, arXiv:2503.14738 (2025) |
| Planck 2018 distance priors | Compressed (R, ℓ_A) from Planck VI | Planck Collaboration VI, *Astron. Astrophys.* **641**, A6 (2020) |

The files under `data/` are formatted derivatives of the public sources above, repackaged for direct loading by the fit scripts. No proprietary data is included.

---

## Reproducibility notes

- **Random seeding**: `fit_clock_exponents.py` uses a fixed seed (`RNG_SEED = 12345`). The other MCMC scripts (`fit_lcdm.py`, `fit_lcos.py`, `fit_wcdm.py`) initialize walkers from `np.random.randn` without an explicit seed; small numerical variations between runs are within the posterior thickness and do not change the reported summary values.
- **MCMC configuration**: 32 walkers, 5000 steps, 1000 burn-in across all fit scripts.
- **Numerical accuracy**: distance integrals use SciPy's `cumulative_trapezoid` on a 4000-point grid in z ∈ [0, 2.5] (or up to z\_max ≈ 2.4 from the BAO range). For the clock-exponent run the grid extends to 1.002 × max(z\_data).
- **Working directory**: scripts are run from the `scripts/` subdirectory; paths resolve via `../data/` and `../results/`.
- **Platform**: tested on macOS 14.x and Ubuntu 22.04 with Python 3.11. No GPU required.

---

## License

This repository is released under the MIT License. See `LICENSE` for the full text.

The Pantheon+ and DESI DR2 BAO data products are redistributed under the terms of their original publications; refer to the linked sources above for their license terms.

---

## Contact

- Author: B. Shatto, bshatto.pe@gmail.com
- Issues and questions: please open a [GitHub issue](https://github.com/dmobius3/lambda-cos/issues).
