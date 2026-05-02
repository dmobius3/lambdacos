# Λcos: Apparent Phantom Crossing as Template Bias

Code and data for the paper:

**B. Shatto, "Apparent Phantom Crossing as Template Bias: A Non-Phantom Test Case with Λcos" (2026).**

This repository contains the analysis pipeline, MCMC chains, and figure-generation scripts needed to reproduce all results in the paper.

---

## Overview

The Λcos model is a one-parameter deformation of the fiducial flat ΛCDM expansion history using a bounded auxiliary variable. It yields

$$\frac{H^2(z)}{H_0^2} = \alpha\,(1+z)^3 \;-\; \beta\,(1+z) \;+\; \Omega_\Lambda$$

with $\alpha$, $\beta$ determined by $s_0$ and a fixed reference $\Omega_\Lambda$. Under the fiducial-matter diagnostic split, the effective residual satisfies $w_\mathrm{eff}(z) > -1$.

Reproducible results in this repository:

- **Joint Pantheon+ + DESI DR2 BAO fit** (§5.2 of the paper)
- **Prior sensitivity test** (§5.3)
- **$\Omega_\Lambda$ sensitivity scan** (§5.4)
- **CMB distance-prior fit** (§5.5)
- **wCDM model comparison** (§5.7)
- **Template-bias mock fits** for CPL, BA, JBP, and three-parameter polynomial (§4)
- **Threshold scan** of CPL recovered parameters across $s_0 \in [0.01, 0.40]$ (§4.3)
- **Clock exponent selection** (Appendix A)

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
├── README.md                        This file
├── LICENSE                          MIT
├── requirements.txt                 Python dependencies
├── data/
│   ├── pantheon_plus.csv            Pantheon+ SNe Ia magnitudes
│   ├── pantheon_plus_cov.npy        1701 × 1701 statistical + systematic covariance
│   ├── desi_dr2_bao.csv             DESI DR2 BAO observables
│   └── desi_dr2_bao_cov.npy         Inter-observable covariance for the 13 BAO points
├── scripts/
│   ├── lambdacos.py                 Λcos H²(z), w_eff(z), distance integrals
│   ├── likelihoods.py               SN, BAO, CMB-distance-prior likelihoods
│   ├── run_mcmc.py                  emcee driver (parameterized over model and dataset)
│   ├── template_bias.py             §4 mock fits and §4.3 threshold scan
│   ├── clock_exponent_appendix.py   Appendix A alternative-clock fits
│   └── make_figures.py              Figures 1–4 from chain outputs
├── chains/
│   ├── lambdacdm_sn_bao.h5          ΛCDM primary fit
│   ├── lambdacos_sn_bao.h5          Λcos primary fit, Ω_Λ = 0.685 (§5.2)
│   ├── lambdacos_omegaL_*.h5        Λcos at Ω_Λ ∈ {0.680, 0.685, 0.690, 0.700, 0.715} (§5.4)
│   ├── lambdacos_cmb_priors.h5      Λcos with compressed Planck priors (§5.5)
│   ├── wcdm_sn_bao.h5               wCDM comparison (§5.7)
│   └── clock_exp_model_*.h5         Models A, B, C, D from Appendix A
├── figures/
│   ├── fig1_template_bias_overlay.pdf   §4.2: w(z) overlays for CPL/BA/JBP/Polynomial
│   ├── fig2_threshold_scan.pdf          §4.3: recovered (w₀, w_a) vs s₀
│   ├── fig3_lcos_corner.pdf             §5.2: Λcos posterior in (s₀, H₀r_d, M_B)
│   └── fig4_hubble_residuals.pdf        §5.2: Pantheon+ binned residuals for ΛCDM and Λcos
└── tables/
    └── clock_exponent_appendix_A_fits.csv
```

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

The headline result of the paper is the joint SN+BAO Λcos fit (§5.2). To reproduce:

```bash
python scripts/run_mcmc.py \
    --model lambdacos \
    --omega_lambda 0.685 \
    --data sn+bao \
    --output chains/lambdacos_sn_bao.h5
```

Expected posterior summary:

```
s_0       = 0.075   [0.024, 0.143]   (median, 68% CI)
H_0 r_d   = 10010   [9973, 10041]    km/s
M_B       = -19.353 [-19.357, -19.350]
chi^2_min = 1772.6  (vs LambdaCDM 1772.4, Delta chi^2 = +0.13)
s_0 95% UL = 0.18 (flat prior)
```

For the ΛCDM baseline:

```bash
python scripts/run_mcmc.py --model lambdacdm --data sn+bao \
    --output chains/lambdacdm_sn_bao.h5
```

---

## Reproducing each figure

After the chains are produced, the figures are generated from the saved chain files:

```bash
# Figure 1 — w(z) recoveries from CPL, BA, JBP, Polynomial
python scripts/make_figures.py --figure 1 \
    --output figures/fig1_template_bias_overlay.pdf

# Figure 2 — CPL threshold scan
python scripts/make_figures.py --figure 2 \
    --scan tables/threshold_scan.csv \
    --output figures/fig2_threshold_scan.pdf

# Figure 3 — corner plot of (s_0, H_0 r_d, M_B)
python scripts/make_figures.py --figure 3 \
    --chain chains/lambdacos_sn_bao.h5 \
    --output figures/fig3_lcos_corner.pdf

# Figure 4 — Pantheon+ residuals
python scripts/make_figures.py --figure 4 \
    --lcdm-chain chains/lambdacdm_sn_bao.h5 \
    --lcos-chain chains/lambdacos_sn_bao.h5 \
    --output figures/fig4_hubble_residuals.pdf
```

Pre-rendered PDFs are included in `figures/` for convenience.

---

## Reproducing the template-bias scan (§4.3)

```bash
python scripts/template_bias.py --scan \
    --s0-range 0.01 0.40 \
    --step 0.01 \
    --parameterization cpl \
    --output tables/threshold_scan.csv
```

Iterates CPL fits across $s_0 \in [0.01, 0.40]$ in steps of $0.01$, recording $(w_0, w_a)$ at each value. The output `threshold_scan.csv` feeds Figure 2.

For single-$s_0$ mock fits across all four parameterizations (Table in §4.2):

```bash
python scripts/template_bias.py --single \
    --s0 0.389 \
    --output tables/single_mock_fits.csv
```

---

## Reproducing Appendix A (clock exponent selection)

```bash
python scripts/clock_exponent_appendix.py \
    --models A B C D \
    --output tables/clock_exponent_appendix_A_fits.csv
```

Models A (n=0), B (n=−1), C (n=+1), D (n=−1/2) are fit with the same MCMC setup as the primary Λcos run. Output is a CSV with one row per model:

| Column | Description |
|---|---|
| Model | A, B, C, D |
| n | clock exponent (S^n in dt/dτ) |
| Best-fit s_0 | posterior median |
| Best-fit H_0 r_d | posterior median, km/s |
| chi2_SN | χ²_SN at best fit |
| chi2_BAO | χ²_BAO at best fit |
| chi2_total | total χ² |
| Delta_chi2_vs_LCDM | relative to ΛCDM baseline 1772.445 |
| Acceptance fraction | MCMC acceptance rate |
| Convergence status | "stable" or boundary diagnostic |

The reference output is included in `tables/clock_exponent_appendix_A_fits.csv`.

---

## Data sources and provenance

| Dataset | Source | Reference |
|---|---|---|
| Pantheon+ SNe Ia | [pantheonplussh0es.github.io](https://pantheonplussh0es.github.io/) | Brout et al., *Astrophys. J.* **938**, 110 (2022) |
| DESI DR2 BAO | [data.desi.lbl.gov](https://data.desi.lbl.gov/) | DESI Collaboration, arXiv:2503.14738 (2025) |
| Planck 2018 distance priors | Compressed (R, ℓ_A) from Planck VI | Planck Collaboration VI, *Astron. Astrophys.* **641**, A6 (2020) |

The files under `data/` are formatted derivatives of the public sources above, repackaged for direct loading by `scripts/likelihoods.py`. No proprietary data is included.

---

## Reproducibility notes

- **Random seeds**: every script accepts a `--seed` argument; default is 42 for deterministic chains.
- **MCMC convergence**: the integrated autocorrelation time τ is reported at the end of each chain. The primary fit converges with τ < 50 across all parameters.
- **Numerical accuracy**: distance integrals use SciPy's `quad` with `epsrel=1e-8`. CMB-prior integrals to z\* = 1090 use the same tolerance with the radiation term included (Ω_r = 9.15 × 10⁻⁵).
- **Platform**: tested on macOS 14.x and Ubuntu 22.04 with Python 3.11. No GPU required.

---

## License

This repository is released under the MIT License. See `LICENSE` for the full text.

The Pantheon+ and DESI DR2 BAO data products are redistributed under the terms of their original publications; refer to the linked sources above for their license terms.

---

## Contact

- Author: B. Shatto, bshatto.pe@gmail.com
- Issues and questions: please open a [GitHub issue](https://github.com/dmobius3/lambda-cos/issues).
