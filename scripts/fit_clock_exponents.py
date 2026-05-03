#!/usr/bin/env python3
"""
Appendix A clock exponent comparison for the Λcos paper.

Loops over four clock rates dt/dτ = S^n with:
    Model A: n = 0
    Model B: n = -1
    Model C: n = +1
    Model D: n = -1/2

For each model, computes the normalized expansion history

    H^2(S) ∝ cos^2(t/2) S^(2n-2) / 4
    S = s0 / (1+z)

then scales the matter sector to (1 - ΩΛ), adds ΩΛ = 0.685,
and fits Pantheon+ + DESI DR2 BAO with the same MCMC setup as fit_lcos.py.

Inputs expected in ../data/:
    pantheon_plus.csv
    pantheon_plus_cov.npy
    desi_dr2_bao.csv
    desi_dr2_bao_cov.npy

Run from repo root:
    python scripts/fit_clock_exponents.py

Outputs to ../results/:
    clock_exponent_results.csv
    clock_exponent_<model>_chain.npy
    clock_exponent_<model>_postburn.csv
"""

import json
from pathlib import Path

import emcee
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("../data")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OMEGA_LAMBDA = 0.685
OMEGA_M = 1.0 - OMEGA_LAMBDA
LCDM_CHI2_MIN = 1772.4

C_LIGHT = 299792.458  # km/s
H0_SN = 70.0          # absorbed into M_B for SN likelihood

NWALKERS = 32
NSTEPS = 5000
BURN = 1000
RNG_SEED = 12345


# -----------------------------
# Data loading
# -----------------------------
pantheon = pd.read_csv(DATA_DIR / "pantheon_plus.csv")
z_sn = pantheon["zHD"].to_numpy(float)
m_b_corr = pantheon["m_b_corr"].to_numpy(float)

cov_sn = np.load(DATA_DIR / "pantheon_plus_cov.npy")
cho_sn = cho_factor(cov_sn, lower=True, check_finite=False)

ones_sn = np.ones_like(z_sn)
cinv_ones = cho_solve(cho_sn, ones_sn, check_finite=False)
ones_cinv_ones = float(ones_sn @ cinv_ones)

bao = pd.read_csv(DATA_DIR / "desi_dr2_bao.csv")
z_bao = bao["z_eff"].to_numpy(float)
bao_values = bao["value"].to_numpy(float)
cov_bao = np.load(DATA_DIR / "desi_dr2_bao_cov.npy")
inv_cov_bao = np.linalg.inv(cov_bao)

zmax = max(float(np.max(z_sn)), float(np.max(z_bao))) * 1.002
z_grid = np.linspace(0.0, zmax, 5000)


# -----------------------------
# Model functions
# -----------------------------
def e2_clock_model(z: np.ndarray, s0: float, n_exp: float) -> np.ndarray:
    """
    Normalized clock-exponent model.

    H^2(S) ∝ (1 - S^2) S^(2n-2)
    S = s0 / (1+z)

    Normalized by the z=0 value:
        R(z) = [(1-S^2) S^(2n-2)] / [(1-s0^2) s0^(2n-2)]

    Full model:
        E^2 = Ω_m R(z) + Ω_Λ
    """
    z = np.asarray(z, dtype=float)
    if not (0.0 < s0 < 1.0):
        return np.full_like(z, np.nan)

    S = s0 / (1.0 + z)
    ratio = ((1.0 - S**2) * S**(2.0 * n_exp - 2.0)) / (
        (1.0 - s0**2) * s0**(2.0 * n_exp - 2.0)
    )
    return OMEGA_M * ratio + OMEGA_LAMBDA


def get_e_and_integral(s0: float, n_exp: float):
    e2 = e2_clock_model(z_grid, s0, n_exp)
    if np.any(~np.isfinite(e2)) or np.any(e2 <= 0):
        return None, None
    e = np.sqrt(e2)
    integral = cumulative_trapezoid(1.0 / e, z_grid, initial=0.0)
    return e, integral


def sn_chi2(s0: float, n_exp: float, M_B: float) -> float:
    e, integral = get_e_and_integral(s0, n_exp)
    if e is None:
        return np.inf

    I = np.interp(z_sn, z_grid, integral)
    d_l_mpc = (1.0 + z_sn) * (C_LIGHT / H0_SN) * I
    if np.any(d_l_mpc <= 0) or np.any(~np.isfinite(d_l_mpc)):
        return np.inf

    mu_model = 5.0 * np.log10(d_l_mpc) + 25.0
    delta = m_b_corr - M_B - mu_model
    return float(delta @ cho_solve(cho_sn, delta, check_finite=False))


def bao_model_vector(s0: float, H0rd: float, n_exp: float) -> np.ndarray:
    e, integral = get_e_and_integral(s0, n_exp)
    if e is None:
        return np.full(len(bao), np.nan)

    I = np.interp(z_bao, z_grid, integral)
    E_bao = np.interp(z_bao, z_grid, e)

    D_M = C_LIGHT / H0rd * I
    D_H = C_LIGHT / H0rd / E_bao
    D_V = (z_bao * D_M**2 * D_H) ** (1.0 / 3.0)

    model = np.empty(len(bao), dtype=float)
    for i, obs in enumerate(bao["observable"]):
        if obs == "DV_rd":
            model[i] = D_V[i]
        elif obs == "DM_rd":
            model[i] = D_M[i]
        elif obs == "DH_rd":
            model[i] = D_H[i]
        else:
            raise ValueError(f"Unknown BAO observable: {obs}")

    return model


def bao_chi2(s0: float, H0rd: float, n_exp: float) -> float:
    model = bao_model_vector(s0, H0rd, n_exp)
    if np.any(~np.isfinite(model)):
        return np.inf
    delta = model - bao_values
    return float(delta @ inv_cov_bao @ delta)


def total_chi2(theta: np.ndarray, n_exp: float):
    s0, H0rd, M_B = theta
    if not (0.001 <= s0 <= 0.99 and 8000.0 <= H0rd <= 12000.0 and -20.0 <= M_B <= -18.0):
        return np.inf
    c_sn = sn_chi2(s0, n_exp, M_B)
    if not np.isfinite(c_sn):
        return np.inf
    c_bao = bao_chi2(s0, H0rd, n_exp)
    return c_sn + c_bao


def log_prob(theta: np.ndarray, n_exp: float) -> float:
    chi2 = total_chi2(theta, n_exp)
    if not np.isfinite(chi2):
        return -np.inf
    return -0.5 * chi2


# -----------------------------
# Initialization helpers
# -----------------------------
def best_initial_point(n_exp: float):
    """
    Fast deterministic pre-optimization to seed walkers.
    """
    starts = [
        np.array([0.05, 10000.0, -19.35]),
        np.array([0.20, 10000.0, -19.35]),
        np.array([0.50, 9500.0, -19.30]),
        np.array([0.85, 9300.0, -19.25]),
        np.array([0.95, 9200.0, -19.10]),
    ]

    best = None
    bounds = [(0.001, 0.99), (8000.0, 12000.0), (-20.0, -18.0)]

    for start in starts:
        res = minimize(
            lambda x: total_chi2(x, n_exp),
            start,
            method="Nelder-Mead",
            options={"maxiter": 1200, "xatol": 1e-8, "fatol": 1e-6},
        )
        res2 = minimize(
            lambda x: total_chi2(x, n_exp),
            res.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1200, "ftol": 1e-9},
        )
        candidate = res2 if res2.fun <= res.fun else res
        if best is None or candidate.fun < best.fun:
            best = candidate

    x = np.array(best.x, dtype=float)
    x[0] = np.clip(x[0], 0.001, 0.99)
    x[1] = np.clip(x[1], 8000.0, 12000.0)
    x[2] = np.clip(x[2], -20.0, -18.0)
    return x


def initialize_walkers(center: np.ndarray, rng: np.random.Generator):
    """
    Create walkers near the best point while respecting priors.
    """
    scales = np.array([0.01, 25.0, 0.01])
    p0 = center + scales * rng.normal(size=(NWALKERS, 3))

    p0[:, 0] = np.clip(p0[:, 0], 0.001, 0.99)
    p0[:, 1] = np.clip(p0[:, 1], 8000.0, 12000.0)
    p0[:, 2] = np.clip(p0[:, 2], -20.0, -18.0)

    return p0


def summarize_chain(chain_post: np.ndarray, n_exp: float):
    """
    Return best-fit point and chi2 split from post-burn chain.
    """
    logp = np.array([log_prob(theta, n_exp) for theta in chain_post])
    best = chain_post[np.argmax(logp)]
    s0, H0rd, M_B = best

    c_sn = sn_chi2(s0, n_exp, M_B)
    c_bao = bao_chi2(s0, H0rd, n_exp)
    c_total = c_sn + c_bao

    return best, c_sn, c_bao, c_total


# -----------------------------
# Main run
# -----------------------------
def main():
    rng = np.random.default_rng(RNG_SEED)

    models = [
        ("ΛCDM baseline", np.nan, None),
        ("A", 0.0, "proper_time"),
        ("B", -1.0, "conformal"),
        ("C", 1.0, "symmetric"),
        ("D", -0.5, "budget"),
    ]

    rows = [{
        "model": "ΛCDM baseline",
        "n": np.nan,
        "best_s0": np.nan,
        "best_H0rd": np.nan,
        "chi2_SN": np.nan,
        "chi2_BAO": np.nan,
        "chi2_total": LCDM_CHI2_MIN,
        "delta_chi2": 0.0,
        "acceptance_fraction": np.nan,
        "tau_max": np.nan,
    }]

    for model_label, n_exp, tag in models[1:]:
        print(f"\nRunning Model {model_label}: n = {n_exp}")

        center = best_initial_point(n_exp)
        print(f"  Initial point: s0={center[0]:.6f}, H0rd={center[1]:.2f}, M_B={center[2]:.5f}")

        p0 = initialize_walkers(center, rng)

        sampler = emcee.EnsembleSampler(
            NWALKERS,
            3,
            lambda theta: log_prob(theta, n_exp),
        )
        sampler.run_mcmc(p0, NSTEPS, progress=True)

        chain = sampler.get_chain()
        chain_post = chain[BURN:].reshape(-1, 3)
        acceptance = float(np.mean(sampler.acceptance_fraction))
        try:
            tau = emcee.autocorr.integrated_time(chain[BURN:], c=5, tol=0)
            tau_max = float(np.max(tau))
        except Exception:
            tau_max = float("nan")

        chain_path = RESULTS_DIR / f"clock_exponent_{model_label}_chain.npy"
        post_path = RESULTS_DIR / f"clock_exponent_{model_label}_postburn.csv"
        np.save(chain_path, chain)

        pd.DataFrame(
            chain_post,
            columns=["s0", "H0rd", "M_B"],
        ).to_csv(post_path, index=False)

        best, c_sn, c_bao, c_total = summarize_chain(chain_post, n_exp)
        delta = c_total - LCDM_CHI2_MIN

        rows.append({
            "model": model_label,
            "n": n_exp,
            "best_s0": best[0],
            "best_H0rd": best[1],
            "chi2_SN": c_sn,
            "chi2_BAO": c_bao,
            "chi2_total": c_total,
            "delta_chi2": delta,
            "acceptance_fraction": acceptance,
            "tau_max": tau_max,
        })

        print(
            f"  Best: s0={best[0]:.6f}, H0rd={best[1]:.2f}, M_B={best[2]:.5f}, "
            f"chi2={c_total:.3f}, Δχ2={delta:.3f}, acceptance={acceptance:.3f}"
        )

    results = pd.DataFrame(rows)
    results_path = RESULTS_DIR / "clock_exponent_results.csv"
    results.to_csv(results_path, index=False)

    print("\nClock exponent comparison:")
    print(results.to_string(index=False))
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
