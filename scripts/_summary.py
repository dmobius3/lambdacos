"""
Shared summary-writing helper for fit_*.py scripts.

Produces a harmonized JSON schema across fit_lcdm, fit_lcos, fit_wcdm,
fit_lcdm_cmb, and fit_lcos_cmb so a referee comparing the deposit sees
the same keys for every model:

    {
      "model": <str>,
      "fixed": {<param>: <val>, ...} | null,
      "param_names": [<str>, ...],
      "best_fit": {<param>: <val>, ...},
      "posterior_quantiles": {
        <param>: {"16": <val>, "median": <val>, "84": <val>},
        ...
      },
      "extras": {<key>: <val>, ...},
      "chi2": {
        "total": <val>,
        "SN": <val>,
        "BAO": <val>,
        "CMB": <val> | null
      },
      "tau_per_param": [<val>, ...],
      "tau_max": <val>,
      "acceptance": <val>
    }

Notes:
  - "best_fit" is the scipy.optimize minimum seeded from the chain argmax,
    not the chain argmax itself (so it is a true local minimum of chi2_total).
  - "extras" carries model-specific quantities that don't fit the common
    schema, e.g., {"s0_95UL": 0.181} for Λcos, or
    {"delta_chi2_vs_LCDM": -13.05, ...} for wCDM.
  - "chi2.CMB" is null for SN+BAO-only fits, present for CMB-prior fits.
"""

import json
import numpy as np
import emcee
from scipy.optimize import minimize


def write_summary(
    out_path,
    model_name,
    param_names,
    post_chain,
    chain_post_burn,
    log_prob_post,
    chi2_func,
    bounds_check,
    acceptance,
    fixed=None,
    extras=None,
):
    """Compute the harmonized summary and write to out_path.

    Args:
        out_path: filesystem path to the .json output.
        model_name: human-readable model label.
        param_names: list of parameter names matching post_chain columns.
        post_chain: post-burn chain reshaped to (N, ndim).
        chain_post_burn: post-burn chain with shape (nsteps_post, nwalkers, ndim),
                         used for emcee.autocorr.integrated_time.
        log_prob_post: log-probability values for post_chain, shape (N,).
        chi2_func: callable(theta) -> dict with keys "total", "SN", "BAO",
                   and optionally "CMB". Used both for the optimizer and for
                   the decomposition at the best-fit point.
        bounds_check: callable(theta) -> bool; the optimizer returns 1e10
                      when bounds_check is False.
        acceptance: scalar mean acceptance fraction across walkers.
        fixed: optional dict of fixed parameters (e.g., {"omega_lambda": 0.685}).
        extras: optional dict of model-specific extras.

    Returns:
        The summary dict (also written to out_path).
    """
    ndim = len(param_names)

    # Best fit via scipy.optimize seeded from the chain argmax.
    chain_argmax = post_chain[np.argmax(log_prob_post)]
    opt = minimize(
        lambda x: chi2_func(x)["total"] if bounds_check(x) else 1e10,
        chain_argmax,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-7, "maxiter": 8000},
    )
    best_fit = {param_names[i]: float(opt.x[i]) for i in range(ndim)}
    decomp = chi2_func(opt.x)

    # Posterior quantiles per parameter.
    post_q = {
        name: {
            "16": float(np.percentile(post_chain[:, i], 16)),
            "median": float(np.percentile(post_chain[:, i], 50)),
            "84": float(np.percentile(post_chain[:, i], 84)),
        }
        for i, name in enumerate(param_names)
    }

    # Integrated autocorrelation time.
    try:
        tau = emcee.autocorr.integrated_time(chain_post_burn, c=5, tol=0)
        tau_per_param = [float(t) for t in tau]
        tau_max = float(np.max(tau))
    except Exception:
        tau_per_param, tau_max = None, None

    chi2_block = {
        "total": float(decomp["total"]),
        "SN": float(decomp.get("SN", float("nan"))),
        "BAO": float(decomp.get("BAO", float("nan"))),
        "CMB": float(decomp["CMB"]) if decomp.get("CMB") is not None else None,
    }

    summary = {
        "model": model_name,
        "fixed": fixed,
        "param_names": list(param_names),
        "best_fit": best_fit,
        "posterior_quantiles": post_q,
        "extras": extras or {},
        "chi2": chi2_block,
        "tau_per_param": tau_per_param,
        "tau_max": tau_max,
        "acceptance": float(acceptance),
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
