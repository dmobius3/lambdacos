"""
Aggregate the §5.4 Ω_Λ sensitivity table from individual fit_lcos.py runs.

Run after fit_lcos.py has been run at each value:
    python fit_lcos.py --omega_lambda 0.680
    python fit_lcos.py --omega_lambda 0.685   # canonical
    python fit_lcos.py --omega_lambda 0.690
    python fit_lcos.py --omega_lambda 0.700
    python fit_lcos.py --omega_lambda 0.715

Then:
    python omega_lambda_scan.py

Output:
    tables/omega_lambda_scan.csv
"""

import json
import numpy as np, pandas as pd

OMEGA_VALUES = [0.680, 0.685, 0.690, 0.700, 0.715]

# Read flat ΛCDM baseline chi2_min from its summary, with a hardcoded fallback.
import os
LCDM_PATH = "../results/lcdm_summary.json"
if os.path.exists(LCDM_PATH):
    with open(LCDM_PATH) as f:
        LCDM_chi2_baseline = json.load(f)["chi2"]["total"]
else:
    LCDM_chi2_baseline = 1772.456

rows = []
for OL in OMEGA_VALUES:
    if abs(OL - 0.685) < 1e-6:
        suffix = ""
        label = f"{OL} (canonical)"
    else:
        suffix = "_omegaL_" + f"{OL:.3f}".replace(".", "p")
        label = f"{OL}"
    path = f"../results/lcos{suffix}_summary.json"
    with open(path) as f:
        s = json.load(f)
    rows.append({
        "Omega_Lambda": OL,
        "label": label,
        "s0_median": s["posterior_quantiles"]["s0"]["median"],
        "s0_16": s["posterior_quantiles"]["s0"]["16"],
        "s0_84": s["posterior_quantiles"]["s0"]["84"],
        "s0_95UL": s["extras"]["s0_95UL"],
        "best_fit_s0": s["best_fit"]["s0"],
        "best_fit_H0rd": s["best_fit"]["H0rd"],
        "best_fit_MB": s["best_fit"]["MB"],
        "chi2_min": s["chi2"]["total"],
        "chi2_SN": s["chi2"]["SN"],
        "chi2_BAO": s["chi2"]["BAO"],
        "Delta_chi2_vs_LCDM": s["chi2"]["total"] - LCDM_chi2_baseline,
        "tau_max": s["tau_max"],
        "acceptance": s["acceptance"],
    })

df = pd.DataFrame(rows)
df.to_csv("../tables/omega_lambda_scan.csv", index=False)
print(df.to_string(index=False))
print(f"\nLCDM baseline chi2_total used: {LCDM_chi2_baseline:.4f}")
