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
LCDM_CHI2_BASELINE = 1772.456  # from fit_lcdm.py optimum (see lcdm_summary.json)

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
        "s0_median": s["s0_median"],
        "s0_95UL": s["s0_95"],
        "best_fit_s0": s["best_fit_s0"],
        "best_fit_H0rd": s["best_fit_H0rd"],
        "best_fit_MB": s["best_fit_MB"],
        "chi2_min": s["chi2_min"],
        "chi2_SN": s["chi2_SN"],
        "chi2_BAO": s["chi2_BAO"],
        "Delta_chi2_vs_LCDM": s["chi2_min"] - LCDM_CHI2_BASELINE,
        "tau_max": s["tau_max"],
        "acceptance": s["acceptance"],
    })

df = pd.DataFrame(rows)
df.to_csv("../tables/omega_lambda_scan.csv", index=False)
print(df.to_string(index=False))
