"""
Plot final results: corner + residuals + summary.

Loads the post-burn posteriors from fit_lcdm.py and fit_lcos.py, builds the
Λcos corner plot (Fig. 3) and the binned Pantheon+ residuals overlay (Fig. 4),
and prints a brief model-comparison summary.

Run:
    python scripts/make_plots.py
Outputs:
    results/lcos_corner.png   Fig. 3
    results/lcos_corner.pdf
    results/residuals.png     Fig. 4
    results/residuals.pdf
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve

# ---------------- CONFIG ----------------
ΩΛ = 0.685
c = 299792.458
H0_ref = 70.0  # km/s/Mpc, used in distance modulus (degenerate with MB)
nbins = 50

data_dir = "../data/"
results_dir = "../results/"

# ---------------- DATA ----------------
sn = pd.read_csv(data_dir+"pantheon_plus.csv")
z_sn, m = sn.zHD.values, sn.m_b_corr.values
C = np.load(data_dir+"pantheon_plus_cov.npy")
cfac = cho_factor(C)

# ---------------- POSTERIORS ----------------
lcdm_post = pd.read_csv(results_dir+"lcdm_post.csv")
lcos_post = pd.read_csv(results_dir+"lcos_post.csv")

Om_best = lcdm_post.Om.mean()
s0_best = lcos_post.s0.mean()

# ---------------- E(z) FOR EACH MODEL ----------------
zgrid = np.linspace(0, 2.5, 4000)

def E_lcdm(z, Om):
    return np.sqrt(Om*(1+z)**3 + (1-Om))

def E_lcos(z, s0):
    Om = 1 - ΩΛ
    return np.sqrt(Om/(1-s0**2)*(1+z)**3 - Om*s0**2/(1-s0**2)*(1+z) + ΩΛ)

def mu_model(Ez_grid, z_sn):
    I = cumulative_trapezoid(1/Ez_grid, zgrid, initial=0)
    return 5*np.log10((1+z_sn)*(c/H0_ref)*np.interp(z_sn, zgrid, I)) + 25

def profiled_MB(mu):
    """Analytic MB profile: m_b - MB - mu = 0 at SN-weighted mean."""
    r = m - mu
    return (np.ones_like(r) @ cho_solve(cfac, r)) / (np.ones_like(r) @ cho_solve(cfac, np.ones_like(r)))

# Per-SN residuals at each model's best fit
mu_lcdm = mu_model(E_lcdm(zgrid, Om_best), z_sn)
mu_lcos = mu_model(E_lcos(zgrid, s0_best), z_sn)
MB_lcdm = profiled_MB(mu_lcdm)
MB_lcos = profiled_MB(mu_lcos)

resid_lcdm = m - MB_lcdm - mu_lcdm
resid_lcos = m - MB_lcos - mu_lcos

# ---------------- BIN RESIDUALS ----------------
z_bins = np.logspace(np.log10(z_sn.min()), np.log10(z_sn.max()), nbins+1)
z_centers = 0.5*(z_bins[:-1] + z_bins[1:])

def bin_residuals(z, r):
    centers, means, errs = [], [], []
    for i in range(nbins):
        mask = (z >= z_bins[i]) & (z < z_bins[i+1])
        if mask.sum() > 0:
            r_bin = r[mask]
            centers.append(z_centers[i])
            means.append(r_bin.mean())
            errs.append(r_bin.std()/np.sqrt(mask.sum()) if mask.sum() > 1 else r_bin.std())
    return np.array(centers), np.array(means), np.array(errs)

zc_lcdm, m_lcdm, e_lcdm = bin_residuals(z_sn, resid_lcdm)
zc_lcos, m_lcos, e_lcos = bin_residuals(z_sn, resid_lcos)

# ---------------- FIGURE 3: CORNER PLOT ----------------
fig = corner.corner(
    lcos_post[["s0","H0rd","MB"]].values,
    labels=[r"$s_0$", r"$H_0 r_d$ [km/s]", r"$M_B$"],
    show_titles=False,
    plot_density=True,
    plot_contours=True,
    fill_contours=True,
    levels=(0.68, 0.95),
)
plt.savefig(results_dir+"lcos_corner.png", dpi=200)
plt.savefig(results_dir+"lcos_corner.pdf")
plt.close()

# ---------------- FIGURE 4: RESIDUALS ----------------
fig, ax = plt.subplots(figsize=(10, 6.5))
offset = 0.0008  # small x-offset so the two series do not overlap exactly
ax.errorbar(zc_lcdm/(1+offset), m_lcdm, yerr=e_lcdm, fmt="o", color="C0",
            markersize=4, lw=1, capsize=0, label="Flat ΛCDM")
ax.errorbar(zc_lcos*(1+offset), m_lcos, yerr=e_lcos, fmt="o", color="C1",
            markersize=4, lw=1, capsize=0, label="Λcos")
ax.axhline(0, color="black", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("z")
ax.set_ylabel(r"$\mu_{\rm data} - \mu_{\rm model}$ (mag)")
ax.set_ylim(-0.15, 0.20)
ax.legend(loc="upper right", frameon=True)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir+"residuals.png", dpi=200)
plt.savefig(results_dir+"residuals.pdf")
plt.close()

# ---------------- SUMMARY ----------------
print("="*60)
print("Model comparison: Λcos vs ΛCDM (Pantheon+ + DESI DR2 BAO)")
print("="*60)
print(f"ΛCDM:    Ωm        = {Om_best:.4f}")
print(f"         M_B       = {MB_lcdm:.4f}  (profiled)")
print()
print(f"Λcos:    s0        = {s0_best:.4f}")
print(f"         s0 95% UL = {np.percentile(lcos_post.s0, 95):.3f}")
print(f"         M_B       = {MB_lcos:.4f}  (profiled)")
print()
print("ΛCDM vs Λcos complete.")
