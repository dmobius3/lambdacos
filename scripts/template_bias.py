"""
Template bias demo (Fig. 1, Table in §4.2).

Generates a noise-free Λcos BAO mock at s0 = 0.389 across the 7 DESI DR2
effective redshifts, fits four w(z) parameterizations (CPL, BA, JBP, and
a three-parameter polynomial) using DESI covariance weighting, and plots
each recovered w(z) overlaid on the exact Λcos w_eff(z).

Run:
    python scripts/template_bias.py
Outputs:
    results/template_bias.csv   best-fit parameters and chi^2 per parameterization
    results/template_bias.png   Fig. 1 (also saved as .pdf)
    results/template_bias.pdf
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

# ---------------- CONFIG ----------------
s0 = 0.389
ΩΛ = 0.685
Om = 1 - ΩΛ          # 0.315
H0 = 67.4            # km/s/Mpc
rd = 147.1           # Mpc
H0rd = H0 * rd       # 9914.54 km/s
c = 299792.458

data_dir = "../data/"
out_dir = "../results/"

# ---------------- DESI DR2 BAO ----------------
bao = pd.read_csv(data_dir+"desi_dr2_bao.csv")
Cbao = np.load(data_dir+"desi_dr2_bao_cov.npy")
Cbao_inv = np.linalg.inv(Cbao)

# ---------------- MODEL HELPERS ----------------
zgrid = np.linspace(0, 2.5, 4000)

def E_lcos(z, s0):
    return np.sqrt(Om/(1-s0**2)*(1+z)**3 - Om*s0**2/(1-s0**2)*(1+z) + ΩΛ)

def bao_obs(Ez_grid):
    """13 DESI BAO observables given E(z) on zgrid."""
    I = cumulative_trapezoid(1/Ez_grid, zgrid, initial=0)
    DM = c/H0rd * np.interp(bao.z_eff, zgrid, I)
    DH = c/H0rd / np.interp(bao.z_eff, zgrid, Ez_grid)
    out = []
    i = 0
    for _, row in bao.iterrows():
        if row.observable == "DV_rd":
            out.append((row.z_eff*DM[i]**2*DH[i])**(1/3))
        elif row.observable == "DM_rd":
            out.append(DM[i])
        else:
            out.append(DH[i])
        i += 1
    return np.array(out)

# ---------------- LAMBDA-COS MOCK ----------------
bao_mock = bao_obs(E_lcos(zgrid, s0))

# ---------------- W(Z) PARAMETERIZATIONS ----------------
def w_cpl(a, w0, wa):  return w0 + wa*(1-a)
def w_ba(a, w0, wa):   return w0 + wa*a*(1-a)/(a**2 - 2*a + 2)
def w_jbp(a, w0, wa):  return w0 + wa*a*(1-a)
def w_poly(a, w0, w1, w2): return w0 + w1*(1-a) + w2*(1-a)**2

def E_w(zgrid, w_func, params):
    """E(z) for a w(z) parameterization at fixed Om = 0.315."""
    a_grid = 1/(1+zgrid)
    integrand = 3*(1 + w_func(a_grid, *params))/(1+zgrid)
    integral = cumulative_trapezoid(integrand, zgrid, initial=0)
    rho_de = np.exp(integral)
    return np.sqrt(Om*(1+zgrid)**3 + (1-Om)*rho_de)

def chi2_against_mock(w_func, params):
    pred = bao_obs(E_w(zgrid, w_func, params))
    d = pred - bao_mock
    return d @ Cbao_inv @ d

# ---------------- FIT EACH PARAMETERIZATION ----------------
fits = []  # list of (name, w_func, best_params, chi2)

for name, w_func, x0 in [
    ("CPL",        w_cpl,  [-1.0, 0.0]),
    ("BA",         w_ba,   [-1.0, 0.0]),
    ("JBP",        w_jbp,  [-1.0, 0.0]),
    ("Polynomial", w_poly, [-1.0, 0.0, 0.0]),
]:
    res = minimize(lambda p: chi2_against_mock(w_func, p), x0=x0,
                   method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6})
    fits.append((name, w_func, res.x, res.fun))

# ---------------- TABLE ----------------
rows = []
for name, _, p, chi2 in fits:
    row = {
        "parameterization": name,
        "w0": p[0],
        "w1_or_wa": p[1] if len(p) >= 2 else np.nan,
        "w2": p[2] if len(p) == 3 else np.nan,
        "chi2": chi2,
        "crosses_w=-1": "Yes" if p[0] < -1 else "No",
    }
    rows.append(row)

pd.DataFrame(rows).to_csv(out_dir+"template_bias.csv", index=False)

# ---------------- LAMBDA-COS EXACT W_EFF(Z) ----------------
def w_eff_lcos(z, s0):
    s2 = s0**2
    X = Om*s2/(1-s2) * ((1+z)**3 - (1+z)) + ΩΛ
    dX_dz = Om*s2/(1-s2) * (3*(1+z)**2 - 1)
    return -1 + (1+z)/(3*X) * dX_dz

# ---------------- FIGURE 1 ----------------
z_plot = np.linspace(0, 2.5, 500)
a_plot = 1/(1+z_plot)

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.plot(z_plot, w_eff_lcos(z_plot, s0), color="C0", lw=2.2, label="Λcos exact")

styles = {
    "CPL":        ("--", "C1"),
    "BA":         ("--", "C2"),
    "JBP":        ("--", "C3"),
    "Polynomial": ("--", "C4"),
}
for name, w_func, params, _ in fits:
    ls, color = styles[name]
    ax.plot(z_plot, w_func(a_plot, *params), ls=ls, color=color, lw=1.8, label=name)

ax.axhline(-1, color="black", lw=1, ls=":")
ax.set_xlabel("z")
ax.set_ylabel("w(z)")
ax.legend(loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig(out_dir+"template_bias.png", dpi=200)
plt.savefig(out_dir+"template_bias.pdf")
