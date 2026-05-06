"""
ΛCDM MCMC fit with Pantheon+ + DESI DR2 BAO + Planck compressed CMB distance priors (§5.5).

Compressed CMB priors (Planck 2018):
    R   = 1.7502 ± 0.0046     shift parameter
    l_A = 301.47 ± 0.09       acoustic scale
    Treated as independent Gaussians.

R   = sqrt(Ω_m) * ∫₀^z* dz/E(z),                    z* = 1090
l_A ≈ π * c / (H0 r_d) * ∫₀^z* dz/E(z),             treating r_d ≈ r_s(z*)
                                                    (sub-percent for standard cosmology)
Radiation Ω_r = 9.15e-5 is included in E(z) for the high-z integral.

Run:
    python fit_lcdm_cmb.py             # flat ΛCDM, 3 params (Ω_m, H0rd, MB)
    python fit_lcdm_cmb.py --non_flat  # non-flat ΛCDM, 4 params (Ω_m, Ω_Λ, H0rd, MB)

Outputs (under ../results/):
    Flat:     lcdm_cmb_chain.npy, lcdm_cmb_post.csv, lcdm_cmb_summary.json, lcdm_cmb_corner.png
    Non-flat: lcdm_cmb_nonflat_*.{npy,csv,json,png}

The summary JSON uses the harmonized schema from _summary.py.
"""

import argparse
import numpy as np, pandas as pd, emcee
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt, corner

from _summary import write_summary

parser = argparse.ArgumentParser(description="ΛCDM + SN + BAO + CMB distance priors.")
parser.add_argument("--non_flat", action="store_true",
                    help="Sample Ω_Λ as a 4th parameter (non-flat ΛCDM).")
args = parser.parse_args()

nwalkers, nsteps, burn = 32, 5000, 1000
data_dir = "../data/"
out_dir = "../results/"
non_flat = args.non_flat
out_prefix = "lcdm_cmb_nonflat" if non_flat else "lcdm_cmb"

print(f"ΛCDM+CMB: non_flat={non_flat}, output prefix = {out_prefix}")

R_obs, R_err = 1.7502, 0.0046
lA_obs, lA_err = 301.47, 0.09
z_star = 1090.0
Omega_r = 9.15e-5
c = 299792.458

sn = pd.read_csv(data_dir+"pantheon_plus.csv")
z_sn, m = sn.zHD.values, sn.m_b_corr.values
C = np.load(data_dir+"pantheon_plus_cov.npy")
cfac = cho_factor(C)

bao = pd.read_csv(data_dir+"desi_dr2_bao.csv")
Cbao = np.load(data_dir+"desi_dr2_bao_cov.npy")
Cbao_inv = np.linalg.inv(Cbao)

zgrid_low = np.linspace(0, 2.5, 4000)
zgrid_high = np.linspace(0, z_star, 6000)

def E_lcdm(z, Om, OL):
    """ΛCDM (possibly non-flat) with radiation; Ω_k = 1 - Ω_m - Ω_Λ - Ω_r."""
    Ok = 1 - Om - OL - Omega_r
    return np.sqrt(Om*(1+z)**3 + Ok*(1+z)**2 + OL + Omega_r*(1+z)**4)

def sn_chi2(Om, OL, MB):
    Ez = E_lcdm(zgrid_low, Om, OL)
    I = cumulative_trapezoid(1/Ez, zgrid_low, initial=0)
    mu = 5*np.log10((1+z_sn)*(c/70)*np.interp(z_sn, zgrid_low, I)) + 25
    d = m - MB - mu
    return float(d @ cho_solve(cfac, d))

def bao_chi2(Om, OL, H0rd):
    Ez = E_lcdm(zgrid_low, Om, OL)
    I = cumulative_trapezoid(1/Ez, zgrid_low, initial=0)
    DM = c/H0rd*np.interp(bao.z_eff, zgrid_low, I)
    DH = c/H0rd/np.interp(bao.z_eff, zgrid_low, Ez)
    pred = []
    i = 0
    for _, row in bao.iterrows():
        if row.observable == "DV_rd":
            pred.append((row.z_eff*DM[i]**2*DH[i])**(1/3))
        elif row.observable == "DM_rd":
            pred.append(DM[i])
        else:
            pred.append(DH[i])
        i += 1
    pred = np.array(pred)
    d = pred - bao.value.values
    return float(d @ Cbao_inv @ d)

def cmb_chi2(Om, OL, H0rd):
    Ez = E_lcdm(zgrid_high, Om, OL)
    I = cumulative_trapezoid(1/Ez, zgrid_high, initial=0)
    int_to_zstar = float(I[-1])
    R_model = np.sqrt(Om) * int_to_zstar
    lA_model = np.pi * c / H0rd * int_to_zstar
    return float(((R_model - R_obs)/R_err)**2 + ((lA_model - lA_obs)/lA_err)**2)

def chi2_split(theta):
    if non_flat:
        Om, OL, H0rd, MB = theta
    else:
        Om, H0rd, MB = theta
        OL = 1 - Om - Omega_r
    csn = sn_chi2(Om, OL, MB)
    cbao = bao_chi2(Om, OL, H0rd)
    ccmb = cmb_chi2(Om, OL, H0rd)
    return {"total": csn + cbao + ccmb, "SN": csn, "BAO": cbao, "CMB": ccmb}

def loglike(theta):
    if non_flat:
        Om, OL, H0rd, MB = theta
        if not (0.05 < Om < 0.6 and 0.4 < OL < 0.9 and 8000 < H0rd < 12000 and -20 < MB < -18):
            return -np.inf
    else:
        Om, H0rd, MB = theta
        if not (0.05 < Om < 0.6 and 8000 < H0rd < 12000 and -20 < MB < -18):
            return -np.inf
    return -0.5 * chi2_split(theta)["total"]

if non_flat:
    bounds_check = lambda x: 0.05 < x[0] < 0.6 and 0.4 < x[1] < 0.9 and 8000 < x[2] < 12000 and -20 < x[3] < -18
    p0 = np.array([0.30, 0.70, 10000, -19.3]) + np.array([5e-3, 5e-3, 50, 1e-2])*np.random.randn(nwalkers, 4)
    ndim = 4
    param_names = ["Om", "OmegaL", "H0rd", "MB"]
    labels = ["Ωm", "Ω_Λ", "H0rd", "MB"]
else:
    bounds_check = lambda x: 0.05 < x[0] < 0.6 and 8000 < x[1] < 12000 and -20 < x[2] < -18
    p0 = np.array([0.30, 10000, -19.3]) + np.array([5e-3, 50, 1e-2])*np.random.randn(nwalkers, 3)
    ndim = 3
    param_names = ["Om", "H0rd", "MB"]
    labels = ["Ωm", "H0rd", "MB"]

sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike)
sampler.run_mcmc(p0, nsteps, progress=True)

chain = sampler.get_chain()
np.save(out_dir+f"{out_prefix}_chain.npy", chain)

post = chain[burn:].reshape(-1, ndim)
pd.DataFrame(post, columns=param_names).to_csv(out_dir+f"{out_prefix}_post.csv", index=False)

log_prob_post = sampler.get_log_prob()[burn:].reshape(-1)
acceptance = float(np.mean(sampler.acceptance_fraction))

write_summary(
    out_path=out_dir+f"{out_prefix}_summary.json",
    model_name="non-flat ΛCDM + CMB priors" if non_flat else "flat ΛCDM + CMB priors",
    param_names=param_names,
    post_chain=post,
    chain_post_burn=chain[burn:],
    log_prob_post=log_prob_post,
    chi2_func=chi2_split,
    bounds_check=bounds_check,
    acceptance=acceptance,
)

corner.corner(post, labels=labels)
plt.savefig(out_dir+f"{out_prefix}_corner.png")

print(f"Done. See {out_dir}{out_prefix}_summary.json")
