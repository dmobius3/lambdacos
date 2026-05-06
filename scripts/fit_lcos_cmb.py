"""
Λcos MCMC fit with Pantheon+ + DESI DR2 BAO + Planck compressed CMB distance priors (§5.5).

Compressed CMB priors (Planck 2018):
    R   = 1.7502 ± 0.0046     shift parameter
    l_A = 301.47 ± 0.09       acoustic scale
    Treated as independent Gaussians.

R   = sqrt(Ω_m) * ∫₀^z* dz/E(z),                    z* = 1090
l_A ≈ π * c / (H0 r_d) * ∫₀^z* dz/E(z),             treating r_d ≈ r_s(z*)
                                                    (sub-percent for standard cosmology)
Radiation Ω_r = 9.15e-5 is included in E(z) for the high-z integral.

Run:
    python fit_lcos_cmb.py                       # Ω_Λ = 0.685 fixed, 3 params (s0, H0rd, MB)
    python fit_lcos_cmb.py --free_omega_lambda   # Ω_Λ free, 4 params (s0, ΩΛ, H0rd, MB)
    python fit_lcos_cmb.py --omega_lambda 0.700  # Ω_Λ fixed at alternative value

Outputs (under ../results/):
    Fixed Ω_Λ = 0.685:  lcos_cmb_chain.npy, lcos_cmb_post.csv, lcos_cmb_summary.json, lcos_cmb_corner.png
    Free Ω_Λ:           lcos_cmb_freeOL_*.{npy,csv,json,png}
    Fixed alt Ω_Λ:      lcos_cmb_omegaL_<v>_*.{npy,csv,json,png}

The summary JSON uses the harmonized schema from _summary.py.
"""

import argparse
import numpy as np, pandas as pd, emcee
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt, corner

from _summary import write_summary

parser = argparse.ArgumentParser(description="Λcos + SN + BAO + CMB distance priors.")
parser.add_argument("--omega_lambda", type=float, default=0.685,
                    help="Cosmological constant Ω_Λ (default 0.685; ignored if --free_omega_lambda).")
parser.add_argument("--free_omega_lambda", action="store_true",
                    help="Sample Ω_Λ as a free parameter (4-param fit).")
args = parser.parse_args()

nwalkers, nsteps, burn = 32, 5000, 1000
data_dir = "../data/"
out_dir = "../results/"
ΩΛ_fixed = args.omega_lambda
free_OL = args.free_omega_lambda

if free_OL:
    suffix = "_freeOL"
elif abs(ΩΛ_fixed - 0.685) < 1e-6:
    suffix = ""
else:
    suffix = "_omegaL_" + f"{ΩΛ_fixed:.3f}".replace(".", "p")
out_prefix = f"lcos_cmb{suffix}"

print(f"Λcos+CMB: free_omega_lambda={free_OL}, ΩΛ_fixed={ΩΛ_fixed}, output prefix = {out_prefix}")

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

def E_lcos(z, s0, OL):
    """Λcos H/H0 with radiation included for the high-z integral."""
    Om = 1 - OL - Omega_r
    return np.sqrt(Om/(1-s0**2)*(1+z)**3 - Om*s0**2/(1-s0**2)*(1+z) + OL + Omega_r*(1+z)**4)

def sn_chi2(s0, OL, MB):
    Ez = E_lcos(zgrid_low, s0, OL)
    I = cumulative_trapezoid(1/Ez, zgrid_low, initial=0)
    mu = 5*np.log10((1+z_sn)*(c/70)*np.interp(z_sn, zgrid_low, I)) + 25
    d = m - MB - mu
    return float(d @ cho_solve(cfac, d))

def bao_chi2(s0, OL, H0rd):
    Ez = E_lcos(zgrid_low, s0, OL)
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

def cmb_chi2(s0, OL, H0rd):
    Ez = E_lcos(zgrid_high, s0, OL)
    I = cumulative_trapezoid(1/Ez, zgrid_high, initial=0)
    int_to_zstar = float(I[-1])
    Om = 1 - OL - Omega_r
    R_model = np.sqrt(Om) * int_to_zstar
    lA_model = np.pi * c / H0rd * int_to_zstar
    return float(((R_model - R_obs)/R_err)**2 + ((lA_model - lA_obs)/lA_err)**2)

def chi2_split(theta):
    if free_OL:
        s0, OL, H0rd, MB = theta
    else:
        s0, H0rd, MB = theta
        OL = ΩΛ_fixed
    csn = sn_chi2(s0, OL, MB)
    cbao = bao_chi2(s0, OL, H0rd)
    ccmb = cmb_chi2(s0, OL, H0rd)
    return {"total": csn + cbao + ccmb, "SN": csn, "BAO": cbao, "CMB": ccmb}

def loglike(theta):
    if free_OL:
        s0, OL, H0rd, MB = theta
        if not (0.001 < s0 < 0.99 and 0.5 < OL < 0.85 and 8000 < H0rd < 12000 and -20 < MB < -18):
            return -np.inf
    else:
        s0, H0rd, MB = theta
        if not (0.001 < s0 < 0.99 and 8000 < H0rd < 12000 and -20 < MB < -18):
            return -np.inf
    return -0.5 * chi2_split(theta)["total"]

if free_OL:
    bounds_check = lambda x: 0.001 < x[0] < 0.99 and 0.5 < x[1] < 0.85 and 8000 < x[2] < 12000 and -20 < x[3] < -18
    p0 = np.array([0.1, 0.7, 10000, -19.3]) + np.array([1e-2, 5e-3, 50, 1e-2])*np.random.randn(nwalkers, 4)
    ndim = 4
    param_names = ["s0", "OmegaL", "H0rd", "MB"]
    labels = ["s₀", "Ω_Λ", "H0rd", "MB"]
    fixed = None
else:
    bounds_check = lambda x: 0.001 < x[0] < 0.99 and 8000 < x[1] < 12000 and -20 < x[2] < -18
    p0 = np.array([0.1, 10000, -19.3]) + np.array([1e-2, 50, 1e-2])*np.random.randn(nwalkers, 3)
    ndim = 3
    param_names = ["s0", "H0rd", "MB"]
    labels = ["s₀", "H0rd", "MB"]
    fixed = {"omega_lambda": float(ΩΛ_fixed)}

sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike)
sampler.run_mcmc(p0, nsteps, progress=True)

chain = sampler.get_chain()
np.save(out_dir+f"{out_prefix}_chain.npy", chain)

post = chain[burn:].reshape(-1, ndim)
pd.DataFrame(post, columns=param_names).to_csv(out_dir+f"{out_prefix}_post.csv", index=False)

log_prob_post = sampler.get_log_prob()[burn:].reshape(-1)
acceptance = float(np.mean(sampler.acceptance_fraction))

# Extras: s0_95UL is meaningful for both fixed and free (it's a property of the s0 marginal)
extras = {"s0_95UL": float(np.percentile(post[:,0], 95))}

if free_OL:
    model_name = "Λcos (Ω_Λ free) + CMB priors"
elif abs(ΩΛ_fixed - 0.685) < 1e-6:
    model_name = "Λcos + CMB priors"
else:
    model_name = f"Λcos (Ω_Λ = {ΩΛ_fixed}) + CMB priors"

write_summary(
    out_path=out_dir+f"{out_prefix}_summary.json",
    model_name=model_name,
    param_names=param_names,
    post_chain=post,
    chain_post_burn=chain[burn:],
    log_prob_post=log_prob_post,
    chi2_func=chi2_split,
    bounds_check=bounds_check,
    acceptance=acceptance,
    fixed=fixed,
    extras=extras,
)

corner.corner(post, labels=labels)
plt.savefig(out_dir+f"{out_prefix}_corner.png")

print(f"Done. See {out_dir}{out_prefix}_summary.json")
