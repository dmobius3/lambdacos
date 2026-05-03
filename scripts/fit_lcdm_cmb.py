"""
ΛCDM MCMC fit with Pantheon+ + DESI DR2 BAO + Planck compressed CMB distance priors (§5.5).

Compressed CMB priors (Planck 2018):
    R   = 1.7502 ± 0.0046     shift parameter
    l_A = 301.47 ± 0.09       acoustic scale
    Treated as independent Gaussians.

Run:
    python fit_lcdm_cmb.py             # flat ΛCDM, 3 params (Ω_m, H0rd, MB)
    python fit_lcdm_cmb.py --non_flat  # non-flat ΛCDM, 4 params (Ω_m, Ω_Λ, H0rd, MB)
                                       # with Ω_k = 1 - Ω_m - Ω_Λ - Ω_r

Outputs (under ../results/):
    Flat:     lcdm_cmb_chain.npy, lcdm_cmb_post.csv, lcdm_cmb_summary.json, lcdm_cmb_corner.png
    Non-flat: lcdm_cmb_nonflat_*.{npy,csv,json,png}
"""

import argparse, json
import numpy as np, pandas as pd, emcee
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt, corner

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
    return ((R_model - R_obs)/R_err)**2 + ((lA_model - lA_obs)/lA_err)**2

def total_chi2(theta):
    if non_flat:
        Om, OL, H0rd, MB = theta
    else:
        Om, H0rd, MB = theta
        OL = 1 - Om - Omega_r
    csn = sn_chi2(Om, OL, MB)
    cbao = bao_chi2(Om, OL, H0rd)
    ccmb = cmb_chi2(Om, OL, H0rd)
    return csn + cbao + ccmb, csn, cbao, ccmb

def loglike(theta):
    if non_flat:
        Om, OL, H0rd, MB = theta
        if not (0.05 < Om < 0.6 and 0.4 < OL < 0.9 and 8000 < H0rd < 12000 and -20 < MB < -18):
            return -np.inf
    else:
        Om, H0rd, MB = theta
        if not (0.05 < Om < 0.6 and 8000 < H0rd < 12000 and -20 < MB < -18):
            return -np.inf
    chi2, _, _, _ = total_chi2(theta)
    return -0.5 * chi2

if non_flat:
    p0 = np.array([0.30, 0.70, 10000, -19.3]) + np.array([5e-3, 5e-3, 50, 1e-2])*np.random.randn(nwalkers, 4)
    ndim = 4
    labels = ["Ωm", "Ω_Λ", "H0rd", "MB"]
else:
    p0 = np.array([0.30, 10000, -19.3]) + np.array([5e-3, 50, 1e-2])*np.random.randn(nwalkers, 3)
    ndim = 3
    labels = ["Ωm", "H0rd", "MB"]

sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike)
sampler.run_mcmc(p0, nsteps, progress=True)

chain = sampler.get_chain()
np.save(out_dir+f"{out_prefix}_chain.npy", chain)

post = chain[burn:].reshape(-1, ndim)
cols = ["Om","OmegaL","H0rd","MB"] if non_flat else ["Om","H0rd","MB"]
pd.DataFrame(post, columns=cols).to_csv(out_dir+f"{out_prefix}_post.csv", index=False)

best_mean = post.mean(axis=0)

try:
    tau = emcee.autocorr.integrated_time(chain[burn:], c=5, tol=0)
    tau_per_param = [float(t) for t in tau]
    tau_max = float(np.max(tau))
except Exception:
    tau_per_param, tau_max = None, None
acceptance = float(np.mean(sampler.acceptance_fraction))

log_prob = sampler.get_log_prob()[burn:].reshape(-1)
chain_argmax = post[np.argmax(log_prob)]

if non_flat:
    bounds_check = lambda x: 0.05 < x[0] < 0.6 and 0.4 < x[1] < 0.9 and 8000 < x[2] < 12000 and -20 < x[3] < -18
else:
    bounds_check = lambda x: 0.05 < x[0] < 0.6 and 8000 < x[1] < 12000 and -20 < x[2] < -18

opt = minimize(lambda x: total_chi2(x)[0] if bounds_check(x) else 1e10,
               chain_argmax, method="Nelder-Mead",
               options={"xatol":1e-8, "fatol":1e-7, "maxiter":8000})
chi2_min, chi2_SN_min, chi2_BAO_min, chi2_CMB_min = total_chi2(opt.x)

summary = {
    "non_flat": bool(non_flat),
    "best_fit": {labels[i]: float(opt.x[i]) for i in range(ndim)},
    "posterior_mean": {labels[i]: float(best_mean[i]) for i in range(ndim)},
    "chi2_min": float(chi2_min),
    "chi2_SN": float(chi2_SN_min),
    "chi2_BAO": float(chi2_BAO_min),
    "chi2_CMB": float(chi2_CMB_min),
    "tau_per_param": tau_per_param,
    "tau_max": tau_max,
    "acceptance": acceptance,
}
if non_flat:
    OL_q = np.percentile(post[:,1], [16, 50, 84])
    summary["OmegaL_quantiles"] = {"16": float(OL_q[0]), "50": float(OL_q[1]), "84": float(OL_q[2])}

json.dump(summary, open(out_dir+f"{out_prefix}_summary.json", "w"), indent=2, ensure_ascii=False)

corner.corner(post, labels=labels)
plt.savefig(out_dir+f"{out_prefix}_corner.png")

print(f"\nDone. chi2_min = {chi2_min:.4f} (SN={chi2_SN_min:.2f}, BAO={chi2_BAO_min:.2f}, CMB={chi2_CMB_min:.2f})")
print(f"      tau_max = {tau_max}, acceptance = {acceptance:.3f}")
