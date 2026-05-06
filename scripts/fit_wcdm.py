"""
wCDM MCMC fit to Pantheon+ + DESI DR2 BAO (constant-w dark energy).

Run:
    python fit_wcdm.py
Outputs (under ../results/):
    wcdm_chain.npy, wcdm_post.csv, wcdm_summary.json, wcdm_corner.png

The summary JSON uses the harmonized schema from _summary.py. Δχ²/ΔAIC/ΔBIC
relative to the deposited flat ΛCDM fit are written into "extras".
"""

import json
import numpy as np, pandas as pd, emcee
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt, corner

from _summary import write_summary

# ---------------- CONFIG ----------------
nwalkers, nsteps, burn = 32, 5000, 1000
data_dir = "../data/"
out_dir = "../results/"
c = 299792.458
N_data = 1714  # 1701 SN + 13 BAO

# Read flat ΛCDM baseline chi2_min from its summary (defensive: handle either schema),
# with a hardcoded fallback if the file is unreadable or absent.
import os
def _lcdm_baseline():
    HARDCODED = 1772.456
    path = out_dir + "lcdm_summary.json"
    if not os.path.exists(path):
        return HARDCODED
    try:
        with open(path) as f:
            d = json.load(f)
        if isinstance(d.get("chi2"), dict) and "total" in d["chi2"]:
            return float(d["chi2"]["total"])  # new schema
        if "chi2_min" in d:
            return float(d["chi2_min"])  # old schema
        return HARDCODED
    except Exception:
        return HARDCODED
LCDM_chi2_baseline = _lcdm_baseline()

# ---------------- DATA ----------------
sn = pd.read_csv(data_dir+"pantheon_plus.csv")
z_sn, m = sn.zHD.values, sn.m_b_corr.values
C = np.load(data_dir+"pantheon_plus_cov.npy")
cfac = cho_factor(C)

bao = pd.read_csv(data_dir+"desi_dr2_bao.csv")
Cbao = np.load(data_dir+"desi_dr2_bao_cov.npy")
Cbao_inv = np.linalg.inv(Cbao)

# ---------------- MODEL ----------------
zgrid = np.linspace(0, 2.5, 4000)

def E(z, Om, w):
    return np.sqrt(Om*(1+z)**3 + (1-Om)*(1+z)**(3*(1+w)))

def sn_chi2(Om, w, MB):
    Ez = E(zgrid, Om, w)
    I = cumulative_trapezoid(1/Ez, zgrid, initial=0)
    mu = 5*np.log10((1+z_sn)*(c/70)*np.interp(z_sn,zgrid,I))+25
    d = m - MB - mu
    return float(d @ cho_solve(cfac, d))

def bao_chi2(Om, w, H0rd):
    Ez = E(zgrid, Om, w)
    I = cumulative_trapezoid(1/Ez, zgrid, initial=0)
    DM = c/H0rd*np.interp(bao.z_eff, zgrid, I)
    DH = c/H0rd/np.interp(bao.z_eff, zgrid, Ez)
    out=[]
    i=0
    for _,row in bao.iterrows():
        if row.observable=="DV_rd":
            out.append((row.z_eff*DM[i]**2*DH[i])**(1/3))
        elif row.observable=="DM_rd":
            out.append(DM[i])
        else:
            out.append(DH[i])
        i+=1
    pred = np.array(out)
    d = pred - bao.value.values
    return float(d @ Cbao_inv @ d)

def chi2_split(theta):
    Om, w, H0rd, MB = theta
    csn = sn_chi2(Om, w, MB)
    cbao = bao_chi2(Om, w, H0rd)
    return {"total": csn + cbao, "SN": csn, "BAO": cbao, "CMB": None}

def loglike(theta):
    Om,w,H0rd,MB = theta
    if not (0.01<Om<0.99 and -3<w<0 and 8000<H0rd<12000 and -20<MB<-18):
        return -np.inf
    return -0.5 * chi2_split(theta)["total"]

bounds_check = lambda x: 0.01<x[0]<0.99 and -3<x[1]<0 and 8000<x[2]<12000 and -20<x[3]<-18

# ---------------- MCMC ----------------
p0 = np.array([0.3, -1.0, 10000, -19.3]) + 1e-2*np.random.randn(nwalkers, 4)
sampler = emcee.EnsembleSampler(nwalkers, 4, loglike)
sampler.run_mcmc(p0, nsteps, progress=True)

chain = sampler.get_chain()
np.save(out_dir+"wcdm_chain.npy",chain)

post = chain[burn:].reshape(-1, 4)
pd.DataFrame(post, columns=["Om","w","H0rd","MB"]).to_csv(out_dir+"wcdm_post.csv", index=False)

log_prob_post = sampler.get_log_prob()[burn:].reshape(-1)
acceptance = float(np.mean(sampler.acceptance_fraction))
chi2_min_at_argmax = -2 * log_prob_post.max()

# extras: Δχ²/ΔAIC/ΔBIC vs flat ΛCDM (1 extra parameter)
delta_chi2 = chi2_min_at_argmax - LCDM_chi2_baseline
delta_AIC = delta_chi2 + 2 * 1  # +2 per extra parameter
delta_BIC = delta_chi2 + np.log(N_data) * 1

extras = {
    "delta_chi2_vs_LCDM": float(delta_chi2),
    "delta_AIC_vs_LCDM": float(delta_AIC),
    "delta_BIC_vs_LCDM": float(delta_BIC),
    "LCDM_chi2_baseline": float(LCDM_chi2_baseline),
}

write_summary(
    out_path=out_dir+"wcdm_summary.json",
    model_name="wCDM",
    param_names=["Om", "w", "H0rd", "MB"],
    post_chain=post,
    chain_post_burn=chain[burn:],
    log_prob_post=log_prob_post,
    chi2_func=chi2_split,
    bounds_check=bounds_check,
    acceptance=acceptance,
    extras=extras,
)

corner.corner(post, labels=["Ωm","w","H0rd","MB"])
plt.savefig(out_dir+"wcdm_corner.png")

print(f"Done. See {out_dir}wcdm_summary.json")
