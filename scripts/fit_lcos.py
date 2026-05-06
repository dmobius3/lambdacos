"""
Λcos MCMC fit at fixed Ω_Λ.

Run:
    python fit_lcos.py                       # canonical Ω_Λ = 0.685 (§5.2)
    python fit_lcos.py --omega_lambda 0.700  # alternative Ω_Λ (§5.4 scan)

Outputs (under ../results/):
    Default Ω_Λ:  lcos_chain.npy, lcos_post.csv, lcos_summary.json, lcos_corner.png
    Other Ω_Λ:    lcos_omegaL_<value>_*.{npy,csv,json,png}
                  (e.g., lcos_omegaL_0p700_chain.npy)

The summary JSON uses the harmonized schema from _summary.py.
"""

import argparse
import numpy as np, pandas as pd, emcee
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt, corner

from _summary import write_summary

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="Λcos MCMC fit at fixed Ω_Λ.")
parser.add_argument("--omega_lambda", type=float, default=0.685,
                    help="Cosmological constant Ω_Λ (default 0.685, the §5.2 canonical value)")
args = parser.parse_args()

# ---------------- CONFIG ----------------
nwalkers, nsteps, burn = 32, 5000, 1000
data_dir = "../data/"
out_dir = "../results/"
c = 299792.458
ΩΛ = args.omega_lambda

# Output suffix: empty for canonical 0.685, else _omegaL_<value>
if abs(ΩΛ - 0.685) < 1e-6:
    suffix = ""
else:
    suffix = "_omegaL_" + f"{ΩΛ:.3f}".replace(".", "p")

print(f"Λcos fit: Ω_Λ = {ΩΛ}, output prefix = lcos{suffix}")

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

def E(z, s0):
    Om = 1-ΩΛ
    return np.sqrt(Om/(1-s0**2)*(1+z)**3 - Om*s0**2/(1-s0**2)*(1+z) + ΩΛ)

def sn_chi2(s0, MB):
    Ez = E(zgrid, s0)
    I = cumulative_trapezoid(1/Ez, zgrid, initial=0)
    mu = 5*np.log10((1+z_sn)*(c/70)*np.interp(z_sn,zgrid,I))+25
    d = m - MB - mu
    return float(d @ cho_solve(cfac, d))

def bao_chi2(s0, H0rd):
    Ez = E(zgrid, s0)
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
    s0, H0rd, MB = theta
    csn = sn_chi2(s0, MB)
    cbao = bao_chi2(s0, H0rd)
    return {"total": csn + cbao, "SN": csn, "BAO": cbao, "CMB": None}

def loglike(theta):
    s0,H0rd,MB = theta
    if not (0.001<s0<0.99 and 8000<H0rd<12000 and -20<MB<-18):
        return -np.inf
    return -0.5 * chi2_split(theta)["total"]

bounds_check = lambda x: 0.001<x[0]<0.99 and 8000<x[1]<12000 and -20<x[2]<-18

# ---------------- MCMC ----------------
p0 = np.array([0.1,10000,-19.3]) + 1e-2*np.random.randn(nwalkers,3)
sampler = emcee.EnsembleSampler(nwalkers,3,loglike)
sampler.run_mcmc(p0,nsteps,progress=True)

chain = sampler.get_chain()
np.save(out_dir+f"lcos{suffix}_chain.npy",chain)

post = chain[burn:].reshape(-1,3)
pd.DataFrame(post,columns=["s0","H0rd","MB"]).to_csv(out_dir+f"lcos{suffix}_post.csv",index=False)

log_prob_post = sampler.get_log_prob()[burn:].reshape(-1)
acceptance = float(np.mean(sampler.acceptance_fraction))

# Special: 95% upper limit on s0 (one-sided).
s0_95UL = float(np.percentile(post[:,0], 95))

write_summary(
    out_path=out_dir+f"lcos{suffix}_summary.json",
    model_name="Λcos",
    param_names=["s0", "H0rd", "MB"],
    post_chain=post,
    chain_post_burn=chain[burn:],
    log_prob_post=log_prob_post,
    chi2_func=chi2_split,
    bounds_check=bounds_check,
    acceptance=acceptance,
    fixed={"omega_lambda": float(ΩΛ)},
    extras={"s0_95UL": s0_95UL},
)

corner.corner(post,labels=["s₀","H0rd","MB"])
plt.savefig(out_dir+f"lcos{suffix}_corner.png")

print(f"Done. See {out_dir}lcos{suffix}_summary.json")
