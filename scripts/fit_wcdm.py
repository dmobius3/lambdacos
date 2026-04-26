"""
wCDM MCMC fit.

Run:
    python scripts/fit_wcdm.py
Outputs:
    chain.npy, postburn.csv, summary.json (with ΔAIC, ΔBIC vs ΛCDM), corner.png
"""

import numpy as np, pandas as pd, emcee, json
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt, corner

# ---------------- CONFIG ----------------
nwalkers, nsteps, burn = 32, 5000, 1000
data_dir = "../data/"
out_dir = "../results/"
LCDM_chi2_baseline = 1772.445
N_data = 1714  # 1701 SN + 13 BAO

# ---------------- DATA ----------------
sn = pd.read_csv(data_dir+"pantheon_plus.csv")
z_sn, m = sn.zHD.values, sn.m_b_corr.values
C = np.load(data_dir+"pantheon_plus_cov.npy")
cfac = cho_factor(C)

bao = pd.read_csv(data_dir+"desi_dr2_bao.csv")
Cbao = np.load(data_dir+"desi_dr2_bao_cov.npy")
Cbao_inv = np.linalg.inv(Cbao)

# ---------------- MODEL ----------------
c = 299792.458
zgrid = np.linspace(0, 2.5, 4000)

def E(z, Om, w):
    return np.sqrt(Om*(1+z)**3 + (1-Om)*(1+z)**(3*(1+w)))

def sn_chi2(Om, w):
    Ez = E(zgrid, Om, w)
    I = cumulative_trapezoid(1/Ez, zgrid, initial=0)
    mu = 5*np.log10((1+z_sn)*(c/70)*np.interp(z_sn,zgrid,I))+25
    r = m - mu
    MB = (np.ones_like(r) @ cho_solve(cfac, r)) / (np.ones_like(r) @ cho_solve(cfac, np.ones_like(r)))
    d = r - MB
    return d @ cho_solve(cfac, d), MB

def bao_model(Om, w, H0rd):
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
    return np.array(out)

def loglike(theta):
    Om,w,H0rd,MB = theta
    if not (0.01<Om<0.99 and -3<w<0 and 8000<H0rd<12000 and -20<MB<-18):
        return -np.inf
    csn,_ = sn_chi2(Om, w)
    d = bao_model(Om, w, H0rd) - bao.value.values
    cbao = d @ Cbao_inv @ d
    return -0.5*(csn+cbao)

# ---------------- MCMC ----------------
p0 = np.array([0.3, -1.0, 10000, -19.3]) + 1e-2*np.random.randn(nwalkers, 4)
sampler = emcee.EnsembleSampler(nwalkers, 4, loglike)
sampler.run_mcmc(p0, nsteps, progress=True)

chain = sampler.get_chain()
np.save(out_dir+"wcdm_chain.npy",chain)

post = chain[burn:].reshape(-1, 4)
pd.DataFrame(post, columns=["Om","w","H0rd","MB"]).to_csv(out_dir+"wcdm_post.csv", index=False)

# Best-fit chi2 from log-prob (max log L = min chi^2 / 2)
log_prob = sampler.get_log_prob()[burn:].reshape(-1)
chi2_min = -2 * log_prob.max()

Δk = 1  # one extra parameter vs flat ΛCDM
Δχ2 = chi2_min - LCDM_chi2_baseline
ΔAIC = Δχ2 + 2
ΔBIC = Δχ2 + np.log(N_data)

best = post.mean(axis=0)
json.dump({
    "Om": float(best[0]),
    "w": float(best[1]),
    "H0rd": float(best[2]),
    "MB": float(best[3]),
    "chi2_min": float(chi2_min),
    "Δχ2": float(Δχ2),
    "ΔAIC": float(ΔAIC),
    "ΔBIC": float(ΔBIC),
}, open(out_dir+"wcdm_summary.json","w"))

corner.corner(post, labels=["Ωm","w","H0rd","MB"])
plt.savefig(out_dir+"wcdm_corner.png")
