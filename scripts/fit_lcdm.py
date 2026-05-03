"""
Flat ΛCDM MCMC fit to Pantheon+ + DESI DR2 BAO.

Run:
    python scripts/fit_lcdm.py
Outputs:
    chain.npy, postburn.csv, summary.json, corner.png
"""

import numpy as np, pandas as pd, emcee, json
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt, corner

# ---------------- CONFIG ----------------
nwalkers, nsteps, burn = 32, 5000, 1000
data_dir = "../data/"
out_dir = "../results/"

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

def E(z, Om):
    return np.sqrt(Om*(1+z)**3 + (1-Om))

def sn_chi2(Om):
    Ez = E(zgrid, Om)
    I = cumulative_trapezoid(1/Ez, zgrid, initial=0)
    mu = 5*np.log10((1+z_sn)*(c/70)*np.interp(z_sn,zgrid,I))+25
    r = m - mu
    MB = (np.ones_like(r) @ cho_solve(cfac, r)) / (np.ones_like(r) @ cho_solve(cfac, np.ones_like(r)))
    d = r - MB
    return d @ cho_solve(cfac, d), MB

def bao_model(Om, H0rd):
    Ez = E(zgrid, Om)
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
    Om,H0rd,MB = theta
    if not (0.01<Om<0.99 and 8000<H0rd<12000 and -20<MB<-18):
        return -np.inf
    csn,_ = sn_chi2(Om)
    d = bao_model(Om,H0rd) - bao.value.values
    cbao = d @ Cbao_inv @ d
    return -0.5*(csn+cbao)

# ---------------- MCMC ----------------
p0 = np.array([0.3,10000,-19.3]) + 1e-2*np.random.randn(nwalkers,3)
sampler = emcee.EnsembleSampler(nwalkers,3,loglike)
sampler.run_mcmc(p0,nsteps,progress=True)

chain = sampler.get_chain()
np.save(out_dir+"lcdm_chain.npy",chain)

post = chain[burn:].reshape(-1,3)
pd.DataFrame(post,columns=["Om","H0rd","MB"]).to_csv(out_dir+"lcdm_post.csv",index=False)

best = post.mean(axis=0)

try:
    tau = emcee.autocorr.integrated_time(chain[burn:], c=5, tol=0)
    tau_per_param = [float(t) for t in tau]
    tau_max = float(np.max(tau))
except Exception:
    tau_per_param, tau_max = None, None
acceptance = float(np.mean(sampler.acceptance_fraction))

json.dump({"Om":float(best[0]),"H0rd":float(best[1]),"MB":float(best[2]),
           "tau_per_param":tau_per_param,"tau_max":tau_max,
           "acceptance":acceptance},
          open(out_dir+"lcdm_summary.json","w"))

corner.corner(post,labels=["Ωm","H0rd","MB"])
plt.savefig(out_dir+"lcdm_corner.png")
