"""
Λcos MCMC fit at fixed Ω_Λ.

Run:
    python fit_lcos.py                       # default Ω_Λ = 0.685 (canonical §5.2 fit)
    python fit_lcos.py --omega_lambda 0.700  # alternative Ω_Λ value (§5.4 scan)

Outputs (under ../results/):
    Default Ω_Λ:  lcos_chain.npy, lcos_post.csv, lcos_summary.json, lcos_corner.png
    Other Ω_Λ:    lcos_omegaL_<value>_*.{npy,csv,json,png}
                  (e.g., lcos_omegaL_0p700_chain.npy)
"""

import argparse, json
import numpy as np, pandas as pd, emcee
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt, corner

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="Λcos MCMC fit at fixed Ω_Λ.")
parser.add_argument("--omega_lambda", type=float, default=0.685,
                    help="Cosmological constant Ω_Λ (default 0.685, the §5.2 canonical value)")
args = parser.parse_args()

# ---------------- CONFIG ----------------
nwalkers, nsteps, burn = 32, 5000, 1000
data_dir = "../data/"
out_dir = "../results/"
ΩΛ = args.omega_lambda

# Output suffix: empty for the canonical 0.685 value, else _omegaL_<value>
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
c = 299792.458
zgrid = np.linspace(0, 2.5, 4000)

def E(z, s0):
    Om = 1-ΩΛ
    return np.sqrt(Om/(1-s0**2)*(1+z)**3 - Om*s0**2/(1-s0**2)*(1+z) + ΩΛ)

def sn_chi2(s0, MB):
    Ez = E(zgrid, s0)
    I = cumulative_trapezoid(1/Ez, zgrid, initial=0)
    mu = 5*np.log10((1+z_sn)*(c/70)*np.interp(z_sn,zgrid,I))+25
    d = m - MB - mu
    return d @ cho_solve(cfac, d)

def bao_model(s0, H0rd):
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
    return np.array(out)

def chi2_total(theta):
    s0, H0rd, MB = theta
    csn = sn_chi2(s0, MB)
    d = bao_model(s0, H0rd) - bao.value.values
    cbao = d @ Cbao_inv @ d
    return csn + cbao, csn, cbao

def loglike(theta):
    s0,H0rd,MB = theta
    if not (0.001<s0<0.99 and 8000<H0rd<12000 and -20<MB<-18):
        return -np.inf
    chi2, _, _ = chi2_total(theta)
    return -0.5*chi2

# ---------------- MCMC ----------------
p0 = np.array([0.1,10000,-19.3]) + 1e-2*np.random.randn(nwalkers,3)
sampler = emcee.EnsembleSampler(nwalkers,3,loglike)
sampler.run_mcmc(p0,nsteps,progress=True)

chain = sampler.get_chain()
np.save(out_dir+f"lcos{suffix}_chain.npy",chain)

post = chain[burn:].reshape(-1,3)
pd.DataFrame(post,columns=["s0","H0rd","MB"]).to_csv(out_dir+f"lcos{suffix}_post.csv",index=False)

s0_95 = np.percentile(post[:,0],95)
s0_med = float(np.percentile(post[:,0],50))
best_mean = post.mean(axis=0)

try:
    tau = emcee.autocorr.integrated_time(chain[burn:], c=5, tol=0)
    tau_per_param = [float(t) for t in tau]
    tau_max = float(np.max(tau))
except Exception:
    tau_per_param, tau_max = None, None
acceptance = float(np.mean(sampler.acceptance_fraction))

# Find precise chi2_min via scipy.optimize seeded from chain argmax
log_prob = sampler.get_log_prob()[burn:].reshape(-1)
chain_argmax = post[np.argmax(log_prob)]
opt = minimize(lambda x: chi2_total(x)[0] if (0.001<x[0]<0.99 and 8000<x[1]<12000 and -20<x[2]<-18) else 1e10,
               chain_argmax, method="Nelder-Mead",
               options={"xatol":1e-8, "fatol":1e-7, "maxiter":5000})
chi2_min, chi2_SN_min, chi2_BAO_min = chi2_total(opt.x)

json.dump({"omega_lambda":float(ΩΛ),
           "s0_mean":float(best_mean[0]),"s0_median":s0_med,
           "H0rd_mean":float(best_mean[1]),"MB_mean":float(best_mean[2]),
           "s0_95":float(s0_95),
           "best_fit_s0":float(opt.x[0]),"best_fit_H0rd":float(opt.x[1]),"best_fit_MB":float(opt.x[2]),
           "chi2_min":float(chi2_min),"chi2_SN":float(chi2_SN_min),"chi2_BAO":float(chi2_BAO_min),
           "tau_per_param":tau_per_param,"tau_max":tau_max,
           "acceptance":acceptance},
          open(out_dir+f"lcos{suffix}_summary.json","w"))

corner.corner(post,labels=["s₀","H0rd","MB"])
plt.savefig(out_dir+f"lcos{suffix}_corner.png")

print(f"Done. chi2_min = {chi2_min:.4f}, s0_95 = {s0_95:.4f}, tau_max = {tau_max}")
