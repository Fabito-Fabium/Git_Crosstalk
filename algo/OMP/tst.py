# %% ###################################################################################################################
import matplotlib
import scipy.optimize

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
# %% ###################################################################################################################
import numpy as np
import scipy.sparse.linalg as la
from source.synth_data import synth_data
from source.subProb_linOp import ownlinOp
import source.pylops as my_pylops
from tqdm import tqdm
from time import time
import random
import source.pyproximal as pyproximal
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import lsq_linear

from matplotlib import animation

random.seed(42)
# %% ###################################################################################################################
def show_lmbd(lmbd_spc, norm_res):
    plt.close('all')
    plt.loglog(lmbd_spc, norm_res)
    plt.xlabel("$\lambda$")
    plt.ylabel("$||f-f_{est}||_2$")
    plt.title(f"Lambda do subproblema Hf = g - f, melhor lmbd: {lmbd_spc[np.argmin(norm_res)]}")

def show_hest(hinv0, tit):
    plt.close("all")
    fig, axs = plt.subplots(ncols=3)

    real = axs[0].imshow(h.reshape(Nh, Nc).T, aspect='auto', cmap='Greys', interpolation='nearest')
    axs[0].set_title(f"h real")
    nonn = axs[1].imshow(hinv0.reshape(Nh, Nc).T, aspect='auto', cmap='Greys', interpolation='nearest')
    axs[1].set_title(f"{tit}, norm={olo.norm(hinv0.ravel() - h.ravel())}")
    # axs[1].set_title(f"FISTA NONNEGATIVE 200 iter, t={t0: .4f}s, norm={norm(hinv0-h.ravel())}")

    diff = axs[2].imshow((h.reshape(Nh, Nc) - (hinv0).reshape(Nh, Nc)).T, aspect='auto', cmap='Greys',
                         interpolation='nearest')
    axs[2].set_title(f"h - hh")

    fig.colorbar(real, ax=axs[0])
    fig.colorbar(nonn, ax=axs[1])
    fig.colorbar(diff, ax=axs[2])

def show_stats(hest, w_g):
    print(f"||hh-h||^2: \t\t\t {olo.norm(hest.reshape(Nh, Nc) - h)}")
    print(f"||w||^2: \t\t\t\t {olo.norm(w_g)}")
    print(f"||F(hh) - g_clean||^2: \t {olo.norm(olo.F(hest) - crs)}")
    print(f"||F(hh) - g||^2: \t\t {olo.norm(olo.F(hest) - (g-f.ravel()))}")

# %%
# Definindo os parâmetros do problema
dtype = np.float32

Fs = 125e6
Fc = 5e6
Nh = 8
Nt = 1875
Ne = 10

mySys = synth_data(Fs, Fc, Nh, Nt, Ne, vmax=1, bw=0.75, sim_dt=False)
f, h, idx, crs = mySys.create_synth()
olo = ownlinOp(Fs, Fc, Nh, Nt, Ne, f, h, idx, remez=True, filt=mySys.get_pulse())
#g = olo.apply_SNR(crs, 40)
g_clean = crs + f.ravel()
g, w_g = olo.apply_SNR(g_clean, 40)

Nc = len(idx)

# %% ###################################################################################################################
homp = np.zeros((Nh, Nc))
t0 = time()
for i in tqdm(range(Nc)):
    AF = la.LinearOperator((Nt * Ne, Nh), matvec=lambda x: olo.Fo(x, i), rmatvec=lambda x: olo.FoT(x, i))
    AFp = my_pylops.LinearOperator(AF)
    homp[:, i] = my_pylops.optimization.sparsity.omp(AFp, g.ravel() - f.ravel(),  niter_outer=1,
                    niter_inner=Ne, sigma=1e-10, normalizecols=True, nonneg=True)[0]
print(time() - t0)

show_hest(homp, "OMP")
# %%
from joblib import Parallel, delayed
homp = np.zeros((Nh, Nc))

def omp_col(ii):
    AF = la.LinearOperator((Nt * Ne, Nh), matvec=lambda x: olo.Fo(x, ii), rmatvec=lambda x: olo.FoT(x, ii))
    AFp = my_pylops.LinearOperator(AF)
    homp_col = my_pylops.optimization.sparsity.omp(AFp, g.ravel() - f.ravel(),  niter_outer=1,
                    niter_inner=200, sigma=1e-20, normalizecols=True, nonneg=True, discard=False)[0]

    return homp_col
output= Parallel(n_jobs=-1)(delayed(omp_col)(i) for i in range(Nc))
homp = np.array(output).T
