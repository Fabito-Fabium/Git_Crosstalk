# %% ###################################################################################################################
import matplotlib
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
# %% No regularization #################################################################################################
# mthd = "L-BFGS-B"
mthd = "CG"
opts = {'maxiter': 200, 'disp': True}

def fun(x):
    res = olo.F(x) - crs
    return olo.norm(res), 2*olo.FT(olo.F(x) - crs)

x0 = np.zeros(Nh*Nc)

AF = la.LinearOperator((Nt*Ne, Nh*Nc), matvec=lambda x: olo.F(x), rmatvec=lambda x: olo.FT(x))
hest = minimize(fun, x0, method=mthd, jac=True, options=opts).x.reshape(Nh, Nc)
# %% No regularization minres ##########################################################################################
AF = la.LinearOperator((Nh*Nc, Nh*Nc), matvec=lambda x: olo.FT(olo.F(x, f), f)) #, rmatvec=lambda x: FT(F(x)))
hest = la.minres(AF, olo.FT(crs).ravel(), x0=np.zeros(Nh*Nc), maxiter=500, show=True, rtol=1e-9)[0].reshape(Nh, Nc)  # 9 s
# %%
show_stats(hest, w_g)
show_hest(hest, "Sem Regularização")
# %% ###################################################################################################################
# Estimando o h FISTA
AF = la.LinearOperator((Nt*Ne, Nc*Nh), matvec=lambda x: olo.F(x), rmatvec=lambda x: olo.FT(x))

l2 = pyproximal.L2(Op=my_pylops.LinearOperator(AF), b=(g.ravel() - f.ravel())) #crosstalk = g - f
l1 = pyproximal.L1()
lmbd_h = np.logspace(-5, 0, 50)
norm_h = np.zeros(len(lmbd_h))
AF = la.LinearOperator((Nh * Nc, Nh * Nc), matvec=lambda x: olo.FT(olo.F(x, f), f))  # , rmatvec=lambda x: FT(F(x)))

for i in tqdm(range(len(lmbd_h))):
    hinv0 = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=np.zeros_like(h.ravel()),
                        epsg=lmbd_h[i], niter=200, show=False, acceleration='fista', nonneg=True)

    #hinv0 = la.minres(AF, FT(g, f), maxiter=500, show=False, rtol=1e-9)[0].reshape(Nh, Nc)  # 9 s
    norm_h[i] = olo.norm(h - hinv0.reshape(Nh, Nc))

# %% ###################################################################################################################
show_lmbd(lmbd_h, norm_h)
# %% ###################################################################################################################
AF = la.LinearOperator((Nt*Ne, Nc*Nh), matvec=lambda x: olo.F(x), rmatvec=lambda x: olo.FT(x))

l2 = pyproximal.L2(Op=my_pylops.LinearOperator(AF), b=g.ravel() - f.ravel())
l1 = pyproximal.L1()
lmbd = lmbd_h[np.argmin(norm_h)]
# lmbd = 0.015
t0 = time()
hinv0 = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=np.zeros_like(h.ravel()),
                            epsg=lmbd, niter=200, show=True, acceleration='fista', nonneg=True)
t0 = time() - t0
# %% ###################################################################################################################
show_hest(hinv0, "Fista")
show_stats(hinv0, w_g)
# %% ###################################################################################################################
homp = np.zeros((Nh, Nc))
# t0 = time()
for i in tqdm(range(Nc)):
    AF = la.LinearOperator((Nt * Ne, Nh), matvec=lambda x: olo.Fo(x, i), rmatvec=lambda x: olo.FoT(x, i))
    AFp = my_pylops.LinearOperator(AF)
    homp[:, i] = my_pylops.optimization.sparsity.omp(AFp, g.ravel() - f.ravel(),  niter_outer=200,
                    niter_inner=Ne, sigma=1e-10, normalizecols=True, nonneg=False, discard=False)[0]
# t0 = time() - t0
# %%
show_stats(homp, w_g)
show_hest(homp, "MOD OMP")
