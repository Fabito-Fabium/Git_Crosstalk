# %% ###################################################################################################################
import matplotlib
matplotlib.use('TkAgg')
########################################################################################################################
# %%
import pyproximal
from time import time
import time as temp
import cv2
import scipy.signal as ss
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la
from scipy import signal
from matplotlib import animation

import pyproximal
import pylops.linearoperator as pl
import pylops
from tqdm import tqdm

from source.synth_data import synth_data
from source.subProb_linOp import ownlinOp
import source.pylops as my_pylops
import random


from scipy.optimize import minimize
from scipy.optimize import Bounds
from source.fetch_data import fetch
from joblib import Parallel, delayed
random.seed(42)
plt.ion()
# %% ###################################################################################################################
# Definindo os parâmetros do problema ##################################################################################
dtype = np.float32

Fs = 125e6
Fc = 5e6
Nh = 8
Nt = 1875
Ne = 5

mySys = synth_data(Fs, Fc, Nh, Nt, Ne, vmax=1, bw=0.75, sim_dt=False)
f, h, idx, crs = mySys.create_synth()
olo = ownlinOp(Fs, Fc, Nh, Nt, Ne, f, h, idx, remez=True, filt=mySys.get_pulse())

g_clean = crs + f.ravel()
crs = olo.apply_SNR(crs, 40)[0]
g, w_g = olo.apply_SNR(g_clean, 40)


Nc = len(idx)

# show_plt()
########################################################################################################################

# %%
def hest_wrong_delay():
    hest = np.zeros((Nh, Nc))
    for i in range(Nc):
        hest[:, i] = np.zeros(Nh)
        d = idx[i][0] - idx[i][1]

        hest[np.random.randint(Nh), i] = fetch().gain()[np.abs(d)]
    # plt.imshow(hest, aspect='auto')
    return hest

def reset_all():
    # hest = np.zeros_like(h)
    # for i in range(Nc):
    #     d = idx[i][0] - idx[i][1]
    #     if abs(d) == 1:
    #         hest[0:1, i] = 0.9
    hest = h.ravel()
    # hest = olo.apply_SNR(hest_wrong_delay(), 5)[0]
    # grafico da primeira estimativa para h
    # fig, axs = plt.subplots(ncols=2)
    # axs[0].imshow((h).T, aspect='auto', cmap = 'Greys')
    # axs[0].set_title('h real')
    # axs[1].imshow((hest).T, aspect='auto', cmap = 'Greys')
    # axs[1].set_title('hest')
    # plt.figure()
    # plt.imshow((h - hest).T, aspect='auto', vmax=1, vmin=-1, cmap='seismic')
    # plt.title(f"norm: {norm(h-hest)}")
    # plt.colorbar()




    Fest_F_norm = np.zeros(numiter)
    Hest_H_norm = np.zeros(numiter)
    Hf_g_norm = np.zeros(numiter)
    Fh_g_norm = np.zeros(numiter)
    Gest_G_norm = np.zeros(numiter)
    fest = (olo.apply_SNR(f, 5)[0]).reshape(Nt, Ne)
    return Fest_F_norm, Hest_H_norm, Hf_g_norm, Fh_g_norm, Gest_G_norm, fest, hest

numiter = 200
# plt.figure()
# plt.imshow(fest, aspect='auto', interpolation='nearest')
# plt.title('Primeiro f estimado')
# %% # Hf = g first ####################################################################################################


lmbd_f_spc = np.logspace(-4, 1, 50)
n_lmbd_f = np.zeros(len(lmbd_f_spc))
mthd = "L-BFGS-B"
opts = {'maxiter': 200, 'disp': False}
from scipy.signal import savgol_filter
g_ft = g.reshape(Nt, Ne).copy()
for i in range(Ne):
    g_ft[:, i] = savgol_filter(g_ft[:, i],  5, 2, mode='nearest')
g_ft = g_ft.ravel()
t0 = time()
def get_lmbd_sub_f(lmbd_f_idx):
    lmbd_f = lmbd_f_spc[lmbd_f_idx]
    print(f"\n current lmbd:{lmbd_f}\n")
    Fest_F_norm, Hest_H_norm, Hf_g_norm, Fh_g_norm, Gest_G_norm, fest, hest = reset_all()
    # hest = olo.apply_SNR(hest, -2)[0]
    for i in range(numiter):
        hest_anim[:, :, i] = hest.reshape(Nh, Nc)

        AH = la.LinearOperator((Nt * Ne, Nt * Ne),
                               matvec=lambda x: olo.HT(olo.H(x, hest), hest) + lmbd_f * olo.DT(olo.D(x)))

        def fun(x):
            res = olo.H(x, hest) - (g_ft - x)
            min = olo.norm(res) + lmbd_f * olo.norm(olo.D(x))
            jac = (AH * x + olo.H(x, hest) + olo.HT(x - g_ft, hest) + (x - g_ft))
            return min, jac

        x0 = fest.ravel()
        # x0 = np.zeros(Nt*Ne)
        fest = minimize(fun, x0, method=mthd, jac=True, options=opts).x.reshape(Nt, Ne)

        Hf_g_norm[i] = olo.norm(olo.H(fest) - crs)
        Fh_g_norm[i] = olo.norm(olo.F(hest) - crs)
        Fest_F_norm[i] = olo.norm(fest.ravel() - f.ravel())
        Hest_H_norm[i] = olo.norm(hest.ravel() - h.ravel())
        Gest_G_norm[i] = olo.norm(olo.H(fest, hest) - crs)
        print(f"f{i: 4d} \t Hf - g: {Hf_g_norm[i]: .5f} \t Fh - g: {Fh_g_norm[i]: .5f} \t fest - f: "
              f"{Fest_F_norm[i]: .5f} \t hest - h: {Hest_H_norm[i]: .5f} \t\t Hest(f) - g: {Gest_G_norm[i]: .5f}")

        homp = np.zeros((Nh, Nc))

        def omp_col(ii):
            AF = la.LinearOperator((Nt * Ne, Nh), matvec=lambda x: olo.Fo(x, ii, fest),
                                   rmatvec=lambda x: olo.FoT(x, ii, fest))
            AFp = my_pylops.LinearOperator(AF)
            homp_col = my_pylops.optimization.sparsity.omp(AFp, g_ft - fest.ravel(),
                                                           niter_outer=300, niter_inner=Ne, sigma=1e-10,
                                                           normalizecols=True, nonneg=False, discard=True)[0]

            return homp_col

        par_out = Parallel(n_jobs=-1)(delayed(omp_col)(i) for i in range(Nc))

        hest = np.array(par_out).T

        Hf_g_norm[i] = olo.norm(olo.H(fest, h) - crs)
        Fh_g_norm[i] = olo.norm(olo.F(hest) - crs)
        Fest_F_norm[i] = olo.norm(fest.ravel() - f.ravel())
        Hest_H_norm[i] = olo.norm(hest.ravel() - h.ravel())
        Gest_G_norm[i] = olo.norm(olo.H(fest, hest) - crs)
        print(f"h{i: 4d} \t Hf - g: {Hf_g_norm[i]: .5f} \t Fh - g: {Fh_g_norm[i]: .5f} \t fest - f: "
              f"{Fest_F_norm[i]: .5f} \t hest - h: {Hest_H_norm[i]: .5f} \t\t Hest(f) - g: {Gest_G_norm[i]: .5f}")
        plt.close('all')

        if olo.norm(hest.ravel() - h.ravel()) < 1e-6:
            break

    return np.mean(Gest_G_norm), np.mean(Hest_H_norm), np.mean(Fest_F_norm)

lmbd_par = Parallel(n_jobs=-1)(delayed(get_lmbd_sub_f)(i) for i in range(len(lmbd_f_spc)))
print(time() - t0)
lmbd_par = np.array(lmbd_par)

plt.loglog(lmbd_f_spc, lmbd_par[:, 2])

# %%

plt.close('all')
plt.loglog(lmbd_f_spc, n_lmbd_f)
plt.xlabel("$\lambda$")
plt.ylabel("$||f-f_{est}||_2$")
plt.title(f"Lambda do subproblema Hf = g - f, melhor lmbd: {lmbd_f_spc[np.argmin(n_lmbd_f)]}")

# %% # Fh = g first ####################################################################################################
Fest_F_norm, Hest_H_norm, Hf_g_norm, Fh_g_norm, Gest_G_norm, fest, hest = reset_all()
lmbd_f = lmbd_f_spc[np.argmin(lmbd_par[:, 2])]
print(lmbd_f)
# lmbd_f = 0.01
mthd = "L-BFGS-B"
opts = {'maxiter': 1000, 'disp': False}

from scipy.signal import savgol_filter
g_ft = g.reshape(Nt, Ne).copy()
for i in range(Ne):
    g_ft[:, i] = savgol_filter(g_ft[:, i],  5, 2, mode='nearest')
g_ft = g_ft.ravel()
b_OMP = g_ft - f.ravel()
plt.ion()
g_tst = g.ravel()
hest_anim = np.zeros((Nh, Nc, numiter))
h_last = 0
f_last = 0
# hest = olo.apply_SNR(hest, -2)[0]
for i in range(numiter):
    hest_anim[:, :, i] = hest.reshape(Nh, Nc)

    AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: olo.HT(olo.H(x, hest), hest) + lmbd_f * olo.DT(olo.D(x)))

    def fun(x):
        res = olo.H(x, hest) - (g_ft - x)
        min = olo.norm(res) + lmbd_f * olo.norm(olo.D(x))
        jac = (AH * x + olo.H(x, hest) + olo.HT(x - g_ft, hest) + (x - g_ft))
        return min, jac


    x0 = fest.ravel()
    # x0 = np.zeros(Nt*Ne)
    fest = minimize(fun, x0, method=mthd, jac=True, options=opts).x.reshape(Nt, Ne)

    Hf_g_norm[i] = olo.norm(olo.H(fest) - crs)
    Fh_g_norm[i] = olo.norm(olo.F(hest) - crs)
    Fest_F_norm[i] = olo.norm(fest.ravel() - f.ravel())
    Hest_H_norm[i] = olo.norm(hest.ravel() - h.ravel())
    Gest_G_norm[i] = olo.norm(olo.H(fest, hest) - crs)
    print(f"f{i: 4d} \t Hf - g: {Hf_g_norm[i]: .5f} \t Fh - g: {Fh_g_norm[i]: .5f} \t fest - f: "
          f"{Fest_F_norm[i]: .5f} \t hest - h: {Hest_H_norm[i]: .5f} \t\t Hest(f) - g: {Gest_G_norm[i]: .5f}")

    homp = np.zeros((Nh, Nc))
    def omp_col(ii):
        AF = la.LinearOperator((Nt * Ne, Nh), matvec=lambda x: olo.Fo(x, ii, fest), rmatvec=lambda x: olo.FoT(x, ii, fest))
        AFp = my_pylops.LinearOperator(AF)
        homp_col = my_pylops.optimization.sparsity.omp(AFp, g_ft - fest.ravel(),
                                        niter_outer=300,niter_inner=Ne, sigma=1e-10,
                                        normalizecols=True, nonneg=False, discard=False)[0]

        return homp_col

    par_out = Parallel(n_jobs=-1)(delayed(omp_col)(i) for i in range(Nc))

    hest = np.array(par_out).T


    Hf_g_norm[i] = olo.norm(olo.H(fest, h) - crs)
    Fh_g_norm[i] = olo.norm(olo.F(hest) - crs)
    Fest_F_norm[i] = olo.norm(fest.ravel() - f.ravel())
    Hest_H_norm[i] = olo.norm(hest.ravel() - h.ravel())
    Gest_G_norm[i] = olo.norm(olo.H(fest, hest) - crs)
    print(f"h{i: 4d} \t Hf - g: {Hf_g_norm[i]: .5f} \t Fh - g: {Fh_g_norm[i]: .5f} \t fest - f: "
          f"{Fest_F_norm[i]: .5f} \t hest - h: {Hest_H_norm[i]: .5f} \t\t Hest(f) - g: {Gest_G_norm[i]: .5f}")
    plt.close('all')

    if olo.norm(hest.ravel() - h.ravel()) < 1e-6:
        break


# %% ###################################################################################################################
plt.close("all")
fig, axs = plt.subplots(nrows=2)
axs[0].plot(Hest_H_norm[:i])
axs[1].set_title("norm Fest - f")
axs[1].plot(Fest_F_norm[:i])
axs[0].set_title("norm Hest - h")

fig, axs = plt.subplots(nrows=2)
axs[0].plot(Hf_g_norm[:i])
axs[0].set_title("norm Hest(f) - g")
axs[1].plot(Fh_g_norm[:i])
axs[1].set_title("norm Fest(h) - g")

plt.figure()
plt.plot(Gest_G_norm[:i])
plt.title("norm Hest(fest) - G ")

fig, axs = plt.subplots(ncols=2)
axs[0].imshow(h.reshape(Nh,Nc).T, aspect='auto', cmap='Greys')
axs[0].set_title("h real")
axs[1].imshow(hest.reshape(Nh,Nc).T, aspect='auto', cmap='Greys')
axs[1].set_title("h estimado")

# %%
from matplotlib import animation
# %%
fig = plt.figure()
im = plt.imshow(hest_anim[:, :, 0].T, cmap='Greys', interpolation='nearest', aspect='auto')
plt.colorbar(im)
plt.show()

def animate(i):
    art = hest_anim[:, :, i].T
    im.set_array(art)

ani = animation.FuncAnimation(fig, animate, frames=50, interval=1e-7)
writervideo = animation.FFMpegWriter(fps=5)
ani.save('hest_c15dB_h-MOD.mp4', writer=writervideo)
plt.close()
