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

import source.pyproximal as pyproximal
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
    hest0 = h.copy()
    hest0[:, 1] = hest0[:, 2]
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
    fest = np.zeros((Nt, Ne))
    return Fest_F_norm, Hest_H_norm, Hf_g_norm, Fh_g_norm, Gest_G_norm, fest, hest0

numiter = 100
# plt.figure()
# plt.imshow(fest, aspect='auto', interpolation='nearest')
# plt.title('Primeiro f estimado')
# %% # Hf = g first ####################################################################################################
# lmbd_f_spc = np.logspace(-3, 0, 100)
hagime = 0.05
owari = 0.1
lmbd_f_spc = np.arange(hagime, owari, (owari - hagime)/200)
n_lmbd_f = np.zeros(len(lmbd_f_spc))
mthd = "CG"
opts = {'maxiter': 300, 'disp': False}
def get_lmbd_sub_f(lmbd_f_idx):
    lmbd_f = lmbd_f_spc[lmbd_f_idx]
    print(f"\n current lmbd:{lmbd_f}\n")
    Fest_F_norm, Hest_H_norm, Hf_g_norm, Fh_g_norm, Gest_G_norm, fest, hest = reset_all()
    b_OMP = olo.apply_SNR(crs, 10)[0]
    fest = np.zeros(Nt*Ne)
    # hest = olo.apply_SNR(hest, -2)[0]
    for i in range(numiter):
        AH = la.LinearOperator((Nt * Ne, Nt * Ne),
                               matvec=lambda x: olo.HT(olo.H(x, hest), hest) + lmbd_f * olo.DT(olo.D(x)))


        def fun(x):
            res = olo.H(x, hest) - (g - x)
            min = olo.norm(res) + lmbd_f * olo.norm(olo.D(x)) # +lmbd_f/(olo.score_teng(x) + 1e-6)
            jac = (AH * x + olo.H(x, hest) + olo.HT(x - g, hest) + (x - g))
            x.reshape(Nt,Ne)[0:10, :] = 0
            return min, jac


        fest = minimize(fun, fest.ravel(), method=mthd, jac=True, options=opts).x.reshape(Nt, Ne)

        def omp_col(ii):
            AF = la.LinearOperator((Nt * Ne, Nh), matvec=lambda x: olo.Fo(x, ii, fest),
                                   rmatvec=lambda x: olo.FoT(x, ii, fest))
            AFp = my_pylops.LinearOperator(AF)
            homp_col = my_pylops.optimization.sparsity.omp(AFp, g.ravel() - fest.ravel(),
                                                           niter_outer=1, niter_inner=200, sigma=1e-20,
                                                           normalizecols=True, nonneg=True)[0]
            return homp_col


        # def fista_col(ii):
        #     AF = la.LinearOperator((Nt * Ne, Nh), matvec=lambda x: olo.Fo(x, ii, fest), rmatvec=lambda x: olo.FoT(x, ii, fest))
        #     AFp = my_pylops.LinearOperator(AF)
        #
        #     l2 = pyproximal.L2(Op=AFp, b=g.ravel() - fest.ravel())
        #     l1 = pyproximal.L1()
        #     hfista_col = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=hest[:, ii],
        #                                                                  epsg=0.014, niter=200,
        #                                                                  show=False, acceleration='fista', nonneg=True)
        #
        #     return hfista_col

        par_out = Parallel(n_jobs=-2)(delayed(omp_col)(i) for i in range(Nc))

        hest = np.array(par_out).T

        Hf_g_norm[i] = olo.norm(olo.H(fest, h) - crs)
        Fh_g_norm[i] = olo.norm(olo.F(hest) - crs)
        Fest_F_norm[i] = olo.norm(fest.ravel() - f.ravel())
        Hest_H_norm[i] = olo.norm(hest.ravel() - h.ravel())
        Gest_G_norm[i] = olo.norm(olo.H(fest, hest) - crs)

        print(f"f{i: 4d} \t Hfest - g: {Hf_g_norm[i]: .5f} \t Fhest - g: {Fh_g_norm[i]: .5f} \t fest - f: "
              f"{Fest_F_norm[i]: .5f} \t hest - h: {Hest_H_norm[i]: .5f} \t\t Hest(fest) - g: {Gest_G_norm[i]: .5f}")

        if olo.norm(hest.ravel() - h.ravel()) < 1e-6:
            break

    fest_diff_norm = olo.norm(fest.ravel() - f.ravel())
    hest_diff_norm = olo.norm(hest.ravel() - h.ravel())
    gest_diff_norm = olo.norm(olo.H(fest, hest) - crs)

    return gest_diff_norm, hest_diff_norm, fest_diff_norm

lmbd_par = Parallel(n_jobs=-3)(delayed(get_lmbd_sub_f)(i) for i in range(len(lmbd_f_spc)))

# %%
n_lmbd_f = np.array(lmbd_par)[:, 1]
plt.close('all')
plt.loglog(lmbd_f_spc, n_lmbd_f)
plt.xlabel("$\lambda$")
plt.ylabel("$||f-f_{est}||_2$")
plt.title(f"Lambda do subproblema Hf = g - f, melhor lmbd: {lmbd_f_spc[np.argmin(n_lmbd_f)]}")

# %% # Fh = g first ####################################################################################################
Fest_F_norm, Hest_H_norm, Hf_g_norm, Fh_g_norm, Gest_G_norm, fest, hest = reset_all()
# lmbd_f = 0.0608
lmbd_f = 0.056
mthd = "CG"
opts = {'maxiter': 300, 'disp': False}
b_OMP = olo.apply_SNR(crs, 10)[0]
plt.ion()
hest_anim = np.zeros((Nh, Nc, numiter+1))
fest_anim = np.zeros((Nt, Ne, numiter+1))
h_last = 0
f_last = 0
# hest = olo.apply_SNR(hest, -2)[0]
fest_anim[:, :, 0] = np.zeros((Nt, Ne))
hest_anim[:, :, 0] = hest.reshape((Nh, Nc))
x0 = np.zeros(Nt*Ne)
for i in range(numiter):
    AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: olo.HT(olo.H(x, hest), hest) + lmbd_f * olo.DT(olo.D(x)))

    def fun(x):
        res = olo.H(x, hest) - (g - x)
        min = olo.norm(res) + lmbd_f * olo.norm(olo.D(x))# + lmbd_f/(olo.score_teng(x) + 1e-6)
        jac = (AH * x + olo.H(x, hest) + olo.HT(x - g, hest) + (x - g))
        x.reshape(Nt, Ne)[0:10, :] = 0
        return min, jac


    fest = minimize(fun, x0, method=mthd, jac=True, options=opts).x.reshape(Nt, Ne)
    fest_anim[:, :, i+1] = fest.reshape(Nt, Ne)
    x0 = fest.reshape(Nt*Ne)


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
        homp_col = my_pylops.optimization.sparsity.omp(AFp, g.ravel() - fest.ravel(),
                                        niter_outer=1, niter_inner=200, sigma=1e-20,
                                        normalizecols=True, nonneg=True)[0]

        return homp_col

    par_out = Parallel(n_jobs=-1)(delayed(omp_col)(i) for i in range(Nc))

    hest = np.array(par_out).T
    hest_anim[:, :, i+1] = hest.reshape(Nh, Nc)

    # def fista_col(ii):
    #     AF = la.LinearOperator((Nt * Ne, Nh), matvec=lambda x: olo.Fo(x, ii), rmatvec=lambda x: olo.FoT(x, ii))
    #     AFp = my_pylops.LinearOperator(AF)
    #
    #     l2 = pyproximal.L2(Op=AFp, b=g.ravel() - fest.ravel())
    #     l1 = pyproximal.L1()
    #     hfista_col = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=hest[:, ii],
    #                                                                  epsg=0.014, niter=200,
    #                                                                  show=False, acceleration='fista', nonneg=True)
    #
    #     return hfista_col
    #
    #
    # par_out = Parallel(n_jobs=-2)(delayed(fista_col)(i) for i in range(Nc))
    # hest = np.array(par_out).T

    Hf_g_norm[i] = olo.norm(olo.H(fest, h) - crs)
    Fh_g_norm[i] = olo.norm(olo.F(hest) - crs)
    Fest_F_norm[i] = olo.norm(fest.ravel() - f.ravel())
    Hest_H_norm[i] = olo.norm(hest.ravel() - h.ravel())
    Gest_G_norm[i] = olo.norm(olo.H(fest, hest) - crs)
    print(f"h{i: 4d} \t Hf - g: {Hf_g_norm[i]: .5f} \t Fh - g: {Fh_g_norm[i]: .5f} \t fest - f: "
          f"{Fest_F_norm[i]: .5f} \t hest - h: {Hest_H_norm[i]: .5f} \t\t Hest(f) - g: {Gest_G_norm[i]: .5f}")
    plt.close('all')

    # if olo.norm(hest.ravel() - h.ravel()) < 1e-6:
    #     break


# %% ###################################################################################################################
plt.close("all")
fig, axs = plt.subplots(nrows=2)
axs[0].plot(g.reshape(Nt, Ne)[:, 0])
axs[1].plot(fest[:, 0])
axs[0].set_title("Comparação entre a primeira coluna de g e fest")

# %%
fig, axs = plt.subplots(nrows=2)
axs[0].plot(Hest_H_norm[:i])
axs[0].set_title(f"norm hest - h, valor final: {Hest_H_norm[99]}")
axs[1].plot(Fest_F_norm[:i])
axs[1].set_title(f"norm fest - f, valor final: {Fest_F_norm[99]}")


fig, axs = plt.subplots(nrows=2)
axs[0].plot(Hf_g_norm[:i])
axs[0].set_title(f"norm Hest(f) - g, valor final:{Hf_g_norm[99]}")
axs[1].plot(Fh_g_norm[:i])
axs[1].set_title(f"norm Fest(h) - g, valor final:{Fh_g_norm[99]}")

plt.figure()
plt.plot(Gest_G_norm[:i])
plt.title(f"norm Hest(fest) - G, valor final:{Hest_H_norm[99]}")

fig, axs = plt.subplots(ncols=2)
h1 = axs[0].imshow(h.reshape(Nh,Nc).T, aspect='auto', cmap='Greys')
axs[0].set_title("h real")
h0 = axs[1].imshow(hest.reshape(Nh,Nc).T, aspect='auto', cmap='Greys')
axs[1].set_title("h estimado")
plt.colorbar(h1, ax=axs[0])
plt.colorbar(h0, ax=axs[1])
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

ani = animation.FuncAnimation(fig, animate, frames=50, interval=50)
writervideo = animation.FFMpegWriter(fps=5)
ani.save('hest_c15dB_h-MOD.mp4', writer=writervideo)
plt.close()
# %%
fig, axs = plt.subplots(ncols=2)
im0 = axs[0].imshow(hest_anim[:, :, 0].T, cmap='Greys', interpolation='nearest', aspect='auto')
im1 = axs[1].imshow(fest_anim[:, :, 0], aspect='auto', interpolation='nearest', vmax=20, vmin=-20)
plt.colorbar(im0)
plt.show()

def animate(i):
    art0 = hest_anim[:, :, i].T
    art1 = fest_anim[:, :, i]
    im0.set_array(art0)
    im1.set_array(art1)
    im0.axes.set_title(f"{i}th hest")
    im1.axes.set_title(f"{i}th fest")

ani = animation.FuncAnimation(fig, animate, frames=101, interval=750)
writervideo = animation.FFMpegWriter(fps=1)
ani.save('hest_c15dB_h-MOD.mp4', writer=writervideo)
plt.close()
# %%
def show_stats(hest, fest, w_g):
    print(f"||hh-h||^2: \t\t\t\t {olo.norm(hest.reshape(Nh, Nc) - h)}")
    print(f"||fh-f||^2: \t\t\t\t {olo.norm(fest.reshape(Nt, Ne) - f)}")
    print(f"||w||^2: \t\t\t\t\t {olo.norm(w_g)}")
    print(f"||F(hh) - crs_clean||^2: \t {olo.norm(olo.F(hest) - crs)}")
    print(f"||F(hh) - crs||^2: \t\t\t {olo.norm(olo.F(hest) - (g-f.ravel()))}")

    print(f"||H(fh) - crt_clean||^2: \t {olo.norm(olo.H(fest, h) - (g_clean - f.ravel()))}")
    print(f"||H(fh) - crt||^2: \t\t\t {olo.norm(olo.H(fest, h) - (g - f.ravel()))}")

    print(f"||Fh(hh) - crs_clean||^2: \t {olo.norm(olo.F(hest, fest) - crs)}")
    print(f"||Fh(hh) - crs||^2: \t\t {olo.norm(olo.F(hest, fest) - (g-f.ravel()))}")