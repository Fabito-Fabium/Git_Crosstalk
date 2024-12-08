#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')

# %%
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from scipy.optimize import minimize
from scipy.optimize import Bounds
import scipy.sparse.linalg as la
from time import time

from source.synth_data import synth_data
from source.subProb_linOp import ownlinOp

import random
random.seed(42)
# %%
# Definindo os parâmetros do problema
dtype = np.float32

Fs = 125e6
Fc = 5e6
Nh = 8
Nt = 1875
Ne = 15

mySys = synth_data(Fs, Fc, Nh, Nt, Ne, vmax=1e5, bw=0.75, sim_dt=False)
f, h, idx, g = mySys.create_synth()
olo = ownlinOp(Fs, Fc, Nh, Nt, Ne, f, h, idx, remez=True, filt=mySys.get_pulse())
g = olo.apply_SNR(g, 40)

Nc = len(idx)

# %%
# seja um hest_{0} onde o max é em elementos com distância 1:
hest = np.zeros_like(h)
for i in range(Nc):
    d = idx[i][0] - idx[i][1]
    if abs(d) == 1:
        hest[0:1, i] = 0.9

# grafico da primeira estimativa para h
fig, axs = plt.subplots(ncols=2)
axs[0].imshow((h).T, aspect='auto', cmap = 'Greys')
axs[0].set_title('h real')
axs[1].imshow((hest).T, aspect='auto', cmap = 'Greys')
axs[1].set_title('hest')
# plt.figure()
# plt.imshow((h - hest).T, aspect='auto', vmax=1, vmin=-1, cmap='seismic')
# plt.title(f"norm: {norm(h-hest)}")
# plt.colorbar()


numiter = 1000

Fest_F_norm = np.zeros(numiter)
Hest_H_norm = np.zeros(numiter)
Hf_g_norm = np.zeros(numiter)
Fh_g_norm = np.zeros(numiter)
Gest_G_norm = np.zeros(numiter)
fest = (g+f.ravel()).reshape(Nt, Ne)
lmbd_h = 0.05
lmbd_f = 0.004
plt.figure()
plt.imshow(fest, aspect='auto')
plt.title('Primeiro f estimado')
# %%
for i in range(numiter):
    # subproblema Fh = g - f
    AF = la.LinearOperator((Nt*Ne, Nc*Nh), matvec=lambda x: F(x, fest), rmatvec=lambda x: FT(x, fest))
    l1 = pyproximal.L1()

    l2 = pyproximal.L2(Op=pylops.LinearOperator(AF), b=g)
    hest = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=hest.ravel(), niter=200, epsg=lmbd_h,show=False, acceleration='fista', nonneg=True)

    # subproblema Hf = g - f

    AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: H(x, hest), rmatvec=lambda x: HT(x, hest))

    l2 = pyproximal.L2(Op=pylops.LinearOperator(AH), b=g.ravel())
    l1 = pyproximal.L1(sigma=lambda x: normalized_energy(x).ravel())

    fest = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=np.zeros(Nt * Ne), epsg=lmbd_f, niter=120,
                                                           show=False, acceleration='fista', nonneg=False).reshape(Nt, Ne)

    # AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: HT(H(x, hest), hest) + lmbd_f*DT(D(x)))
    # fest = la.minres(AH, HT(g, hest), x0=fest.ravel(), maxiter=200, rtol=1e-20, show=False)[0].reshape(Nt, Ne)  # 9 s
    # if (abs(norm(H(fest_new, hest)-g)) < abs(norm(H(fest, hest)-g))) | (np.mean(H(fest_new, hest)-g)**2 < np.mean(norm(H(fest, hest)-g)))**2:
    #     fest = fest_new

    Hf_g_norm[i] = norm(H(f, hest) - g)
    Fh_g_norm[i] = norm(F(h, fest) - g)
    Fest_F_norm[i] = norm(fest.ravel() - f.ravel())
    Hest_H_norm[i] = norm(hest.ravel() - h.ravel())
    Gest_G_norm[i] = norm(H(fest, hest) - g)
    print(f"{i: 4d} \t Hf - g: {Hf_g_norm[i]: .5f} \t Fh - g: {Fh_g_norm[i]: .5f} \t fest - f: {Fest_F_norm[i]: .5f} \t hest - h: {Hest_H_norm[i]: .5f} \t\t Hest(f) - g: {Gest_G_norm[i]: .5f}")


# %%
plt.close("all")
fig, axs = plt.subplots(nrows=2)
axs[0].plot(Hest_H_norm[:i])
axs[0].set_title("norm Fest - f")
axs[1].plot(Fest_F_norm[:i])
axs[1].set_title("norm Hest - h")


fig, axs = plt.subplots(nrows=2)
axs[0].plot(Hf_g_norm[:i])
axs[0].set_title("norm Hest(f) - g")
axs[1].plot(Fh_g_norm[:i])
axs[1].set_title("norm Fest(h) - g")

plt.figure()
plt.plot(Gest_G_norm[:i])
plt.title("norm Hest(fest) - G ")


fig, axs = plt.subplots(ncols=2)
axs[0].imshow(h.reshape(Nh,Nc).T, aspect='auto', vmax=1, vmin=0, cmap='Greys')
axs[0].set_title("h real")
axs[1].imshow(hest.reshape(Nh,Nc).T, aspect='auto', vmax=1, vmin=0, cmap='Greys')
axs[1].set_title("h estimado")