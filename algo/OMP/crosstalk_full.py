#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')
from statsmodels.sandbox.distributions.genpareto import method
from tqdm import tqdm


import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import signal

from scipy.optimize import minimize
from scipy.optimize import Bounds
import scipy.sparse.linalg as la
import pyproximal
import pylops

from time import time
import math


dtype = np.float32

Fs = 50e6
Fc = 5e6
Nh = 20
Nt = 1875//5
t = np.arange(Nt, dtype=dtype)/Fs
Ne = 5
bw = .75

f = np.zeros((Nt, Ne), dtype=dtype)
pulse = np.zeros((Nt, Ne), dtype=dtype)

for ne in range(Ne-1):
    #tau = t - 15*ne/Fs
    tau = (t - (ne/Ne + 1/Ne)*Nt/Fs)
    f[:, ne] = ss.gausspulse(tau, Fc, bw)

    F = np.fft.fft(f[:, ne])
    # Pulse = np.zeros_like(F) + np.max(np.abs(F))
    # Pulse -= np.abs(F)
    # pulse[:, ne] = np.fft.ifft(Pulse)

    t0 = np.abs(tau).argmin()
    pulse[t0, ne] = np.max(np.abs(F))
    pulse[:, ne] -= f[:, ne]

    #
    # Pulse = np.fft.fft(pulse[:, ne-2])
    # ff = np.fft.fftfreq(len(t), 1/Fs)
    # Pulse = np.fft.fft(pulse[:, ne])
    # ff = np.fft.fftshift(np.fft.fftfreq(len(t), 1/Fs))
    # ff = (Fs / 2) * np.arange(len(F)) / len(F)
    # plt.plot(ff, np.abs(np.fft.fftshift(F)), ff, np.abs(np.fft.fftshift(Pulse)))

#plt.imshow(f, aspect='auto', interpolation='nearest')
# plt.plot(pulse)

# Simulação do problema (seção referente ao h)
#idx = list(combinations(range(Ne), 2))
idx = [(i, j) for i, j in combinations(range(Ne), 2) if abs(i-j) < 30]

Nc = len(idx)
h = np.zeros((Nh, Nc))
for i in range(Nc):
    h[:, i] = np.zeros(Nh)
    d = idx[i][0] - idx[i][1]
    h[min(Nh-1, np.abs(d)-1), i] = np.exp(-d**2/10)


# band = [5e6*0.5, 5e6*1.5]  # Desired stop band, Hz
# trans_width = 3e5    # Width of transition from pass to stop, Hz
# numtaps = 555        # Size of the FIR filter.
# edges = [0, band[0] - trans_width, band[0], band[1],
#          band[1] + trans_width, 0.5*Fs]
# b = signal.remez(numtaps, edges, [1, 0, 1], fs=Fs)
# # w, h = signal.freqz(b, [1], worN=2000, fs=Fs)
# # plot_response(w, h, "Band-stop Filter")

def D(x):
    x_ = x.reshape((Nt, Ne))
    y = np.zeros((Nt, Ne), dtype=dtype)
    for i in range(Ne):
        y[:, i] = ss.convolve(x_[:, i], pulse[:, i], mode="same", method="fft")
    return y.ravel()

def DT(x):
    x_ = x.reshape((Nt, Ne))
    y = np.zeros((Nt, Ne), dtype=dtype)
    for i in range(Ne):
        y[:, i] = ss.correlate(x_[:, i], pulse[:, i], mode="same", method="fft")
    return y.ravel()


dCss = 3

def H(x_, op=h):
    h = op.reshape(Nh, Nc, )
    x = x_.reshape(Nt, Ne, )
    y = np.zeros((Nt, Ne))
    for i in range(Nc):
        y[dCss:, idx[i][0]] += ss.convolve(x[:, idx[i][1]], h[:, i], mode="same", method="fft")[:-dCss]
        y[dCss:, idx[i][1]] += ss.convolve(x[:, idx[i][0]], h[:, i], mode="same", method="fft")[:-dCss]
    return y.ravel()

# Hmtx = np.zeros((Nt*Ne, Nt*Ne))
# ff = np.zeros(Nt*Ne)
# for i in range(Nt*Ne):
#     ff[i] = 1
#     Hmtx[:, i] = H(ff.reshape(Nt, Ne), h).ravel()
#     ff[i] = 0

if -((Nh-1)//2)+dCss >= 0:
    slcH = slice(dCss+Nh//2,Nt+dCss+Nh//2)
else:
    slcH = slice(dCss + Nh // 2, -((Nh-1)//2)+dCss)


print(slcH)
def HT(x_, op=h):
    h = op.reshape(Nh, Nc, )
    x = x_.reshape(Nt, Ne, )
    y = np.zeros((Nt, Ne))
    for i in range(Nc):
        y[:, idx[i][1]] += ss.correlate(x[:, idx[i][0]], h[:, i], mode="full")[slcH]
        y[:, idx[i][0]] += ss.correlate(x[:, idx[i][1]], h[:, i], mode="full")[slcH]
    return y.ravel()


def F(x_, f=f):
    x = x_.reshape(Nh, Nc)
    y = np.zeros((Nt, Ne), dtype=dtype)
    for i in range(Nc):
        y[dCss:, idx[i][0]] += ss.convolve(f[:, idx[i][1]], x[:, i], mode="same")[:-dCss]
        y[dCss:, idx[i][1]] += ss.convolve(f[:, idx[i][0]], x[:, i], mode="same")[:-dCss]
    return y.ravel()


# Fmtx = np.zeros((Nt*Ne, Nh*Nc))
# hh = np.zeros(Nh*Nc)
# for i in tqdm(range(Nh*Nc)):
#     hh[i] = 1
#     Fmtx[:, i] = F(hh.reshape(Nh, Nc), f).ravel()
#     hh[i] = 0

slcFT = slice(dCss+(Nt+1)//2 - (Nh+1)//2 + (Nt+1) % 2, -((Nt)//2 - (Nh)//2 - (Nt+1) % 2)+dCss)


def FT(x_, f=f):
    x = x_.reshape(Nt, Ne)
    y = np.zeros((Nh, Nc), dtype=dtype)
    for i in range(Nc):
        y[:, i] += ss.correlate(x[:, idx[i][0]], f[:, idx[i][1]], mode="same")[slcFT]
        y[:, i] += ss.correlate(x[:, idx[i][1]], f[:, idx[i][0]], mode="same")[slcFT]
    return y.ravel()

def norm(x):
    return np.sum(x**2)


def normalized_energy(x):
    return x/np.sqrt(np.sum(x**2)+1e-20)
# %%
# Definindo "g"
SNR = 40  # dB

g_clean = H(f, h)

s2g = np.mean(g_clean**2)
s2w = s2g * 10**(-SNR/10)

w = np.sqrt(s2w) * np.random.randn(Nt*Ne)

SNR_calc = 10*np.log10(np.mean(g_clean**2) / np.mean(w**2))

g = g_clean + w

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