# %%
import matplotlib
matplotlib.use('TkAgg')

########################################################################################################################
# %%
import pyproximal
import time
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
import source.pylops as pylops
from tqdm import tqdm

from source.synth_data import synth_data
from source.subProb_linOp import ownlinOp
import random


from scipy.optimize import minimize
from scipy.optimize import Bounds

random.seed(42)
plt.ion()
# %% ###################################################################################################################
def show_el(fest):
    #fest = D(fest).reshape(Nt, Ne)
    plt.close('all')
    plt.figure()
    plt.plot(f[:, 1])
    plt.plot(fest[:, 1])
    plt.title(f'norm fest - f: {olo.norm(fest - f)}')

    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(f[:, 3])
    axs[0].set_title(f"f el 3, norm f3 - fest3={olo.norm(fest[:, 3]-f[:, 3])}")
    axs[1].plot(fest[:, 3])
    axs[1].set_title(f"fest el 3")

    fig, axs = plt.subplots(ncols=3)
    axs[0].imshow(f, aspect='auto', interpolation='nearest')
    axs[0].set_title(f"real f")
    axs[1].imshow(fest, aspect='auto', interpolation='nearest')
    axs[1].set_title(f"estimated f")
    diff = axs[2].imshow(fest - f, aspect='auto', interpolation='nearest')
    axs[2].set_title(f"f - fest, $||f - fest||_2^2$ = {olo.norm(fest - f)}")
    fig.colorbar(diff, ax=axs[2])


def show_plt():
    print(f'norm_vec f: {olo.norm_vec(f)}, norm_vec g: {olo.norm_vec(g)}')
    plt.close('all')
    fig1, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=False)

    ax1.imshow(20 * np.log(abs(f) + 1), interpolation='nearest', aspect='auto')
    ax1.set_title(f'b-scan sem crosstalk')
    ax2.imshow(20 * np.log(abs(g) + 1).reshape(Nt, Ne), interpolation='nearest', aspect='auto')
    ax2.set_title(f'b-scan com crosstalk')

def show_lmbd(lmbd_spc, norm_res):
    plt.close('all')
    plt.loglog(lmbd_spc, norm_res)
    plt.xlabel("$\lambda$")
    plt.ylabel("$||f-f_{est}||_2$")
    plt.title(f"Lambda do subproblema Hf = g - f, melhor lmbd: {lmbd_spc[np.argmin(norm_res)]}")

def mimshow(data):
    aux = np.array(data)
    plt.cla()
    if len(aux.ravel()) == Nt*Ne:
        plt.imshow(aux.reshape(Nt, Ne), aspect='auto', interpolation='nearest')
    else:
        plt.imshow(aux.reshape(Nh, Nc), aspect='auto', interpolation='nearest')


def print_stats(fest):
    print(f"||fh-f||^2: \t\t\t {olo.norm(fest - f)}")
    print(f"||w||^2: \t\t\t\t {olo.norm(w_g)}")
    print(f"||H(fh) - crt_clean||^2: {olo.norm(olo.H(fest, h) - (g_clean - f.ravel()))}")
    print(f"||H(fh) - crt||^2: \t\t {olo.norm(olo.H(fest, h) - (g - f.ravel()))}")

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
# minres no regularization
HL = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: olo.HT(olo.H(x)))
fest = la.minres(HL, olo.HT(crs),x0=np.zeros(Nt*Ne), maxiter=200, rtol=1e-20, show=True)[0]
fest = fest.reshape((Nt, Ne), )  # 9 s

# %%
# L2 regularization using mires
# multi-valued lambda
lmbd_spc = np.logspace(-15, 0, 200)
mse_lmbd = np.zeros(len(lmbd_spc))
norm_res = np.zeros(len(lmbd_spc))
norm_fest = np.zeros(len(lmbd_spc))
norm_f = np.zeros(len(lmbd_spc))

for i in tqdm(range(len(lmbd_spc))):
##### Old remez ########################################################################################################
    # AH = la.LinearOperator((Nt * Ne, Nt * Ne),
    #                        matvec=lambda x: olo.HT(olo.H(x, h), h) + lmbd_spc[i]*olo.DT(olo.D(x)),
    #                        rmatvec=lambda x: olo.HT(olo.H(x, h), h) + lmbd_spc[i]*olo.DT(olo.D(x)))
    #
    # fest = la.minres(AH, olo.HT(crs, h), x0=np.zeros(Nt*Ne), maxiter=200, rtol=1e-20, show=False)[0].reshape((Nt, Ne))
##### variable crs #####################################################################################################
    HA = la.LinearOperator((Nt * Ne, Nt * Ne),
        matvec=lambda x: olo.HT(olo.H(x, h), h) + lmbd_spc[i] * olo.DT(olo.D(x)) + olo.H(x) + olo.HT(x) + x,
        rmatvec=lambda x: olo.HT(olo.H(x, h), h) + lmbd_spc[i] * olo.DT(olo.D(x)) + olo.H(x) + olo.HT(x) + x)

    fest = la.minres(HA, olo.HT(g, h) + g, x0=np.zeros(Nt*Ne), maxiter=200, rtol=1e-20, show=False)[0].reshape((Nt, Ne))
########################################################################################################################
    mse_lmbd[i] = np.mean(f - fest)**2
    norm_res[i] = olo.norm(fest - f)
    norm_fest[i] = olo.norm(fest)
    norm_f[i] = olo.norm(f)
# %% ###################################################################################################################
show_lmbd(lmbd_spc, norm_res)
# %% ###################################################################################################################
# Minres, single valued lambda
lmbd = lmbd_spc[np.argmin(norm_res)]
# old ##################################################################################################################
# AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: olo.HT(olo.H(x, h), h) + lmbd * olo.DT(olo.D(x)),
#                        rmatvec=lambda x: olo.HT(olo.H(x, h), h) + lmbd * olo.DT(olo.D(x)))
#
# fest = la.minres(AH, olo.HT(crs, h), x0=np.zeros(Nt*Ne), maxiter=200, rtol=1e-20, show=False)[0].reshape((Nt, Ne))
##### variable crs #####################################################################################################
HA = la.LinearOperator((Nt * Ne, Nt * Ne),
    matvec=lambda x: olo.HT(olo.H(x, h), h) + lmbd * olo.DT(olo.D(x)) + olo.H(x) + olo.HT(x) + x,
    rmatvec=lambda x: olo.HT(olo.H(x, h), h) + lmbd * olo.DT(olo.D(x)) + olo.H(x) + olo.HT(x) + x)

fest = la.minres(HA, olo.HT(g, h) + g, x0=np.zeros(Nt*Ne), maxiter=200, rtol=1e-20, show=False)[0].reshape((Nt, Ne))
# %%####################################################################################################################
fest = fest.reshape((Nt, Ne))
print_stats(fest)
show_el(fest)
# %% ###################################################################################################################
# L2 regularization with minimize
x0 = np.zeros((Nt, Ne), dtype=dtype).ravel()


lmbd_spc = np.logspace(-10, 1, 50)
mse_lmbd = np.zeros(len(lmbd_spc))
norm_res = np.zeros(len(lmbd_spc))
norm_fest = np.zeros(len(lmbd_spc))
norm_f = np.zeros(len(lmbd_spc))

mthd = "CG"
opts = {'maxiter': 200, 'disp': False}

for i in tqdm(range(len(lmbd_spc))):
    AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: olo.HT(olo.H(x, h), h) + lmbd_spc[i]*olo.DT(olo.D(x)),
                           rmatvec=lambda x: olo.HT(olo.H(x, h), h) + lmbd[i]*olo.DT(olo.D(x)))
    def fun(x):
        res = olo.H(x) - (g - x)
        min = olo.norm(res) + lmbd_spc[i] * olo.norm(olo.D(x))
        jac = (AH * x + olo.H(x) + olo.HT(x-g) + (x - g))
        return min, jac

    fh = minimize(fun, x0, method=mthd, jac=True, options=opts).x.reshape(Nt, Ne)  # 17 s
    mse_lmbd[i] = np.mean(f - fh)**2
    norm_res[i] = olo.norm(fh - f)
    norm_fest[i] = olo.norm(fh)
    norm_f[i] = olo.norm(f)

# %% ###################################################################################################################
show_lmbd(lmbd_spc, norm_res)
# %%
lmbd = lmbd_spc[np.argmin(norm_res)]
# lmbd = 0.1

mthd = "CG"
opts = {'maxiter': 500, 'disp': False}

AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: olo.HT(olo.H(x, h), h) + lmbd * olo.DT(olo.D(x)))

def fun(x):
    res = olo.H(x) - (g-x)
    min = olo.norm(res) + lmbd * olo.norm(olo.D(x))
    jac = (AH * x + olo.H(x) + olo.HT(x-g) + (x - g))
    return min, jac

x0 = np.zeros((Nt, Ne), dtype=dtype).ravel()
fest = minimize(fun, x0, method=mthd, jac=True, options=opts).x.reshape(Nt, Ne)
# fest = la.minres(AH, olo.HT(crs), x0=fest.ravel(), maxiter=200, rtol=1e-20, show=False)[0].reshape(Nt, Ne)
print(olo.norm(fest - f))

# %%####################################################################################################################
fest = fest.reshape((Nt, Ne))
print_stats(fest)
show_el(fest)
# %%
# Fista
# single valued lambda
AH = la.LinearOperator((Nt*Ne, Nt*Ne), matvec=lambda x: olo.H(x), rmatvec=lambda x: olo.HT(x))

l2 = pyproximal.L2(Op=pylops.LinearOperator(AH), b=crs.ravel())
l1 = pyproximal.L1(sigma=lambda x: olo.normalized_energy(x))
# l1 = pyproximal.L1(sigma=lambda x:  1/(olo.score_teng(x) + 1e-150))
lmbd = 1e-5
#lmbd = lmbd_spc[np.argmin(norm_res)]
fest = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=np.zeros(Nt*Ne), epsg=lmbd, niter=120, show=True, acceleration='fista', nonneg=False).reshape(Nt, Ne)

# %%
from joblib import Parallel, delayed
lmbd_spc = np.logspace(-10, 1, 50)
mse_lmbd = np.zeros(len(lmbd_spc))
norm_res = np.zeros(len(lmbd_spc))
norm_fest = np.zeros(len(lmbd_spc))
norm_f = np.zeros(len(lmbd_spc))

mthd = "CG"
opts = {'maxiter': 200, 'disp': False}

for i in tqdm(range(len(lmbd_spc))):
    def fista_col(ii):
        AH = la.LinearOperator((Nt, Nt),
                               matvec=lambda x: olo.HoT(olo.Ho(x, ii), ii) + olo.HoT(olo.B(x, ii), ii),
                               rmatvec=lambda x: olo.HoT(olo.Ho(x, ii), ii) + olo.BT(olo.Ho(x, ii), ii))

        l2 = pyproximal.L2(Op=pylops.LinearOperator(AH), b=olo.HoT(crs.ravel(), ii))
        l1 = pyproximal.L1(sigma=lambda x: olo.normalized_energy(x))
        # l1 = pyproximal.L1(sigma=lambda x:  1/(olo.score_teng(x) + 1e-150))
        lmbd = lmbd_spc[i]
        # lmbd = lmbd_spc[np.argmin(norm_res)]
        fest = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=np.zeros(Nt), epsg=lmbd, niter=500,
                                                               show=True, acceleration='fista', nonneg=False)

        return fest


    fest = Parallel(n_jobs=-1)(delayed(fista_col)(i) for i in range(Ne))
    fh = np.array(fest).T

    mse_lmbd[i] = np.mean(f - fh)**2
    norm_res[i] = olo.norm(fh - f)
    norm_fest[i] = olo.norm(fh)
    norm_f[i] = olo.norm(f)
# %%
def fista_col(ii):
    AH = la.LinearOperator((Nt, Nt),
                           matvec=lambda x: olo.HoT(olo.Ho(x, ii), ii)+ olo.HoT(olo.B(x, ii), ii),
                           rmatvec=lambda x: olo.HoT(olo.Ho(x, ii), ii) + olo.BT(olo.Ho(x, ii), ii))

    l2 = pyproximal.L2(Op=pylops.LinearOperator(AH), b=olo.HoT(crs.ravel(), ii))
    l1 = pyproximal.L1(sigma=lambda x: olo.normalized_energy(x))
    # l1 = pyproximal.L1(sigma=lambda x:  1/(olo.score_teng(x) + 1e-150))
    lmbd = lmbd_spc[np.argmin(norm_res)]
    # lmbd = lmbd_spc[np.argmin(norm_res)]
    fest = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=np.zeros(Nt), epsg=lmbd, niter=500,
                                                           show=True, acceleration='fista', nonneg=False)

    return fest
fest = Parallel(n_jobs=-1)(delayed(fista_col)(i) for i in range(Ne))
fest = np.array(fest).T
print(np.mean(f - fh)**2)
# %%

mthd = "L-BFGS-B"
opts = {'maxiter': 500, 'disp': False}
hfista = np.zeros((Nh, Nc))
crs_real = g.ravel() - f.ravel()
lmbd = 1
bounds = [(-1, 1)]
def B(x, col):
    x_ = np.zeros((Nt, Ne), dtype=dtype)
    x_[:, col] = x
    x_ = x_.ravel()
    return x_

def BT(x, col):
    return x.reshape(Nt, Ne)[:, col]

def L2_col(ii):
    AH = la.LinearOperator((Nt, Nt),
                           matvec=lambda x: olo.HoT(olo.Ho(x, ii), ii)+ olo.HoT(olo.B(x, ii), ii),
                           rmatvec=lambda x: olo.HoT(olo.Ho(x, ii), ii) + olo.BT(olo.Ho(x, ii), ii))

    WH = la.LinearOperator((Nt * Ne, Nt),
                           matvec=lambda x: lmbd*olo.D(olo.B(x, ii)),
                           rmatvec=lambda x: lmbd*olo.BT(olo.DT(x), ii))
    def fun(x):
        g_ = g - B(x, ii)

        if (g_ == g).all():
            jac = 2 * (AH.T * (AH * x) + WH.T * (WH * x) - AH.T * olo.HoT(g.ravel(), ii))
            return np.inf, jac
        else:
            gw = g
            gw[g_==g] = 0

        jac = 2 * (AH.T * (AH * x) + WH.T * (WH * x) - AH.T * olo.HoT(g_.ravel(), ii))
        res = AH*x - olo.HoT(g_, ii)
        min = olo.norm(res)**2 + lmbd * olo.norm((WH*x)) #+ 1/(olo.norm(res)+1e-150)
        return min, jac

    x0 = np.zeros((Nt), dtype=dtype).ravel()
    fest = minimize(fun, x0, method=mthd, jac=True, options=opts, bounds=bounds).x.reshape(Nt)
    return fest
fest = Parallel(n_jobs=-1)(delayed(L2_col)(i) for i in range(Ne))
fest = np.array(fest).T
print(olo.norm(fest.ravel() - (f.ravel())))
# %%
AH = la.LinearOperator((Nt * Ne, Nt), matvec=lambda x: olo.Ho(x, 1), rmatvec=lambda x: olo.HoT(x, 1))
tst = AH @ f.reshape(Nt ,Ne)[:, 1]
plt.plot(tst)

# %%
def s_colT(x, col):
    h_ =  np.zeros((Nt))
    h_[(col)] = 1
    x_ = ss.correlate(x, h_.ravel(), mode='full')
    return x_
