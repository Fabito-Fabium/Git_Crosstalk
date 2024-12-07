# %%
import matplotlib
matplotlib.use('TkAgg')

# %%
import pyproximal
from time import time
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
# %%
# Definindo os parâmetros do problema
dtype = np.float32

Fs = 50e6
Fc = 5e6
Nh = 8
Nt = 1875
Ne = 15

mySys = synth_data(Fs, Fc, Nh, Nt, Ne)
f, h, idx, g = mySys.create_synth()
olo = ownlinOp(Fs, Fc, Nh, Nt, Ne, f, h, idx, remez=True)
g = olo.apply_SNR(g, 40)

Nc = len(idx)

# %%
print(f'norm_vec f: {olo.norm_vec(f)}, norm_vec g: {olo.norm_vec(g)}')
plt.close('all')
fig1, (ax1, ax2)= plt.subplots(ncols=2, sharex = True, sharey = False)
ax1.imshow(f, interpolation ='nearest', aspect = 'auto')
ax1.set_title(f'b-scan sem crosstalk')
ax2.imshow((f+g.reshape(Nt, Ne)).reshape(Nt, Ne), interpolation ='nearest', aspect = 'auto')
ax2.set_title(f'b-scan com crosstalk')

# for (j,i),label in np.ndenumerate(normalized_energy(f)):
#     ax1.text(i,j,label,ha='center',va='center')
#
# for (j,i),label in np.ndenumerate(normalized_energy(g+f)):
#     ax2.text(i,j,label,ha='center',va='center')

# %%
# Estimando o f
# single valued lambda
lmbd = 0.008858
#lmbd = lmbd_spc[np.argmin(norm_res)]
AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: olo.HT(olo.H(x, h), h) + lmbd * olo.DT(olo.D(x)))
fest = la.minres(AH, olo.HT(g, h), x0=np.zeros(Nt*Ne), maxiter=200, rtol=1e-20, show=False)[0].reshape((Nt, Ne), )  # 9 s

print("norm diff: {}".format(olo.norm(fest - f)))

plt.figure()
plt.plot(f[:, 1])
plt.plot(fest[:, 1])
plt.title(f'norm fest - f: {olo.norm(fest - f)}')

fig, axs = plt.subplots(nrows=2)
axs[0].plot(f[:, 3])
axs[0].set_title(f"f el 1, norm f1 - fest1={olo.norm(fest[:, 2]-f[:, 2])}")
axs[1].plot(fest[:, 3])
axs[1].set_title(f"fest el 1")

# %%
# Estimando o h
# single valued lambda
AH = la.LinearOperator((Nt*Ne, Nt*Ne), matvec=lambda x: olo.H(x), rmatvec=lambda x: olo.HT(x))

l2 = pyproximal.L2(Op=pylops.LinearOperator(AH), b=g.ravel())
#l1 = pyproximal.L1(sigma=lambda x: normalized_energy(x))
l1 = pyproximal.L1(sigma=lambda x:  1/(olo.score_teng(x) + 1e-5))
lmbd = 0.02
#lmbd = lmbd_spc[np.argmin(norm_res)]
t0 = time()
fest = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=np.zeros(Nt*Ne), epsg=lmbd, niter=120, show=True, acceleration='fista', nonneg=False).reshape(Nt, Ne)
t0 = time() - t0

# %%
#fest = D(fest).reshape(Nt, Ne)
plt.figure()
plt.plot(f[:, 1])
plt.plot(fest[:, 1])
plt.title(f'norm fest - f: {olo.norm(fest - f)}')

fig, axs = plt.subplots(nrows=2)
axs[0].plot(f[:, 3])
axs[0].set_title(f"f el 1, norm f1 - fest1={olo.norm(fest[:, 2]-f[:, 2])}")
axs[1].plot(fest[:, 3])
axs[1].set_title(f"fest el 1")

# %%
print(f"||fh-f||^2: \t\t\t {norm(fest - f)}")
print(f"||w||^2: \t\t\t\t {norm(w)}")
print(f"||H(fh) - g_clean||^2: \t {norm(H(fest, h) - g_clean)}")
print(f"||H(fh) - g||^2: \t\t {norm(H(fest, h) - g.ravel())}")

# %%
# Estimando o f
# multi-valued lambda
lmbd_spc = np.logspace(-10, -3, 50)
mse_lmbd = np.zeros(len(lmbd_spc))
norm_res = np.zeros(len(lmbd_spc))
norm_fest = np.zeros(len(lmbd_spc))
norm_f = np.zeros(len(lmbd_spc))

for i in tqdm(range(len(lmbd_spc))):
    AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: H(x), rmatvec=lambda x: HT(x))

    l2 = pyproximal.L2(Op=pylops.LinearOperator(AH), b=g.ravel())
    #l1 = pyproximal.L1(sigma=lambda x: normalized_energy(x).ravel())
    l1 = pyproximal.L1(sigma=lambda x:  1/(score_teng(x)+1e-5))
    fest = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=np.zeros(Nt * Ne), epsg=lmbd_spc[i], niter=120,
                                                           show=False, acceleration='fista', nonneg=False).reshape(Nt, Ne)
    # AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: HT(H(x, h), h) + lmbd_spc[i]*DT(D(x)))# rmatvec=lambda x: HT(H(x, h), h) + lmbd[i]*D(D(x)))
    # fest = la.minres(AH, HT(g, h), x0=np.zeros(Nt*Ne), maxiter=200, rtol=1e-20, show=False)[0].reshape((Nt, Ne), )
    mse_lmbd[i] = np.mean(f - fest)**2
    norm_res[i] = norm(fest - f)
    norm_fest[i] = norm(fest)
    norm_f[i] = norm(f)

# %%
plt.close('all')
plt.loglog(lmbd_spc, norm_res)
plt.xlabel("$\lambda$")
plt.ylabel("$||f-f_{est}||_2$")
plt.title(f"Lambda do subproblema Hf = g - f, melhor lmbd: {lmbd_spc[np.argmin(norm_res)]}")
