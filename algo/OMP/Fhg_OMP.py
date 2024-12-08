# %%
import matplotlib
matplotlib.use('TkAgg')
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la
from source.synth_data import synth_data
from source.subProb_linOp import ownlinOp
import source.pylops as my_pylops
from tqdm import tqdm
from time import time
import random

from matplotlib import animation

random.seed(42)
# %%
# Definindo os parâmetros do problema
dtype = np.float32

Fs = 125e6
Fc = 5e6
Nh = 8
Nt = 1875
Ne = 15

mySys = synth_data(Fs, Fc, Nh, Nt, Ne, vmax=1, bw=0.75, sim_dt=False)
f, h, idx, g = mySys.create_synth()
olo = ownlinOp(Fs, Fc, Nh, Nt, Ne, f, h, idx, remez=True, filt=mySys.get_pulse())
g = olo.apply_SNR(g, 40)

Nc = len(idx)
# %%
# minres no regularization
AF = la.LinearOperator((Nh*Nc, Nh*Nc), matvec=lambda x: olo.FT(olo.F(x)))
hh = la.minres(AF, olo.FT(g), maxiter=500, show=True, rtol=1e-15)[0].reshape(Nh, Nc)  # 9 s

# %%
homp = np.zeros((Nh, Nc))
t0 = time()
for i in tqdm(range(Nc)):
    AF = la.LinearOperator((Nt * Ne, Nh), matvec=lambda x: olo.Fo(x, i), rmatvec=lambda x: olo.FoT(x, i))
    AFp = my_pylops.LinearOperator(AF)
    homp[:, i] = my_pylops.optimization.sparsity.omp(AFp, g.ravel(), niter_outer=200, niter_inner=Ne, sigma=1e-10, normalizecols=True, nonneg=True, discard=True)[0]

t0 = time() - t0
print(olo.norm(homp - h))

# %%
plt.close("all")
fig, axs = plt.subplots(ncols=2)
real = axs[0].imshow(h.reshape(Nh, Nc).T, aspect='auto', cmap='Greys', interpolation='nearest')
axs[0].set_title(f"h real")
nonn = axs[1].imshow(homp.reshape(Nh, Nc).T, aspect='auto', cmap='Greys', interpolation='nearest')
axs[1].set_title(f"OMP 200 iter, t={t0: .4f}s, norm={olo.norm(homp-h)}")
