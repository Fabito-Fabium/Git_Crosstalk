# %%
import matplotlib
matplotlib.use('TkAgg')
import pyproximal
from time import time

import cv2
import scipy.signal as ss
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy import signal
from matplotlib import animation

import pyproximal
import pylops.linearoperator as pl
import pylops
from scipy import misc
from tqdm import tqdm
from scipy.linalg.interpolative import estimate_spectral_norm as spectral_norm
import scipy.linalg as sl

# %%
# Definindo os parâmetros do problema
dtype = np.float32
H_gain = np.load(r'/home/lilpc/Documents/Lassip/Crosstalk/main_new/matrix_gain.npy')
H_gain[58, 1:] = H_gain[58, :-1]
H_gain_new = np.zeros_like(H_gain)

for i in range(32):
    if i == 0:
        H_gain_new[i] = H_gain[i]
        H_gain_new[63-i] = H_gain[63-i, ::-1]
    else:
        H_gain_new[i, :-i] = H_gain[i, i:]
        H_gain_new[63-i, i:] = H_gain[63-i, :-i]
        H_gain_new[63-i] = H_gain_new[63-i, ::-1]
plt.imshow(H_gain_new)
summed_H_gain = np.sum(H_gain_new, axis=0)

for i in range(64):
    summed_H_gain[i] = summed_H_gain[i]/np.count_nonzero(H_gain_new[:,i])

for i in range(63):
    if summed_H_gain[i] < summed_H_gain[i+1]:
        summed_H_gain[i+1] = summed_H_gain[i] - 2*summed_H_gain[i]/1000

plt.figure()
summed_H_gain = summed_H_gain/np.max(summed_H_gain)

plt.plot(summed_H_gain, color='r')
# %%


Fs = 125e6
Fc = 5e6
Nh = 8
Nt = 1875
t = np.arange(Nt, dtype=dtype)/Fs
Ne = 15
bw = .75
Tp = 75


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


#plt.imshow(f, aspect='auto', interpolation='nearest')
# plt.plot(pulse)

# Simulação do problema (seção referente ao h)
idx = list(combinations(range(Ne), 2))
# idx = [(i, j) for i, j in combinations(range(Ne), 2) if abs(i-j) < 30]

Nc = len(idx)
h = np.zeros((Nh, Nc))
for i in range(Nc):
    h[:, i] = np.zeros(Nh)
    d = idx[i][0] - idx[i][1]
    h[min(Nh-1, np.abs(d)-1), i] = summed_H_gain[np.abs(d)]

plt.figure()
plt.imshow(h.T, aspect='auto', interpolation='nearest', cmap='Greys')
plt.colorbar()
# %%
# plt.imshow(h, aspect='auto', interpolation='nearest')

# Definição das operações/operadores:

band = [5e6*0.1, 5e6*1.99]  # Desired stop band, Hz
trans_width = 3e5    # Width of transition from pass to stop, Hz
numtaps = 555        # Size of the FIR filter.
edges = [0, band[0] - trans_width, band[0], band[1],
         band[1] + trans_width, 0.5*Fs]
b = signal.remez(numtaps, edges, [1, 0, 1], fs=Fs)
# ww, hh = signal.freqz(b, [1], worN=2000, fs=Fs)
# plot_response(ww, hh, "Band-stop Filter")


def D(x):
    x_ = x.reshape((Nt, Ne), )
    y = np.zeros((Nt, Ne), dtype=dtype)
    for i in range(Ne):
        #y[:, i] = ss.convolve(x_[:, i], pulse[:, i], mode="same", method="fft")
        y[:, i] = ss.convolve(x_[:, i], b, mode="same", method="fft")
    return y.ravel()

# # plt.plot(D(D(fest)).reshape(Nt, Ne)[:, 5])
# print('Dmtx')
# Dmtx = np.zeros((Nt*Ne, Nt*Ne))
# hh = np.zeros(Nt*Ne)
# for i in tqdm(range(Nt*Ne)):
#     hh[i] = 1
#     Dmtx[:, i] = D(hh).ravel()
#     hh[i] = 0
# %%


def DT(x):
    x_ = x.reshape((Nt, Ne), )
    y = np.zeros((Nt, Ne), dtype=dtype)
    for i in range(Ne):
        #y[:-1, i] = ss.correlate(x_[:, i], pulse[:, i], mode="same", method="fft")[1:]
        y[:, i] = ss.convolve(x_[:, i], b, mode="same", method="fft")
    return y.ravel()


# plt.plot(DT(w), label='func')
# plt.plot(Dmtx.T @ w.ravel(), label='Dmtx')
# plt.legend()
# plt.figure()
# plt.plot(DT(w) - Dmtx.T @ w.ravel())
# print(norm(DT(w) - Dmtx.T @ w.ravel()))

# %%
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

# %%
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


def norm(x):
    return np.sum(x**2)


# print(norm(((Hmtx.T @ g.ravel()) - HT(g))))
# plt.plot(Hmtx.T @ g.ravel())
# plt.plot(HT(g))
# plt.figure()
# plt.imshow((Hmtx.T @ g.ravel() - HT(g)).reshape(Nt, Ne), aspect='auto', interpolation='nearest')

# %%
def norm_vec(x):
    norm_vec = np.zeros(Ne, dtype=dtype)
    x_ = (x/np.sqrt(np.sum(x**2))).reshape(Nt, Ne)
    for i in range(Ne):
        norm_vec[i] = norm(x_[:, i]**2)

    return norm(norm_vec)

def normalized_energy(x):
    return x/np.sqrt(np.sum(x**2)+1e-20)

def score_teng(img):
    """Implements the Tenengrad (TENG) focus measure operator.
    Based on the gradient of the image.

    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    return numpy.mean(gaussianX * gaussianX +
                      gaussianY * gaussianY)

# %%
# Definindo "g"
SNR = 40  # dB

g_clean = H(f, h)
# g_clean = g.ravel()
s2g = np.mean(g_clean**2)
s2w = s2g * 10**(-SNR/10)

w = np.sqrt(s2w) * np.random.randn(Nt*Ne)

SNR_calc = 10*np.log10(np.mean(g_clean**2) / np.mean(w**2))

g = g_clean + w
g = g.reshape(Nt, Ne, )

# %%

print(f'norm_vec f: {norm_vec(f)}, norm_vec g: {norm_vec(g)}')
plt.close('all')
fig1, (ax1, ax2)= plt.subplots(ncols=2, sharex = True, sharey = False)
ax1.imshow(20*np.log(abs(f)+1), interpolation ='nearest', aspect = 'auto')
ax1.set_title(f'b-scan sem crosstalk')
ax2.imshow(20*np.log(abs((f+g.reshape(Nt, Ne)).reshape(Nt, Ne)) + 1), interpolation ='nearest', aspect = 'auto')
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
AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: HT(H(x, h), h) + lmbd * DT(D(x)))
fest = la.minres(AH, HT(g, h), x0=np.zeros(Nt*Ne), maxiter=200, rtol=1e-20, show=False)[0].reshape((Nt, Ne), )  # 9 s

50

# %%
# Estimando o h
# single valued lambda
AH = la.LinearOperator((Nt*Ne, Nt*Ne), matvec=lambda x: H(x), rmatvec=lambda x: HT(x))

l2 = pyproximal.L2(Op=pylops.LinearOperator(AH), b=g.ravel())
l1 = pyproximal.L1(sigma=lambda x: normalized_energy(x).ravel())
lmbd = lmbd_spc[np.argmin(norm_res)]
#lmbd = 0.005
t0 = time()
fest = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=np.zeros(Nt*Ne), epsg=lmbd, niter=120, show=True, acceleration='fista', nonneg=False).reshape(Nt, Ne)
t0 = time() - t0

# %%
#fest = D(fest).reshape(Nt, Ne)
plt.figure()
plt.plot(f[:, 1])
plt.plot(fest[:, 1])
plt.title(f'norm fest - f: {norm(fest[:] - f[:])}')

fig, axs = plt.subplots(nrows=2)
axs[0].plot(f[:, 0])
axs[0].set_title(f"f el 1, norm f1 - fest1={norm(fest[:, 2]-f[:, 2])}")
axs[1].plot(fest[:, 0])
axs[1].set_title(f"fest el 1")

    # %%
print(f"||fh-f||^2: \t\t\t {norm(fest - f)}")
print(f"||w||^2: \t\t\t\t {norm(w)}")
print(f"||H(fh) - g_clean||^2: \t {norm(H(fest, h) - g_clean)}")
print(f"||H(fh) - g||^2: \t\t {norm(H(fest, h) - g.ravel())}")

# %%
# Estimando o f
# multi-valued lambda
lmbd_spc = np.logspace(-8, 2, 50)
mse_lmbd = np.zeros(len(lmbd_spc))
norm_res = np.zeros(len(lmbd_spc))
norm_fest = np.zeros(len(lmbd_spc))
norm_f = np.zeros(len(lmbd_spc))

for i in tqdm(range(len(lmbd_spc))):
    # AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: H(x), rmatvec=lambda x: HT(x))
    #
    # l2 = pyproximal.L2(Op=pylops.LinearOperator(AH), b=g.ravel())
    # l1 = pyproximal.L1(sigma=lambda x: normalized_energy(x).ravel())
    #
    # fest = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=np.zeros(Nt * Ne), epsg=lmbd_spc[i], niter=120,
    #                                                        show=False, acceleration='fista', nonneg=False).reshape(Nt, Ne)
    AH = la.LinearOperator((Nt * Ne, Nt * Ne), matvec=lambda x: HT(H(x, h), h) + lmbd_spc[i]*DT(D(x)))# rmatvec=lambda x: HT(H(x, h), h) + lmbd[i]*D(D(x)))
    fest = la.minres(AH, HT(g, h), x0=np.zeros(Nt*Ne), maxiter=200, rtol=1e-20, show=False)[0].reshape((Nt, Ne), )
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
