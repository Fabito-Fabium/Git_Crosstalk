import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
matplotlib.use('TkAgg')
import scipy.signal as ss
from tqdm import tqdm
from joblib import Parallel, delayed

class sim_parameters:
    def __init__(self, Lx=20, Ly=20, Lt=100e-3, res=1):
        # ROI dimension
        self.Lxyt = [Lx, Ly, Lt]
        self.Dxyt = np.array([50*10**(-3*res), 50*10**(-3*res), 100*10**(-6*res)])
        self.Nxyt = np.array(np.round(self.Lxyt/self.Dxyt), dtype=int)
        self.speed_Sound = 340 * np.ones((int(self.Nxyt[1]), int(self.Nxyt[0])))  # [m/s]
        CFL = np.max(self.speed_Sound) ** 2 * self.Dxyt[2] ** 2 * (1 / self.Dxyt[0] ** 2 + 1 / self.Dxyt[1] ** 2)
        CFL_max = 1  # Courant stability criterion
        if CFL > CFL_max:
            raise ValueError(f"Unstable! CFL number = {CFL} > {CFL_max}")
        else:
            print(f"Stable: CFL = {CFL} < {CFL_max}")

        self.kx = np.array([[1, -2, 1]])
        self.ky = self.kx.T

    def lap(self, p):
        return ss.convolve(p, self.kx, mode='same') / self.Dxyt[0] ** 2 + ss.convolve(p, self.ky, mode='same') / self.Dxyt[1] ** 2

    def wave(self, pxy=[2, 2], show=False):
        fc = 440  # [Hz]
        t0 = 5 / fc
        t = np.arange(self.Nxyt[2]) * self.Dxyt[2]
        s = ss.gausspulse(t - t0, fc)

        nxySource = np.round(self.Nxyt[:-1]/pxy)

        nxLocus = np.arange(self.Nxyt[0], dtype=int)
        nyLocus = np.zeros(self.Nxyt[0], dtype=int)
        print(int(self.Nxyt[2]), len(nxLocus))
        sensorRead = np.zeros((self.Nxyt[2], len(nxLocus)))
        pressureField = np.zeros((self.Nxyt[::-1]))
        if show:
            plt.close('all')
            # Visualization of pressure field
            fig, ax = plt.subplots(ncols=2)
            vmaxmin = .1
            im0 = ax[0].imshow(pressureField[0], vmin=-vmaxmin, vmax=vmaxmin, extent=[0, self.Lxyt[0], self.Lxyt[1], 0])
            im1 = ax[1].imshow(sensorRead, aspect='auto', vmin=-vmaxmin, vmax=vmaxmin, extent=[0, self.Lxyt[0], self.Lxyt[2], 0])

            ax[0].set(xlabel='x', ylabel='y')
            ax[1].set(xlabel='x', ylabel='t')

            print(self.speed_Sound.shape)
            for it in tqdm(range(2, self.Nxyt[2])):
                pressureField[it] = (2 * pressureField[it - 1] - pressureField[it - 2] + self.Dxyt[2] ** 2 * self.speed_Sound ** 2 * self.lap(pressureField[it - 1]))
                pressureField[it, int(nxySource[1]), int(nxySource[0])] += s[it]
                sensorRead[it] = pressureField[it, nyLocus, nxLocus]
                im0.set_data(pressureField[it])
                im1.set_data(sensorRead)
                fig.suptitle(f"Simulation time = {it * self.Dxyt[2]: .3f}")
                ax[0].set_title("Pressure field.")
                plt.title("Sensor readings")
                plt.pause(1e-12)

        else:
            for it in tqdm(range(2, self.Nxyt[2])):
                pressureField[it] = (2 * pressureField[it - 1] - pressureField[it - 2] + self.Dxyt[
                    2] ** 2 * self.speed_Sound ** 2 * self.lap(pressureField[it - 1]))
                pressureField[it, int(nxySource[1]), int(nxySource[0])] += s[it]
                sensorRead[it] = pressureField[it, nyLocus, nxLocus]

        return sensorRead

dude = sim_parameters(20, 20, res=1)
tst = dude.wave(show=False, pxy=[3])
plt.imshow(tst[:, 64:128], aspect='auto', interpolation='nearest')
plt.pause(1)
# %%
plt.close('all')
# Visualization of pressure field
fig, ax = plt.subplots(ncols=2)
vmaxmin = .1
im0 = ax[0].imshow(p[0], vmin=-vmaxmin, vmax=vmaxmin, extent=[0, Lx, Ly, 0])
im1 = ax[1].imshow(m, aspect='auto', vmin=-vmaxmin, vmax=vmaxmin,extent=[0, Lx, Lt, 0])

ax[0].set(xlabel='x', ylabel='y')
ax[1].set(xlabel='x', ylabel='t')

for it in tqdm(range(2, Nt)):
    p[it] = 2*p[it-1] - p[it-2] + Dt**2 * c**2 * lap(p[it - 1])
    p[it, nys, nxs] += s[it]
    m[it] = p[it, nym, nxm]
    im0.set_data(p[it])
    im1.set_data(m)
    fig.suptitle(f"Simulation time = {it * Dt: .3f}")
    ax[0].set_title("Pressure field.")
    plt.title("Sensor readings")
    plt.pause(1e-12)

#%%
plt.figure()
plt.imshow(m, extent=[0, Lx, Lt, 0], aspect='auto')
plt.title("Sensor readings")
plt.xlabel('x')
plt.ylabel('t')

plt.show()

# %% solution with FISTA
import pylops
import pyproximal
import scipy.sparse.linalg as la

Nh = 2*Nt

def norm(x):
    return np.sum(x ** 2)

def P(m):
    for it in range(2, Nt):
        p[it] = 2 * p[it - 1] - p[it - 2] + Dt ** 2 * c ** 2 * lap(p[it - 1])
        p[it, nys, nxs] += s[it]
        m[it] = p[it, nym, nxm]
    return m

m = P(m)
# Let h[k] be a signal composed by impulses and S be a matrix s.t. m[k] = s*h[k] = Sh[k]
hk = np.random.randn(Nh*Nx)
def sConv(s, h):
    y = ss.convolve(s, h, mode='same')
    return y

def S(h_, s=s):
    h = h_.reshape(Nh, Nx)
    b = np.zeros_like(m)
    for i in range(Nx):
        b[:, i] = sConv(s, h[:, i])
    return b.ravel()
# From here, since we will use a linear operator, we are obligated to deduce the expression of the transpose of S,
# for that, we just need to find the matrix form of S as follows
# Smtx = np.zeros((Nt*Nx, Nh*Nx))
# hh = np.zeros(Nh*Nx)
# for i in tqdm(range(Nh*Nx)):
#     hh[i] = 1
#     Smtx[:, i] = S(hh).ravel()
#     hh[i] = 0

# Since the correlation function of scipy.sparse introduces an unwanted delay, we need to create a window so that
# the norm of Smtx.T @ (signal) - ST(signal).ravel() is close to zero

def sCorr(s, hk):
    y = ss.correlate(hk, s, mode='full')
    return y

def ST(h_, s=s):
    h = h_.reshape(Nt, Nx)
    b = np.zeros((Nh, Nx))
    for i in range(Nx):
        b[:-1, i] = sCorr(s, h[:, i])
    return b

# plt.plot(Smtx.T @ m.ravel(), label='mtx')
# plt.plot(ST(m).ravel())
# plt.legend()

# with S and ST, we just need to formulate the problem to fit into the FISTA algorithm
# %%
AF = la.LinearOperator((Nt*Nx, Nh*Nx), matvec=lambda x: S(x), rmatvec=lambda x: ST(x))
l1 = pyproximal.L1()
l2 = pyproximal.L2(Op=pylops.LinearOperator(AF), b=m.ravel())

# lmbd = lmbd_spc[np.argmin(norm_res)]
hest = pyproximal.optimization.primal.ProximalGradient(l2, l1, x0=np.zeros(Nh*Nx), niter=200, epsg=0.1, show=True,
                                                       acceleration='fista', nonneg=False)

# %%
fig, ax = plt.subplots(ncols=2)
im0 = ax[0].imshow(m, extent=[0, Lx, Lt, 0], aspect='auto')
im1 = ax[1].imshow(S(hest).reshape(m.shape), extent=[0, Lx, Lt, 0], aspect='auto')
ax[0].set_title("real")
ax[1].set_title("estimated")

print(f'norm diff between m and mest: {norm(S(hest).ravel() - m.ravel())}')
# %%
plt.figure()
plt.imshow(hest.reshape(Nh, Nx).T, aspect='auto', interpolation='nearest')