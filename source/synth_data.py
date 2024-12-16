import numpy as np
import scipy.signal as ss
from itertools import combinations
import matplotlib.pyplot as plt
from scipy import signal
from source.subProb_linOp import ownlinOp
from source.fetch_data import fetch

from joblib import Parallel, delayed
# %%
# Definindo os parâmetros do problema
dtype = np.float32

class synth_data:
    def __init__(self, Fs, Fc, Nh, Nt, Ne, bw, vmax=1, sim_dt=True):
        self.Fs = Fs
        self.Fc = Fc
        self.Nh = Nh
        self.Nt = Nt
        self.Ne = Ne
        self.sim_dt = sim_dt
        self.bw = bw
        self.vmax = vmax
        self.pulse = None

    def _get_(self):
        Fs = self.Fs
        Fc = self.Fc
        Nh = self.Nh
        Nt = self.Nt
        Ne = self.Ne
        return Fs, Fc, Nh, Nt, Ne

    def create_synth(self, wave=False):
        Fs, Fc, Nh, Nt, Ne = self._get_()

        if not(self.sim_dt):
            real_gain = fetch().gain()
        else:
            real_gain = None

        t = np.arange(Nt, dtype=dtype)/Fs

        if wave == False:
            f = np.zeros((Nt, Ne), dtype=dtype)
            pulse = np.zeros((Nt, Ne), dtype=dtype)

            for ne in range(Ne - 1):
                # tau = t - 15*ne/Fs
                tau = (t - (ne / Ne + 1 / Ne) * Nt / Fs)
                f[:, ne] = self.vmax*ss.gausspulse(tau, Fc, self.bw)

                F = np.fft.fft(f[:, ne])
                # Pulse = np.zeros_like(F) + np.max(np.abs(F))
                # Pulse -= np.abs(F)
                # pulse[:, ne] = np.fft.ifft(Pulse)

                t0 = np.abs(tau).argmin()
                pulse[t0, ne] = np.max(np.abs(F))
                pulse[:, ne] -= f[:, ne]

            self.pulse = pulse
        else:


        # plt.imshow(f, aspect='auto', interpolation='nearest')
        # plt.plot(pulse)

        # Simulação do problema (seção referente ao h)
        idx = list(combinations(range(Ne), 2))
        # idx = [(i, j) for i, j in combinations(range(Ne), 2) if abs(i-j) < 30]

        Nc = len(idx)
        h = np.zeros((Nh, Nc))
        for i in range(Nc):
            h[:, i] = np.zeros(Nh)
            d = idx[i][0] - idx[i][1]
            if self.sim_dt:
                h[min(Nh-1, np.abs(d)-1), i] = np.exp(-d**2/10)
            else:
                h[min(Nh-1, np.abs(d)-1), i] = real_gain[np.abs(d)]


        mylin = ownlinOp(Fs, Fc, Nh, Nt, Ne, f, h, idx)
        g = mylin.F(h, f)

        return f, h, idx, g

    def get_pulse(self):
        return self.pulse

    def plot_response(self, w, h, title):
        "Utility function to plot response functions"
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(w, 20 * np.log10(np.abs(h)))
        ax.set_ylim(-40, 5)
        ax.grid(True)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.set_title(title)

    def show_response(self, pulse, el=0):
        Fs = self.Fs
        try:
            ff, Pulse = signal.freqz(pulse[:, 0], [1], worN=2000, fs=Fs)
        except:
            ff, Pulse = signal.freqz(pulse, [1], worN=2000, fs=Fs)

        self.plot_response(ff, Pulse, "Filter")
