import numpy as np
import scipy.signal as ss
from itertools import combinations
import matplotlib.pyplot as plt
from scipy import signal
from source.subProb_linOp import ownlinOp
# %%
# Definindo os parâmetros do problema
dtype = np.float32

class synth_data:
    def __init__(self, Fs, Fc, Nh, Nt, Ne):
        self.Fs = Fs
        self.Fc = Fc
        self.Nh = Nh
        self.Nt = Nt
        self.Ne = Ne

    def _get_(self):
        Fs = self.Fs
        Fc = self.Fc
        Nh = self.Nh
        Nt = self.Nt
        Ne = self.Ne
        return Fs, Fc, Nh, Nt, Ne

    def create_synth(self):
        Fs, Fc, Nh, Nt, Ne = self._get_()

        t = np.arange(Nt, dtype=dtype)/Fs

        # Simulação do problema (seção referente ao f)
        f = np.zeros((Nt, Ne), dtype=dtype)

        for ne in range(Ne):
            #f[:, ne] = ss.gausspulse((t - 15*ne/Fs)/1, Fc)
            f[:, ne] = ss.gausspulse((t - (ne/Ne)*Nt/Fs), Fc)

        # plt.imshow(f, aspect='auto', interpolation='nearest')

        # Simulação do problema (seção referente ao h)
        # idx = list(combinations(range(Ne), 2))
        idx = [(i, j) for i, j in combinations(range(Ne), 2) if abs(i-j) < 30]

        Nc = len(idx)
        h = np.zeros((Nh, Nc))
        for i in range(Nc):
            h[:, i] = np.zeros(Nh)
            d = idx[i][0] - idx[i][1]
            h[min(Nh-1, np.abs(d)-1), i] = np.exp(-d**2/10)

        # plt.figure()
        # plt.imshow(h.T, aspect='auto', interpolation='nearest', cmap='Greys')
        # plt.colorbar()

        # plt.imshow(h, aspect='auto', interpolation='nearest')

        mylin = ownlinOp(Fs, Fc, Nh, Nt, Ne, f, h, idx)

        g = mylin.F(h)

        return f, h, idx, g

    def create_pulse(self):
        Fs, Fc, Nh, Nt, Ne = self._get_()
        t = np.arange(Nt, dtype=dtype) / Fs
        bw = .75

        f = np.zeros((Nt, Ne), dtype=dtype)
        pulse = np.zeros((Nt, Ne), dtype=dtype)

        for ne in range(Ne - 1):
            # tau = t - 15*ne/Fs
            tau = (t - (ne / Ne + 1 / Ne) * Nt / Fs)
            f[:, ne] = ss.gausspulse(tau, Fc, bw)

            F = np.fft.fft(f[:, ne])
            # Pulse = np.zeros_like(F) + np.max(np.abs(F))
            # Pulse -= np.abs(F)
            # pulse[:, ne] = np.fft.ifft(Pulse)

            t0 = np.abs(tau).argmin()
            pulse[t0, ne] = np.max(np.abs(F))
            pulse[:, ne] -= f[:, ne]

        return pulse

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
