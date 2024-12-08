import numpy as np
import scipy.signal as ss
import cv2
import random
from scipy import signal

dtype = np.float32
random.seed(42)


class ownlinOp:

    def __init__(self, Fs, Fc, Nh, Nt, Ne, f, h, idx, remez=True, filt=None):
        self.Fs = Fs
        self.Fc = Fc
        self.Nh = Nh
        self.Nt = Nt
        self.Ne = Ne
        self.f = f
        self.h = h
        self.idx = idx
        self.Nc = len(idx)
        self.remez = remez
        if remez:
            self.filt = self.remez_filt()
        else:
            self.filt = filt
        self.dCss = 3

    def _get_(self):
        return self.f, self.h, self.idx, self.Nt, self.Ne, self.Nc, self.Nh

    def Fo(self, x_, i, f=None):
        dCss = 3
        op, _, idx, Nt, Ne, _, _ = self._get_()

        if f is None:
            f = op

        f = f.reshape(Nt, Ne)

        y = np.zeros((Nt, Ne), dtype=dtype)

        y[dCss:, idx[i][0]] += ss.convolve(f[:, idx[i][1]], x_, mode="same")[:-dCss]
        y[dCss:, idx[i][1]] += ss.convolve(f[:, idx[i][0]], x_, mode="same")[:-dCss]

        return y.ravel()

    def F(self, x_, f=None):
        _, _, _, Nt, Ne, Nc, Nh = self._get_()

        x = x_.reshape(Nh, Nc)
        y = np.zeros((Nt, Ne), dtype=dtype)
        for i in range(Nc):
            y += self.Fo(x[:, i], i, f).reshape(Nt, Ne)
        return y.ravel()


    def FoT(self, x_, i, f=None):
        op, _, idx, Nt, Ne, _, Nh = self._get_()

        if f is None:
            f = op

        f = f.reshape(Nt, Ne)
        dCss = 3
        slcFT = slice(dCss + (Nt + 1) // 2 - (Nh + 1) // 2 + (Nt + 1) % 2,
                      -((Nt) // 2 - (Nh) // 2 - (Nt + 1) % 2) + dCss)

        x = x_.reshape(Nt, Ne)
        y = np.zeros(Nh, dtype=dtype)

        y += ss.correlate(x[:, idx[i][0]], f[:, idx[i][1]], mode="same")[slcFT]
        y += ss.correlate(x[:, idx[i][1]], f[:, idx[i][0]], mode="same")[slcFT]

        return y.ravel()

    def FT(self, x_, f=None):
        _, _, _, _, _, Nc, Nh = self._get_()

        y = np.zeros((Nh, Nc), dtype=dtype)
        for i in range(Nc):
            y[:, i] += self.FoT(x_, i, f)
        return y.ravel()

    def H(self, x_, op=None):
        try:
            h = op.reshape(self.Nh, self.Nc, )
        except AttributeError:
            h = self.h.reshape(self.Nh, self.Nc, )

        x = x_.reshape(self.Nt, self.Ne, )
        y = np.zeros((self.Nt, self.Ne))
        for i in range(self.Nc):
            y[self.dCss:, self.idx[i][0]] += ss.convolve(x[:, self.idx[i][1]], h[:, i], mode="same", method="fft")[:-self.dCss]
            y[self.dCss:, self.idx[i][1]] += ss.convolve(x[:, self.idx[i][0]], h[:, i], mode="same", method="fft")[:-self.dCss]
        return y.ravel()

    def HT(self, x_, op=None):
        try:
            h = op.reshape(self.Nh, self.Nc, )
        except AttributeError:
            h = self.h.reshape(self.Nh, self.Nc, )

        if -((self.Nh - 1) // 2) + self.dCss >= 0:
            slcH = slice(self.dCss + self.Nh // 2, self.Nt + self.dCss + self.Nh // 2)
        else:
            slcH = slice(self.dCss + self.Nh // 2, -((self.Nh - 1) // 2) + self.dCss)

        h = h.reshape(self.Nh, self.Nc, )
        x = x_.reshape(self.Nt, self.Ne, )
        y = np.zeros((self.Nt, self.Ne))
        for i in range(self.Nc):
            y[:, self.idx[i][1]] += ss.correlate(x[:, self.idx[i][0]], h[:, i], mode="full")[slcH]
            y[:, self.idx[i][0]] += ss.correlate(x[:, self.idx[i][1]], h[:, i], mode="full")[slcH]
        return y.ravel()

    def norm(self, x):
        return np.sum(x ** 2)

    def apply_SNR(self, x, SNR):
        x_shape = x.shape
        x_clean = x.ravel()

        s2x = np.mean(x_clean ** 2)
        s2w = s2x * 10 ** (-SNR / 10)

        w = np.sqrt(s2w) * np.random.randn(len(x_clean))

        SNR_calc = 10 * np.log10(np.mean(x_clean ** 2) / np.mean(w ** 2))

        print("white noise with SNR: {}dB".format(SNR_calc))

        x_SNR = x_clean + w

        return x_SNR.reshape(x_shape), w

    def score_teng(self, img):
        """Implements the Tenengrad (TENG) focus measure operator.
        Based on the gradient of the image.

        :param img: the image the measure is applied to
        :type img: numpy.ndarray
        :returns: numpy.float32 -- the degree of focus
        """
        gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        return np.mean(gaussianX * gaussianX +
                       gaussianY * gaussianY)

    def remez_filt(self):
        Fc = self.Fc
        Fs = self.Fs

        band = [Fc * 0.1, Fc * 1.99]  # Desired stop band, Hz
        trans_width = 3e5  # Width of transition from pass to stop, Hz
        numtaps = 555  # Size of the FIR filter.
        edges = [0, band[0] - trans_width, band[0], band[1],
                 band[1] + trans_width, 0.5 * Fs]
        b = signal.remez(numtaps, edges, [1, 0, 1], fs=Fs)

        return b

    def D(self, x):
        Nt = self.Nt
        Ne = self.Ne
        remez = self.remez

        x_ = x.reshape((Nt, Ne), )
        y = np.zeros((Nt, Ne), dtype=dtype)

        if remez:
            for i in range(Ne):
                y[:, i] = ss.convolve(x_[:, i], self.filt, mode="same", method="fft")
        else:
            for i in range(Ne):
                y[:, i] = ss.convolve(x_[:, i], self.filt[:, i], mode="same", method="fft")

        return y.ravel()

    def DT(self, x):
        Nt = self.Nt
        Ne = self.Ne
        remez = self.remez

        x_ = x.reshape((Nt, Ne), )
        y = np.zeros((Nt, Ne), dtype=dtype)


        if remez:
            for i in range(Ne):
                y[:, i] = ss.convolve(x_[:, i], self.filt, mode="same", method="fft")
        else:
            for i in range(Ne):
                y[:-1, i] = ss.correlate(x_[:, i], self.filt[:, i], mode="same", method="fft")[1:]

        return y.ravel()

    def norm_vec(self,x):
        Ne = self.Ne
        Nt = self.Nt

        norm_vec = np.zeros(Ne, dtype=dtype)
        x_ = (x / np.sqrt(np.sum(x ** 2))).reshape(Nt, Ne)
        for i in range(Ne):
            norm_vec[i] = self.norm(x_[:, i] ** 2)

        return self.norm(norm_vec)

    def normalized_energy(self, x):
        return x / np.sqrt(np.sum(x ** 2) + 1e-20)
