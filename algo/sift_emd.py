# %%
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
########################################################################################################################
# %%
import pyproximal
from time import time
import cv2
import scipy.signal as ss
import numpy as np
from itertools import combinations
import scipy.sparse.linalg as la
from scipy import signal
from matplotlib import animation

import pyproximal
import pylops.linearoperator as pl
import pylops
from tqdm import tqdm

from source.synth_data import synth_data
from source.subProb_linOp import ownlinOp
import random


from scipy.optimize import minimize
from scipy.optimize import Bounds

random.seed(42)
# %% ###################################################################################################################
# Definindo os parâmetros do problema ##################################################################################
dtype = np.float32

Fs = 125e6
Fc = 5e6
Nh = 8
Nt = 1875
Ne = 10

mySys = synth_data(Fs, Fc, Nh, Nt, Ne, vmax=1, bw=0.75, sim_dt=False)
f, h, idx, crs = mySys.create_synth()
olo = ownlinOp(Fs, Fc, Nh, Nt, Ne, f, h, idx, remez=False, filt=mySys.get_pulse())

g_clean = crs + f.ravel()
crs = olo.apply_SNR(crs, 40)[0]
g, w_g = olo.apply_SNR(g_clean, 40)


Nc = len(idx)
# %% ###################################################################################################################
import emd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

x = g.reshape(Nt, Ne).ravel('f')

plt.figure()
plt.plot(x, 'k')
peak_locs, peak_mags = emd.sift.get_padded_extrema(x, pad_width=0, mode='peaks')
trough_locs, trough_mags = emd.sift.get_padded_extrema(x, pad_width=0, mode='troughs')

plt.figure()
plt.plot(x, 'k')
plt.plot(peak_locs, peak_mags, 'o')
plt.plot(trough_locs, trough_mags, 'o')
plt.xlim(0, Nt)
proto_imf = x.copy()
# Compute upper and lower envelopes
upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')

# Compute average envelope
avg_env = (upper_env+lower_env) / 2

plt.figure()
plt.plot(x, 'k')
plt.plot(upper_env)
plt.plot(lower_env)
plt.plot(avg_env)
plt.xlim(0, Nt)
plt.legend(['Signal', 'Upper Env', 'Lower Env', 'Avg Env'])

# Subtract slow dynamics from previous cell
proto_imf = x - avg_env

# Compute upper and lower envelopes
upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')

# Compute average envelope
avg_env = (upper_env+lower_env) / 2

plt.figure()
plt.subplot(211)
plt.plot(x, 'k')
plt.xlim(0, Nt)
plt.legend(['Original Signal'])

plt.subplot(212)
plt.plot(proto_imf, 'k')
plt.plot(upper_env)
plt.plot(lower_env)
plt.plot(avg_env)
plt.xlim(0, Nt)
plt.legend(['Proto IMF', 'Upper Env', 'Lower Env', 'Avg Env'])


def my_get_next_imf(x, zoom=None, sd_thresh=0.1):

    proto_imf = x.copy()  # Take a copy of the input so we don't overwrite anything
    continue_sift = True  # Define a flag indicating whether we should continue sifting
    niters = 0            # An iteration counter

    if zoom is None:
        zoom = (0, x.shape[0])

    # Main loop - we don't know how many iterations we'll need so we use a ``while`` loop
    while continue_sift:
        niters += 1  # Increment the counter

        # Compute upper and lower envelopes
        upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
        lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')

        # Compute average envelope
        avg_env = (upper_env+lower_env) / 2

        # Add a summary subplot
        plt.subplot(5, 1, niters)
        plt.plot(proto_imf[zoom[0]:zoom[1]], 'k')
        plt.plot(upper_env[zoom[0]:zoom[1]])
        plt.plot(lower_env[zoom[0]:zoom[1]])
        plt.plot(avg_env[zoom[0]:zoom[1]])

        # Should we stop sifting?
        stop, val = emd.sift.stop_imf_sd(proto_imf-avg_env, proto_imf, sd=sd_thresh)

        # Remove envelope from proto IMF
        proto_imf = proto_imf - avg_env

        # and finally, stop if we're stopping
        if stop:
            continue_sift = False

    # Return extracted IMF
    return proto_imf

plt.figure()
plt.subplots_adjust(hspace=0.5)
imf1 = my_get_next_imf(x, zoom=(100, Nt))

plt.subplot(311)
plt.plot(x)
plt.title('Signal')
plt.xticks([])

plt.subplot(312)
plt.plot(imf1)
plt.title('IMF 1')
plt.xticks([])

plt.subplot(313)
plt.plot(x - imf1)
plt.title('Signal - IMF 1')
plt.xticks([])
# sphinx_gallery_thumbnail_number = 8