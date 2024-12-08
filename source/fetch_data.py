import numpy as np

dtype = np.float32

class fetch:
    def __init__(self):
        pass

    def gain(self):
        H_gain = np.load('data/matrix_gain.npy')
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

        summed_H_gain = np.sum(H_gain_new, axis=0)

        for i in range(64):
            summed_H_gain[i] = summed_H_gain[i]/np.count_nonzero(H_gain_new[:,i])

        for i in range(63):
            if summed_H_gain[i] < summed_H_gain[i+1]:
                summed_H_gain[i+1] = summed_H_gain[i] - 2*summed_H_gain[i]/1000

        summed_H_gain = summed_H_gain/np.max(summed_H_gain)

        return summed_H_gain