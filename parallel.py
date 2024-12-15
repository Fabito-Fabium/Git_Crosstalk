from joblib import Parallel, delayed
import numpy as np
from time import sleep

def tst(i):
    return i, i**2

output = Parallel(n_jobs=-1)(delayed(tst)(i) for i in range(10))