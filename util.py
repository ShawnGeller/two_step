import numpy as np
import scipy
import scipy.special

def clopper_upper(successes, trials, alpha):
    x, n, a = np.broadcast_arrays(successes, trials, alpha)
    retval = np.ones_like(n, dtype=float)
    mask = x != n
    retval[mask] = scipy.special.betaincinv(x[mask]+1, n[mask]-x[mask], 1-a[mask])
    return retval

def clopper_lower(successes, trials, alpha):
    x, n, a = np.broadcast_arrays(successes, trials, alpha)
    retval = np.zeros_like(n, dtype=float)
    mask = x != 0
    retval[mask] = scipy.special.betaincinv(x[mask], n[mask]-x[mask]+1, a[mask])
    return retval
