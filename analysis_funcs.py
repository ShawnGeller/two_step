import numpy as np
import scipy.special
import scipy.stats
import scipy
import scipy.optimize

from util import clopper_upper, clopper_lower


def point_at_nominal(nominal_meas, nominal_prep, other_meas=None, i=0):
    if other_meas is None:
        other_meas = nominal_meas
    # Column sums are 1.
    if i:
        tmp = nominal_meas
        nominal_meas = other_meas
        other_meas = tmp
    return np.array([
        [[1-nominal_prep-2*nominal_meas, nominal_prep],
         [nominal_meas, other_meas]],
        [[nominal_meas, other_meas],
         [nominal_prep, 1-nominal_prep-2*other_meas]]
    ])


def point_to_data(n0, n1, point):
    a0 = np.around(n0*point[..., 0]).astype(int)
    a0[0, 0] = n0 - np.sum(np.ravel(a0)[1:])
    a1 = np.around(n1 * point[..., 1]).astype(int)
    a1[1, 1] = n1 - np.sum(np.ravel(a1)[:-1])
    return np.stack((a0, a1), axis=-1)


def data_to_point(data):
    return np.stack([data[..., 0]/np.sum(data[..., 0]), data[..., 1]/np.sum(data[..., 1])], axis=-1)


def not_i(i):
    return (i+1)%2


def bare(point, i):
    ni = not_i(i)
    return point[ni, i, i]


def upperb(r, c, i):
    ni = not_i(i)
    return r[ni, i, i]*(r[ni, i, i] + r[ni, ni, i] + r[i, ni, ni])/c**6 + \
           r[ni, ni, i]*r[ni, i, i]**2*(r[ni, i, i]+r[i, ni, ni])/c**12


def lowerb(r, c, i):
    ni = not_i(i)
    return r[ni,ni,i]/c**3*(r[i,ni,ni]/c**3+r[ni,i,i]**2/c**6+r[ni,i,i]*r[i,ni,ni]/c**6) +\
    r[ni,i,i]**2/c**6*(r[ni,i,i]/c**3 + r[i,ni,ni]/c**3)


def upper_from_data(data, alpha):
    return clopper_upper(data, np.sum(data, axis=(0, 1)), alpha)


def lower_from_data(data, alpha):
    return clopper_lower(data, np.sum(data, axis=(0, 1)), alpha)


def upperb_at_upper(data, alpha, c, i):
    return upperb(upper_from_data(data, alpha), c, i)


def lowerb_at_upper(data, alpha, c, i):
    return lowerb(upper_from_data(data, alpha), c, i)


def bare_at_upper(data, alpha, i):
    return bare(upper_from_data(data, alpha), i)


def bare_at_lower(data, alpha, i):
    return bare(lower_from_data(data, alpha), i)


def total_upper(data, alpha, beta, c, i):
    return upperb_at_upper(data, beta, c, i) + bare_at_upper(data, alpha - 3*beta, i)


def total_lower(data, alpha, beta, c, i):
    return max(bare_at_lower(data, alpha - 3*beta, i) - lowerb_at_upper(data, beta, c, i), 0)


def upper_reference_point(r, c, i):
    return bare(r, i) + upperb(r, c, i)


def lower_reference_point(r, c, i):
    return max(bare(r, i) - lowerb(r, c, i), 0)


def optimal_upper_beta(data, alpha, c, i):
    def fn(beta):
        return total_upper(data, alpha, beta, c, i)
    return scipy.optimize.minimize_scalar(fn, alpha/6, bounds=(0., alpha/3), method='bounded').x


def optimal_lower_beta(data, alpha, c, i):
    def fn(beta):
        return -total_lower(data, alpha, beta, c, i)
    return scipy.optimize.minimize_scalar(fn, alpha/6, bounds=(0., alpha/3), method='bounded').x


def upper_loss(data, alpha, beta, c, i):
    upp_p = upper_reference_point(data_to_point(data), c, i)
    beta_opt = optimal_upper_beta(data, alpha, c, i)
    d1 = total_upper(data, alpha, beta, c, i) - upp_p
    d2 = total_upper(data, alpha, beta_opt, c, i) - upp_p
    return 0 if np.isclose(upp_p, 0) else np.abs(d1 - d2) / upp_p


def lower_loss(data, alpha, beta, c, i):
    low_p = lower_reference_point(data_to_point(data), c, i)
    beta_opt = optimal_lower_beta(data, alpha, c, i)
    d1 = total_lower(data, alpha, beta, c, i) - low_p
    d2 = total_lower(data, alpha, beta_opt, c, i) - low_p
    return 0 if np.isclose(low_p, 0) else np.abs(d1 - d2) / low_p


def exclude_0(max_range, num_points):
    return np.linspace(max_range / (num_points + 1), max_range, num=num_points, endpoint=False)


def max_loss(max_point, alpha, beta, n0, n1, c, i, num_points=10):
    ni = not_i(i)
    r110_max = max_point[ni, ni, i]
    r100_max = max_point[ni, i, i]
    r011_max = max_point[i, ni, ni]
    upp_l = 0.
    low_l = 0.
    low_max_point = max_point
    upp_max_point = max_point
    for r100 in exclude_0(r100_max, num_points):
        for r110 in exclude_0(r110_max, num_points):
            for r011 in exclude_0(r011_max, num_points):
                point = point_at_nominal(r100, r110, r011, i)
                data = point_to_data(n0, n1, point)
                upp_cand = upper_loss(data, alpha, beta, c, i)
                low_cand = lower_loss(data, alpha, beta, c, i)
                if upp_cand > upp_l:
                    upp_l = upp_cand
                    upp_max_point = point
                if low_cand > low_l:
                    low_l = low_cand
                    low_max_point = point
    return low_l, upp_l, low_max_point, upp_max_point
