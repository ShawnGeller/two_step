import numpy as np
import scipy.special
import scipy.stats
import scipy
import scipy.optimize

from util import clopper_upper, clopper_lower


def point_at_nominal(nominal_meas, nominal_prep):
    return np.array([
        [[1-nominal_prep-2*nominal_meas, nominal_prep],
         [nominal_meas, nominal_meas]],
        [[nominal_meas, nominal_meas],
         [nominal_prep, 1-nominal_prep-2*nominal_meas]]
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
    return bare_at_lower(data, alpha - 3*beta, i) - lowerb_at_upper(data, beta, c, i)


def upper_reference_point(r, c, i):
    return bare(r, i) + upperb(r, c, i)


def lower_reference_point(r, c, i):
    return bare(r, i) - lowerb(r, c, i)


def optimal_upper_beta(data, alpha, c, i):
    def fn(beta):
        return total_upper(data, alpha, beta, c, i)
    return scipy.optimize.minimize_scalar(fn, alpha/6, bounds=(0., alpha/3), method='bounded').x


def optimal_lower_beta(data, alpha, c, i):
    def fn(beta):
        return -total_lower(data, alpha, beta, c, i)
    return scipy.optimize.minimize_scalar(fn, alpha/6, bounds=(0., alpha/3), method='bounded').x


def log_width_cost(alpha_lower, alpha_upper, n0, n1, c, i, beta_lower, beta_upper):
    def fn(r):
        d = point_to_data(n0, n1, r)
        u = total_upper(d, alpha_upper, beta_upper, c, i)
        l = total_lower(d, alpha_lower, beta_lower, c, i)
        return np.log(u-l)
    return fn


def width_cost(alpha_lower, alpha_upper, n0, n1, c, i, beta_lower, beta_upper):
    def fn(r):
        d = point_to_data(n0, n1, r)
        u = total_upper(d, alpha_upper, beta_upper, c, i)
        l = total_lower(d, alpha_lower, beta_lower, c, i)
        return u-l
    return fn


def get_optimal_betas(prior, cost, n0, n1, alpha_lower, alpha_upper, c, i):
    def fn(beta_lower, beta_upper):
        return prior.expect(cost(alpha_lower, alpha_upper, n0, n1, c, i, beta_lower, beta_upper))
    return scipy.optimize.minimize(fn, np.array([alpha_lower/2, alpha_upper/2]), bounds=[(0., alpha_lower/2), (0., alpha_upper/2)]).x
