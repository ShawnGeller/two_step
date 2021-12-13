r"""An implementation of statistical analysis in the supplementary material of
[arXiv]_

This is a Python implementation of the two-step model used in the supplementary
information of the paper "High-fidelity indirect readout of trapped-ion
hyperfine qubits". We refer to that document for details.

.. [arXiv] arXiv xxxx.xxxxx
"""
import numpy as np
import scipy.special
import scipy.stats
import scipy
import scipy.optimize

from util import clopper_upper, clopper_lower


def point_at_nominal(nominal_meas, nominal_prep, other_meas=None, i=0):
    r"""From a pair of "nominal" measurement and state prep errors, makes a
    distribution described by those errors.

    For a "nominal measurement error" :math:`e_m` and a "nominal preparation
    error" :math:`e_p`, this function prepares the distribution

    .. math::

        r(\neg i, \neg i|i) &= e_p \\
        r(i, i, |\neg i) &= e_p \\
        r(\neg i, i|i) &= e_m \\
        r(i, \neg i|i) &= e_m \\
        r(i,\neg i|\neg i) &= e_m \\
        r(\neg i,i|\neg i) &= e_m \\

    and the other two components of the probability distribution such that the
    appropriate normalization constraints are satisfied. If ``other_meas`` is
    specified, :math:`r(i,\neg i|\neg i)` and :math:`r(\neg i,i|\neg i)` are
    set to that instead.

    :param float nominal_meas: Should be larger than 0, less than 1.
    :param other_meas: Defaults to ``nominal_meas``
    :return: A probability distribution, with axes such that
        ``point_at_nominal(a, b)[x, y, z] = r(x, y|z)``
    """
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
    r"""From a pair of "total number of experiments in prep 0",
    "total number of experiments in prep 1" and a point :math:`r`, returns
    counts that round to the given :math:`r`.

    The rounding is such that the total numbers :math:`N_i` are preserved.
    This is done by subtracting any discrepancy from :math:`n(i, i, i)`.

    :param n0: Total number of experiments with prep 0.
        This quantity is :math:`N_0` in the supplement.
    :param n1: Total number of experiments with prep 1.
        This quantity is :math:`N_1` in the supplement.
    :param point: The point :math:`r` to create data from.
    :return: Counts. Indexed in the same way that ``point`` is, namely::

            point_to_data(n0, n1, point)[x, y, z] = n(x, y, z)

        where ``n`` in the above refers to the :math:`n` in the supplement.
    """
    a0 = np.around(n0*point[..., 0]).astype(int)
    a0[0, 0] = n0 - np.sum(np.ravel(a0)[1:])
    a1 = np.around(n1 * point[..., 1]).astype(int)
    a1[1, 1] = n1 - np.sum(np.ravel(a1)[:-1])
    return np.stack((a0, a1), axis=-1)


def data_to_point(data):
    r"""From a set of data :math:`n`, returns the corresponding :math:`\hat{r}`.

    This implements equation S.20.

    :param data: Counts.
    :return: The frequencies corresponding to the counts.
    """
    return np.stack([data[..., 0]/np.sum(data[..., 0]), data[..., 1]/np.sum(data[..., 1])], axis=-1)


def not_i(i):
    r"""
    ``return (i+1)%2``
    """
    return (i+1)%2


def bare(point, i):
    r"""The "bare" measurement error.

    This is the lowest order estimate of the measurement infidelity, given by
    :math:`\hat{r}(\neg i, i|i)`.

    Literally returns ``return point[not_i(i), i, i]``
    """
    ni = not_i(i)
    return point[ni, i, i]


def upperb(r, c, i):
    r"""Computes an upper bound to the higher order corrections for the
    estimate of infidelity.

    Equation S.16 in the supplement.

    :param r: The frequencies to compute the upper bound for.
    :param c: The parameter used to bound the infidelities.
    :param i: Initial prep.
    :return: :math:`u(r, c, i)`
    """
    ni = not_i(i)
    return r[ni, i, i]*(r[ni, i, i] + r[ni, ni, i] + r[i, ni, ni])/c**6 + \
           r[ni, ni, i]*r[ni, i, i]**2*(r[ni, i, i]+r[i, ni, ni])/c**12


def lowerb(r, c, i):
    r"""Computes an upper bound to the higher order corrections for the
    estimate of infidelity.

    Equation S.18 in the supplement.

    :param r: The frequencies to compute the upper bound for.
    :param c: The parameter used to bound the infidelities.
    :param i: Initial prep.
    :return: :math:`l(r, c, i)`
    """
    ni = not_i(i)
    return r[ni,ni,i]/c**3*(r[i,ni,ni]/c**3+r[ni,i,i]**2/c**6+r[ni,i,i]*r[i,ni,ni]/c**6) +\
    r[ni,i,i]**2/c**6*(r[ni,i,i]/c**3 + r[i,ni,ni]/c**3)


def upper_from_data(data, alpha):
    r"""Computes the `Clopper-Pearson`_ confidence upper bounds from the data.

    The returned array is of the same shape as the input.

    .. seealso:: :py:func:`~redundant_fidelity.util.clopper_upper`
    .. _`Clopper-Pearson`: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval

    :param data: :math:`n(x, y, z)` in the supplement.
    :param alpha: The significance level to compute the upper bounds at.
    :return: :math:`\overline{r(x, y|z)}(\alpha, n)`
    """
    return clopper_upper(data, np.sum(data, axis=(0, 1)), alpha)


def lower_from_data(data, alpha):
    r"""Computes the `Clopper-Pearson`_ confidence lower bounds from the data.

    The returned array is of the same shape as the input.

    .. seealso:: :py:func:`~redundant_fidelity.util.clopper_lower`
    .. _`Clopper-Pearson`: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval

    :param data: :math:`n(x, y, z)` in the supplement.
    :param alpha: The significance level to compute the lower bounds at.
    :return: The confidence lower bounds.
    """
    return clopper_lower(data, np.sum(data, axis=(0, 1)), alpha)


def upperb_at_upper(data, alpha, c, i):
    r"""Computes :math:`u(\overline{r}(\alpha, n), c, x)`.

    This is used in Equation S.22

    :param data: :math:`n`
    :param alpha: The significance level to compute the confidence upper bound
        at.
    :param c: :math:`c`
    :param i: Intended preparation.
    :return: The upper bounds
        ``return upperb(upper_from_data(data, alpha), c, i)``
    """
    return upperb(upper_from_data(data, alpha), c, i)


def lowerb_at_upper(data, alpha, c, i):
    r"""Computes :math:`l(\underline{r}(\alpha, n), c, x)`.

    Similar to
    .. seealso:: :py:func:`~redundant_fidelity.analysis_funcs.upperb_at_upper`
    """
    return lowerb(upper_from_data(data, alpha), c, i)


def bare_at_upper(data, alpha, i):
    r"""The lowest order measurement infidelity at the confidence upper bound.

    .. seealso:: :py:func:`~redundant_fidelity.analysis_funcs.bare`

    :param data: Counts to compute for.
    :param alpha: Significance level
    :param i: Indended prep
    :return: Literally returns ``return bare(upper_from_data(data, alpha), i)``
    """
    return bare(upper_from_data(data, alpha), i)


def bare_at_lower(data, alpha, i):
    r"""The lowest order measurement infidelity at the confidence lower bound.

    Similar to

    .. seealso:: :py:func:`~redundant_fidelity.analysis_funcs.bare_at_upper`
    """
    return bare(lower_from_data(data, alpha), i)


def total_upper(data, alpha, beta, c, i):
    r"""The confidence upper bound for the measurement infidelity.

    Computed using the union bound.
    This is Equation S.21 in the supplement for :math:`\gamma = 3\beta`.

    :param data: The counts to compute for.
    :param alpha: Total significance level.
    :param beta: The significance to "give to" the higher order terms.
    :param c: :math:`c`, the coarse lower bound on all the fidelities.
    :param i: Intended prep
    :return: The upper bounds.
    """
    return upperb_at_upper(data, beta, c, i) + bare_at_upper(data, alpha - 3*beta, i)


def total_lower(data, alpha, beta, c, i):
    r"""The confidence lower bound for the measurement infidelity.

    If we obtain something negative from subtracting off the higher order
    corrections, we return 0.

    .. seealso:: :py:func:`~redundant_fidelity.analysis_funcs.total_upper`
    """
    return max(bare_at_lower(data, alpha - 3*beta, i) - lowerb_at_upper(data, beta, c, i), 0)


def upper_reference_point(r, c, i):
    r"""

    This is
    :math:`g(r, c, x) \equiv r(\neg x, x|x) + u(r,c,x)`

    :param r: :math:`r`
    :param c: :math:`c`
    :param i: Intended prep.
    :return: :math:`f`
    """
    return bare(r, i) + upperb(r, c, i)


def lower_reference_point(r, c, i):
    r"""

    This is
    :math:`f(r, c, x) \equiv r(\neg x, x|x) - l(r,c,x)`

    .. seealso:: :py:func:`~redundant_fidelity.analysis_funcs.upper_reference_point`
    """
    return max(bare(r, i) - lowerb(r, c, i), 0)


def optimal_upper_beta(data, alpha, c, i):
    r"""Given data and a fixed total significance, finds the optimal
    significance to assign to the higher order corrections.

    Optimal in the sense of giving the tightest upper bound.

    Note that this is
    a function exclusively used before the true data is seen, only used to find
    a good choice for :math:`\beta` . Using this function after the data is
    seen is cheating, and does not give a legitimate confidence interval.

    This is equation S.24

    Uses ``scipy.optimize.minimize_scalar``.

    :param data: Training data to compute confidence upper bound for.
    :param alpha: Significance level.
    :param c: :math:`c`
    :param i: Intended preparation.
    :return: :math:`\beta^*(n, \alpha, c, i)`
    """
    def fn(beta):
        return total_upper(data, alpha, beta, c, i)
    return scipy.optimize.minimize_scalar(fn, alpha/6, bounds=(0., alpha/3), method='bounded').x


def optimal_lower_beta(data, alpha, c, i):
    r"""Given data and a fixed total significance, finds the optimal
    significance to assign to the higher order corrections.

    Optimal in the sense of giving the tightest lower bound.

    .. seealso:: :py:func:`~redundant_fidelity.analysis_funcs.optimal_upper_beta`
    """
    def fn(beta):
        return -total_lower(data, alpha, beta, c, i)
    return scipy.optimize.minimize_scalar(fn, alpha/6, bounds=(0., alpha/3), method='bounded').x


def upper_loss(data, alpha, beta, c, i):
    r"""A loss function to analyze quality of the choice for :math:`\beta`.

    Equation S.25.

    Note that this function is only used for training purposes, we use this only
    on artificial training counts.

    :param data: The data to compute for.
    :param alpha: The total significance.
    :param beta: Significance for the higher order terms.
    :param c: :math:`c`
    :param i: Intended preparation.
    :return: :math:`d(\beta, n, \alpha, c, i)`
    """
    upp_p = upper_reference_point(data_to_point(data), c, i)
    beta_opt = optimal_upper_beta(data, alpha, c, i)
    d1 = total_upper(data, alpha, beta, c, i) - upp_p
    d2 = total_upper(data, alpha, beta_opt, c, i) - upp_p
    return 0 if np.isclose(upp_p, 0) else np.abs(d1 - d2) / d2


def lower_loss(data, alpha, beta, c, i):
    r"""A loss function to analyze quality of the choice for :math:`\beta`.

    .. seealso:: :py:func:`~redundant_fidelity.analysis_funcs.upper_loss`
    """
    low_p = lower_reference_point(data_to_point(data), c, i)
    beta_opt = optimal_lower_beta(data, alpha, c, i)
    d1 = low_p - total_lower(data, alpha, beta, c, i)
    d2 = low_p - total_lower(data, alpha, beta_opt, c, i)
    return 0 if np.isclose(low_p, 0) else np.abs(d1 - d2) / d2


def exclude_0(max_range, num_points):
    return np.linspace(max_range / (num_points), max_range, num=num_points)


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
