import util
import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import pytest


def other_clopper_lower(x, n, alpha):
    if x == 0:
        return 0
    def fn(p):
        return (1 - scipy.stats.binom.cdf(x - 1, n, p)) - alpha
    return scipy.optimize.root_scalar(fn, bracket=[x / n / 1000, x / n]).root


def other_clopper_upper(x, n, alpha):
    if x == n:
        return 1
    def fn(p):
        return (scipy.stats.binom.cdf(x, n, p)) - alpha
    return scipy.optimize.root_scalar(fn, bracket=[x / n, 1-.0000001]).root


@pytest.mark.parametrize("x, n, alpha",
                         [
                             (1, 5, .05),
                             (0, 5, .05),
                             (5, 5, .05),
                         ])
def test_clopper_lower(x, n, alpha):
    test1 = util.clopper_lower(x, n, alpha)
    test2 = other_clopper_lower(x, n, alpha)
    assert np.isclose(test1, test2)


@pytest.mark.parametrize("xs, ns, alphas",
                         [
                             (np.array([1, 0, 5]), np.array([5, 5, 5]), np.array([.05, .05, .05])),
                             (np.array([1, 0, 5]), np.array([5]), np.array([.05])),
                             (np.array([1, 0, 5]), np.array([5, 4, 5]), np.array([.05])),
                             (np.array([1, 0, 5]), np.array([5, 4, 5]), np.array([.05, .32, .16])),
                         ])
def test_clopper_broadcast(xs, ns, alphas):
    test1 = util.clopper_lower(xs, ns, alphas)
    xs, ns, alphas, test1 = np.broadcast_arrays(xs, ns, alphas, test1)
    for x, n, alpha, l in zip(*list(map(np.ravel, (xs, ns, alphas, test1)))):
        test2 = other_clopper_lower(x, n, alpha)
        assert np.isclose(test2, l)


@pytest.mark.parametrize("x, n, alpha",
                         [
                             (1, 5, .05),
                             (0, 5, .05),
                             (5, 5, .05),
                         ])
def test_clopper_upper(x, n, alpha):
    test1 = util.clopper_upper(x, n, alpha)
    test2 = other_clopper_upper(x, n, alpha)
    assert np.isclose(test1, test2)


@pytest.mark.parametrize("xs, ns, alphas",
                         [
                             (np.array([1, 0, 5]), np.array([5, 5, 5]), np.array([.05, .05, .05])),
                             (np.array([1, 0, 5]), np.array([5]), np.array([.05])),
                             (np.array([1, 0, 5]), np.array([5, 4, 5]), np.array([.05])),
                             (np.array([1, 0, 5]), np.array([5, 4, 5]), np.array([.05, .32, .16])),
                         ])
def test_clopper_broadcast(xs, ns, alphas):
    test1 = util.clopper_lower(xs, ns, alphas)
    xs, ns, alphas, test1 = np.broadcast_arrays(xs, ns, alphas, test1)
    for x, n, alpha, l in zip(*list(map(np.ravel, (xs, ns, alphas, test1)))):
        test2 = other_clopper_lower(x, n, alpha)
        assert np.isclose(test2, l)
