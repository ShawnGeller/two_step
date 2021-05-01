import numpy as np
import scipy.special
import scipy.stats
import scipy
import scipy.optimize
from itertools import cycle
import analysis_funcs as af
import matplotlib.pyplot as plt



def one_sided_vary_beta(alpha, data, c, i, beta_range):
    return np.array([af.total_upper(data, alpha, beta, c, i) for beta in beta_range])


def two_sided_vary_beta(alpha, data, c, i, beta_range):
    return np.transpose(np.array([[af.total_upper(data, alpha, beta, c, i), af.total_lower(data, alpha, beta, c, i)] for beta in beta_range]))


def compute_beta_range(alphas, data, c, i, num_points=30):
    beta_max = 0
    for alpha in alphas:
        beta_max = max(beta_max, af.optimal_upper_beta(data, alpha, c, i))
    beta_max = beta_max * 2
    beta_range = np.linspace(beta_max / (num_points + 1), beta_max, num=num_points, endpoint=False)
    return beta_range


def vary_beta_plot(alphas, data, c, i, colors=None, num_points=30):
    beta_range = compute_beta_range(alphas, data, c, i, num_points=num_points)
    lower_point = af.lower_reference_point(af.data_to_point(data), c, i)
    upper_point = af.upper_reference_point(af.data_to_point(data), c, i)
    lower_point = np.array([lower_point for beta in beta_range])
    upper_point = np.array([upper_point for beta in beta_range])
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(beta_range, lower_point, 'k-')
    ax.plot(beta_range, upper_point, 'k-')
    if colors is None:
        colors = cycle(['r', 'g', 'b', 'y'])
    else:
        colors = cycle(colors)
    for alpha, color in zip(alphas, colors):
        ax.plot(beta_range, one_sided_vary_beta(alpha, data, c, i, beta_range), color=color, linestyle='-')
        for line in two_sided_vary_beta(alpha/2, data, c, i, beta_range):
            ax.plot(beta_range, line, color=color, linestyle='--')
    ax.set_ylim(bottom=0)
    plt.show()


def main():
    n0 = 100000
    n1 = 25000
    c = .95
    i = 1
    one_sigma_alpha = 1-(scipy.stats.norm.cdf(1) - scipy.stats.norm.cdf(-1))
    two_sigma_alpha = 1-(scipy.stats.norm.cdf(2) - scipy.stats.norm.cdf(-2))
    alphas = [one_sigma_alpha, two_sigma_alpha]
    # alphas = [one_sigma_alpha, .1]
    data = af.point_to_data(n0, n1, af.point_at_nominal(1e-4, 1e-2))
    vary_beta_plot(alphas, data, c, i)


if __name__ == "__main__":
    main()