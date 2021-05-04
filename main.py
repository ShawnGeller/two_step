import numpy as np
import scipy.special
import scipy.stats
import scipy
import scipy.optimize
from itertools import cycle
import analysis_funcs as af
import matplotlib.pyplot as plt
import argparse



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
    ax.plot(beta_range, lower_point, 'k-', label="lower reference")
    ax.plot(beta_range, upper_point, 'k-', label="upper reference")
    if colors is None:
        colors = cycle(['r', 'g', 'b', 'y'])
    else:
        colors = cycle(colors)
    for alpha, color in zip(alphas, colors):
        for xx in range(1, 3):
            opt_upp_bet = af.optimal_upper_beta(data, alpha/xx, c, i)
            t_upp = af.total_upper(data, alpha/xx, opt_upp_bet, c, i)
            ax.plot(opt_upp_bet, t_upp, 'mo')
            if xx == 2:
                opt_low_bet = af.optimal_lower_beta(data, alpha/xx, c, i)
                t_low = af.total_lower(data, alpha/xx, opt_low_bet, c, i)
                ax.plot(opt_low_bet, t_low, 'mo')
        ax.plot(beta_range, one_sided_vary_beta(alpha, data, c, i, beta_range), color=color, linestyle='-', label=r"one sided, level = {:.2f}".format(1-alpha))
        first = True
        for line in two_sided_vary_beta(alpha/2, data, c, i, beta_range):
            if first:
                ax.plot(beta_range, line, color=color, linestyle='--', label=r"two sided, level = {:.2f}".format(1-alpha))
            else:
                ax.plot(beta_range, line, color=color, linestyle='--')
            first = False
    ax.set_xlim(left=beta_range[0], right=beta_range[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("beta")
    ax.set_ylabel("estimate")
    ax.set_title("c = {}, initial prep = {},\n r(10|0) = r(01|0) = 1e-5, r(01|1) = r(10|1) = 1e-4, r(11|0) = r(00|1) = 1e-2".format(c, i))
    ax.legend()
    plt.show()


def main(i, c=.95, n0=100000, n1=25000, alpha=None):

    one_sigma_alpha = 1-(scipy.stats.norm.cdf(1) - scipy.stats.norm.cdf(-1))
    two_sigma_alpha = 1-(scipy.stats.norm.cdf(2) - scipy.stats.norm.cdf(-2))
    alphas = [one_sigma_alpha, two_sigma_alpha]

    if alpha is not None:
        alpha = float(alpha)
        alphas.append(alpha)
    data = af.point_to_data(n0, n1, af.point_at_nominal(1e-5, 1e-2, 1e-4))
    vary_beta_plot(alphas, data, c, i)
    for alpha in alphas:
        print("alpha = {:.2e}, optimal upper beta = {:.2e}".format(alpha, af.optimal_upper_beta(data, alpha, c, i)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Computes confidence intervals and plots.")
    parser.add_argument("--n0", type=int, default=100000, help="Number of experiments with intended prep = 1. Defaults to 100000.")
    parser.add_argument("--n1", type=int, default=25000, help="Number of experiments with intended prep = 1. Defaults to 25000.")
    parser.add_argument("--alpha", type=float, help="Will always compute 1 sigma and 2 sigma levels for both two sided and one sided, if alpha is specified will compute at this level in addition.")
    parser.add_argument("--c", type=float, default=.95, help="The parameter c, given a lower bound on correct state prep, correct measurement, QND-ness")
    parser.add_argument("i", type=int, help="The initial prep, either 0 or 1.")
    args = parser.parse_args()
    main(args.i, args.c, args.n0, args.n1, args.alpha)
