import numpy as np
import scipy.special
import scipy.stats
import scipy
import scipy.optimize
from itertools import cycle
import analysis_funcs as af
import matplotlib.pyplot as plt
import argparse
import os


def prepare_filename(filename):
    direc, fn = os.path.split(filename)
    if not direc:
        direc = os.path.curdir
    if not os.path.exists(direc):
        os.makedirs(filename)
    fn, ext = os.path.splitext(filename)
    if not ext:
        ext = ".png"
    return fn + ext


def one_sided_vary_beta(alpha, data, c, i, beta_range):
    return np.array(
        [af.total_upper(data, alpha, beta, c, i) for beta in beta_range])


def two_sided_vary_beta(alpha, data, c, i, beta_range):
    return np.transpose(np.array([[af.total_upper(data, alpha, beta, c, i),
                                   af.total_lower(data, alpha, beta, c, i)] for
                                  beta in beta_range]))


def compute_beta_range(alphas, data, c, i, num_points=30):
    beta_max = 0
    for alpha in alphas:
        beta_max = max(beta_max, af.optimal_upper_beta(data, alpha, c, i))
    beta_max = beta_max * 2
    beta_range = np.linspace(beta_max / (num_points + 1), beta_max,
                             num=num_points, endpoint=False)
    return beta_range


def vary_beta_plot(alphas, data, c, i, colors=None, num_points=30,
                   filename=None):
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
            opt_upp_bet = af.optimal_upper_beta(data, alpha / xx, c, i)
            t_upp = af.total_upper(data, alpha / xx, opt_upp_bet, c, i)
            ax.plot(opt_upp_bet, t_upp, 'mo')
            if xx == 2:
                opt_low_bet = af.optimal_lower_beta(data, alpha / xx, c, i)
                t_low = af.total_lower(data, alpha / xx, opt_low_bet, c, i)
                ax.plot(opt_low_bet, t_low, 'mo')
        ax.plot(beta_range, one_sided_vary_beta(alpha, data, c, i, beta_range),
                color=color, linestyle='-',
                label=r"one sided, level = {:.2f}".format(1 - alpha))
        first = True
        for line in two_sided_vary_beta(alpha / 2, data, c, i, beta_range):
            if first:
                ax.plot(beta_range, line, color=color, linestyle='--',
                        label=r"two sided, level = {:.2f}".format(1 - alpha))
            else:
                ax.plot(beta_range, line, color=color, linestyle='--')
            first = False
    ax.set_xlim(left=beta_range[0], right=beta_range[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("beta")
    ax.set_ylabel("estimate")
    point = af.data_to_point(data)
    ni = af.not_i(i)
    ax.set_title("c = {:.2f}, initial prep = {},\n "
                 "r(10|0) = r(01|0) = {:.1e}, "
                 "r(01|1) = r(10|1) = {:.1e}, "
                 "r(11|0) = r(00|1) = {:.1e}".format(
        c, i, point[ni, i, i], point[i, ni, ni], point[ni, ni, i]))
    ax.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def main(args):
    if args.filename is not None:
        args.filename = prepare_filename(args.filename)

    one_sigma_alpha = 1 - (scipy.stats.norm.cdf(1) - scipy.stats.norm.cdf(-1))
    two_sigma_alpha = 1 - (scipy.stats.norm.cdf(2) - scipy.stats.norm.cdf(-2))
    alphas = [one_sigma_alpha, two_sigma_alpha]

    if args.alpha is not None:
        args.alpha = float(args.alpha)
        alphas.append(args.alpha)
    data = af.point_to_data(args.n0, args.n1,
                            af.point_at_nominal(args.r100, args.r001,
                                                args.r011))
    vary_beta_plot(alphas, data, args.c, args.i, filename=args.filename)
    for alpha in alphas:
        print("alpha = {:.2e}, optimal upper beta = {:.2e}".format(alpha,
                                                                   af.optimal_upper_beta(
                                                                       data,
                                                                       alpha,
                                                                       args.c,
                                                                       args.i)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Computes confidence intervals and plots.")
    parser.add_argument("--n0", type=int, default=100000,
                        help="Number of experiments with intended prep = 1. Defaults to 100000.")
    parser.add_argument("--n1", type=int, default=25000,
                        help="Number of experiments with intended prep = 1. Defaults to 25000.")
    parser.add_argument("--alpha", type=float,
                        help="Will always compute 1 sigma and 2 sigma levels for both two sided and one sided, if alpha is specified will compute at this level in addition.")
    parser.add_argument("--c", type=float, default=.95,
                        help="The parameter c, given a lower bound on correct state prep, correct measurement, QND-ness")
    parser.add_argument("-o", "--filename", type=str,
                        help="File to save to. If unspecified, will not save.")
    parser.add_argument("--r100", type=float, default=1e-5,
                        help="Nominal measurement error to compute at. Will take r100 = r010")
    parser.add_argument("--r011", type=float, default=1e-4,
                        help="Nominal measurement error to compute at. Will take r011 = r101")
    parser.add_argument("--r001", type=float, default=1e-4,
                        help="Nominal prep error to compute at. Will take r001 = r110")
    parser.add_argument("i", type=int, help="The initial prep, either 0 or 1.")
    args = parser.parse_args()
    main(args)
