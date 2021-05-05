import os
import csv
import argparse
import numpy as np
import scipy.stats
import sys

import analysis_funcs as af


def prepare_filename(filename):
    direc, fn = os.path.split(filename)
    if not direc:
        direc = os.path.curdir
    if not os.path.exists(direc):
        os.makedirs(filename)
    fn, ext = os.path.splitext(filename)
    if not ext:
        ext = ".csv"
    return fn + ext


def parse_input(nstr):
    ns = nstr.split(",")
    ns = np.array([int(n) for n in ns]).reshape((2, 2, 2))
    ns = np.moveaxis(ns, (0, 1, 2), (2, 1, 0))
    return ns


def parse_input_file(input_filename):
    with open(input_filename) as f:
        reader = csv.reader(f)
        data = np.array([int(row[0]) for row in reader])
    data = data.reshape((2, 2, 2))
    data = np.moveaxis(data, (0, 1, 2), (2, 1, 0))
    return data


def main(args):
    if args.counts:
        data = parse_input(args.counts)
    elif args.input_filename:
        data = parse_input_file(args.input_filename)
    else:
        raise ValueError("Must specify either n or input file")
    one_sigma_alpha = 1 - (scipy.stats.norm.cdf(1) - scipy.stats.norm.cdf(-1))
    two_sigma_alpha = 1 - (scipy.stats.norm.cdf(2) - scipy.stats.norm.cdf(-2))
    alphas = [one_sigma_alpha, two_sigma_alpha]

    allbounds = np.array(
        [
            [i] +
            [f(af.data_to_point(data), args.c, i) for f in [af.upper_reference_point, af.lower_reference_point]] +
            [f(data, alpha, args.beta, args.c, i) for alpha in alphas for f in [af.total_lower, af.total_upper]]
            for i in range(2)
        ])
    filename = prepare_filename(args.filename)
    np.savetxt(filename, allbounds, header="Initial prep" + ",Lower reference,Upper reference" +  "".join([",{0:.3e} lower,{0:.3e} upper".format(1-alpha) for alpha in alphas]), delimiter=",", comments="")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Computes confidence intervals at 1 and 2 sigma.")
    parser.add_argument("--c", type=float, default=.95,
                        help="The parameter c, given a lower bound on correct state prep, correct measurement, QND-ness. Default is .95")
    parser.add_argument("--beta", type=float, default=.001,
                        help="The significance level to compute the small parameter confidence intervals at.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-i", "--input-filename", type=str,
                        help="CSV file with counts. Format is a single column, with counts in order"
                             "n000 \n n100 \n n010 \n n110 \n n001 \n n101 \n n011 \n n111")
    group.add_argument("-n", "--counts", type=str,
                        help="Input a string of counts, with format\n"
                             "n000,n100,n010,n110,n001,n101,n011,n111")
    parser.add_argument("filename", type=str,
                        help="CSV File to save to. If unspecified, will not save.")
    args = parser.parse_args()
    main(args)
    sys.exit(0)
