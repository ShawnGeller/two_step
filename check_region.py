#%%
import numpy as np
import sys

import analysis_funcs as af
import util
import scipy.stats

def main():
    # %%
    n0 = 100000
    n1 = 25000
    me =  3e-5
    me1 = 3e-4
    pe =  3e-2
    c = .95
    alpha = .05
    i = 0
    # %%
    max_point = af.point_at_nominal(me, pe, me1)
    # %%
    ml32 = af.max_loss(max_point, .32 / 2, .001, n0, n1, c, i)
    print(".32, {}".format(ml32))
    # %%
    ml05 = af.max_loss(max_point, alpha / 2, .001, n0, n1, c, i)
    print(".05, {}".format(ml05))
    # %%
    ml321 = af.max_loss(max_point, .32 / 2, .001, n0, n1, c, 1)
    print(".32, i=1, {}".format(ml321))
    # %%
    ml051 = af.max_loss(max_point, alpha / 2, .001, n0, n1, c, 1)
    print(".05, i=1, {}".format(ml051))


if __name__ == "__main__":
    main()
    sys.exit(0)
