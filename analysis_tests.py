#%%
import numpy as np
import analysis_funcs as af
import util
import scipy.stats

#%%
n0 = 100000
n1 = 25000
me = 1e-4
pe = 1e-2
c = .95
alpha = .05
i = 0
point = af.point_at_nominal(me, pe)
datapoint = af.point_to_data(n0, n1, point)


#%%

def test_point_at_nominal():
    me = 1e-4
    pe = 1e-2
    pan = af.point_at_nominal(me, pe)
    assert np.allclose(pan, np.array([[[9.898e-01, 1.000e-02], [1.000e-04, 1.000e-04]], [[1.000e-04, 1.000e-04], [1.000e-02, 9.898e-01]]]))


#%%
def test_not_i():
    assert af.not_i(1) == 0
    assert af.not_i(0) == 1


#%%
def test_bare():
    assert af.bare(point, i) == me

#%%
af.upperb(point, c, i)
af.lowerb(point, c, i)


#%%
def test_upper_from_data():
    assert np.all(af.upper_from_data(datapoint, alpha) > point)


#%%
def test_lower_from_data():
    assert np.all(af.lower_from_data(datapoint, alpha) < point)


#%%
def test_upperb_at_upper():
    assert af.upperb_at_upper(datapoint, alpha, c, i) > af.upperb(point, c, i)

#%%
def test_lowerb_at_upper():
    assert af.lowerb_at_upper(datapoint, alpha, c, i) > af.lowerb(point, c, i)

#%%
def test_bare_at_lower():
    ni = af.not_i(i)
    assert af.bare_at_lower(datapoint, alpha, i) == util.clopper_lower(datapoint[ni, i, i], n0, alpha)

#%%
def test_total_upper():
    tu = af.total_upper(datapoint, alpha, alpha/4, c, i)
    assert tu > af.bare_at_upper(datapoint, alpha, i)
    assert tu < 1.

#%%
def test_total_lower():
    tl = af.total_lower(datapoint, alpha, alpha/4, c, i)
    assert tl < af.bare_at_lower(datapoint, alpha, i)

#%%
def test_upper_reference_point():
    rp = af.upper_reference_point(point, c, i)
    assert rp > af.bare(point, i)
    tu = af.total_upper(datapoint, alpha, alpha / 4, c, i)
    assert rp < tu

#%%
def test_lower_reference_point():
    rp = af.lower_reference_point(point, c, i)
    assert rp < af.bare(point, i)
    tl = af.total_lower(datapoint, alpha, alpha / 4, c, i)
    assert rp > tl

#%%
def test_optimal_upper_beta():
    opt_beta = af.optimal_upper_beta(datapoint, alpha, c, i)
    tu = af.total_upper(datapoint, alpha, alpha / 4, c, i)
    assert af.total_upper(datapoint, alpha, opt_beta, c, i) <= tu


#%%
def test_optimal_lower_beta():
    opt_beta = af.optimal_lower_beta(datapoint, alpha, c, i)
    tl = af.total_lower(datapoint, alpha, alpha / 4, c, i)
    assert af.total_lower(datapoint, alpha, opt_beta, c, i) >= tl


#%%
prior = scipy.stats.dirichlet(np.ravel(datapoint[:, :, 0]) + np.ones(4)/4.)
af.get_optimal_betas(prior, af.log_width_cost, n0, n1, alpha/2, alpha/2, c, i)