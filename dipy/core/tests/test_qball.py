""" Testing qball 

"""

import numpy as np

import dipy.core.qball as qball

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric

@parametric
def test_sph_harm_ind_list():
    m_list, n_list = qball.sph_harm_ind_list(8)
    yield assert_equal(m_list.shape, n_list.shape)
    yield assert_equal(m_list.ndim, 2)
    yield assert_equal(m_list.shape, (45,1))
    yield assert_true(np.all(np.abs(m_list) <= n_list))
    yield assert_array_equal(n_list % 2, 0)
    yield assert_raises(ValueError, qball.sph_harm_ind_list, 1)

@parametric
def test_real_sph_harm():
    # Tests derived from tables in
    # http://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    # where real spherical harmonic $Y^m_n$ is defined to be:
    #    Real($Y^m_n$) * sqrt(2) if m > 0
    #    $Y^m_n$                 if m == 0
    #    Imag($Y^m_n$) * sqrt(2) if m < 0
 
    rsh = qball.real_sph_harm
    pi = np.pi
    exp = np.exp
    sqrt = np.sqrt
    sin = np.sin
    cos = np.cos
    yield assert_array_almost_equal(rsh(0,0,0,0),
           0.5/sqrt(pi))
    yield assert_array_almost_equal(rsh(2,2,pi/3,pi/5),
           0.25*sqrt(15./(2.*pi))*
           (sin(pi/5.))**2.*cos(0+2.*pi/3)*sqrt(2))
    yield assert_array_almost_equal(rsh(-2,2,pi/3,pi/5),
           0.25*sqrt(15./(2.*pi))*
           (sin(pi/5.))**2.*sin(0-2.*pi/3)*sqrt(2))
    yield assert_array_almost_equal(rsh(2,2,pi,pi/2),
           0.25*sqrt(15/(2.*pi))*
           cos(2.*pi)*sin(pi/2.)**2.*sqrt(2))
    yield assert_array_almost_equal(rsh(-2,4,pi/4.,pi/3.),
           (3./8.)*sqrt(5./(2.*pi))*
           sin(0-2.*pi/4.)*
           sin(pi/3.)**2.*
           (7.*cos(pi/3.)**2.-1)*sqrt(2))
    yield assert_array_almost_equal(rsh(4,4,pi/8.,pi/6.),
           (3./16.)*sqrt(35./(2.*pi))*
           cos(0+4.*pi/8.)*sin(pi/6.)**4.*sqrt(2))
    yield assert_array_almost_equal(rsh(-4,4,pi/8.,pi/6.),
           (3./16.)*sqrt(35./(2.*pi))*
           sin(0-4.*pi/8.)*sin(pi/6.)**4.*sqrt(2))

