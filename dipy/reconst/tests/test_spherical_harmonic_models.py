
import numpy as np
from dipy.core.triangle_subdivide import create_half_unit_sphere
from dipy.reconst.dti import design_matrix, _compact_tensor

from dipy.reconst.recspeed import peak_finding_edges
from nose.tools import assert_equal, assert_raises, assert_true, assert_false
from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.reconst.spherical_harmonic_models import qball_opdf_fit, \
        real_sph_harm, sph_harm_ind_list, cartesian2polar, qball_odf_fit, \
        qball_design, _robust_peaks, _closest_peak, OpdfModel, \
        normalize_data, ClosestPeakSelector

def test_sph_harm_ind_list():
    m_list, n_list = sph_harm_ind_list(8)
    assert_equal(m_list.shape, n_list.shape)
    assert_equal(m_list.shape, (45,))
    assert_true(np.all(np.abs(m_list) <= n_list))
    assert_array_equal(n_list % 2, 0)
    assert_raises(ValueError, sph_harm_ind_list, 1)


def test_real_sph_harm():
    # Tests derived from tables in
    # http://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    # where real spherical harmonic $Y^m_n$ is defined to be:
    #    Real($Y^m_n$) * sqrt(2) if m > 0
    #    $Y^m_n$                 if m == 0
    #    Imag($Y^m_n$) * sqrt(2) if m < 0

    rsh = real_sph_harm
    pi = np.pi
    exp = np.exp
    sqrt = np.sqrt
    sin = np.sin
    cos = np.cos
    assert_array_almost_equal(rsh(0,0,0,0),
           0.5/sqrt(pi))
    assert_array_almost_equal(rsh(2,2,pi/3,pi/5),
           0.25*sqrt(15./(2.*pi))*
           (sin(pi/5.))**2.*cos(0+2.*pi/3)*sqrt(2))
    assert_array_almost_equal(rsh(-2,2,pi/3,pi/5),
           0.25*sqrt(15./(2.*pi))*
           (sin(pi/5.))**2.*sin(0-2.*pi/3)*sqrt(2))
    assert_array_almost_equal(rsh(2,2,pi,pi/2),
           0.25*sqrt(15/(2.*pi))*
           cos(2.*pi)*sin(pi/2.)**2.*sqrt(2))
    assert_array_almost_equal(rsh(-2,4,pi/4.,pi/3.),
           (3./8.)*sqrt(5./(2.*pi))*
           sin(0-2.*pi/4.)*
           sin(pi/3.)**2.*
           (7.*cos(pi/3.)**2.-1)*sqrt(2))
    assert_array_almost_equal(rsh(4,4,pi/8.,pi/6.),
           (3./16.)*sqrt(35./(2.*pi))*
           cos(0+4.*pi/8.)*sin(pi/6.)**4.*sqrt(2))
    assert_array_almost_equal(rsh(-4,4,pi/8.,pi/6.),
           (3./16.)*sqrt(35./(2.*pi))*
           sin(0-4.*pi/8.)*sin(pi/6.)**4.*sqrt(2))
    aa = np.ones((3,1,1,1))
    bb = np.ones((1,4,1,1))
    cc = np.ones((1,1,5,1))
    dd = np.ones((1,1,1,6))
    assert_equal(rsh(aa, bb, cc, dd).shape, (3, 4, 5, 6))

peak_values = np.array([1, .9, .8, .7, .6, .2, .1])
peak_points = np.array([[1., 0., 0.],
                        [0., .9, .1],
                        [0., 1., 0.],
                        [.9, .1, 0.],
                        [0., 0., 1.],
                        [1., 1., 0.],
                        [0., 1., 1.]])
norms = np.sqrt((peak_points*peak_points).sum(-1))
peak_points = peak_points/norms[:, None]

def test_robust_peaks():
    good_peaks = _robust_peaks(peak_points, peak_values, .5, .9)
    assert_array_equal(good_peaks, peak_points[[0,1,4]])

def test_closest_peak():
    prev = np.array([1, -.9, 0])
    prev = prev/np.sqrt(np.dot(prev, prev))
    cp = _closest_peak(peak_points, prev, .5)
    assert_array_equal(cp, peak_points[0])
    cp = _closest_peak(peak_points, -prev, .5)
    assert_array_equal(cp, -peak_points[0])
    cp = _closest_peak(peak_points, prev, .75)
    assert_equal(cp, False)

def test_set_angle_limit():
    bval = np.ones(100)
    bval[0] = 0
    bvec = np.ones((3, 100))
    sig = np.zeros(100)
    v = np.ones((200, 3))
    e = None
    opdf_fitter = OpdfModel(6, bval, bvec, sampling_points=v, sampling_edges=e)
    norm_sig = normalize_data(sig, bval, min_signal=0)
    stepper = ClosestPeakSelector(opdf_fitter, norm_sig, gfa_limit=0,
                                  angle_limit=55)
    assert_raises(ValueError, stepper._set_angle_limit, 99)
    assert_raises(ValueError, stepper._set_angle_limit, -1.1)

def make_fake_signal():
    v, e, f = create_half_unit_sphere(4)
    vecs_xy = v[np.flatnonzero(v[:, 2] == 0)]
    evals = np.array([1.8, .2, .2])*10**-3
    evecs_moveing = np.empty((len(vecs_xy), 3, 3))
    evecs_moveing[:, :, 0] = vecs_xy
    evecs_moveing[:, :, 1] = [0, 0, 1]
    evecs_moveing[:, :, 2] = np.cross(evecs_moveing[:, :, 0],
                                      evecs_moveing[:, :, 1])
    assert ((evecs_moveing * evecs_moveing).sum(1) - 1 < .001).all()
    assert ((evecs_moveing * evecs_moveing).sum(2) - 1 < .001).all()

    gtab = np.empty((len(v) + 1, 3))
    bval = np.empty(len(v) + 1)
    bval[0] = 0
    bval[1:] = 2000
    gtab[0] = [0, 0, 0]
    gtab[1:] = v
    bvec = gtab.T
    B = design_matrix(bvec, bval)

    tensor_moveing = np.empty_like(evecs_moveing)
    for ii in xrange(len(vecs_xy)):
        tensor_moveing[ii] = np.dot(evecs_moveing[ii]*evals,
                                    evecs_moveing[ii].T)
    D_moveing = _compact_tensor(tensor_moveing)
    tensor_fixed = np.diag(evals)
    D_fixed = _compact_tensor(tensor_fixed)

    sig = np.exp(np.dot(D_moveing, B.T)) + 1.1*np.exp(np.dot(B, D_fixed))
    return v, e, vecs_xy, bval, bvec, sig


def test_ClosestPeakSelector():
    v, e, vecs_xy, bval, bvec, sig = make_fake_signal()
    opdf_fitter = OpdfModel(6, bval, bvec, sampling_points=v, sampling_edges=e)
    norm_sig = normalize_data(sig, bval, min_signal=0)
    stepper = ClosestPeakSelector(opdf_fitter, norm_sig, gfa_limit=0,
                                  angle_limit=55)

    for ii in xrange(len(vecs_xy)):
        step = stepper.next_step(ii, [0, 1., 0])
        if np.dot(vecs_xy[ii], [0, 1., 0]) < .57:
            assert_false(step)
        else:
            s2 = stepper.next_step(ii, vecs_xy[ii])
            assert_array_equal(vecs_xy[ii], step)
            step = stepper.next_step(ii, [1., 0, 0.])
            assert_array_equal([1., 0, 0.], step)

