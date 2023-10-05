import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_raises)
from dipy.data import get_fnames, dsi_voxels, default_sphere
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.reconst.odf import gfa
from dipy.direction.peaks import peak_directions
from dipy.sims.voxel import sticks_and_ball
from dipy.core.gradients import gradient_table
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.core.sphere_stats import angular_similarity


def test_dsi():
    # load repulsion 724 sphere
    sphere = default_sphere

    # load icosahedron sphere
    sphere2 = create_unit_sphere(5)
    btable = np.loadtxt(get_fnames('dsi515btable'))
    gtab = gradient_table(btable[:, 0], btable[:, 1:])
    data, golden_directions = sticks_and_ball(gtab, d=0.0015, S0=100,
                                              angles=[(0, 0), (90, 0)],
                                              fractions=[50, 50], snr=None)

    ds = DiffusionSpectrumModel(gtab)

    # repulsion724
    dsfit = ds.fit(data)
    odf = dsfit.odf(sphere)

    directions, _, _ = peak_directions(odf, sphere)
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions),
                        2, 1)

    # 5 subdivisions
    dsfit = ds.fit(data)
    odf2 = dsfit.odf(sphere2)
    directions, _, _ = peak_directions(odf2, sphere2)
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions),
                        2, 1)

    assert_equal(dsfit.pdf().shape, 3 * (ds.qgrid_size, ))
    sb_dummies = sticks_and_ball_dummies(gtab)
    for sbd in sb_dummies:
        data, golden_directions = sb_dummies[sbd]
        odf = ds.fit(data).odf(sphere2)
        directions, _, _ = peak_directions(odf, sphere2)
        if len(directions) <= 3:
            assert_equal(len(directions), len(golden_directions))
        if len(directions) > 3:
            assert_equal(gfa(odf) < 0.1, True)

    assert_raises(ValueError, DiffusionSpectrumModel, gtab, qgrid_size=16)


def test_multivox_dsi():
    data, gtab = dsi_voxels()
    DS = DiffusionSpectrumModel(gtab)

    DSfit = DS.fit(data)
    PDF = DSfit.pdf()
    assert_equal(data.shape[:-1] + (17, 17, 17), PDF.shape)
    assert_equal(np.all(np.isreal(PDF)), True)


def test_multib0_dsi():
    data, gtab = dsi_voxels()
    # Create a new data-set with a b0 measurement:
    new_data = np.concatenate([data, data[..., 0, None]], -1)
    new_bvecs = np.concatenate([gtab.bvecs, np.zeros((1, 3))])
    new_bvals = np.concatenate([gtab.bvals, [0]])
    new_gtab = gradient_table(new_bvals, new_bvecs)
    ds = DiffusionSpectrumModel(new_gtab)
    dsfit = ds.fit(new_data)
    pdf = dsfit.pdf()
    dsfit.odf(default_sphere)
    assert_equal(new_data.shape[:-1] + (17, 17, 17), pdf.shape)
    assert_equal(np.all(np.isreal(pdf)), True)

    # And again, with one more b0 measurement (two in total):
    new_data = np.concatenate([data, data[..., 0, None]], -1)
    new_bvecs = np.concatenate([gtab.bvecs, np.zeros((1, 3))])
    new_bvals = np.concatenate([gtab.bvals, [0]])
    new_gtab = gradient_table(new_bvals, new_bvecs)
    ds = DiffusionSpectrumModel(new_gtab)
    dsfit = ds.fit(new_data)
    pdf = dsfit.pdf()
    dsfit.odf(default_sphere)
    assert_equal(new_data.shape[:-1] + (17, 17, 17), pdf.shape)
    assert_equal(np.all(np.isreal(pdf)), True)


def sticks_and_ball_dummies(gtab):
    sb_dummies = {}
    S, sticks = sticks_and_ball(gtab, d=0.0015, S0=100,
                                angles=[(0, 0)],
                                fractions=[100], snr=None)
    sb_dummies['1fiber'] = (S, sticks)
    S, sticks = sticks_and_ball(gtab, d=0.0015, S0=100,
                                angles=[(0, 0), (90, 0)],
                                fractions=[50, 50], snr=None)
    sb_dummies['2fiber'] = (S, sticks)
    S, sticks = sticks_and_ball(gtab, d=0.0015, S0=100,
                                angles=[(0, 0), (90, 0), (90, 90)],
                                fractions=[33, 33, 33], snr=None)
    sb_dummies['3fiber'] = (S, sticks)
    S, sticks = sticks_and_ball(gtab, d=0.0015, S0=100,
                                angles=[(0, 0), (90, 0), (90, 90)],
                                fractions=[0, 0, 0], snr=None)
    sb_dummies['isotropic'] = (S, sticks)
    return sb_dummies
