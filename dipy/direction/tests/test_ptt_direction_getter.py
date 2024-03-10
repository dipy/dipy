"""Test file for Parallel Transport Tracking Algorithm."""
import warnings

import numpy as np
import numpy.testing as npt
from dipy.core.sphere import unit_octahedron
from dipy.data import get_fnames, default_sphere
from dipy.direction import PTTDirectionGetter
from dipy.io.image import load_nifti
from dipy.reconst.shm import (
    SphHarmFit,
    SphHarmModel,
    sh_to_sf,
    descoteaux07_legacy_msg,
    tournier07_legacy_msg,
)
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines


def test_ptt_tracking():
    # Test PTT direction getter generate 100 streamlines with more than 1 pts.
    fod_fname, seed_coordinates_fname, _ = get_fnames('ptt_minimal_dataset')
    fod, affine = load_nifti(fod_fname)
    seed_coordinates = np.loadtxt(seed_coordinates_fname)[:10, :]
    sf = sh_to_sf(fod, default_sphere, basis_type='tournier07', sh_order_max=8,
                  legacy=False)
    sf[sf < 0] = 0
    sc = BinaryStoppingCriterion(np.ones(fod.shape[:3]))

    dg_default = PTTDirectionGetter.from_pmf(sf, sphere=default_sphere,
                                             max_angle=20)
    dg_count2 = PTTDirectionGetter.from_pmf(sf, sphere=default_sphere,
                                            max_angle=20,
                                            probe_count=2,
                                            probe_radius=0.2)
    dg_quality10 = PTTDirectionGetter.from_pmf(sf, sphere=default_sphere,
                                               max_angle=20,
                                               probe_quality=10)
    dg_length2 = PTTDirectionGetter.from_pmf(sf, sphere=default_sphere,
                                             max_angle=20,
                                             probe_length=2)

    for dg in [dg_default, dg_count2, dg_quality10, dg_length2]:
        streamline_generator = LocalTracking(direction_getter=dg,
                                             step_size=0.2,
                                             stopping_criterion=sc,
                                             seeds=seed_coordinates,
                                             affine=affine)
        streamlines = Streamlines(streamline_generator)
        npt.assert_equal(len(streamlines), 10)
        npt.assert_(np.all([len(s) > 1 for s in streamlines]))

    # Test with zeros pmf
    dg = PTTDirectionGetter.from_pmf(np.zeros(sf.shape), sphere=default_sphere,
                                     max_angle=20)
    streamline_generator = LocalTracking(direction_getter=dg,
                                         step_size=0.2,
                                         stopping_criterion=sc,
                                         seeds=seed_coordinates,
                                         affine=affine)
    streamlines = Streamlines(streamline_generator)
    npt.assert_equal(len(streamlines), 10)
    npt.assert_(np.all([len(s) == 1 for s in streamlines]))

    # Test with maximum length reach
    dg = PTTDirectionGetter.from_pmf(sf, sphere=default_sphere,
                                     max_angle=20)
    streamline_generator = LocalTracking(direction_getter=dg,
                                         step_size=0.2,
                                         stopping_criterion=sc,
                                         seeds=seed_coordinates,
                                         affine=affine,
                                         maxlen=1,
                                         minlen=1)
    streams = Streamlines(streamline_generator)
    npt.assert_almost_equal(np.linalg.norm(streams[0][0] - streams[0][1]), 0.2,
                            decimal=1)
    npt.assert_equal(len(streams), 10)
    npt.assert_(np.all([len(s) <= 3 for s in streams]))

    streamline_generator = LocalTracking(direction_getter=dg,
                                         step_size=0.2,
                                         stopping_criterion=sc,
                                         seeds=seed_coordinates,
                                         affine=affine,
                                         maxlen=2,
                                         fixedstep=False)

    # Check fixedstep ValueError
    npt.assert_raises(ValueError, Streamlines, streamline_generator)


def test_PTTDirectionGetter():
    # Test the constructors and errors of the PTTDirectionGetter
    class SillyModel(SphHarmModel):

        def fit(self, data, mask=None):
            coeff = np.zeros(data.shape[:-1] + (15,))
            return SphHarmFit(self, coeff, mask=None)

    silly_model = SillyModel(gtab=None)
    data = np.zeros((3, 3, 3, 7))
    fit = silly_model.fit(data)
    point = np.zeros(3)
    dir = unit_octahedron.vertices[0].copy()

    # Make ptt_dg from shm_coeffs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        dg = PTTDirectionGetter.from_shcoeff(fit.shm_coeff, 90,
                                             unit_octahedron)
    npt.assert_equal(dg.get_direction(point, dir), 1)

    # Make ptt_dg from pmf
    pmf = np.zeros((3, 3, 3, unit_octahedron.theta.shape[0]))
    dg = PTTDirectionGetter.from_pmf(pmf, 90, unit_octahedron)
    npt.assert_equal(dg.get_direction(point, dir), 1)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=tournier07_legacy_msg,
            category=PendingDeprecationWarning)

        # Check probe_length ValueError
        npt.assert_raises(ValueError,
                          PTTDirectionGetter.from_shcoeff,
                          fit.shm_coeff, 90, unit_octahedron,
                          basis_type="tournier07",
                          probe_length=0)

        # Check probe_radius ValueError
        npt.assert_raises(ValueError,
                          PTTDirectionGetter.from_shcoeff,
                          fit.shm_coeff, 90, unit_octahedron,
                          basis_type="tournier07",
                          probe_radius=-1)

        # Check probe_quality ValueError
        npt.assert_raises(ValueError,
                          PTTDirectionGetter.from_shcoeff,
                          fit.shm_coeff, 90, unit_octahedron,
                          basis_type="tournier07",
                          probe_quality=1)

        # Check probe_length ValueError
        npt.assert_raises(ValueError,
                          PTTDirectionGetter.from_shcoeff,
                          fit.shm_coeff, 90, unit_octahedron,
                          basis_type="tournier07",
                          probe_count=0)
