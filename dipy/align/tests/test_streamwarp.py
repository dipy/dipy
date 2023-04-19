from numpy.testing import assert_equal
from dipy.align.streamwarp import (bundlewarp,
                                   bundlewarp_shape_analysis,
                                   bundlewarp_vector_filed)

from dipy.data import two_cingulum_bundles
from dipy.tracking.streamline import set_number_of_points, Streamlines


def test_bundlewarp():

    cingulum_bundles = two_cingulum_bundles()

    cb1 = Streamlines(cingulum_bundles[0])
    cb1 = set_number_of_points(cb1, 20)

    cb2 = Streamlines(cingulum_bundles[1])
    cb2 = set_number_of_points(cb2, 20)

    deformed_bundle, affine_bundle, dists, mp, warp = bundlewarp(cb1, cb2)

    assert_equal(len(affine_bundle), len(cb2))
    assert_equal(len(deformed_bundle), len(cb2))
    assert_equal(len(deformed_bundle), len(affine_bundle))

    assert_equal(dists.shape, (len(cb2), len(cb1)))

    assert_equal(len(cb2), len(mp))

    assert_equal(len(cb2), len(warp))


def test_bundlewarp_vector_filed():

    cingulum_bundles = two_cingulum_bundles()

    cb1 = Streamlines(cingulum_bundles[0])
    cb1 = set_number_of_points(cb1, 20)

    cb2 = Streamlines(cingulum_bundles[1])
    cb2 = set_number_of_points(cb2, 20)

    deformed_bundle, affine_bundle, dists, mp, warp = bundlewarp(cb1, cb2)

    offsets, directions, colors = bundlewarp_vector_filed(affine_bundle,
                                                          deformed_bundle)

    assert_equal(len(offsets), len(cb2.get_data()))
    assert_equal(len(directions), len(cb2.get_data()))
    assert_equal(len(colors), len(cb2.get_data()))

    assert_equal(len(offsets), len(deformed_bundle.get_data()))
    assert_equal(len(directions), len(deformed_bundle.get_data()))
    assert_equal(len(colors), len(deformed_bundle.get_data()))


def test_bundle_shape_profile():

    cingulum_bundles = two_cingulum_bundles()

    cb1 = Streamlines(cingulum_bundles[0])
    cb1 = set_number_of_points(cb1, 20)

    cb2 = Streamlines(cingulum_bundles[1])
    cb2 = set_number_of_points(cb2, 20)

    deformed_bundle, affine_bundle, dists, mp, warp = bundlewarp(cb1, cb2)

    n = 10
    shape_profile, stdv = bundlewarp_shape_analysis(affine_bundle,
                                                    deformed_bundle,
                                                    no_disks=n)

    assert_equal(len(shape_profile), n)
    assert_equal(len(stdv), n)

    n = 100
    shape_profile, stdv = bundlewarp_shape_analysis(affine_bundle,
                                                    deformed_bundle,
                                                    no_disks=n)

    assert_equal(len(shape_profile), n)
    assert_equal(len(stdv), n)
