""" Testing sphere makers
"""

import numpy as np

from ..spheremakers import sphere_vf_from

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises

def test_spheremakers():
    # Test inputs to spheremakers
    # Example data given string
    sphere = sphere_vf_from('symmetric362')
    v, f = sphere.vertices, sphere.faces
    assert_equal(f.shape[1], 3)
    assert_equal(v.shape[1], 3)
    # Given tuple
    vdash, fdash = sphere_vf_from((v, f))
    assert_array_equal(vdash, v)
    assert_array_equal(fdash, f)
    # Given dict
    sphere = sphere_vf_from({'vertices': v})
    vdash, fdash = sphere.vertices, sphere.faces
    assert_array_almost_equal(vdash, v)
    assert_array_almost_equal(fdash, f)
