import numpy as np
import numpy.testing as nt
import warnings

from dipy.core.sphere import Sphere, HemiSphere
from dipy.core.triangle_subdivide import create_unit_sphere
from dipy.core.geometry import cart2sphere, sphere2cart

verts, edges, sides = create_unit_sphere(5)
r, theta, phi = cart2sphere(*verts.T)


def test_sphere_construct_args():
    nt.assert_raises(ValueError, Sphere)
    nt.assert_raises(ValueError, Sphere, x=1, theta=1)
    nt.assert_raises(ValueError, Sphere, xyz=1, theta=1)
    nt.assert_raises(ValueError, Sphere, xyz=1, theta=1, phi=1)


def test_edges_faces():
    nt.assert_raises(ValueError, Sphere, xyz=1, edges=1, faces=None)
    Sphere(xyz=[0, 0, 1], faces=[0, 0, 0])
    Sphere(xyz=[[0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]],
           edges=[[0, 1],
                  [1, 2],
                  [2, 0]],
           faces=[0, 1, 2])


def test_sphere_not_unit():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        nt.assert_raises(UserWarning, Sphere, xyz=[0, 0, 1.5])


def test_sphere_construct():
    s0 = Sphere(xyz=verts)
    s1 = Sphere(theta=theta, phi=phi)
    s2 = Sphere(*verts.T)

    nt.assert_array_almost_equal(s0.theta, s1.theta)
    nt.assert_array_almost_equal(s0.theta, s2.theta)
    nt.assert_array_almost_equal(s0.theta, theta)

    nt.assert_array_almost_equal(s0.phi, s1.phi)
    nt.assert_array_almost_equal(s0.phi, s2.phi)
    nt.assert_array_almost_equal(s0.phi, phi)


if __name__ == "__main__":
    nt.run_module_suite()
