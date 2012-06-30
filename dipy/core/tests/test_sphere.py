import numpy as np
import numpy.testing as nt
import warnings

from dipy.core.sphere import (Sphere, HemiSphere, unique_edges, unique_faces,
                              faces_from_sphere_vertices)
from dipy.core.triangle_subdivide import (create_unit_sphere,
                                          octahedron_vertices,
                                          octahedron_edges,
                                          octahedron_triangles)
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


def freeze_array(a):
    return set(frozenset(i) for i in a)

def test_unique_edges():
    u = unique_edges([[0, 1, 2],
                      [1, 2, 0]])
    u = freeze_array(u)

    e = freeze_array([[1, 2],
                      [0, 1],
                      [0, 2]])

    nt.assert_equal(e, u)


def test_unique_faces():
    u = unique_faces([[0, 1, 2],
                      [1, 2, 0],
                      [0, 2, 1],
                      [1, 2, 3]])
    u = freeze_array(u)

    e = freeze_array([[0, 1, 2],
                      [1, 2, 3]])

    nt.assert_equal(e, u)


def test_faces_from_sphere_vertices():
    faces = faces_from_sphere_vertices(octahedron_vertices)
    faces = set(frozenset(i) for i in faces)
    expected = set(frozenset(i) for i in octahedron_edges[octahedron_triangles, 0])
    nt.assert_equal(faces, expected)


if __name__ == "__main__":
    nt.run_module_suite()
