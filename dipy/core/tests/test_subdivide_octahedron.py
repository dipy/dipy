from numpy.testing import assert_array_almost_equal

from dipy.core.subdivide_octahedron import create_unit_sphere


def test_create_unit_sphere():
    sphere = create_unit_sphere(7)
    v, e, f = sphere.vertices, sphere.edges, sphere.faces
    assert_array_almost_equal((v*v).sum(1), 1)


def create_half_unit_sphere():
    sphere = create_half_unit_sphere(7)
    v, e, f = sphere.vertices, sphere.edges, sphere.faces
    assert_array_almost_equal((v*v).sum(1), 1)
