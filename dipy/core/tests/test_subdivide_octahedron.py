from numpy.testing import assert_allclose

from dipy.core.subdivide_octahedron import create_unit_sphere


def test_create_unit_sphere():
    sphere = create_unit_sphere(recursion_level=7)
    v, _, _ = sphere.vertices, sphere.edges, sphere.faces
    assert_allclose((v * v).sum(1), 1)


def create_half_unit_sphere():
    sphere = create_half_unit_sphere(7)
    v, _, _ = sphere.vertices, sphere.edges, sphere.faces
    assert_allclose((v * v).sum(1), 1)
