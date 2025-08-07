import warnings

import numpy as np
import numpy.testing as nt

from dipy.core.geometry import cart2sphere, vector_norm
from dipy.core.sphere import (
    HemiSphere,
    Sphere,
    _get_forces,
    _get_forces_alt,
    disperse_charges,
    disperse_charges_alt,
    faces_from_sphere_vertices,
    fibonacci_sphere,
    hemi_icosahedron,
    unique_edges,
    unique_sets,
    unit_icosahedron,
    unit_octahedron,
)
from dipy.core.sphere_stats import random_uniform_on_sphere
from dipy.testing.decorators import set_random_number_generator

verts = unit_octahedron.vertices
edges = unit_octahedron.edges
oct_faces = unit_octahedron.faces
r, theta, phi = cart2sphere(*verts.T)


def test_sphere_construct_args():
    nt.assert_raises(ValueError, Sphere)
    nt.assert_raises(ValueError, Sphere, x=1, theta=1)
    nt.assert_raises(ValueError, Sphere, xyz=1, theta=1)
    nt.assert_raises(ValueError, Sphere, xyz=1, theta=1, phi=1)


def test_sphere_edges_faces():
    nt.assert_raises(ValueError, Sphere, xyz=1, edges=1, faces=None)
    Sphere(xyz=[0, 0, 1], faces=[0, 0, 0])
    Sphere(
        xyz=[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        edges=[[0, 1], [1, 2], [2, 0]],
        faces=[0, 1, 2],
    )


def test_sphere_not_unit():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nt.assert_raises(UserWarning, Sphere, xyz=[0, 0, 1.5])


def test_bad_edges_faces():
    nt.assert_raises(ValueError, Sphere, xyz=[0, 0, 1.5], edges=[[1, 2]])


def test_sphere_construct():
    s0 = Sphere(xyz=verts)
    s1 = Sphere(theta=theta, phi=phi)
    # Unpack verts.T into x, y, z and pass as keyword arguments
    x, y, z = verts.T
    s2 = Sphere(x=x, y=y, z=z)

    nt.assert_array_almost_equal(s0.theta, s1.theta)
    nt.assert_array_almost_equal(s0.theta, s2.theta)
    nt.assert_array_almost_equal(s0.theta, theta)

    nt.assert_array_almost_equal(s0.phi, s1.phi)
    nt.assert_array_almost_equal(s0.phi, s2.phi)
    nt.assert_array_almost_equal(s0.phi, phi)


def array_to_set(a):
    return {frozenset(i) for i in a}


def test_unique_edges():
    faces = np.array([[0, 1, 2], [1, 2, 0]])
    e = array_to_set([[1, 2], [0, 1], [0, 2]])

    u = unique_edges(faces)
    nt.assert_equal(e, array_to_set(u))

    u, m = unique_edges(faces, return_mapping=True)
    nt.assert_equal(e, array_to_set(u))
    edges = [[[0, 1], [1, 2], [2, 0]], [[1, 2], [2, 0], [0, 1]]]
    nt.assert_equal(np.sort(u[m], -1), np.sort(edges, -1))


def test_unique_sets():
    sets = np.array([[0, 1, 2], [1, 2, 0], [0, 2, 1], [1, 2, 3]])
    e = array_to_set([[0, 1, 2], [1, 2, 3]])

    # Run without inverse
    u = unique_sets(sets)
    nt.assert_equal(len(u), len(e))
    nt.assert_equal(array_to_set(u), e)

    # Run with inverse
    u, m = unique_sets(sets, return_inverse=True)
    nt.assert_equal(len(u), len(e))
    nt.assert_equal(array_to_set(u), e)
    nt.assert_equal(np.sort(u[m], -1), np.sort(sets, -1))


def test_faces_from_sphere_vertices():
    faces = faces_from_sphere_vertices(verts)
    faces = array_to_set(faces)
    expected = array_to_set(oct_faces)
    nt.assert_equal(faces, expected)


def test_sphere_attrs():
    s = Sphere(xyz=verts)
    nt.assert_array_almost_equal(s.vertices, verts)
    nt.assert_array_almost_equal(s.x, verts[:, 0])
    nt.assert_array_almost_equal(s.y, verts[:, 1])
    nt.assert_array_almost_equal(s.z, verts[:, 2])


def test_edges_faces():
    s = Sphere(xyz=verts)
    faces = oct_faces
    nt.assert_equal(array_to_set(s.faces), array_to_set(faces))
    nt.assert_equal(array_to_set(s.edges), array_to_set(edges))

    s = Sphere(xyz=verts, faces=[[0, 1, 2]])
    nt.assert_equal(array_to_set(s.faces), array_to_set([[0, 1, 2]]))
    nt.assert_equal(array_to_set(s.edges), array_to_set([[0, 1], [1, 2], [0, 2]]))

    s = Sphere(xyz=verts, faces=[[0, 1, 2]], edges=[[0, 1]])
    nt.assert_equal(array_to_set(s.faces), array_to_set([[0, 1, 2]]))
    nt.assert_equal(array_to_set(s.edges), array_to_set([[0, 1]]))


def test_sphere_subdivide():
    sphere1 = unit_octahedron.subdivide(n=4)
    sphere2 = Sphere(xyz=sphere1.vertices)
    nt.assert_equal(sphere1.faces.shape, sphere2.faces.shape)
    nt.assert_equal(array_to_set(sphere1.faces), array_to_set(sphere2.faces))

    sphere1 = unit_icosahedron.subdivide(n=4)
    sphere2 = Sphere(xyz=sphere1.vertices)
    nt.assert_equal(sphere1.faces.shape, sphere2.faces.shape)
    nt.assert_equal(array_to_set(sphere1.faces), array_to_set(sphere2.faces))

    # It might be good to also test the vertices somehow if we can think of a
    # good test for them.


def test_sphere_find_closest():
    sphere1 = unit_octahedron.subdivide(n=4)
    for ii in range(sphere1.vertices.shape[0]):
        nt.assert_equal(sphere1.find_closest(sphere1.vertices[ii]), ii)


def test_hemisphere_find_closest():
    hemisphere1 = hemi_icosahedron.subdivide(n=4)
    for ii in range(hemisphere1.vertices.shape[0]):
        nt.assert_equal(hemisphere1.find_closest(hemisphere1.vertices[ii]), ii)
        nt.assert_equal(hemisphere1.find_closest(-hemisphere1.vertices[ii]), ii)
        nt.assert_equal(hemisphere1.find_closest(hemisphere1.vertices[ii] * 2), ii)


def test_hemisphere_subdivide():
    def flip(vertices):
        x, y, z = vertices.T
        f = (z < 0) | ((z == 0) & (y < 0)) | ((z == 0) & (y == 0) & (x < 0))
        return 1 - 2 * f[:, None]

    decimals = 6
    # Test HemiSphere.subdivide
    # Create a hemisphere by dividing a hemi-icosahedron
    hemi1 = HemiSphere.from_sphere(unit_icosahedron).subdivide(n=4)
    vertices1 = np.round(hemi1.vertices, decimals)
    vertices1 *= flip(vertices1)
    order = np.lexsort(vertices1.T)
    vertices1 = vertices1[order]

    # Create a hemisphere from a subdivided sphere
    sphere = unit_icosahedron.subdivide(n=4)
    hemi2 = HemiSphere.from_sphere(sphere)
    vertices2 = np.round(hemi2.vertices, decimals)
    vertices2 *= flip(vertices2)
    order = np.lexsort(vertices2.T)
    vertices2 = vertices2[order]

    # The two hemispheres should have the same vertices up to their order
    nt.assert_array_equal(vertices1, vertices2)

    # Create a hemisphere from vertices
    hemi3 = HemiSphere(xyz=hemi1.vertices)
    nt.assert_array_equal(hemi1.faces, hemi3.faces)
    nt.assert_array_equal(hemi1.edges, hemi3.edges)


def test_hemisphere_constructor():
    s0 = HemiSphere(xyz=verts)
    s1 = HemiSphere(theta=theta, phi=phi)
    # Unpack verts.T into x, y, z and pass as keyword arguments
    x, y, z = verts.T
    s2 = HemiSphere(x=x, y=y, z=z)

    uniq_verts = verts[::2].T
    rU, thetaU, phiU = cart2sphere(*uniq_verts)

    nt.assert_array_almost_equal(s0.theta, s1.theta)
    nt.assert_array_almost_equal(s0.theta, s2.theta)
    nt.assert_array_almost_equal(s0.theta, thetaU)

    nt.assert_array_almost_equal(s0.phi, s1.phi)
    nt.assert_array_almost_equal(s0.phi, s2.phi)
    nt.assert_array_almost_equal(s0.phi, phiU)


def test_mirror():
    verts = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, -1, -1]]
    verts = np.array(verts, "float")
    verts = verts / np.sqrt((verts * verts).sum(-1)[:, None])
    faces = [[0, 1, 3], [0, 2, 3], [1, 2, 3]]

    h = HemiSphere(xyz=verts, faces=faces)
    s = h.mirror()

    nt.assert_equal(len(s.vertices), 8)
    nt.assert_equal(len(s.faces), 6)
    verts = s.vertices

    def _angle(a, b):
        return np.arccos(np.dot(a, b))

    for triangle in s.faces:
        a, b, c = triangle
        nt.assert_(_angle(verts[a], verts[b]) <= np.pi / 2)
        nt.assert_(_angle(verts[a], verts[c]) <= np.pi / 2)
        nt.assert_(_angle(verts[b], verts[c]) <= np.pi / 2)


def test_hemisphere_faces():
    t = (1 + np.sqrt(5)) / 2
    vertices = np.array(
        [
            [-t, -1, 0],
            [-t, 1, 0],
            [1, 0, t],
            [-1, 0, t],
            [0, t, 1],
            [0, -t, 1],
        ]
    )
    vertices /= vector_norm(vertices, keepdims=True)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 4],
            [1, 3, 4],
            [2, 3, 4],
            [1, 2, 5],
            [0, 3, 5],
            [2, 3, 5],
            [0, 4, 5],
            [1, 4, 5],
        ]
    )
    edges = np.array(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 5),
        ]
    )

    h = HemiSphere(xyz=vertices)
    nt.assert_equal(len(h.edges), len(edges))
    nt.assert_equal(array_to_set(h.edges), array_to_set(edges))
    nt.assert_equal(len(h.faces), len(faces))
    nt.assert_equal(array_to_set(h.faces), array_to_set(faces))


def test_get_force():
    charges = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    force, pot = _get_forces(charges)
    nt.assert_array_almost_equal(force, 0)

    charges = np.array([[1, -0.1, 0], [1, 0, 0]])
    force, pot = _get_forces(charges)
    nt.assert_array_almost_equal(force[1, [0, 2]], 0)
    nt.assert_(force[1, 1] > 0)


def test_disperse_charges():
    charges = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    d_sphere, pot = disperse_charges(HemiSphere(xyz=charges), 10)
    nt.assert_array_almost_equal(charges, d_sphere.vertices)

    charges = np.array([[3.0 / 5, 4.0 / 5, 0], [4.0 / 5, 3.0 / 5, 0]])
    expected_charges = np.array([[0, 1.0, 0], [1.0, 0, 0]])
    d_sphere, pot = disperse_charges(HemiSphere(xyz=charges), 1000, const=0.2)
    nt.assert_array_almost_equal(expected_charges, d_sphere.vertices)
    for ii in range(1, len(pot)):
        # check that the potential of the system is going down
        nt.assert_(pot[ii] - pot[ii - 1] <= 0)

    # Check that the disperse_charges does not blow up with a large constant
    d_sphere, pot = disperse_charges(HemiSphere(xyz=charges), 1000, const=20.0)
    nt.assert_array_almost_equal(expected_charges, d_sphere.vertices)
    for ii in range(1, len(pot)):
        # check that the potential of the system is going down
        nt.assert_(pot[ii] - pot[ii - 1] <= 0)

    # check that the function seems to work with a larger number of charges
    charges = np.arange(21).reshape(7, 3)
    norms = np.sqrt((charges * charges).sum(-1))
    charges = charges / norms[:, None]
    d_sphere, pot = disperse_charges(HemiSphere(xyz=charges), 1000, const=0.05)
    for ii in range(1, len(pot)):
        # check that the potential of the system is going down
        nt.assert_(pot[ii] - pot[ii - 1] <= 0)
    # check that the resulting charges all lie on the unit sphere
    d_charges = d_sphere.vertices
    norms = np.sqrt((d_charges * d_charges).sum(-1))
    nt.assert_array_almost_equal(norms, 1)


def test_disperse_charges_alt():
    # Create a random set of points
    num_points = 3
    init_pointset = random_uniform_on_sphere(n=num_points, coords="xyz")

    # Compute the associated electrostatic potential
    init_pointset = init_pointset.reshape(init_pointset.shape[0] * 3)
    init_charges_potential = _get_forces_alt(init_pointset)

    # Disperse charges
    init_pointset = init_pointset.reshape(3, 3)
    dispersed_pointset = disperse_charges_alt(init_pointset, 10)

    # Compute the associated electrostatic potential
    dispersed_pointset = dispersed_pointset.reshape(init_pointset.shape[0] * 3)
    dispersed_charges_potential = _get_forces_alt(dispersed_pointset)

    # Verify that the potential of the optimal configuration is smaller than
    # that of the original configuration
    nt.assert_array_less(dispersed_charges_potential, init_charges_potential)


@set_random_number_generator()
def test_fibonacci_sphere(rng):
    # Test that the number of points is correct
    points = fibonacci_sphere(n_points=724, rng=rng)
    nt.assert_equal(len(points), 724)

    # Test randomization
    points1 = fibonacci_sphere(n_points=100, randomize=True, rng=rng)
    points2 = fibonacci_sphere(n_points=100, randomize=True, rng=rng)
    with nt.assert_raises(AssertionError):
        nt.assert_array_equal(points1, points2)

    # Check for near closeness to 0
    nt.assert_almost_equal(np.mean(np.mean(points, axis=0)), 0, decimal=2)


@set_random_number_generator()
def test_fibonacci_hemisphere(rng):
    # Test that the number of points is correct
    points = fibonacci_sphere(n_points=724, hemisphere=True, rng=rng)
    nt.assert_equal(len(points), 724)

    # Test randomization
    points1 = fibonacci_sphere(n_points=100, hemisphere=True, randomize=True, rng=rng)
    points2 = fibonacci_sphere(n_points=100, hemisphere=True, randomize=True, rng=rng)
    with nt.assert_raises(AssertionError):
        nt.assert_array_equal(points1, points2)

    # Check for near closeness to 0
    nt.assert_almost_equal(np.mean(points, axis=0)[2], 0, decimal=2)
