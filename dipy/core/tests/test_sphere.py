import numpy as np
import numpy.testing as nt
import warnings

from dipy.core.sphere import (Sphere, HemiSphere, unique_edges, unique_sets,
                              faces_from_sphere_vertices, HemiSphere,
                              disperse_charges, _get_forces)
from dipy.core.subdivide_octahedron import (create_unit_sphere,
                                            octahedron_vertices,
                                            octahedron_edges,
                                            octahedron_triangles)
from dipy.core.geometry import cart2sphere, sphere2cart

verts, edges, sides = octahedron_vertices, octahedron_edges, octahedron_triangles
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


def test_bad_edges_faces():
    nt.assert_raises(ValueError, Sphere, xyz=[0, 0, 1.5], edges=[[1, 2]])


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


def array_to_set(a):
    return set(frozenset(i) for i in a)


def test_unique_edges():
    u = unique_edges([[0, 1, 2],
                      [1, 2, 0]])
    u = array_to_set(u)

    e = array_to_set([[1, 2],
                      [0, 1],
                      [0, 2]])

    nt.assert_equal(e, u)


def test_unique_sets():
    u = unique_sets([[0, 1, 2],
                     [1, 2, 0],
                     [0, 2, 1],
                     [1, 2, 3]])

    e = array_to_set([[0, 1, 2],
                      [1, 2, 3]])

    nt.assert_equal(len(u), len(e))
    nt.assert_equal(array_to_set(u), e)


def test_faces_from_sphere_vertices():
    faces = faces_from_sphere_vertices(verts)
    faces = array_to_set(faces)
    expected = array_to_set(edges[sides, 0])
    nt.assert_equal(faces, expected)


def test_sphere_attrs():
    s = Sphere(xyz=verts)
    nt.assert_array_almost_equal(s.vertices, verts)
    nt.assert_array_almost_equal(s.x, verts[:, 0])
    nt.assert_array_almost_equal(s.y, verts[:, 1])
    nt.assert_array_almost_equal(s.z, verts[:, 2])


def test_edges_faces():
    s = Sphere(xyz=verts)
    faces = edges[sides, 0]
    nt.assert_equal(array_to_set(s.faces), array_to_set(faces))
    nt.assert_equal(array_to_set(s.edges), array_to_set(edges))

    s = Sphere(xyz=verts, faces=[[0, 1, 2]])
    nt.assert_equal(array_to_set(s.faces), array_to_set([[0, 1, 2]]))
    nt.assert_equal(array_to_set(s.edges),
                    array_to_set([[0, 1], [1, 2], [0, 2]]))

    s = Sphere(xyz=verts, faces=[[0, 1, 2]], edges=[[0, 1]])
    nt.assert_equal(array_to_set(s.faces), array_to_set([[0, 1, 2]]))
    nt.assert_equal(array_to_set(s.edges),
                    array_to_set([[0, 1]]))


def test_hemisphere_constructor():
    s0 = HemiSphere(xyz=verts)
    s1 = HemiSphere(theta=theta, phi=phi)
    s2 = HemiSphere(*verts.T)

    uniq_verts = verts[::2].T
    rU, thetaU, phiU = cart2sphere(*uniq_verts)

    nt.assert_array_almost_equal(s0.theta, s1.theta)
    nt.assert_array_almost_equal(s0.theta, s2.theta)
    nt.assert_array_almost_equal(s0.theta, thetaU)

    nt.assert_array_almost_equal(s0.phi, s1.phi)
    nt.assert_array_almost_equal(s0.phi, s2.phi)
    nt.assert_array_almost_equal(s0.phi, phiU)


def test_mirror():
    verts = [[0, 0, 1],
             [0, 1, 0],
             [1, 0, 0],
             [-1, -1, -1]]
    verts = np.array(verts, 'float')
    verts /= (verts * verts).sum(-1)[:, None]
    faces = [[0, 1, 3],
             [0, 2, 3],
             [1, 2, 3]]

    h = HemiSphere(xyz=verts, faces=faces)
    s = h.mirror()

    nt.assert_equal(len(s.vertices), 8)
    nt.assert_equal(len(s.faces), 6)
    verts = s.vertices

    def _angle(a, b):
        return np.arccos(a.dot(b))

    for triangle in s.faces:
        a, b, c = triangle
        nt.assert_(_angle(verts[a], verts[b]) <= np.pi/2)
        nt.assert_(_angle(verts[a], verts[c]) <= np.pi/2)
        nt.assert_(_angle(verts[b], verts[c]) <= np.pi/2)


def test_hemisphere_faces():
    sphere = create_unit_sphere(2)
    faces = sphere.faces[::2] // 2
    h = HemiSphere(xyz=sphere.vertices)

    nt.assert_equal(len(h.faces), len(faces))
    nt.assert_equal(len(h.faces), len(array_to_set(h.faces)))
    nt.assert_equal(array_to_set(h.faces), array_to_set(faces))

def test_get_force():
    charges = np.array([[1., 0, 0],
                        [0, 1., 0],
                        [0, 0, 1.]])
    force, pot = _get_forces(charges)
    nt.assert_array_almost_equal(force, 0)

    charges = np.array([[1, -.1, 0],
                        [1, 0, 0]])
    force, pot = _get_forces(charges)
    nt.assert_array_almost_equal(force[1, [0, 2]], 0)
    nt.assert_(force[1, 1] > 0)

def test_disperse_charges():
    charges = np.array([[1., 0, 0],
                        [0, 1., 0],
                        [0, 0, 1.]])
    d_sphere, pot = disperse_charges(HemiSphere(xyz=charges), 10)
    nt.assert_array_almost_equal(charges, d_sphere.vertices)

    a = np.sqrt(3)/2
    charges = np.array([[3./5, 4./5, 0],
                        [4./5, 3./5, 0]])
    expected_charges =  np.array([[0, 1., 0],
                                  [1., 0, 0]])
    d_sphere, pot = disperse_charges(HemiSphere(xyz=charges), 1000, .2)
    nt.assert_array_almost_equal(expected_charges, d_sphere.vertices)
    for ii in xrange(1, len(pot)):
        #check that the potential of the system is either going down or
        #stayting almost the same
        nt.assert_(pot[ii] - pot[ii-1] < 1e-12)

    #check that the function seems to work with a larger number of charges
    charges = np.arange(21).reshape(7,3)
    norms = np.sqrt((charges*charges).sum(-1))
    charges = charges / norms[:, None]
    d_sphere, pot = disperse_charges(HemiSphere(xyz=charges), 1000, .05)
    for ii in xrange(1, len(pot)):
        #check that the potential of the system is either going down or
        #stayting almost the same
        nt.assert_(pot[ii] - pot[ii-1] < 1e-12)
    #check that the resulting charges all lie on the unit sphere
    d_charges = d_sphere.vertices
    norms = np.sqrt((d_charges*d_charges).sum(-1))
    nt.assert_array_almost_equal(norms, 1)

if __name__ == "__main__":
    nt.run_module_suite()
