__all__ = ['Sphere', 'HemiSphere', 'faces_from_sphere_vertices', 'unique_edges',
           'unique_faces', 'reduce_antipodal']

import numpy as np
import warnings

from dipy.core.geometry import cart2sphere, sphere2cart
from dipy.core.onetime import auto_attr
from dipy.reconst.recspeed import remove_similar_vertices


def _all_specified(*args):
    for a in args:
        if a is None:
            return False
    return True


def _some_specified(*args):
    for a in args:
        if a is not None:
            return True
    return False


def faces_from_sphere_vertices(vertices):
    """
    Triangulate a set of vertices on the sphere.

    Parameters
    ----------
    vertices : (M, 3) ndarray
        XYZ coordinates of vertices on the sphere.

    Returns
    -------
    faces : (N, 3) ndarray
        Indices into vertices; forms triangular faces.

    """
    from scipy.spatial import Delaunay
    return Delaunay(vertices).convex_hull


def unique_edges(faces):
    """Extract all unique edges from given triangular faces.

    Parameters
    ----------
    faces : (N, 3) ndarray
        Vertex indices forming triangular faces.

    Returns
    -------
    edges : (N, 2) ndarray
        Unique edges.

    """
    edges = set()
    for a, b, c in faces:
        edges.add(frozenset((a, b)))
        edges.add(frozenset((a, c)))
        edges.add(frozenset((b, c)))
    edges = [tuple(e) for e in edges]
    return np.array(edges)


def unique_sets(sets):
    """Remove duplicate sets.

    Parameters
    ----------
    sets : array (N, k)
        N sets of size k.

    Return
    ------
    sets : array
        Unique sets.

    """
    sets = set(frozenset(s) for s in sets)
    sets = [tuple(s) for s in sets]
    return np.array(sets)


def reduce_antipodal(points, faces, tol=0):
    # Check that points have expected symmetry
    n = len(points) // 2
    if not np.allclose(points[:n], -points[n:], tol):
        raise ValueError("points don't have the expected symmetry")
    new_points = points[:n]
    new_faces = unique_faces(faces % n)
    new_edges = unique_edges(new_faces)
    return new_points, new_edges, new_faces


class Sphere(object):
    """Points on the unit sphere.

    The sphere can be constructed using one of three conventions::

      Sphere(x, y, z)
      Sphere(xyz=xyz)
      Sphere(theta=theta, phi=phi)

    Parameters
    ----------
    x, y, z : 1-D array_like
        Vertices as x-y-z coordinates.
    theta, phi : 1-D array_like
        Vertices as spherical coordinates.  Theta and phi are the inclination
        and azimuth angles respectively.
    xyz : (N, 3) ndarray
        Vertices as x-y-z coordinates.

    faces : (N, 3) ndarray
        Indices into vertices that form triangular faces.  If unspecified,
        the faces are computed using a Delaunay triangulation.
    edges : (N, 2) ndarray
        Edges between vertices.  If unspecified, the edges are
        derived from the faces.

    """

    def __init__(self, x=None, y=None, z=None,
                 theta=None, phi=None,
                 xyz=None,
                 faces=None, edges=None):

        all_specified = _all_specified(x, y, z) + _all_specified(xyz) + \
                        _all_specified(theta, phi)
        one_complete = _some_specified(x, y, z) + _some_specified(xyz) + \
                       _some_specified(theta, phi)

        if not (all_specified == 1 and one_complete == 1):
            raise ValueError("Sphere must be constructed using either "
                             "(x,y,z), (theta, phi) or xyz.")

        if edges is not None and faces is None:
            raise ValueError("Either specify both faces and "
                             "edges, only faces, or neither.")

        if edges is not None:
            self.edges = np.asarray(edges)
        if faces is not None:
            self.faces = np.asarray(faces)

        if theta is not None:
            self.theta, self.phi = np.asarray(theta), np.asarray(phi)
            return

        if xyz is not None:
            xyz = np.asarray(xyz)
            x, y, z = xyz.T

        x, y, z = (np.asarray(t) for t in (x, y, z))
        r, self.theta, self.phi = cart2sphere(x, y, z)

        if not np.allclose(r, 1):
            warnings.warn("Vertices are not on the unit sphere.")

    @auto_attr
    def vertices(self):
        return np.column_stack(sphere2cart(1, self.theta, self.phi))

    @property
    def x(self):
        return self.vertices[:, 0]

    @property
    def y(self):
        return self.vertices[:, 1]

    @property
    def z(self):
        return self.vertices[:, 2]

    @auto_attr
    def faces(self):
        return faces_from_sphere_vertices(self.vertices)

    @auto_attr
    def edges(self):
        return unique_edges(self.faces)


class HemiSphere(Sphere):
    """Points on the unit sphere.

    A HemiSphere is similar to a Sphere but it takes anitpodal symmetry into
    account. Antipodal symmetry means that point v on a HemiSphere is the same
    as the point -v. Duplicate points are discarded when constructing a
    HemiSphere (including antipodal duplicates). `edges` and `faces` are
    remapped to the remaining points as closely as possible.

    The HemiSphere can be constructed using one of three conventions::

      HemiSphere(x, y, z)
      HemiSphere(xyz=xyz)
      HemiSphere(theta=theta, phi=phi)

    Parameters
    ----------
    x, y, z : 1-D array_like
        Vertices as x-y-z coordinates.
    theta, phi : 1-D array_like
        Vertices as spherical coordinates.  Theta and phi are the inclination
        and azimuth angles respectively.
    xyz : (N, 3) ndarray
        Vertices as x-y-z coordinates.

    faces : (N, 3) ndarray
        Indices into vertices that form triangular faces.  If unspecified,
        the faces are computed using a Delaunay triangulation.
    edges : (N, 2) ndarray
        Edges between vertices.  If unspecified, the edges are
        derived from the faces.
    tol : float
        Angle in degrees. Vertices that are less than tol degrees apart are
        treated as duplicates.

    See Also
    --------
    Sphere

    """
    def __init__(self, x=None, y=None, z=None,
                 theta=None, phi=None,
                 xyz=None,
                 faces=None, edges=None, tol=1e-5):
        """Create a HemiSphere from points"""

        sphere = Sphere(x=x, y=y, z=z, theta=theta, phi=phi, xyz=xyz)
        uniq_vertices, mapping = remove_similar_vertices(sphere.vertices, tol)
        if faces is not None:
            faces = np.asarray(faces)
            faces = unique_sets(mapping[faces])
        if edges is not None:
            edges = np.asarray(edges)
            edges = unique_sets(mapping[edges])
        Sphere.__init__(self, xyz=uniq_vertices, edges=edges, faces=faces)

    def mirror(self):
        """Create a full Sphere from a HemiSphere"""
        n = len(self.vertices)
        vertices = np.vstack([self.vertices, -self.vertices])

        edges = np.vstack([self.edges, n + self.edges])
        _switch_vertex(edges[:,0], edges[:,1], vertices)

        faces = np.vstack([self.faces, n + self.faces])
        _switch_vertex(faces[:,0], faces[:,1], vertices)
        _switch_vertex(faces[:,0], faces[:,2], vertices)
        return Sphere(xyz=vertices, edges=edges, faces=faces)

    @auto_attr
    def faces(self):
        vertices = np.vstack([self.vertices, -self.vertices])
        faces = faces_from_sphere_vertices(vertices)
        return unique_sets(faces % len(self.vertices))

def _switch_vertex(index1, index2, vertices):
    """When we mirror an edge (a, b). We can either create (a, b) and (a', b')
    OR (a, b') and (a', b). The angles of edges (a, b) and (a, b') are
    supplementary, so we choose the two new edges such that their angles are
    less than 90 degrees.
    """
    n = len(vertices)
    A = vertices[index1]
    B = vertices[index2]
    is_far = (A * B).sum(-1) < 0
    index2 += n/2 * (is_far)
    index2 %= n

