__all__ = ['Sphere', 'HemiSphere', 'faces_from_sphere_vertices', 'unique_edges',
           'unique_faces']

import numpy as np
import warnings

from dipy.core.geometry import cart2sphere, sphere2cart
from dipy.core.auto_attr import auto_attr


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


def unique_faces(faces):
    """Remove duplicate faces.

    Parameters
    ----------
    faces : (N, 3) ndarray
        Vertex indices forming triangular faces.

    Returns
    -------
    faces : (N, 3) ndarray
        Unique faces.

    """
    faces = set(frozenset(f) for f in faces)
    faces = [tuple(f) for f in faces]
    return np.array(faces)


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
                 edges=None, faces=None):

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

        self._edges = edges
        self._faces = faces

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
    def faces(self):
        pass

    @auto_attr
    def edges(self):
        pass


class HemiSphere(Sphere):
    pass

