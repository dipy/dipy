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


def reduce_antipodal(points, faces, tol=1e-5):
    hs = HemiSphere(xyz=points, faces=faces, tol=tol)
    return hs.vertices, hs.edges, hs.faces

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

    A HemiSphere is similar to a Sphere but it takes antipodal symmetry into
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

    @classmethod
    def from_sphere(klass, sphere, tol=1e-5):
        """Create instance from a Sphere"""
        return klass(theta=sphere.theta, phi=shere.phi,
                     edges=sphere.edges, faces=sphere.faces, tol=tol)

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
    index2[is_far] += n/2
    index2 %= n


def _get_forces(charges):
    r"""Given a set of charges on the surface of the sphere gets total force
    those charges exert on each other.

    The force exerted by one charge on another is given by Coulomb's law. For
    this simulation we use charges of equal magnitude so this force can be
    written as $\vec{r}/r^3$, up to a constant factor, where $\vec{r}$ is the
    separation of the two charges and $r$ is the magnitude of $\vec{r}$. Forces
    are additive so the total force on each of the charges is the sum of the
    force exerted by each other charge in the system. Charges do not exert a
    force on themselves. The electric potential can similarly be written as
    $1/r$ and is also additive.
    """

    all_charges = np.concatenate((charges, -charges))
    all_charges = all_charges[:, None]
    r = charges - all_charges
    r_mag = np.sqrt((r*r).sum(-1))[:, :, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        force = r / r_mag**3
        potential = 1. / r_mag

    d = np.arange(len(charges))
    force[d,d] = 0
    force = force.sum(0)
    force_r_comp = (charges*force).sum(-1)[:, None]
    f_theta = force - force_r_comp*charges
    potential[d,d] = 0
    potential = 2*potential.sum()
    return f_theta, potential

def disperse_charges(hemi, iters, const=.05):
    """Models electrostatic repulsion on the unit sphere

    Places charges on a sphere and simulates the repulsive forces felt by each
    one. Allows the charges to move for some number of iterations and returns
    their final location as well as the total potential of the system at each
    step.

    Parameters
    ----------
    hemi : HemiSphere
        Points on a unit sphere
    iters : int
        Number of iterations to run
    const : float
        Using a smaller const could provide a more accurate result, but will
        need more iterations to converge.

    Returns
    -------
    hemi : HemiSphere
        distributed points on a unit sphere
    potential : ndarray
        The electrostatic potential at each iteration. This can be useful to
        check if the repulsion converged to a minimum.

    Note:
    -----
    This function is meant to be used with diffusion imaging so antipodal
    symmetry is assumed. Therefor each charge must not only be unique, but if
    there is a charge at +x, there cannot be a charge at -x. These are treated
    as the same location and because the distance between the two charges will
    be zero, the result will be unstable.
    """
    if not isinstance(hemi, HemiSphere):
        raise ValueError("expecting HemiSphere")
    charges = hemi.vertices.copy()
    forces, v = _get_forces(charges)
    force_mag = np.sqrt((forces*forces).sum())
    max_force = force_mag.max()
    if max_force > 1:
        const = const/max_force
    potential = np.empty(iters)

    for ii in xrange(iters):
        forces, potential[ii] = _get_forces(charges)
        charges += forces * const
        norms = np.sqrt((charges*charges).sum(-1))
        charges /= norms[:, None]
    return HemiSphere(xyz=charges), potential


def interp_rbf(data, sphere_origin, sphere_target,
               rbf='multiquadric', epsilon=None):
    """Interpolate data on the sphere, using radial basis functions.

    Parameters
    ----------
    data : (N,) ndarray
        Function values on the unit sphere.
    sphere_origin : Sphere
        Positions of data values.
    sphere_target : Sphere
        M target positions for which to interpolate.

    family : {'multiquadric', 'inverse', 'gaussian'}
        Radial basis function.
    epsilon : float
        Radial basis function spread parameter.

    Returns
    -------
    v : (M,) ndarray
        Interpolated values.

    See Also
    --------
    scipy.interpolate.Rbf

    """
    from scipy.interpolate import Rbf

    # Workaround for bug in SciPy that doesn't allow
    # specification of epsilon None
    if epsilon is not None:
        kwargs = {'function': rbf,
                  'epsilon': epsilon}
    else:
        kwargs = {'function': rbf}

    rbfi = Rbf(sphere_origin.x, sphere_origin.y, sphere_origin.z, data,
               **kwargs)
    return rbfi(sphere_target.x, sphere_target.y, sphere_target.z)
