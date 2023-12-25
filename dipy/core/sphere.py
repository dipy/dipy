import numpy as np
import warnings

from scipy import optimize

from dipy.core.geometry import cart2sphere, sphere2cart, vector_norm
from dipy.core.onetime import auto_attr
from dipy.reconst.recspeed import remove_similar_vertices

__all__ = ['Sphere', 'HemiSphere', 'faces_from_sphere_vertices',
           'unique_edges']


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
    faces = Delaunay(vertices).convex_hull
    if len(vertices) < 2**16:
        return np.asarray(faces, np.uint16)
    else:
        return faces


def unique_edges(faces, return_mapping=False):
    """Extract all unique edges from given triangular faces.

    Parameters
    ----------
    faces : (N, 3) ndarray
        Vertex indices forming triangular faces.
    return_mapping : bool
        If true, a mapping to the edges of each face is returned.

    Returns
    -------
    edges : (N, 2) ndarray
        Unique edges.
    mapping : (N, 3)
        For each face, [x, y, z], a mapping to it's edges [a, b, c].
        ::

                y
                /\
               /  \
             a/    \b
             /      \
            /        \
           /__________\
          x      c     z

    """
    faces = np.asarray(faces)
    edges = np.concatenate([faces[:, 0:2], faces[:, 1:3], faces[:, ::2]])
    if return_mapping:
        ue, inverse = unique_sets(edges, return_inverse=True)
        return ue, inverse.reshape((3, -1)).T
    else:
        return unique_sets(edges)


def unique_sets(sets, return_inverse=False):
    """Remove duplicate sets.

    Parameters
    ----------
    sets : array (N, k)
        N sets of size k.
    return_inverse : bool
        If True, also returns the indices of unique_sets that can be used
        to reconstruct `sets` (the original ordering of each set may not be
        preserved).

    Returns
    -------
    unique_sets : array
        Unique sets.
    inverse : array (N,)
        The indices to reconstruct `sets` from `unique_sets`.

    """
    sets = np.sort(sets, 1)
    order = np.lexsort(sets.T)
    sets = sets[order]
    flag = np.ones(len(sets), 'bool')
    flag[1:] = (sets[1:] != sets[:-1]).any(-1)
    uniqsets = sets[flag]
    if return_inverse:
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        index = flag.cumsum() - 1
        return uniqsets, index[inverse]
    else:
        return uniqsets


class Sphere:
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
        one_complete = (_some_specified(x, y, z) + _some_specified(xyz) +
                        _some_specified(theta, phi))

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
            self.theta = np.array(theta, copy=False, ndmin=1)
            self.phi = np.array(phi, copy=False, ndmin=1)
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
        faces = faces_from_sphere_vertices(self.vertices)
        return faces

    @auto_attr
    def edges(self):
        return unique_edges(self.faces)

    def subdivide(self, n=1):
        r"""Subdivides each face of the sphere into four new faces.

        New vertices are created at a, b, and c. Then each face [x, y, z] is
        divided into faces [x, a, c], [y, a, b], [z, b, c], and [a, b, c].

        ::

                y
                /\
               /  \
             a/____\b
             /\    /\
            /  \  /  \
           /____\/____\
          x      c     z

        Parameters
        ----------
        n : int, optional
            The number of subdivisions to perform.

        Returns
        -------
        new_sphere : Sphere
            The subdivided sphere.

        """
        vertices = self.vertices
        faces = self.faces
        for _ in range(n):
            edges, mapping = unique_edges(faces, return_mapping=True)
            new_vertices = vertices[edges].sum(1)
            new_vertices /= vector_norm(new_vertices, keepdims=True)
            mapping += len(vertices)
            vertices = np.vstack([vertices, new_vertices])

            x, y, z = faces.T
            a, b, c = mapping.T
            face1 = np.column_stack([x, a, c])
            face2 = np.column_stack([y, b, a])
            face3 = np.column_stack([z, c, b])
            face4 = mapping
            faces = np.concatenate([face1, face2, face3, face4])

        if len(vertices) < 2**16:
            faces = np.asarray(faces, dtype='uint16')
        return Sphere(xyz=vertices, faces=faces)

    def find_closest(self, xyz):
        """
        Find the index of the vertex in the Sphere closest to the input vector

        Parameters
        ----------
        xyz : array-like, 3 elements
            A unit vector

        Returns
        -------
        idx : int
            The index into the Sphere.vertices array that gives the closest
            vertex (in angle).
        """
        cos_sim = np.dot(self.vertices, xyz)
        return np.argmax(cos_sim)


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
        uniq_vertices, mapping = remove_similar_vertices(sphere.vertices, tol,
                                                         return_mapping=True)
        uniq_vertices *= 1 - 2*(uniq_vertices[:, -1:] < 0)
        if faces is not None:
            faces = np.asarray(faces)
            faces = unique_sets(mapping[faces])
        if edges is not None:
            edges = np.asarray(edges)
            edges = unique_sets(mapping[edges])
        Sphere.__init__(self, xyz=uniq_vertices, edges=edges, faces=faces)

    @classmethod
    def from_sphere(cls, sphere, tol=1e-5):
        """Create instance from a Sphere"""
        return cls(theta=sphere.theta, phi=sphere.phi,
                   edges=sphere.edges, faces=sphere.faces, tol=tol)

    def mirror(self):
        """Create a full Sphere from a HemiSphere"""
        n = len(self.vertices)
        vertices = np.vstack([self.vertices, -self.vertices])

        edges = np.vstack([self.edges, n + self.edges])
        _switch_vertex(edges[:, 0], edges[:, 1], vertices)

        faces = np.vstack([self.faces, n + self.faces])
        _switch_vertex(faces[:, 0], faces[:, 1], vertices)
        _switch_vertex(faces[:, 0], faces[:, 2], vertices)
        return Sphere(xyz=vertices, edges=edges, faces=faces)

    @auto_attr
    def faces(self):
        vertices = np.vstack([self.vertices, -self.vertices])
        faces = faces_from_sphere_vertices(vertices)
        return unique_sets(faces % len(self.vertices))

    def subdivide(self, n=1):
        """Create a more subdivided HemiSphere

        See Sphere.subdivide for full documentation.

        """
        sphere = self.mirror()
        sphere = sphere.subdivide(n)
        return HemiSphere.from_sphere(sphere)

    def find_closest(self, xyz):
        """
        Find the index of the vertex in the Sphere closest to the input vector,
        taking into account antipodal symmetry

        Parameters
        ----------
        xyz : array-like, 3 elements
            A unit vector

        Returns
        -------
        idx : int
            The index into the Sphere.vertices array that gives the closest
            vertex (in angle).
        """
        cos_sim = abs(np.dot(self.vertices, xyz))
        return np.argmax(cos_sim)


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
    index2[is_far] = index2[is_far] + (n / 2.0)
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
    force[d, d] = 0
    force = force.sum(0)
    force_r_comp = (charges*force).sum(-1)[:, None]
    f_theta = force - force_r_comp*charges
    potential[d, d] = 0
    potential = 2*potential.sum()
    return f_theta, potential


def disperse_charges(hemi, iters, const=.2):
    """Models electrostatic repulsion on the unit sphere

    Places charges on a sphere and simulates the repulsive forces felt by each
    one. Allows the charges to move for some number of iterations and returns
    their final location as well as the total potential of the system at each
    step.

    Parameters
    ----------
    hemi : HemiSphere
        Points on a unit sphere.
    iters : int
        Number of iterations to run.
    const : float
        Using a smaller const could provide a more accurate result, but will
        need more iterations to converge.

    Returns
    -------
    hemi : HemiSphere
        Distributed points on a unit sphere.
    potential : ndarray
        The electrostatic potential at each iteration. This can be useful to
        check if the repulsion converged to a minimum.

    Notes
    -----
    This function is meant to be used with diffusion imaging so antipodal
    symmetry is assumed. Therefore, each charge must not only be unique, but if
    there is a charge at +x, there cannot be a charge at -x. These are treated
    as the same location and because the distance between the two charges will
    be zero, the result will be unstable.
    """
    if not isinstance(hemi, HemiSphere):
        raise ValueError("expecting HemiSphere")
    charges = hemi.vertices
    forces, v = _get_forces(charges)
    force_mag = np.sqrt((forces*forces).sum())
    const = const / force_mag.max()
    potential = np.empty(iters)
    v_min = v

    for ii in range(iters):
        new_charges = charges + forces * const
        norms = np.sqrt((new_charges**2).sum(-1))
        new_charges /= norms[:, None]
        new_forces, v = _get_forces(new_charges)
        if v <= v_min:
            charges = new_charges
            forces = new_forces
            potential[ii] = v_min = v
        else:
            const /= 2.
            potential[ii] = v_min

    return HemiSphere(xyz=charges), potential


def fibonacci_sphere(n_points, randomize=True):
    """
    Generate points on the surface of a sphere using Fibonacci Spiral.

    Parameters
    ----------
    n_points : int
        The number of points to generate on the sphere surface.
    randomize : bool, optional
        If True, randomize the starting point on the sphere. Default is True.

    Returns
    -------
    points : ndarray
        An array of 3D points representing coordinates on the sphere surface.

    """
    if not isinstance(n_points, int) or n_points <= 4:
        raise ValueError(
            "Number of points must be a positive integer greater than 4."
        )

    random_shift = 0
    if randomize:
        random_generator = np.random.default_rng()
        random_shift = random_generator.integers(0, n_points)

    indices = np.arange(n_points)
    offset = 2.0 / n_points
    increment = np.pi * (3. - np.sqrt(5.))

    y = ((indices * offset) - 1) + (offset / 2)
    r = np.sqrt(1 - y**2)
    phi = ((indices + random_shift) % n_points) * increment

    x = np.cos(phi) * r
    z = np.sin(phi) * r

    points = np.column_stack((x, y, z))
    return points


def _equality_constraints(vects):
    """Spherical equality constraint. Returns 0 if vects lies on the unit
    sphere. Note that a flattened array is returned because `scipy.optimize`
    expects a 1-D array.

    Parameters
    ----------
    vects : array-like shape (N * 3)
        Points on the sphere.
    Returns
    -------
    array-like (N,)
        Difference between squared vector norms and 1.

    """

    N = vects.shape[0] // 3
    vects = vects.reshape((N, 3))
    return (vects ** 2).sum(1) - 1.0

def _grad_equality_constraints(vects):
    r"""Return normals to the surface constraint (which corresponds to
    the gradient of the implicit function).

    Parameters
    ----------
    vects : array-like (N * 3)
        Points on the sphere.
    Returns
    -------
    array-like (N, N * 3)
        grad[i, j] contains :math:`\partial f_i / \partial x_j`.

    """

    N = vects.shape[0] // 3
    vects = vects.reshape((N, 3))
    vects = (vects.T / np.sqrt((vects ** 2).sum(1))).T
    grad = np.zeros((N, N * 3))
    for i in range(3):
        grad[:, i * N:(i + 1) * N] = np.diag(vects[:, i])
    return grad


def _get_forces_alt(vects, alpha=2.0, **kwargs):
    """Electrostatic-repulsion objective function. The alpha parameter
    controls the power repulsion (energy varies as $1 / r^\alpha$) [1]_. For
    $\alpha = 1.0$, this corresponds to electrostatic interaction energy.
    The weights ensure equal importance of each shell to the objective
    function [2]_ [3]_.

    Parameters
    ----------
    vects : array-like (N * 3,)
        Points on the sphere.
    alpha : float
        Controls the power of the repulsion. Default is 1.0.
    weights : array-like (N, N)
        Weight values to the electrostatic energy.

    Returns
    -------
    energy : float
        Sum of all interactions between any two vectors.

    References
    ----------
    .. [1] Papadakis, N. G., et al. "Minimal gradient encoding for robust
           estimation of diffusion anisotropy." Magnetic Resonance Imaging
           2000 Jul; 18(6): 671-679.
    .. [2] Cook, P. A., Symms, M. Boulby, P. A., Alexander, D. C. "Optimal
           acquisition orders of diffusion‐weighted MRI measurements." Journal
           of Magnetic Resonance Imaging 2007 Apr; 25(5): 1051-1058.
    .. [3] Caruyer, E., Lenglet, C., Sapiro, G. and Deriche, R. "Design of
           multishell sampling schemes with uniform coverage in diffusion
           MRI." Magnetic Resonance in Medicine 2013 Jun; 69(6): 1534-1540.

    """

    nb_points = vects.shape[0] // 3
    weights = kwargs.get('weights', np.ones((nb_points, nb_points)))
    charges = vects.reshape((nb_points, 3))
    all_charges = np.concatenate((charges, -charges))
    all_charges = all_charges[:, None]
    r = charges - all_charges
    r_mag = np.sqrt((r * r).sum(-1))[:, :, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        potential = 1 / r_mag ** alpha

    d = np.arange(len(charges))
    potential[d, d] = 0
    potential = potential[:nb_points] + potential[nb_points:]
    potential = weights * potential.sum(-1)
    potential = potential.sum()
    return potential


def _get_grad_forces_alt(vects, alpha=2.0, **kwargs):
    """1st-order derivative of electrostatic-like repulsion energy [1]_.
    The weights ensure equal importance of each shell to the objective
    function [2]_ [3]_.

    Parameters
    ----------
    vects : array-like (N * 3,)
        Points on the sphere.
    alpha : float
        Controls the power of the repulsion. Default is 1.0.
    weights : array-like (N, N)
        Weight values to the electrostatic energy.

    Returns
    -------
    grad : array-like (N * 3,)
        Gradient of the objective function.

    References
    ----------
    .. [1] Papadakis, N. G., et al. "Minimal gradient encoding for robust
           estimation of diffusion anisotropy." Magnetic Resonance Imaging
           2000 Jul; 18(6): 671-679.
    .. [2] Cook, P. A., Symms, M. Boulby, P. A., Alexander, D. C. "Optimal
           acquisition orders of diffusion‐weighted MRI measurements." Journal
           of Magnetic Resonance Imaging 2007 Apr; 25(5): 1051-1058.
    .. [3] Caruyer, E., Lenglet, C., Sapiro, G. and Deriche, R. "Design of
           multishell sampling schemes with uniform coverage in diffusion
           MRI." Magnetic Resonance in Medicine 2013 Jun; 69(6): 1534-1540.

    """

    nb_points = vects.shape[0] // 3
    weights = kwargs.get('weights', np.ones((nb_points, nb_points)))
    charges = vects.reshape((nb_points, 3))
    all_charges = np.concatenate((charges, -charges))
    all_charges = all_charges[:, None]
    r = charges - all_charges
    r_mag = np.sqrt((r * r).sum(-1))[:, :, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forces = -2 * alpha * r / r_mag ** (alpha + 2.)

    d = np.arange(len(charges))
    forces[d, d] = 0
    forces = forces[:nb_points] + forces[nb_points:]
    forces = forces * weights.reshape((nb_points, nb_points, 1))
    forces = forces.sum(0)
    return forces.reshape((nb_points * 3))


def disperse_charges_alt(init_pointset, iters, tol=1.0e-3):
    """Reimplementation of disperse_charges making use of
    `scipy.optimize.fmin_slsqp`.

    Parameters
    ----------
    init_pointset : (N, 3) ndarray
        Points on a unit sphere.
    iters : int
        Number of iterations to run.
    tol : float
        Tolerance for the optimization.

    Returns
    -------
    array-like (N, 3)
        Distributed points on a unit sphere.

    """

    K = init_pointset.shape[0]
    vects = optimize.fmin_slsqp(_get_forces_alt, init_pointset.reshape(K * 3),
                                f_eqcons=_equality_constraints,
                                fprime=_get_grad_forces_alt, iter=iters,
                                acc=tol, args=(), iprint=0)
    return vects.reshape((K, 3))


def euler_characteristic_check(sphere, chi=2):
    r"""Checks the euler characteristic of a sphere

    If $f$ = number of faces, $e$ = number_of_edges and $v$ = number of
    vertices, the Euler formula says $f-e+v = 2$ for a mesh on a sphere. More
    generally, whether $f -e + v == \chi$ where $\chi$ is the Euler
    characteristic of the mesh.

    - Open chain (track) has $\chi=1$
    - Closed chain (loop) has $\chi=0$
    - Disk has $\chi=1$
    - Sphere has $\chi=2$
    - HemiSphere has $\chi=1$

    Parameters
    ----------
    sphere : Sphere
        A Sphere instance with vertices, edges and faces attributes.
    chi : int, optional
       The Euler characteristic of the mesh to be checked

    Returns
    -------
    check : bool
       True if the mesh has Euler characteristic $\chi$

    Examples
    --------
    >>> euler_characteristic_check(unit_octahedron)
    True
    >>> hemisphere = HemiSphere.from_sphere(unit_icosahedron)
    >>> euler_characteristic_check(hemisphere, chi=1)
    True

    """
    v = sphere.vertices.shape[0]
    e = sphere.edges.shape[0]
    f = sphere.faces.shape[0]
    return (f - e + v) == chi


octahedron_vertices = np.array(
    [[1.0, 0.0, 0.0],
     [-1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, -1.0, 0.0],
     [0.0,  0.0, 1.0],
     [0.0,  0.0, -1.0], ])
octahedron_faces = np.array(
    [[0, 4, 2],
     [1, 5, 3],
     [4, 2, 1],
     [5, 3, 0],
     [1, 4, 3],
     [0, 5, 2],
     [0, 4, 3],
     [1, 5, 2], ], dtype='uint16')

t = (1 + np.sqrt(5)) / 2
icosahedron_vertices = np.array(
    [[t,  1,  0],     # 0
     [-t,  1,  0],    # 1
     [t, -1,  0],     # 2
     [-t, -1,  0],    # 3
     [1,  0,  t],     # 4
     [1,  0, -t],     # 5
     [-1,  0,  t],    # 6
     [-1,  0, -t],    # 7
     [0,  t,  1],     # 8
     [0, -t,  1],     # 9
     [0,  t, -1],     # 10
     [0, -t, -1], ])   # 11

icosahedron_vertices /= vector_norm(icosahedron_vertices, keepdims=True)
icosahedron_faces = np.array(
    [[8,  4,  0],
     [2,  5,  0],
     [2,  5, 11],
     [9,  2, 11],
     [2,  4,  0],
     [9,  2,  4],
     [10,  8,  1],
     [10,  8,  0],
     [10,  5,  0],
     [6,  3,  1],
     [9,  6,  3],
     [6,  8,  1],
     [6,  8,  4],
     [9,  6,  4],
     [7, 10,  1],
     [7, 10,  5],
     [7,  3,  1],
     [7,  3, 11],
     [9,  3, 11],
     [7,  5, 11], ], dtype='uint16')

unit_octahedron = Sphere(xyz=octahedron_vertices, faces=octahedron_faces)
unit_icosahedron = Sphere(xyz=icosahedron_vertices, faces=icosahedron_faces)
hemi_icosahedron = HemiSphere.from_sphere(unit_icosahedron)
