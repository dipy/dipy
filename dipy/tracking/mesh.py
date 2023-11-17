import numpy as np


def random_coordinates_from_surface(nb_triangles, nb_seed, triangles_mask=None,
                                    triangles_weight=None, rand_gen=None):
    """Generate random triangles_indices and trilinear_coord

    Triangles_indices probability are weighted by triangles_weight,
        for each triangles inside the given triangles_mask

    Parameters
    ----------
    nb_triangles : int (n)
        The amount of triangles in the mesh
    nb_seed : int
        The number of random indices and coordinates generated.
    triangles_mask : [n] numpy array
        Specifies which triangles should be chosen (or not)
    triangles_weight : [n] numpy array
        Specifies the weight/probability of choosing each triangle
    rand_gen : int
        The seed for the random seed generator (numpy.random.seed).

    Returns
    -------
    triangles_idx: [s] array
        Randomly chosen triangles_indices
    trilin_coord: [s,3] array
        Randomly chosen trilinear_coordinates

    See Also
    --------
    seeds_from_surface_coordinates, random_seeds_from_mask
    """
    # Compute triangles_weight in vts_mask
    if triangles_mask is not None:
        if triangles_weight is None:
            triangles_weight = triangles_mask.astype(float)
        else:
            triangles_weight *= triangles_mask.astype(float)

    # Normalize weights to have probability (sum to 1)
    if triangles_weight is not None:
        triangles_weight /= triangles_weight.sum()

    # Set the random seed generator
    rng = np.random.default_rng(rand_gen)

    # Choose random triangles
    triangles_idx = \
        rng.choice(nb_triangles, size=nb_seed, p=triangles_weight)

    # Choose random trilinear coordinates
    # https://mathworld.wolfram.com/TrianglePointPicking.html
    trilin_coord = rng.random((nb_seed, 3))
    is_upper = (trilin_coord[:, 1:].sum(axis=-1) > 1.0)
    trilin_coord[is_upper] = 1.0 - trilin_coord[is_upper]
    trilin_coord[:, 0] = 1.0 - (trilin_coord[:, 1] + trilin_coord[:, 2])

    return triangles_idx, trilin_coord


def seeds_from_surface_coordinates(triangles, vts_values,
                                   triangles_idx, trilinear_coord):
    """Compute points from triangles_indices and trilinear_coord

    Parameters
    ----------
    triangles : [n, 3] -> m array
        A list of triangles from a mesh
    vts_values : [m, .] array
        List of values to interpolates from coordinates along vertices,
        (vertices, vertices_normal, vertices_colors ...)
    triangles_idx : [s] array
        Specifies which triangles should be chosen (or not)
    trilinear_coord : [s, 3] array
        Specifies the weight/probability of choosing each triangle

    Returns
    -------
    pts : [s, ...] array
        Interpolated values of vertices with triangles_idx and trilinear_coord

    See Also
    --------
    random_coordinates_from_surface
    """
    if vts_values.ndim == 1:
        vts_values = np.reshape(vts_values, (-1, 1))

    # Compute the vertices for each chosen triangle
    tris_vals = vts_values[triangles[triangles_idx]]

    # Interpolate values for each trilinear coordinates
    return np.squeeze(np.einsum('ijk,ij...->ik', tris_vals, trilinear_coord))


def triangles_area(triangles, vts):
    """Compute the local area of each triangle

    Parameters
    ----------
    triangles : [n, 3] -> m array
        A list of triangles from a mesh
    vts : [m, .] array
        List of vertices

    Returns
    -------
    triangles_area : [m] array
        List of area for each triangle in the mesh

    See Also
    --------
    random_coordinates_from_surface
    """
    e1 = vts[triangles[:, 1]] - vts[triangles[:, 0]]
    e2 = vts[triangles[:, 2]] - vts[triangles[:, 0]]

    # Compute the area of each triangle (e.q. 0.5 * length(scipy.cross(e1, e2))
    sqr_norms = np.sum(np.square(e1), axis=-1) * np.sum(np.square(e2), axis=-1)
    u_dot_v = np.sum(e1 * e2, axis=-1)
    tri_area = 0.5 * np.sqrt(sqr_norms - np.square(u_dot_v))
    return tri_area


def vertices_to_triangles_values(triangles, vts_values):
    """Change from values per vertex to values per triangle

    Parameters
    ----------
    triangles : [n, 3] -> m array
        A list of triangles from a mesh
    vts_values : [m, .] array
        List of values to interpolates from coordinates along vertices,
        (vertices, vertices_normal, vertices_colors ...)

    Returns
    -------
    triangles_values : [m] array
        List of values for each triangle in the mesh

    See Also
    --------
    random_coordinates_from_surface
    """
    return np.mean(vts_values[triangles], axis=1)
