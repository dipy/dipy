import numpy as np

try:
    from enthought.mayavi import mlab
except ImportError:
    from mayavi import mlab

from dipy.utils.spheremakers import sphere_vf_from


def show_odfs(odfs, vertices_faces, image=None, colormap='jet',
              scale=2.2, norm=True, radial_scale=True):
    """
    Display a grid of ODFs.

    Parameters
    ----------
    odfs : (X, Y, Z, M) ndarray
        A 3-D arrangement of orientation distribution functions (ODFs).  At
        each ``(x, y, z)`` position, it contains the the values of the
        corresponding ODF evaluated on the M vertices.
    vertices_faces : str or tuple of (vertices, faces)
        A named sphere from `dipy.data.get_sphere`, or a combination of
        `(vertices, faces)`.
    image : (X, Y) ndarray
        Background image (e.g., fractional anisotropy) do display behind the
        ODFs.
    colormap : str
        Color mapping.
    scale : float
        Increasing the scale spaces ODFs further apart.
    norm : bool
        Whether or not to normalize each individual ODF (divide by its maximum
        absolute value).
    radial_scale : bool
        Whether or not to change the radial shape of the ODF according to its
        scalar value.  If set to False, the ODF is displayed as a sphere.

    Notes
    -----
    Mayavi gets really slow when `triangular_mesh` is called too many times,
    so this function stacks ODF data and calls `triangular_mesh` once.

    Examples
    --------
    >>> from dipy.data import get_sphere
    >>> verts, faces = get_sphere('symmetric724')

    >>> angle = np.linspace(0, 2*np.pi, len(verts))
    >>> odf1 = np.sin(angle)
    >>> odf2 = np.cos(angle)
    >>> odf3 = odf1**2 * odf2
    >>> odf4 = odf1 + odf2**2

    >>> odfs = [[[odf1, odf2],
    ...          [odf3, odf4]]]

    >>> show_odfs(odfs, (verts, faces), scale=5)

    """
    vertices, faces = sphere_vf_from(vertices_faces)

    odfs = np.asarray(odfs)
    grid_shape = np.array(odfs.shape[:3])
    faces = np.asarray(faces, dtype=int)

    xx, yy, zz, ff, mm = [], [], [], [], []
    count = 0

    for ijk in np.ndindex(*grid_shape):
        m = odfs[ijk]

        if norm:
            m /= abs(m).max()

        if radial_scale:
            xyz = verts.T * m
        else:
            xyz = verts.T.copy()

        xyz += scale * (ijk - grid_shape / 2.)[:, None]

        x, y, z = xyz
        ff.append(count + faces)
        xx.append(x)
        yy.append(y)
        zz.append(z)
        mm.append(m)

        count += len(x)

    ff, xx, yy, zz, mm = (np.concatenate(arrs) for arrs in (ff, xx, yy, zz, mm))
    mlab.triangular_mesh(xx, yy, zz, ff, scalars=mm, colormap=colormap)

    if image is not None:
        mlab.imshow(image, colormap='gray', interpolate=False)

    mlab.colorbar()
    mlab.show()



if __name__ == "__main__":
    from dipy.data import get_sphere
    verts, faces = get_sphere('symmetric724')

    angle = np.linspace(0, 2*np.pi, len(verts))
    odf1 = np.sin(angle)
    odf2 = np.cos(angle)
    odf3 = odf1**2 * odf2
    odf4 = odf1 + odf2**2

    odfs = [[[odf1, odf2],
             [odf3, odf4]]]

    show_odfs(odfs, (verts, faces), scale=5, radial_scale=True)
