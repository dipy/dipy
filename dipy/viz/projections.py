"""

Visualization tools for 2D projections of 3D functions on the sphere, such as
ODFs.

"""

import numpy as np
import scipy.interpolate as interp
from dipy.utils.optpkg import optional_package
import dipy.core.geometry as geo
from dipy.testing.decorators import doctest_skip_parser

matplotlib, has_mpl, setup_module = optional_package("matplotlib")
plt, _, _ = optional_package("matplotlib.pyplot")
tri, _, _ = optional_package("matplotlib.tri")
bm, has_basemap, _ = optional_package("mpl_toolkits.basemap")


@doctest_skip_parser
def sph_project(vertices, val, ax=None, vmin=None, vmax=None, cmap=None,
                cbar=True, tri=False, boundary=False, **basemap_args):
    """Draw a signal on a 2D projection of the sphere.

    Parameters
    ----------

    vertices : (N,3) ndarray
                unit vector points of the sphere

    val: (N) ndarray
        Function values.

    ax : mpl axis, optional
        If specified, draw onto this existing axis instead.

    vmin, vmax : floats
       Values to cut the z

    cmap : mpl colormap

    cbar: Whether to add the color-bar to the figure

    triang : Whether to display the plot triangulated as a pseudo-color plot.

    boundary : Whether to draw the boundary around the projection
    in a black line

    Returns
    -------
    ax : axis
        Matplotlib figure axis

    Examples
    --------
    >>> from dipy.data import default_sphere
    >>> verts = default_sphere.vertices
    >>> ax = sph_project(verts.T, np.random.rand(len(verts.T))) # skip if not has_basemap
    """
    if ax is None:
        fig, ax = plt.subplots(1)

    if cmap is None:
        cmap = matplotlib.cm.hot

    basemap_args.setdefault('projection', 'ortho')
    basemap_args.setdefault('lat_0', 0)
    basemap_args.setdefault('lon_0', 0)
    basemap_args.setdefault('resolution', 'c')

    from mpl_toolkits.basemap import Basemap

    m = Basemap(**basemap_args)
    if boundary:
        m.drawmapboundary()

    # Rotate the coordinate system so that you are looking from the north pole:
    verts_rot = np.array(
        np.dot(np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]), vertices))

    # To get the orthographic projection, when the first coordinate is
    # positive:
    neg_idx = np.where(verts_rot[0] > 0)

    # rotate the entire b-vector around to point in the other direction:
    verts_rot[:, neg_idx] *= -1

    _, theta, phi = geo.cart2sphere(verts_rot[0], verts_rot[1], verts_rot[2])
    lat, lon = geo.sph2latlon(theta, phi)
    x, y = m(lon, lat)

    my_min = np.nanmin(val)
    if vmin is not None:
        my_min = vmin

    my_max = np.nanmax(val)
    if vmax is not None:
        my_max = vmax

    if tri:
        m.pcolor(x, y, val, vmin=my_min, vmax=my_max, tri=True, cmap=cmap)

    else:
        cmap_data = cmap._segmentdata
        red_interp, blue_interp, green_interp = (
            interp.interp1d(np.array(cmap_data[gun])[:, 0],
                            np.array(cmap_data[gun])[:, 1]) for gun in
            ['red', 'blue', 'green'])

        r = (val - my_min) / float(my_max - my_min)

        # Enforce the maximum and minimum boundaries, if there are values
        # outside those boundaries:
        r[r < 0] = 0
        r[r > 1] = 1

        for this_x, this_y, this_r in zip(x, y, r):
            red = red_interp(this_r)
            blue = blue_interp(this_r)
            green = green_interp(this_r)
            m.plot(this_x, this_y, 'o',
                   c=[red.item(), green.item(), blue.item()])

    if cbar:
        mappable = matplotlib.cm.ScalarMappable(cmap=cmap)
        mappable.set_array([my_min, my_max])
        # setup colorbar axes instance.
        pos = ax.get_position()
        l, b, w, h = pos.bounds
        # setup colorbar axes
        cax = fig.add_axes([l + w + 0.075, b, 0.05, h], frameon=False)
        fig.colorbar(mappable, cax=cax)  # draw colorbar

    return ax
