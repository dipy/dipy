"""

Visualization tools for 2D projections of 3D functions on the sphere, such as
ODFs.

"""

import numpy as np
import scipy.interpolate as interp

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import dipy.core.geometry as geo

def sph2latlon(theta, phi):
    """Convert spherical coordinates to latitude and longitude.

    Returns
    -------
    lat, lon : ndarray
        Latitude and longitude.

    """
    return np.rad2deg(theta - np.pi/2), np.rad2deg(phi - np.pi)

def sph_project(vertices, val, ax=None, vmin=None, vmax=None,
            cmap=matplotlib.cm.hot, cbar=True, tri=False,
            boundary=True, **basemap_args):

    """Draw a signal on a 2D projection of the sphere.

    Parameters
    ----------

    vertices : (N,3) ndarray
                unit vector points of the sphere

    r : (M, N) ndarray
        Function values.

    theta : (M,) ndarray
        Inclination / polar angles of function values.

    phi : (N,) ndarray
        Azimuth angles of function values.

    ax : mpl axis, optional
        If specified, draw onto this existing axis instead.

    cmap: mpl colormap

    cbar: Whether to add the color-bar to the figure

    basemap_args : dict
        Parameters used to initialise the basemap, e.g. ``projection='ortho'``.

    Returns
    -------
    m : basemap
        The newly created matplotlib basemap.
    fig : figure
        Matplotlib figure

    Examples
    --------
    >>> from dipy.data import get_sphere
    >>> verts,faces=get_sphere('symmetric724')
    >>> sph_project(verts,np.random.rand(len(verts)))

    """
    bvecs=bvecs.T

    if ax is None:
        fig, ax = plt.subplots(1)

    basemap_args.setdefault('projection', 'ortho')
    basemap_args.setdefault('lat_0', 0)
    basemap_args.setdefault('lon_0', 0)
    basemap_args.setdefault('resolution', 'c')

    m = Basemap(**basemap_args)
    if boundary:
        m.drawmapboundary()

    # Rotate the coordinate system so that you are looking from the north pole:
    bvecs_rot = np.array(np.dot(np.matrix([[0,0,-1],[0,1,0],[1,0,0]]), bvecs))

    # To get the orthographic projection, when the first coordinate is positive:
    neg_idx = np.where(bvecs_rot[0]>0)

    # rotate the entire bvector around to point in the other direction:
    bvecs_rot[:, neg_idx] *= -1

    _, theta, phi = geo.cart2sphere(bvecs_rot[0], bvecs_rot[1], bvecs_rot[2])
    lat, lon = sph2latlon(theta, phi)
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
        interp.interp1d(np.array(cmap_data[gun])[:,0],
                        np.array(cmap_data[gun])[:,1]) for gun in
                                                  ['red', 'blue','green'])

        r = (val - my_min)/float(my_max-my_min)

        # Enforce the maximum and minumum boundaries, if there are values
        # outside those boundaries:
        r[r<0]=0
        r[r>1]=1

        for this_x, this_y, this_r in zip(x,y,r):
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
        cax = fig.add_axes([l+w+0.075, b, 0.05, h], frameon=False)
        fig.colorbar(mappable, cax=cax) # draw colorbar

    return m, fig
