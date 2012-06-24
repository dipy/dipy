"""

Visualization tools for 2D projections of 3D functions on the sphere, such as
ODFs.

"""

import numpy as np
import scipy.interpolate as interp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import dipy.core.geometry as geo

def sph_project(vertices, val, ax=None, vmin=None, vmax=None,
		cmap=matplotlib.cm.hot, cbar=True, triang=False):

    """Draw a signal on a 2D projection of the sphere.

    Parameters
    ----------

    vertices : (N,3) ndarray
                unit vector points of the sphere

    val: (N) ndarray
        Function values.

    ax : mpl axis, optional
        If specified, draw onto this existing axis instead.

    vmin, vmax: floats
       Values to cut the z

    cmap: mpl colormap

    cbar: Whether to add the color-bar to the figure

    triang: Whether to display the plot triangulated as a pseudo-color plot.

    Returns
    -------
    fig : figure
        Matplotlib figure

    Examples
    --------
    >>> from dipy.data import get_sphere
    >>> verts,faces=get_sphere('symmetric724')
    >>> sph_project(verts,np.random.rand(len(verts)))

    """
    if ax is None:
        fig, ax = plt.subplots(1)

    x = vertices[:, 0]
    y = vertices[:, 1]

    my_min = np.nanmin(val)
    if vmin is not None:
        my_min = vmin

    my_max = np.nanmax(val)
    if vmax is not None:
        my_max = vmax

    r = (val - my_min)/float(my_max-my_min)

    # Enforce the maximum and minumum boundaries, if there are values
    # outside those boundaries:
    r[r<0]=0
    r[r>1]=1

    if triang:
	triang = tri.Triangulation(x, y)
	plt.tripcolor(triang, r, cmap=cmap)
    else:
	cmap_data = cmap._segmentdata
	red_interp, blue_interp, green_interp = (
		interp.interp1d(np.array(cmap_data[gun])[:,0],
				np.array(cmap_data[gun])[:,1]) for gun in
		                ['red', 'blue','green'])


	for this_x, this_y, this_r in zip(x,y,r):
	    red = red_interp(this_r)
	    blue = blue_interp(this_r)
	    green = green_interp(this_r)
	    ax.plot(this_x, this_y, 'o',
		    c=[red.item(), green.item(), blue.item()])


    plt.axis('equal')
    plt.axis('off')
    if cbar:
        mappable = matplotlib.cm.ScalarMappable(cmap=cmap)
        mappable.set_array([my_min, my_max])
        # setup colorbar axes instance.
        pos = ax.get_position()
        l, b, w, h = pos.bounds
        # setup colorbar axes
        cax = fig.add_axes([l+w+0.075, b, 0.05, h], frameon=False)
        fig.colorbar(mappable, cax=cax) # draw colorbar

    return fig
