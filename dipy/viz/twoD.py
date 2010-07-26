import pylab as pl
import numpy as np
def imshow(array, cmap='gray',interpolation='nearest', alpha=1.0,
            vmin=None, vmax=None, origin=None, extent=None):
    """
    Wrapper for pylab.imshow that displays array values as well
    coordinate values with mouse over.
    """
    pl.imshow(array.T, cmap=cmap, interpolation=interpolation, alpha=alpha,
            vmin=vmin, vmax=vmax, origin=origin, extent=extent)
    ax = pl.gca()
    ax.format_coord = __report_pixel

def __report_pixel(x, y):
    x = np.round(x)
    y = np.round(y)
    v = pl.gca().get_images()[0].get_array()[y, x]
    return "x = %d y = %d v = %5.3f" % (x, y, v)


