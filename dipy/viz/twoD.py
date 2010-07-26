import pylab as pl
import numpy as np
def imshow(array, cmap='gray'):
    """
    Wrapper for pylab.imshow that displays array values as well
    coordinate values with mouse over.
    """
    pl.imshow(array, cmap=cmap)
    ax = pl.gca()
    ax.format_coord = __report_pixel

def __report_pixel(x, y):
    v = pl.gca().get_images()[0].get_array()[x, y]
    return "x = %d y = %d v = %5.3f" % (x, y, v)


