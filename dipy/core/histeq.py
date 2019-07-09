import numpy as np


def histeq(arr, num_bins=256):
    """ Performs an histogram equalization on ``arr``.
    This was taken from:
    http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    Parameters
    ----------
    arr : ndarray
        Image on which to perform histogram equalization.
    num_bins : int
        Number of bins used to construct the histogram.

    Returns
    -------
    result : ndarray
        Histogram equalized image.
    """
    # get image histogram
    histo, bins = np.histogram(arr.flatten(), num_bins, density=True)
    cdf = histo.cumsum()
    cdf = 255 * cdf / cdf[-1]

    # use linear interpolation of cdf to find new pixel values
    result = np.interp(arr.flatten(), bins[:-1], cdf)

    return result.reshape(arr.shape)
