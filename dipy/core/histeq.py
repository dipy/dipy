import numpy as np


def histeq(im, num_bins=256):
    """
    Performs an histogram equalization on ``img``.
    This was taken from:
    http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    Input
    -----
    im : ndarray
        Image on which to perform histogram equalization.
    num_bins : int
        Number of bins used to construct the histogram.
    """
    #get image histogram
    histo, bins = np.histogram(im.flatten(), num_bins, normed=True)
    cdf = histo.cumsum()
    cdf = 255 * cdf / cdf[-1]

    #use linear interpolation of cdf to find new pixel values
    result = np.interp(im.flatten(), bins[:-1], cdf)

    return result.reshape(im.shape)
