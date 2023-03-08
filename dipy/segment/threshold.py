import numpy as np


def otsu(image, nbins=256):
    """
    Return threshold value based on Otsu's method.
    Copied from scikit-image to remove dependency.

    Parameters
    ----------
    image : array
        Input image.
    nbins : int
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    threshold : float
        Threshold value.
    """
    hist, bin_centers = np.histogram(image, nbins)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers[1:]) / weight1
    mean2 = (np.cumsum((hist * bin_centers[1:])[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


def upper_bound_by_rate(data, rate=0.05):
    r""" Adjusts upper intensity boundary using rates

    It calculates the image intensity histogram, and based on the rate value it
    decide what is the upperbound value for intensity normalization, usually
    lower bound is 0. The rate is the ratio between the amount of pixels in
    every bins and the bins with highest pixel amount

    Parameters
    ----------
    data : float
        Input intensity value data
    rate : float
        representing the threshold whether a specific histogram bin that should
        be count in the normalization range

    Returns
    -------
    high : float

        the upper_bound value for normalization
    """

    g, h = np.histogram(data)
    m = np.zeros((10, 3))
    high = data.max()
    for i in np.array(range(10)):
        m[i, 0] = g[i]
        m[i, 1] = h[i]
        m[i, 2] = h[i + 1]

    g = sorted(g,reverse = True)
    sz = np.size(g)

    Index = 0

    for i in np.array(range(sz)):
        if g[i] / g[0] > rate:
            Index = Index + 1

    for i in np.array(range(10)):
        for j in np.array(range(Index)):
            if g[j] == m[i, 0]:
                high = m[i, 2]
    return high


def upper_bound_by_percent(data, percent=1):
    """ Find the upper bound for visualization of medical images

    Calculate the histogram of the image and go right to left until you find
    the bound that contains more than a percentage of the image.

    Parameters
    ----------
    data : ndarray
    percent : float

    Returns
    -------
    upper_bound : float
    """

    percent = percent / 100.
    values, bounds = np.histogram(data, 20)
    total_voxels = np.prod(data.shape)
    agg = 0

    for i in range(len(values) - 1, 0, -1):
        agg += values[i]
        if agg / float(total_voxels) > percent:
            return bounds[i]
