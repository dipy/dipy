from __future__ import division, print_function

import numpy as np
from dipy.denoise.denspeed import nlmeans_3d


def nlmeans(arr, patch_radius=1, block_radius=5, sigma=None, rician=True):

    if arr.ndim == 3:

        return nlmeans_3d(arr, patch_radius, block_radius, sigma, rician)
