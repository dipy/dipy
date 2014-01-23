from __future__ import division, print_function

import numpy as np
from dipy.denoise.denspeed import nlmeans_3d


def nlmeans(arr, mask, sigma=None, patch_radius=1, block_radius=5, rician=True):
	"""

	"""

    if arr.ndim == 3:

        return nlmeans_3d(arr, mask, sigma, patch_radius, block_radius, rician)


