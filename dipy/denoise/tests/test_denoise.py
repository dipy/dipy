import numpy as np
import numpy.testing as npt
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.denoise.nlmeans import nlmeans
import dipy.data as dpd
import nibabel as nib


def test_denoise():
    """

    """
    fdata, fbval, fbvec = dpd.get_data()
    # Test on 4D image:
    data = nib.load(fdata).get_data()
    sigma1 = estimate_sigma(data)
    denoised = nlmeans(data, sigma=sigma1)

    # Test on 3D image:
    data = data[..., 0]
    sigma2 = estimate_sigma(data)
    denoised = nlmeans(data, sigma=sigma2)
