from dipy.denoise.noise_estimate import estimate_sigma
from dipy.denoise.nlmeans import nlmeans
import dipy.data as dpd
from dipy.io.image import load_nifti


def test_denoise():
    """

    """
    fdata, fbval, fbvec = dpd.get_fnames()
    # Test on 4D image:
    data, _ = load_nifti(fdata)
    sigma1 = estimate_sigma(data)
    nlmeans(data, sigma=sigma1)

    # Test on 3D image:
    data = data[..., 0]
    sigma2 = estimate_sigma(data)
    nlmeans(data, sigma=sigma2)
