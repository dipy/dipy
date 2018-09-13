import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
auto_response)
from dipy.core.geometry import cart2sphere
from dipy.segment.mask import median_otsu
from dipy.direction import peaks_from_model
from dipy.data import get_sphere
from dipy.core.ndindex import ndindex

fname, fscanner = get_data('small_NODDIx_data')
params = np.loadtxt(fscanner)

img = nib.load(fname)
data = img.get_data()
affine = img.affine
bvecs = params[:, 0:3]
G = params[:, 3] / 10 ** 6 # gradient strength
big_delta = params[:, 4]
small_delta = params[:, 5]
gamma = 2.675987 * 10 ** 8
bvals = gamma ** 2 * G ** 2 * small_delta ** 2 * (big_delta - small_delta / 3.)
gtab = gradient_table(bvals, bvecs, big_delta=big_delta, 
                      small_delta=small_delta,
                      b0_threshold=0, atol=1e-2)

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)

sphere = get_sphere('repulsion724')
maskdata, mask = median_otsu(data, 3, 1, False, vol_idx=range(10, 50),
                             dilate=2)
mask = data[:, :, :, 0]
csd_peaks = peaks_from_model(model=csd_model,
                             data=data,
                             sphere=sphere,
                             mask=mask,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True)

theta_angle = np.zeros((data.shape[0], data.shape[1], 1, 5))
phi_angle = np.zeros((data.shape[0], data.shape[1], 1, 5))
num_peaks = np.zeros((data.shape[0], data.shape[1], 1))

for i, j, k in ndindex((data.shape[0], data.shape[1], 1)):
    if mask[i, j, 0] > 0:
        n = 0
        for m in range(5):
            x = np.squeeze(csd_peaks.peak_dirs[i, j, k, m, 0])
            y = np.squeeze(csd_peaks.peak_dirs[i, j, k, m, 1])
            z = np.squeeze(csd_peaks.peak_dirs[i, j, k, m, 2])
            
            if (x**2 + y**2 + z**2) > 0:
                r, theta_angle[i, j, k, m], phi_angle[i, j, k, m] = \
                cart2sphere(x, y, z)
                phi_angle[i, j, k, m] = phi_angle[i, j, k, m] + np.pi
                theta_angle[i, j, k, m] = np.pi - theta_angle[i, j, k, m]
                if phi_angle[i, j, k, m] > np.pi:
                    phi_angle[i, j, k, m] = phi_angle[i, j, k, m] - np.pi
                    theta_angle[i, j, k, m] = np.pi - theta_angle[i, j, k, m]
                n = n + 1
                num_peaks[i, j, k] = n
                