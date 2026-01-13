"""
=====================================================================
Robust fitting with iteratively reweighted least squares.
=====================================================================


Both :footcite:t:`Coveney2025` and :footcite:t:`Coveney2025b` show
robust fitting using iteretively reweighted least squares (IRLS).
Methods for fitting DTI and DKI data are shown in the relevant examples.

It is possible to construct novel weighting functions and novel outlier functions
in DiPy in order to do robust fitting with IRLS.

Here we show an example of multi-voxel outlier detection (MVOD) :footcite:t:`Coveney2025`.
"""

from dipy.viz.plotting import compare_maps

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

import dipy.reconst.dti as dti

from dipy.data import get_fnames

"""
We will use the same data from the DTI example.
"""

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")

data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

from dipy.segment.mask import median_otsu

maskdata, mask = median_otsu(
    data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=True, dilate=2
)

"""
Firstly let's fit with WLS.
"""

tenmodel = dti.TensorModel(gtab, fit_method="WLS")
tenfit_orig = tenmodel.fit(maskdata[:, :, 38:39], mask=mask[:, :, 38:39])

"""
The DTI and DKI examples both show how to use robust fitting.
Let's corrupt the data by off-setting a bunch of images.
Of course, in this case shot rejection could be used,
but this is just an easy way to make an example. 
Let's see how WLS and Robust WLS (RWLS) perform.
"""

# corrupt data by shifting 10 images by 30 voxels
maskdata[30:, 30:, 38:39, 10:20] = maskdata[0:-30, 0:-30, 38:39, 10:20]

tenmodel = dti.TensorModel(gtab, fit_method="WLS")
tenfit = tenmodel.fit(maskdata[:, :, 38:39], mask=mask[:, :, 38:39])

tenmodel_rwls = dti.TensorModel(gtab, fit_method="RWLS")
tenfit_rwls = tenmodel_rwls.fit(maskdata[:, :, 38:39], mask=mask[:, :, 38:39])

"""
Here are MD and FA for the different fits:
"""

compare_maps(
    [tenfit_orig, tenfit, tenfit_rwls],
    ["md", "fa"],
    fit_labels=["WLS orig", "WLS", "RWLS"],
    map_kwargs=[{"vmin": 0.000, "vmax": 0.002}, {"vmin": 0, "vmax": 1.0}],
    filename="Compare_WLS_and_RWLS.png",
)


"""
We might be able to do better by designing our own weight function to use on
each iteration of IRLS, as well as by designing our own
outlier rejection functions.

Here, for simplicity, we will load the ```weights_method_wls_m_est``` function
(take a look at this function to understand how you might write your own),
but write our own outlier rejection function that depends on neighbouring voxels.
"""

from dipy.utils.volume import adjacency_calc
from dipy.reconst.weights_method import weights_method_wls_m_est
from dipy.reconst.weights_method import simple_cutoff, two_eyes_cutoff

"""
Firstly, let's calculate adjacency, respecting the mask. For each voxel,
we obtain the index (into a masked array of neighbouring voxels), with
voxel-distance less than ```cutoff```. 
"""

adjacency = adjacency_calc(mask.shape[0:2], mask=mask[:, :, 38:39], cutoff=3)

"""
Now we will write a multi-voxel outlier detection (MVOD) function.
"""


def MVOD_cutoff(residuals, log_residuals, pred_sig, design_matrix, leverages, C, cutoff, linear, adjacency):
    """ MVOD outlier detection (includes "two-eyes" SVOD condition as well)"""

    # SVOD: single voxel outlier detection
    cond = simple_cutoff(residuals, log_residuals, pred_sig, design_matrix, leverages, C, cutoff)
    # an alternative outlier definition for single voxels is:
    #cond = two_eyes_cutoff(residuals, log_residuals, pred_sig, design_matrix, leverages, C, cutoff)

    # MVOD: multiple voxel outlier detection
    # ======================================
    MV_cond = np.zeros_like(cond)
    leverages[np.isclose(leverages, 1.0)] = 0.9999
    HAT_factor = np.sqrt(1 - leverages)
    p = design_matrix.shape[1]
    factor = 1.4826  # FIXME: the outlier routines will need access to design_matrix to properly calculate factor = 1.4826 * np.sqrt(N / (N - p))

    # if adjacency is None, then weights are the same for all voxels - solve for first voxel, copy weights across later
    adj_full = False
    if adjacency is None:
        adjacency = [[vox for vox in range(residuals.shape[0])]]
        adj_full = True

    for vox in range(len(adjacency)):

        if linear:
            HAT, HAT_vox = HAT_factor[adjacency[vox]], HAT_factor[vox]
        else:
            HAT, HAT_vox = HAT_factor, HAT_factor

        # calculate C from all adjacent voxels
        residuals_vox = residuals[adjacency[vox]]
        log_residuals_vox = log_residuals[adjacency[vox]]
        pred_sig_vox = pred_sig[adjacency[vox]]

        yy = residuals_vox / HAT
        # NOTE: C is a robust estimate of stdev of error
        # here, uses all images and all voxels in adjacency
        C = factor * np.median(np.abs(yy - np.median(yy)))

        # multivariate outlier condition: RMSE
        MV_cutoff = cutoff
        RMSE = np.sqrt((yy**2).mean(axis=0))
        MED = np.median(RMSE)
        MAD = 1.4826 * np.median(np.abs(RMSE - MED))
        cond_RMSE = (RMSE > (MED + MV_cutoff*MAD)) #| (RMSE < (MED - MV_cutoff*MAD))
        MV_cond[vox] = MV_cond[vox] | cond_RMSE

        # multivariate outlier condition: SE
        MV_cutoff = cutoff
        SE = np.sum(yy, axis=0) # sum of residuals
        MED = np.median(SE)
        MAD = 1.4826 * np.median(np.abs(SE - MED))
        cond_SE = (SE > (MED + MV_cutoff*MAD)) | (SE < (MED - MV_cutoff*MAD))
        MV_cond[vox] = MV_cond[vox] | cond_SE

    if adj_full:
        MV_cond[1:] = MV_cond[0]

    # check condition number that will result... though, this is just design_matrix, WLS solution a bit different
    for vox in range(cond.shape[0]): #len(adjacency)):
        u, s, vh = np.linalg.svd(design_matrix[np.invert(MV_cond[vox] | cond[vox])])
        min_s = s.min()
        if min_s < 1e-2: 
            #print("eval:", min_s)
            pass
        else:
            # NOTE: include MV_cond only if it doesn't make the design_matrix nearly singular
            cond[vox] = cond[vox] | MV_cond[vox]

    return cond


"""
Outlier condition functions in DiPy must only take the following arguments:
```residuals, log_residuals, pred_sig, design_matrix, leverages, C, cutoff```

This is so that weighting functions, which make calls to outlier condition functions,
are always able to pass the required arguments to the outlier functions.

To handle the fact that our outlier function depends on two more arguments, 
we create a function as follows (adjacency is in scope here):
"""

def mvod(*args):
    return MVOD_cutoff(*args, linear=True, adjacency=adjacency)

"""
Now we are free to define our weights function. Even though we will
use ```weights_method_wls_m_est```, we will specify a new weights function
that defines the outlier condition, and other additional arguments, as follows:
"""

def wm(*args):
    return weights_method_wls_m_est(
        *args, m_est="gm", cutoff=3, outlier_condition_func=mvod)

"""
This is because weights functions in DiPy must only take arguments ```data, pred_sig, design_matrix, leverages, rdx, TDX, robust` (and must return only `w, robust```.
"""


tenmodel_rwls_mvod = dti.TensorModel(gtab, fit_method="IRLS", return_S0_hat=True,
                 weights_method=wm,
                 fit_type="WLS",
                 num_iter=10)
tenfit_rwls_mvod = tenmodel_rwls_mvod.fit(maskdata[:, :, 38:39], mask=mask[:, :, 38:39])

"""
Robustly fitting (via IRLS) with the new weights function gives much better results.
"""

compare_maps(
    [tenfit_orig, tenfit, tenfit_rwls_mvod],
    ["md", "fa"],
    fit_labels=["WLS orig", "WLS", "RWLS + MVOD"],
    map_kwargs=[{"vmin": 0.000, "vmax": 0.002}, {"vmin": 0, "vmax": 1.0}],
    filename="Compare_WLS_and_RWLS_MVOD.png",
)

"""
Of course you are free to write your own weights functions from scratch.
Since weights are calculated after each iteration of fitting over all masked
voxels, both the weights function and the outlier condition function can
calculate weights and define outliers for the current voxels using information
from other voxels.
"""


"""
References
----------
.. footbibliography::

"""
