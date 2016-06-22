from __future__ import division, print_function, absolute_import

import logging
import os.path
from glob import glob

import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dti import (TensorModel, color_fa, fractional_anisotropy,
                              geodesic_anisotropy, mean_diffusivity,
                              axial_diffusivity, radial_diffusivity,
                              lower_triangular, mode as get_mode)
from dipy.workflows.utils import choose_create_out_dir, glob_or_value

def dti_metrics_flow(input_files, bvalues, bvectors, mask_files=None,
                     b0_threshold=0.0, out_dir='', out_tensor='tensors.nii.gz',
                     out_fa='fa.nii.gz', out_ga='ga.nii.gz', out_rgb='rgb.nii.gz',
                     out_md='md.nii.gz', out_ad='ad.nii.gz', out_rd='rd.nii.gz',
                     out_mode='mode.nii.gz', out_evec='evecs.nii.gz', out_eval='evals.nii.gz'):

    """ Workflow for tensor reconstruction and DTI metrics computing.
    It a tensor recontruction on the files by 'globing' ``input_files`` and
    saves the dti metrics in a directory specified by ``out_dir``.

    Parameters
    ----------
    input_files : string
        Path to the input volumes. This path may contain wildcards to process
        multiple inputs at once.
    bvalues : string
        Path to the bvalues files. This path may contain wildcards to use
        multiple bvalues files at once.
    bvectors : string
        Path to the bvalues files. This path may contain wildcards to use
        multiple bvalues files at once.
    mask_files : string
        Path to the input masks. This path may contain wildcards to use
        multiple masks at once. (default: No mask used)
    b0_threshold : float, optional
        Threshold used to find b=0 directions (default 0.0)
    out_dir : string, optional
        Output directory (default input file directory)
    out_tensor : string, optional
        Name of the tensors volume to be saved (default 'tensors.nii.gz')
    out_fa : string, optional
        Name of the fractionnal anisotropy volume to be saved (default 'fa.nii.gz')
    out_ga : string, optional
        Name of the geodesic anisotropy volume to be saved (default 'ga.nii.gz')
    out_rgb : string, optional
        Name of the color fa volume to be saved (default 'rgb.nii.gz')
    out_md : string, optional
        Name of the mean diffusivity volume to be saved (default 'md.nii.gz')
    out_ad : string, optional
        Name of the axial diffusivity volume to be saved (default 'ad.nii.gz')
    out_rd : string, optional
        Name of the radial diffusivity volume to be saved (default 'rd.nii.gz')
    out_mode : string, optional
        Name of the mode volume to be saved (default 'mode.nii.gz')
    out_evecs : string, optional
        Name of the eigen vectors volume to be saved (default 'evecs.nii.gz')
    out_evals : string, optional
        Name of the eigen vvalues to be saved (default 'evals.nii.gz')
    """

    globed_dwi, globed_mask = glob_or_value(input_files, mask_files)
    for dwi, mask, bval, bvec in zip(globed_dwi,
                                     globed_mask,
                                     glob(bvalues),
                                     glob(bvectors)):

        logging.info('Computing dti metrics for {0}'.format(dwi))
        img = nib.load(dwi)
        data = img.get_data()
        affine = img.get_affine()

        if mask is None:
            mask = None
        else:
            mask = nib.load(mask).get_data().astype(np.bool)

        tenfit, _ = get_fitted_tensor(data, mask, bval, bvec, b0_threshold)

        out_dir_path = choose_create_out_dir(out_dir, dwi)

        FA = fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0
        FA = np.clip(FA, 0, 1)

        tensor_vals = lower_triangular(tenfit.quadratic_form)
        correct_order = [0, 1, 3, 2, 4, 5]
        tensor_vals_reordered = tensor_vals[..., correct_order]
        fiber_tensors = nib.Nifti1Image(tensor_vals_reordered.astype(
            np.float32), affine)
        nib.save(fiber_tensors, os.path.join(out_dir_path, out_tensor))

        fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
        nib.save(fa_img, os.path.join(out_dir_path, out_fa))

        GA = geodesic_anisotropy(tenfit.evals)
        ga_img = nib.Nifti1Image(GA.astype(np.float32), affine)
        nib.save(ga_img, os.path.join(out_dir_path, out_ga))

        RGB = color_fa(FA, tenfit.evecs)
        rgb_img = nib.Nifti1Image(np.array(255 * RGB, 'uint8'), affine)
        nib.save(rgb_img, os.path.join(out_dir_path, out_rgb))

        MD = mean_diffusivity(tenfit.evals)
        md_img = nib.Nifti1Image(MD.astype(np.float32), affine)
        nib.save(md_img, os.path.join(out_dir_path, out_md))

        AD = axial_diffusivity(tenfit.evals)
        ad_img = nib.Nifti1Image(AD.astype(np.float32), affine)
        nib.save(ad_img, os.path.join(out_dir_path, out_ad))

        RD = radial_diffusivity(tenfit.evals)
        rd_img = nib.Nifti1Image(RD.astype(np.float32), affine)
        nib.save(rd_img, os.path.join(out_dir_path, out_rd))

        MODE = get_mode(tenfit.quadratic_form)
        mode_img = nib.Nifti1Image(MODE.astype(np.float32), affine)
        nib.save(mode_img, os.path.join(out_dir_path, out_mode))

        evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), affine)
        nib.save(evecs_img, os.path.join(out_dir_path, out_evec))

        evals_img = nib.Nifti1Image(tenfit.evals.astype(np.float32), affine)
        nib.save(evals_img, os.path.join(out_dir_path, out_eval))

        logging.info('All dti metrics saved in {0}'.format(out_dir_path))


def get_fitted_tensor(data, mask, bval, bvec, b0_threshold=0):
    logging.info('Tensor estimation...')
    bvals, bvecs = read_bvals_bvecs(bval, bvec)
    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)

    return tenfit, gtab