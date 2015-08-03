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


def compute_dti_metrics(dwis, masks, bvalues, bvectors, out_dir, tensor,
                        fa, ga, rgb, md, ad, rd, mode, evec, eval):
    dwi_paths = glob(dwis)
    if masks is None:
        masks = [masks] * len(dwi_paths)
    else:
        masks = glob(masks)

    for dwi, mask, bval, bvec in zip(dwi_paths,
                                     masks,
                                     glob(bvalues),
                                     glob(bvectors)):

        img = nib.load(dwi)
        data = img.get_data()
        affine = img.get_affine()

        if mask is None:
            mask = None
        else:
            mask = nib.load(mask).get_data().astype(np.bool)

        # Get tensors
        print('Tensor estimation...')
        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        if bvals.min() != 0:
            if bvals.min() > 20:
                raise ValueError('The minimal bvalue is greater than 20. ' +
                                 'This is highly suspicious. Please check ' +
                                 'your data to ensure everything is correct.\n' +
                                 'Value found: {0}'.format(bvals.min()))
            else:
                gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())
        else:
            gtab = gradient_table(bvals, bvecs)

        tenmodel = TensorModel(gtab)
        tenfit = tenmodel.fit(data, mask)

        if out_dir == '':
            out_dir_path = os.path.dirname(dwi)
        elif not os.path.isabs(out_dir):
            out_dir_path = os.path.join(os.path.dirname(dwi), out_dir)
            if not os.path.exists(out_dir_path):
                os.makedirs(out_dir_path)
        else:
            out_dir_path = out_dir

        FA = fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0
        FA = np.clip(FA, 0, 1)

        if tensor:
            # Get the Tensor values and format them for visualisation
            # in the Fibernavigator.
            tensor_vals = lower_triangular(tenfit.quadratic_form)
            correct_order = [0, 1, 3, 2, 4, 5]
            tensor_vals_reordered = tensor_vals[..., correct_order]
            fiber_tensors = nib.Nifti1Image(tensor_vals_reordered.astype(
                np.float32), affine)
            nib.save(fiber_tensors, tensor)

        if fa:
            fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
            nib.save(fa_img, os.path.join(out_dir_path, fa))

        if ga:
            GA = geodesic_anisotropy(tenfit.evals)
            ga_img = nib.Nifti1Image(GA.astype(np.float32), affine)
            nib.save(ga_img, os.path.join(out_dir_path, ga))

        if rgb:
            RGB = color_fa(FA, tenfit.evecs)
            rgb_img = nib.Nifti1Image(np.array(255 * RGB, 'uint8'), affine)
            nib.save(rgb_img, os.path.join(out_dir_path, rgb))

        if md:
            MD = mean_diffusivity(tenfit.evals)
            md_img = nib.Nifti1Image(MD.astype(np.float32), affine)
            nib.save(md_img, os.path.join(out_dir_path, md))

        if ad:
            AD = axial_diffusivity(tenfit.evals)
            ad_img = nib.Nifti1Image(AD.astype(np.float32), affine)
            nib.save(ad_img, os.path.join(out_dir_path, ad))

        if rd:
            RD = radial_diffusivity(tenfit.evals)
            rd_img = nib.Nifti1Image(RD.astype(np.float32), affine)
            nib.save(rd_img, os.path.join(out_dir_path, rd))

        if mode:
            MODE = get_mode(tenfit.quadratic_form)
            mode_img = nib.Nifti1Image(MODE.astype(np.float32), affine)
            nib.save(mode_img, os.path.join(out_dir_path, mode))

        if evec:
            evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), affine)
            nib.save(evecs_img, os.path.join(out_dir_path, evec))

        if eval:
            evals_img = nib.Nifti1Image(tenfit.evals.astype(np.float32), affine)
            nib.save(evals_img, os.path.join(out_dir_path, evec))
