#!/usr/bin/env python

from __future__ import division

import os
from glob import glob
import nibabel as nib
import numpy as np

from dipy.tracking.eudx import EuDX
from dipy.data import get_sphere
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
from dipy.workflows.reconst import get_fitted_tensor
from dipy.workflows.utils import choose_create_out_dir
from dipy.reconst.dti import fractional_anisotropy
from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.trackvis import save_trk





def track_EuDX(ref, peaks_values, peaks_idx, out_dir=''):
    # Load Data volumes
    for ref_path, peaks_values_path, peaks_idx_path in zip(glob(ref),
                                                           glob(peaks_values),
                                                           glob(peaks_idx)):
        ref_img = nib.load(ref_path)
        ref_vol = ref_img.get_data()
        voxel_size = ref_img.get_header().get_zooms()[:3]

        peaks_idx = nib.load(peaks_idx_path).get_data()
        peaks_value = nib.load(peaks_values_path).get_data()

        # Run Dipy EuDX Tracking
        tracks_iter = EuDX(peaks_value,
                           peaks_idx,
                           odf_vertices=get_sphere('symmetric362').vertices,
                           seeds=100000)

        # Save streamlines (tracking results)
        streamlines_trk = [(sl, None, None) for sl in tracks_iter]
        print 'after list comp'

        hdr = nib.trackvis.empty_header()
        hdr['voxel_size'] = voxel_size
        hdr['voxel_order'] = 'LAS'
        hdr['dim'] = ref_vol.shape
        hdr['n_count'] = len(streamlines_trk)

        if out_dir == '':
            out_dir_path = os.path.dirname(ref_path)
        elif not os.path.isabs(out_dir):
            out_dir_path = os.path.join(os.path.dirname(ref_path), out_dir)
            if not os.path.exists(out_dir_path):
                os.makedirs(out_dir_path)
        else:
            out_dir_path = out_dir

        nib.trackvis.write(os.path.join(out_dir_path, 'tractogram.trk'),
                           streamlines_trk,
                           hdr)

def deterministic_tracking(dwi_paths, mask_paths, bvalues, bvectors, out_dir=''):

    for dwi, mask, bval, bvec in zip(glob(dwi_paths),
                                     glob(mask_paths),
                                     glob(bvalues),
                                     glob(bvectors)):
        dwi_img = nib.load(dwi)
        affine = dwi_img.get_affine()
        print affine
        dwi_data = dwi_img.get_data()
        voxel_size = dwi_img.get_header().get_zooms()[:3]
        print 'voxelsize', voxel_size
        mask_img = nib.load(mask)
        mask_data = mask_img.get_data()

        tenfit, gtab = get_fitted_tensor(dwi_data, mask_data, bval, bvec)
        FA = fractional_anisotropy(tenfit.evals)

        seed_mask = FA > 0.4
        seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)

        print seeds.shape
        csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
        csd_fit = csd_model.fit(dwi_data, mask=mask_data)

        classifier = ThresholdTissueClassifier(FA, .2)

        detmax_dg = \
            DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                             max_angle=30.,
                                                             sphere=default_sphere)

        streamlines = \
            LocalTracking(detmax_dg, classifier, seeds, affine, step_size=.5)

        out_dir_path = choose_create_out_dir(out_dir, dwi)

        streamlines_trk = [(sl, None, None) for sl in streamlines]

        hdr = nib.trackvis.empty_header()
        hdr['voxel_size'] = np.array(voxel_size) * np.diagonal(affine)[:-1]
        hdr['voxel_order'] = 'LAS'
        hdr['dim'] = dwi_img.shape[:-1]
        hdr['n_count'] = len(streamlines_trk)
        hdr['origin'] = affine[:3, 3]

        nib.trackvis.write(os.path.join(out_dir_path, 'tractogram.trk'),
                           streamlines_trk,  hdr)











