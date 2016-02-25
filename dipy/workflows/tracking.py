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


def EuDX_tracking_flow(ref, peaks_values, peaks_indexes, out_dir='',
                       tractogram='tractogram.trk'):

    """ Workflow for Eulder delta crossings tracking. Tracking is done by
        'globing' ``input_files``, ``peaks_values``, and ``peaks_indexes``
         and saves the tracks in a directory specified by ``out_dir``.

    Parameters
    ----------
    ref : string
        Path to the reference volume files. This path may contain wildcards to process
        multiple inputs at once.
    peaks_values : string
        Path to the peaks values files. This path may contain wildcards to use
        multiple masks at once.
    peaks_indexes : string
        Path to the peaks indexes files. This path may contain wildcards to use
        multiple bvalues files at once.
    out_dir : string, optional
        Output directory (default input file directory)
    tractogram : string, optional
        Name of the tractogram file to be saved (default 'tractogram.trk')
    Outputs
    -------
    tractogram : tck file
        This file contains the resulting tractogram.
    """
    for ref_path, peaks_values_path, peaks_idx_path in zip(glob(ref),
                                                           glob(peaks_values),
                                                           glob(peaks_indexes)):
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

        hdr = nib.trackvis.empty_header()
        hdr['voxel_size'] = voxel_size
        hdr['voxel_order'] = 'LAS'
        hdr['dim'] = ref_vol.shape
        hdr['n_count'] = len(streamlines_trk)

        out_dir_path = choose_create_out_dir(out_dir, ref)
        nib.trackvis.write(os.path.join(out_dir_path, tractogram),
                           streamlines_trk,
                           hdr)

def deterministic_tracking_flow(dwi_paths, mask_paths, bvalues, bvectors, out_dir=''):

    for dwi, mask, bval, bvec in zip(glob(dwi_paths),
                                     glob(mask_paths),
                                     glob(bvalues),
                                     glob(bvectors)):

        print('Deterministic tracking on {0}'.format(dwi))
        dwi_img = nib.load(dwi)
        affine = dwi_img.get_affine()
        dwi_data = dwi_img.get_data()
        voxel_size = dwi_img.get_header().get_zooms()[:3]
        mask_img = nib.load(mask)
        mask_data = mask_img.get_data()

        tenfit, gtab = get_fitted_tensor(dwi_data, mask_data, bval, bvec)
        FA = fractional_anisotropy(tenfit.evals)

        seed_mask = FA > 0.2
        seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)
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











