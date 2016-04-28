#!/usr/bin/env python
from __future__ import division

import logging
import os
from glob import glob
import nibabel as nib
import numpy as np

from tractconverter.formats.trk import TRK
from tractconverter.formats.tck import TCK
from tractconverter import convert as convert_tracks

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


def EuDX_tracking_flow(peaks_values, peaks_indexes, out_dir='',
                       out_tractogram='tractogram.trk'):

    """ Workflow for Eulder delta crossings tracking. Tracking is done by
        'globing' ``input_files``, ``peaks_values``, and ``peaks_indexes``
         and saves the tracks in a directory specified by ``out_dir``.

    Parameters
    ----------
    peaks_values : string
        Path to the peaks values files. This path may contain wildcards to use
        multiple masks at once.
    peaks_indexes : string
        Path to the peaks indexes files. This path may contain wildcards to use
        multiple bvalues files at once.
    out_dir : string, optional
        Output directory (default input file directory)
    out_tractogram : string, optional
        Name of the tractogram file to be saved (default 'tractogram.trk')
    """
    for peaks_values_path, peaks_idx_path in zip(glob(peaks_values),
                                                           glob(peaks_indexes)):
        logging.info('EuDX tracking on {0}'.format(peaks_values_path))

        peaks_idx_img = nib.load(peaks_idx_path)
        voxel_size = peaks_idx_img.get_header().get_zooms()[:3]
        peaks_idx = peaks_idx_img.get_data()
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
        hdr['dim'] = peaks_idx_img.shape[:3]
        hdr['n_count'] = len(streamlines_trk)

        out_dir_path = choose_create_out_dir(out_dir, peaks_values)
        tractogram_path = os.path.join(out_dir_path, out_tractogram)
        nib.trackvis.write(tractogram_path, streamlines_trk, hdr, points_space='voxel')
        logging.info('Saved {0}'.format(tractogram_path))


def deterministic_tracking_flow(input_files, mask_files, bvalues, bvectors,
                                out_dir='',
                                out_tractogram='deterministic_tractogram.trk'):
    """ Workflow for deterministic tracking. Tracking is done by
        'globing' ``input_files``, ``peaks_values``, and ``peaks_indexes``
         and saves the tracks in a directory specified by ``out_dir``.

    Parameters
    ----------
    input_files : string
        Path to the input volumes. This path may contain wildcards to process
        multiple inputs at once.
    mask_files : string
        Path to the input masks. This path may contain wildcards to use
        multiple masks at once.
    bvalues : string
        Path to the bvalues files. This path may contain wildcards to use
        multiple bvalues files at once.
    bvectors : string
        Path to the bvalues files. This path may contain wildcards to use
        multiple bvalues files at once.
    out_dir : string, optional
        Output directory (default input file directory)
    out_tractogram : string, optional
        Name of the tractogram file to be saved (default 'tractogram.trk')
    """
    for dwi, mask, bval, bvec in zip(glob(input_files),
                                     glob(mask_files),
                                     glob(bvalues),
                                     glob(bvectors)):

        logging.info('Deterministic tracking on {0}'.format(dwi))
        dwi_img = nib.load(dwi)
        affine = dwi_img.get_affine()
        dwi_data = dwi_img.get_data()
        voxel_size = dwi_img.get_header().get_zooms()[:3]
        mask_img = nib.load(mask)
        mask_data = mask_img.get_data()

        tenfit, gtab = get_fitted_tensor(dwi_data, mask_data, bval, bvec)
        FA = fractional_anisotropy(tenfit.evals)

        seed_mask = FA > 0.8
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
        hdr['voxel_size'] = voxel_size
        hdr['voxel_order'] = 'LAS'
        hdr['dim'] = dwi_img.shape[:-1]
        hdr['n_count'] = len(streamlines_trk)

        tractogram_path = os.path.join(out_dir_path, out_tractogram)
        nib.trackvis.write(tractogram_path, streamlines_trk,  hdr, points_space='voxel')

        logging.info('Saved {0}'.format(tractogram_path))
