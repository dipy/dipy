#!/usr/bin/env python
from __future__ import division

import logging
import inspect

import nibabel as nib
import numpy as np

from dipy.io.trackvis import save_trk
from dipy.io.image import load_nifti
from dipy.io.peaks import load_peaks
from dipy.tracking.eudx import EuDX
from dipy.data import get_sphere
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
from dipy.workflows.reconst import get_fitted_tensor
from dipy.reconst.dti import fractional_anisotropy
from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.workflows.multi_io import io_iterator_


def det_track_flow(peaks_files, stopping_files, use_sh=False,
                   out_dir='', out_tractogram='tractogram.trk'):

    """ Workflow for deterministic tracking

    Parameters
    ----------
    peaks_files : string
       Path to the peaks values files. This path may contain wildcards to use
       multiple masks at once.
    stopping_files : string
        Path of FA or other images used for stopping criteria for tracking.
    use_sh : bool
        Use spherical harmonics saved in peask to find the maximum peak cone.
    out_dir : string, optional
       Output directory (default input file directory)
    out_tractogram : string, optional
       Name of the tractogram file to be saved (default 'tractogram.trk')
    """
    io_it = io_iterator_(inspect.currentframe(), EuDX_tracking_flow,
                         input_structure=False)

    for peaks_path, stopping_path, out_tract in io_it:
        logging.info('Deterministic tracking on {0}'.format(peaks_path))
        pam = load_peaks(peaks_path)
        stop, affine = load_nifti(stopping_path)
        classifier = ThresholdTissueClassifier(stop, .25)
        seed_mask = stop > .2
        seeds = utils.seeds_from_mask(
                seed_mask, density=[2, 2, 2], affine=affine)

        if use_sh:
            detmax_dg = \
                DeterministicMaximumDirectionGetter.from_shcoeff(
                    pam.shm_coeff,
                    max_angle=30.,
                    sphere=pam.sphere)

            streamlines = \
                LocalTracking(detmax_dg, classifier, seeds, affine,
                              step_size=.5)

        else:
            streamlines = LocalTracking(pam, classifier,
                                        seeds, affine, step_size=.5)

        # Compute streamlines and store as a list.
        streamlines = list(streamlines)
        save_trk(out_tract, streamlines)


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
    io_it = io_iterator_(inspect.currentframe(), EuDX_tracking_flow,
                         input_structure=False)

    for peaks_values_path, peaks_idx_path, out_tract in io_it:
        logging.info('EuDX tracking on {0}'.format(peaks_values_path))

        peaks_idx_img = nib.load(peaks_idx_path)
        peaks_idx = peaks_idx_img.get_data()
        peaks_value = nib.load(peaks_values_path).get_data()

        # Run Dipy EuDX Tracking
        tracks_iter = EuDX(peaks_value,
                           peaks_idx,
                           odf_vertices=get_sphere('symmetric362').vertices,
                           seeds=100000)

        # Save streamlines (tracking results)
        streamlines_trk = [(sl) for sl in tracks_iter]

        translation = peaks_idx_img.get_affine()
        translation[:3, 3] = 0
        save_trk(out_tract, streamlines_trk, transfo=translation)
        logging.info('Saved {0}'.format(out_tract))

    return io_it


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

    io_it = io_iterator_(inspect.currentframe(), deterministic_tracking_flow,
                         input_structure=False)

    for dwi, mask, bval, bvec, out_tract in io_it:

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

        streamlines_trk = [(sl) for sl in streamlines]
        translation = np.eye(4)
        translation[:3, 3] = dwi_img.get_affine()[:3, 3] * -1.0
        save_trk(out_tract, streamlines_trk, transfo=translation)

        logging.info('Saved {0}'.format(out_tract))

    return io_it
