#!/usr/bin/env python
from __future__ import division

import logging
import numpy as np

import nibabel as nib
from nibabel.streamlines import save, Tractogram

from dipy.data import get_sphere
from dipy.direction import DeterministicMaximumDirectionGetter, PeaksAndMetrics
from dipy.io.image import load_nifti
from dipy.io.peaks import load_peaks
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier,
                                 LocalTracking)
from dipy.workflows.workflow import Workflow


class GenericTrackFlow(Workflow):
    def _core_run(self, stopping_path, stopping_thr, seeding_path, seed_density,
                  use_sh, pam, out_tract):

        stop, affine = load_nifti(stopping_path)
        classifier = ThresholdTissueClassifier(stop, stopping_thr)

        seed_mask, _ = load_nifti(seeding_path)
        seeds = \
            utils.seeds_from_mask(
                seed_mask,
                density=[seed_density, seed_density, seed_density],
                affine=affine)

        direction_getter = pam

        if use_sh:
            direction_getter = \
                DeterministicMaximumDirectionGetter.from_shcoeff(
                    pam.shm_coeff,
                    max_angle=30.,
                    sphere=pam.sphere)

        streamlines = LocalTracking(direction_getter, classifier,
                                    seeds, affine, step_size=.5)

        tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        save(tractogram, out_tract)

        logging.info('Saved {0}'.format(out_tract))


class DetTrackPAMFlow(GenericTrackFlow):
    @classmethod
    def get_short_name(cls):
        return 'tracking'

    def run(self, pam_files, stopping_files, seeding_files,
            stopping_thr=0.2,
            seed_density=1,
            use_sh=False,
            out_dir='',
            out_tractogram='tractogram.trk'):

        """ Workflow for deterministic using a saved peaks and metrics (PAM)
        file as input.

        Parameters
        ----------
        pam_files : string
           Path to the peaks and metrics files. This path may contain
           wildcards to use multiple masks at once.
        stopping_files : string
            Path of FA or other images used for stopping criteria for tracking.
        seeding_files : string
            A binary image showing where we need to seed for tracking.
        stopping_thr : float, optional
            Threshold applied to stopping volume's data to identify where
            tracking has to stop. (default 0.25)
        seed_density : int, optional
            Number of seeds per dimension inside voxel (default 1).
        use_sh : bool, optional
            Use spherical harmonics saved in peask to find the
            maximum peak cone. (default False)
        out_dir : string, optional
           Output directory (default input file directory)
        out_tractogram : string, optional
           Name of the tractogram file to be saved
           (default 'tractogram.trk')
        """
        io_it = self.get_io_iterator()

        for pams_path, stopping_path, seeding_path, out_tract in io_it:

            logging.info('Deterministic tracking on {0}'
                         .format(pams_path))

            pam = load_peaks(pams_path, verbose=False)

            self._core_run(stopping_path, stopping_thr, seeding_path,
                           seed_density, use_sh, pam, out_tract)


class DetTrackPeaksFlow(GenericTrackFlow):
    @classmethod
    def get_short_name(cls):
        return 'tracking'

    def run(self, peaks_values, peaks_idxs, peaks_dirs, stopping_files,
            seeding_files, stopping_thr=0.2, seed_density=1, out_dir='',
            out_tractogram='tractogram.trk'):

        """ Workflow for deterministic tracking using peaks

        Parameters
        ----------
        peaks_values : string
           Path to the peaks values files. This path may contain
           wildcards to use multiple masks at once.
        peaks_idxs : string
           Path to the peaks indices files. This path may contain
           wildcards to use multiple masks at once.
        peaks_dirs : string
           Path to the peaks directions files. This path may contain
           wildcards to use multiple masks at once.
        stopping_files : string
            Path of FA or other images used for stopping criteria for tracking.
        seeding_files : string
            A binary image showing where we need to seed for tracking.
        stopping_thr : float, optional
            Threshold applied to stopping volume's data to identify where
            tracking has to stop. (default 0.25)
        seed_density : int, optional
            Number of seeds per dimension inside voxel (default 1).
        out_dir : string, optional
           Output directory (default input file directory)
        out_tractogram : string, optional
           Name of the tractogram file to be saved
           (default 'tractogram.trk')
        """
        io_it = self.get_io_iterator()

        for peaks_vals_path, peaks_idx_path, peaks_dirs_path, stopping_path,\
            seeding_path, out_tract in io_it:

            logging.info('Deterministic tracking on {0}'
                         .format(peaks_vals_path))

            sphere = get_sphere('symmetric362')
            pam = PeaksAndMetrics()
            pam.sphere = sphere
            pam.peak_dirs = nib.load(peaks_dirs_path).get_data()
            pam.peak_values = nib.load(peaks_vals_path).get_data()
            pam.peak_indices = nib.load(peaks_idx_path).get_data()

            self._core_run(stopping_path, stopping_thr, seeding_path,
                           seed_density, False, pam, out_tract)


class DetTrackSHFlow(GenericTrackFlow):
    @classmethod
    def get_short_name(cls):
        return 'tracking'

    def run(self, sh_files, stopping_files, seeding_files, stopping_thr=0.2,
            seed_density=1, out_dir='', out_tractogram='tractogram.trk'):

        """ Workflow for deterministic tracking using spherical harmonics

        Parameters
        ----------
        sh_files : string
           Path to the peaks values files. This path may contain
           wildcards to use multiple masks at once.
        stopping_files : string
            Path of FA or other images used for stopping criteria for tracking.
        seeding_files : string
            A binary image showing where we need to seed for tracking.
        stopping_thr : float, optional
            Threshold applied to stopping volume's data to identify where
            tracking has to stop. (default 0.25)
        seed_density : int, optional
            Number of seeds per dimension inside voxel (default 1).
        out_dir : string, optional
           Output directory (default input file directory)
        out_tractogram : string, optional
           Name of the tractogram file to be saved
           (default 'tractogram.trk')
        """
        io_it = self.get_io_iterator()

        for sh_path, stopping_path, seeding_path, out_tract in io_it:

            logging.info('Deterministic tracking on {0}'
                         .format(sh_path))

            sphere = get_sphere('symmetric362')
            pam = PeaksAndMetrics()
            pam.sphere = sphere
            pam.shm_coeff = nib.load(sh_path).get_data().astype('float64')

            self._core_run(stopping_path, stopping_thr, seeding_path,
                           seed_density, True, pam, out_tract)
