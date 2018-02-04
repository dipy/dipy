#!/usr/bin/env python
from __future__ import division

import logging
import numpy as np

from nibabel.streamlines import save, Tractogram

from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.image import load_nifti
from dipy.io.peaks import load_peaks
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier,
                                 LocalTracking)
from dipy.workflows.workflow import Workflow


class GenericTrackFlow(Workflow):
    def _core_run(self, stopping_path, stopping_thr, seeding_path,
                  seed_density, use_sh, pam, out_tract):

        stop, affine = load_nifti(stopping_path)
        classifier = ThresholdTissueClassifier(stop, stopping_thr)
        logging.info('classifier done')
        seed_mask, _ = load_nifti(seeding_path)
        seeds = \
            utils.seeds_from_mask(
                seed_mask,
                density=[seed_density, seed_density, seed_density],
                affine=affine)
        logging.info('seeds done')
        direction_getter = pam

        if use_sh:
            direction_getter = \
                DeterministicMaximumDirectionGetter.from_shcoeff(
                    pam.shm_coeff,
                    max_angle=30.,
                    sphere=pam.sphere)

        streamlines = LocalTracking(direction_getter, classifier,
                                    seeds, affine, step_size=.5)
        logging.info('LocalTracking initiated')

        tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        save(tractogram, out_tract)

        logging.info('Saved {0}'.format(out_tract))


class DetTrackPAMFlow(GenericTrackFlow):
    @classmethod
    def get_short_name(cls):
        return 'det_track'

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
             tracking has to stop (default 0.25).
        seed_density : int, optional
            Number of seeds per dimension inside voxel (default 1).
             For example, seed_density of 2 means 8 regularly distributed points
             in the voxel. And seed density of 1 means 1 point at the center
             of the voxel.
        use_sh : bool, optional
            Use spherical harmonics saved in peaks to find the
             maximum peak cone. (default False)
        out_dir : string, optional
           Output directory (default input file directory)
        out_tractogram : string, optional
           Name of the tractogram file to be saved (default 'tractogram.trk')

        References
        ----------
        Garyfallidis, University of Cambridge, PhD thesis 2012.
        Amirbekian, University of California San Francisco, PhD thesis 2017.
        """
        io_it = self.get_io_iterator()

        for pams_path, stopping_path, seeding_path, out_tract in io_it:

            logging.info('Deterministic tracking on {0}'
                         .format(pams_path))

            pam = load_peaks(pams_path, verbose=False)

            self._core_run(stopping_path, stopping_thr, seeding_path,
                           seed_density, use_sh, pam, out_tract)

