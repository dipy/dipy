#!/usr/bin/env python
from __future__ import division

import logging

from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.image import load_nifti
from dipy.io.peaks import load_peaks
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier,
                                 LocalTracking)
from dipy.workflows.workflow import Workflow

import numpy as np
from nibabel.streamlines import save, Tractogram


class GenericTrackFlow(Workflow):

    def _core_run(self, stopping_path, stopping_thr, seeding_path, seed_density,
                  use_sh, shm_coeff, sphere, out_tract):
        stop, affine = load_nifti(stopping_path)
        classifier = ThresholdTissueClassifier(stop, stopping_thr)

        seed_mask, _ = load_nifti(seeding_path)
        seeds = \
            utils.seeds_from_mask(
                seed_mask,
                density=[seed_density, seed_density, seed_density],
                affine=affine)

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

        tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        save(tractogram, out_tract)

        logging.info('Saved {0}'.format(out_tract))


class DetTrackFlow(GenericTrackFlow):
    @classmethod
    def get_short_name(cls):
        return 'tracking'

    def run(self, pam_files, stopping_files, seeding_files,
            stopping_thr=0.2,
            seed_density=1,
            use_sh=False,
            out_dir='',
            out_tractogram='tractogram.trk'):

        """ Workflow for deterministic tracking

        Parameters
        ----------
        pam_files : string
           Path to the peaks values files. This path may contain
           wildcards to use multiple masks at once.
        stopping_files : string
            Path of FA or other images used for stopping criteria for tracking.
        seeding_files : string
            A binary image showing where we need to seed for tracking.
        stopping_thr : float
            Threshold applied to stopping volume's data to identify where
            tracking has to stop. (default 0.25)
        seed_density : int
            Number of seeds per dimension inside voxel (default 1).
        use_sh : bool
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

            pam = load_peaks(pams_path)

            stop, affine = load_nifti(stopping_path)
            classifier = ThresholdTissueClassifier(stop, stopping_thr)

            seed_mask, _ = load_nifti(seeding_path)
            seeds = \
                utils.seeds_from_mask(
                    seed_mask,
                    density=[seed_density, seed_density, seed_density],
                    affine=affine)

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

            tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
            save(tractogram, out_tract)

            logging.info('Saved {0}'.format(out_tract))

