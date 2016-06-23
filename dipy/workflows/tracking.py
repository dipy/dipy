#!/usr/bin/env python
from __future__ import division

import logging

from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.image import load_nifti
from dipy.io.peaks import load_peaks
from dipy.io.trackvis import save_trk
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier,
                                 LocalTracking)
from dipy.workflows.workflow import Workflow


class DetTrackFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'tracking'

    def run(self, peaks_files, stopping_files, seeding_files,
            stopping_thr=0.2,
            seed_density=1,
            use_sh=False,
            out_dir='',
            out_tractogram='tractogram.trk'):

        """ Workflow for deterministic tracking

        Parameters
        ----------
        peaks_files : string
           Path to the peaks values files. This path may contain
           wildcards to use multiple masks at once.
        stopping_files : string
            Path of FA or other images used for stopping criteria
            for tracking.
        seeding_files : string
            A binary image showing where we need to seed for
            tracking.
        seed_density : int
            Number of seeds per dimension inside voxel
            (default is 1).
        stopping_thr : float
            Default is 0.25
        use_sh : bool
            Use spherical harmonics saved in peask to find the
            maximum peak cone.
        out_dir : string, optional
           Output directory (default input file directory)
        out_tractogram : string, optional
           Name of the tractogram file to be saved
           (default 'tractogram.trk')
        """
        io_it = self.get_io_iterator()

        for peaks_path, stopping_path, seeding_path, \
                out_tract in io_it:
            logging.info('Deterministic tracking on {0}'
                         .format(peaks_path))
            pam = load_peaks(peaks_path)
            stop, affine = load_nifti(stopping_path)
            classifier = ThresholdTissueClassifier(stop,
                                                   stopping_thr)
            # seed_mask = stop > .2
            seed_mask, _ = load_nifti(seeding_path)
            seeds = utils.seeds_from_mask(
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

            # Compute streamlines and store as a list.
            streamlines = list(streamlines)

            # Currently not working with 2x2x2 vols.
            save_trk(out_tract, streamlines, affine_to_rasmm=affine)
            logging.info('Saved {0}'.format(out_tract))
