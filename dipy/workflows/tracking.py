#!/usr/bin/env python
from __future__ import division

import logging
import numpy as np

from nibabel.streamlines import save, Tractogram

from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter,
                            ClosestPeakDirectionGetter)
from dipy.io.image import load_nifti
from dipy.io.peaks import load_peaks
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking,
                                 CmcTissueClassifier,
                                 ParticleFilteringTracking)
from dipy.workflows.workflow import Workflow


class LocalFiberTrackingPAMFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'track_local'

    def _get_direction_getter(self, strategy_name, pam, pmf_threshold=0.1,
                              max_angle=30.):
        """Get Tracking Direction Getter object.

        Parameters
        ----------
        strategy_name: str
            string representing direction getter name

        Returns
        -------
        direction_getter : instance of DirectionGetter
            Used to get directions for fiber tracking.

        """
        dg, msg = None, ''
        if strategy_name.lower() in ["deterministic", "det"]:
            msg = "Deterministic"
            dg = DeterministicMaximumDirectionGetter.from_shcoeff(
                pam.shm_coeff,
                sphere=pam.sphere,
                max_angle=max_angle,
                pmf_threshold=pmf_threshold)
        elif strategy_name.lower() in ["probabilistic", "prob"]:
            msg = "Probabilistic"
            dg = ProbabilisticDirectionGetter.from_shcoeff(
                pam.shm_coeff,
                sphere=pam.sphere,
                max_angle=max_angle,
                pmf_threshold=pmf_threshold)
        elif strategy_name.lower() in ["closestpeaks", "cp"]:
            msg = "ClosestPeaks"
            dg = ClosestPeakDirectionGetter.from_shcoeff(
                pam.shm_coeff,
                sphere=pam.sphere,
                max_angle=max_angle,
                pmf_threshold=pmf_threshold)
        elif strategy_name.lower() in ["eudx", ]:
            msg = "Eudx"
            dg = pam
        else:
            msg = "No direction getter defined. Deterministic"
            dg = DeterministicMaximumDirectionGetter.from_shcoeff(
                pam.shm_coeff,
                sphere=pam.sphere,
                max_angle=max_angle,
                pmf_threshold=pmf_threshold)

        logging.info('{0} direction getter strategy selected'.format(msg))
        return dg

    def _core_run(self, stopping_path, stopping_thr, seeding_path,
                  seed_density, direction_getter, out_tract):

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

        streamlines = LocalTracking(direction_getter, classifier,
                                    seeds, affine, step_size=.5)
        logging.info('LocalTracking initiated')

        tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        save(tractogram, out_tract)

        logging.info('Saved {0}'.format(out_tract))

    def run(self, pam_files, stopping_files, seeding_files,
            stopping_thr=0.2,
            seed_density=1,
            tracking_method="deterministic",
            pmf_threshold=0.1,
            max_angle=30.,
            out_dir='',
            out_tractogram='tractogram.trk'):
        """Workflow for Local Fiber Tracking.

        This workflow use a saved peaks and metrics (PAM) file as input.

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
             For example, seed_density of 2 means 8 regularly distributed
             points in the voxel. And seed density of 1 means 1 point at the
             center of the voxel.
        tracking_method : string, optional
            Select direction getter strategy:
             - "eudx" (Uses the peaks saved in the pam_files)
             - "deterministic" or "det" for a deterministic tracking
               (Uses the sh saved in the pam_files, default)
             - "probabilistic" or "prob" for a Probabilistic tracking
               (Uses the sh saved in the pam_files)
             - "closestpeaks" or "cp" for a ClosestPeaks tracking
               (Uses the sh saved in the pam_files)
        pmf_threshold : float, optional
            Threshold for ODF functions. (default 0.1)
        max_angle : float, optional
            Maximum angle between tract segments. This angle can be more
            generous (larger) than values typically used with probabilistic
            direction getters. The angle range is (0, 90)
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

            logging.info('Local tracking on {0}'
                         .format(pams_path))

            pam = load_peaks(pams_path, verbose=False)
            dg = self._get_direction_getter(tracking_method, pam,
                                            pmf_threshold=pmf_threshold,
                                            max_angle=max_angle)

            self._core_run(stopping_path, stopping_thr, seeding_path,
                           seed_density, dg, out_tract)


class PFTrackingPAMFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'track_pft'

    def run(self, pam_files, wm_files, gm_files, csf_files, seeding_files,
            step_size=0.2,
            back_tracking_dist=2,
            front_tracking_dist=1,
            max_trial=20,
            particle_count=15,
            seed_density=1,
            pmf_threshold=0.1,
            max_angle=30.,
            out_dir='',
            out_tractogram='tractogram.trk'):
        """Workflow for Particle Filtering Tracking.

        This workflow use a saved peaks and metrics (PAM) file as input.

        Parameters
        ----------
        pam_files : string
           Path to the peaks and metrics files. This path may contain
            wildcards to use multiple masks at once.
        wm_files : string
            Path of White matter for stopping criteria for tracking.
        gm_files : string
            Path of grey matter for stopping criteria for tracking.
        csf_files : string
            Path of cerebrospinal fluid for stopping criteria for tracking.
        seeding_files : string
            A binary image showing where we need to seed for tracking.
        step_size : float, optional
            Step size used for tracking.
        back_tracking_dist : float, optional
            Distance in mm to back track before starting the particle filtering
            tractography. The total particle filtering tractography distance is
            equal to back_tracking_dist + front_tracking_dist.
            By default this is set to 2 mm.
        front_tracking_dist : float, optional
            Distance in mm to run the particle filtering tractography after the
            the back track distance. The total particle filtering tractography
            distance is equal to back_tracking_dist + front_tracking_dist. By
            default this is set to 1 mm.
        max_trial : int, optional
            Maximum number of trial for the particle filtering tractography
            (Prevents infinite loops, default=20).
        particle_count : int, optional
            Number of particles to use in the particle filter. (default 15)
        seed_density : int, optional
            Number of seeds per dimension inside voxel (default 1).
             For example, seed_density of 2 means 8 regularly distributed
             points in the voxel. And seed density of 1 means 1 point at the
             center of the voxel.
        pmf_threshold : float, optional
            Threshold for ODF functions. (default 0.1)
        max_angle : float, optional
            Maximum angle between tract segments. This angle can be more
            generous (larger) than values typically used with probabilistic
            direction getters. The angle range is (0, 90)
        out_dir : string, optional
           Output directory (default input file directory)
        out_tractogram : string, optional
           Name of the tractogram file to be saved (default 'tractogram.trk')

        References
        ----------
        Girard, G., Whittingstall, K., Deriche, R., & Descoteaux, M.
               Towards quantitative connectivity analysis: reducing
               tractography biases. NeuroImage, 98, 266-278, 2014..

        """
        io_it = self.get_io_iterator()

        for pams_path, wm_path, gm_path, csf_path, seeding_path, out_tract \
                in io_it:

            logging.info('Particle Filtering tracking on {0}'
                         .format(pams_path))

            pam = load_peaks(pams_path, verbose=False)

            wm, affine, voxel_size = load_nifti(wm_path, return_voxsize=True)
            gm, _ = load_nifti(gm_path)
            csf, _ = load_nifti(csf_path)
            avs = sum(voxel_size) / len(voxel_size)  # average_voxel_size
            classifier = CmcTissueClassifier.from_pve(wm, gm, csf,
                                                      step_size=step_size,
                                                      average_voxel_size=avs)
            logging.info('classifier done')
            seed_mask, _ = load_nifti(seeding_path)
            seeds = utils.seeds_from_mask(seed_mask,
                                          density=[seed_density, seed_density,
                                                   seed_density],
                                          affine=affine)
            logging.info('seeds done')
            dg = ProbabilisticDirectionGetter

            direction_getter = dg.from_shcoeff(pam.shm_coeff,
                                               max_angle=max_angle,
                                               sphere=pam.sphere,
                                               pmf_threshold=pmf_threshold)

            streamlines = ParticleFilteringTracking(
                direction_getter,
                classifier,
                seeds, affine,
                step_size=step_size,
                pft_back_tracking_dist=back_tracking_dist,
                pft_front_tracking_dist=front_tracking_dist,
                pft_max_trial=max_trial,
                particle_count=particle_count)

            logging.info('ParticleFilteringTracking initiated')

            tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
            save(tractogram, out_tract)

            logging.info('Saved {0}'.format(out_tract))
