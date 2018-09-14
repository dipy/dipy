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
        return 'lf_track'

    def _get_direction_getter(self, strategy_name):
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
        direction_getter = DeterministicMaximumDirectionGetter
        msg = ''
        if strategy_name.lower() in ["deterministic", "det"]:
            msg = "Deterministic"
            direction_getter = DeterministicMaximumDirectionGetter
        elif strategy_name.lower() in ["probabilistic", "prob"]:
            msg = "Probabilistic"
            direction_getter = ProbabilisticDirectionGetter
        elif strategy_name.lower() in ["closestpeaks", "cp"]:
            msg = "ClosestPeaks"
            direction_getter = ClosestPeakDirectionGetter
        else:
            msg = "No direction getter defined. Deterministic"

        logging.info('{0} direction getter strategy selected'.format(msg))
        return direction_getter

    def _core_run(self, stopping_path, stopping_thr, seeding_path,
                  seed_density, use_sh, dg, pam, out_tract):

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
            direction_getter = dg.from_shcoeff(pam.shm_coeff,
                                               max_angle=30.,
                                               sphere=pam.sphere)

        streamlines = LocalTracking(direction_getter, classifier,
                                    seeds, affine, step_size=.5)
        logging.info('LocalTracking initiated')

        tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        save(tractogram, out_tract)

        logging.info('Saved {0}'.format(out_tract))

    def run(self, pam_files, stopping_files, seeding_files,
            stopping_thr=0.2,
            seed_density=1,
            use_sh=False,
            sh_strategy="deterministic",
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
        use_sh : bool, optional
            Use spherical harmonics saved in peaks to find the
             maximum peak cone. (default False)
        sh_strategy : string, optional
            Select direction getter strategy:
             - "deterministic" or "det" for a deterministic tracking
             - "probabilistic" or "prob" for a Probabilistic tracking
             - "closestpeaks" or "cp" for a ClosestPeaks tracking
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
            dg = self._get_direction_getter(sh_strategy)

            self._core_run(stopping_path, stopping_thr, seeding_path,
                           seed_density, use_sh, dg, pam,
                           out_tract)


class PFTrackingPAMFlow(LocalFiberTrackingPAMFlow):
    @classmethod
    def get_short_name(cls):
        return 'pf_track'

    def _core_run(self, wm_path, gm_path, csf_path, seeding_path,
                  d_back, d_front, max_trial, particle_count, seed_density,
                  use_sh, dg, pam, out_tract):

        wm, affine, voxel_size = load_nifti(wm_path, return_voxsize=True)
        gm, _ = load_nifti(gm_path)
        csf, _ = load_nifti(csf_path)
        classifier = CmcTissueClassifier(wm, gm, csf,
                                         step_size=0.2,
                                         average_voxel_size=voxel_size)
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
            direction_getter = dg.from_shcoeff(pam.shm_coeff,
                                               max_angle=30.,
                                               sphere=pam.sphere)

        streamlines = ParticleFilteringTracking(direction_getter, classifier,
                                                seeds, affine, step_size=.5,
                                                pft_back_tracking_dist=d_back,
                                                pft_front_tracking_dist=d_front,
                                                pft_max_trial=max_trial,
                                                particle_count=particle_count)
        logging.info('ParticleFilteringTracking initiated')

        tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        save(tractogram, out_tract)

        logging.info('Saved {0}'.format(out_tract))

    def run(self, pam_files, wm_files, gm_files, csf_files, seeding_files,
            back_tracking_dist=2,
            front_tracking_dist=1,
            max_trial=20,
            particle_count=15,
            seed_density=1,
            use_sh=False,
            sh_strategy="deterministic",
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
            (Prevents infinite loops).
        particle_count : int, optional
            Number of particles to use in the particle filter.
        seed_density : int, optional
            Number of seeds per dimension inside voxel (default 1).
             For example, seed_density of 2 means 8 regularly distributed
             points in the voxel. And seed density of 1 means 1 point at the
             center of the voxel.
        use_sh : bool, optional
            Use spherical harmonics saved in peaks to find the
             maximum peak cone. (default False)
        sh_strategy : string, optional
            Select direction getter strategy:
             - "deterministic" or "det" for a deterministic tracking
             - "probabilistic" or "prob" for a Probabilistic tracking
             - "closestpeaks" or "cp" for a ClosestPeaks tracking
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

        for pams_path, wm_path, gm_path, csf_path, seeding_path, out_tract in io_it:

            logging.info('Particle Filtering tracking on {0}'
                         .format(pams_path))

            pam = load_peaks(pams_path, verbose=False)
            dg = self._get_direction_getter(sh_strategy)

            self._core_run(wm_path, gm_path, csf_path, seeding_path,
                           back_tracking_dist, front_tracking_dist, max_trial,
                           particle_count, seed_density, use_sh, dg, pam,
                           out_tract)
