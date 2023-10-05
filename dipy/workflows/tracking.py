#!/usr/bin/env python3

import logging

from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter,
                            ClosestPeakDirectionGetter)
from dipy.io.image import load_nifti
from dipy.io.peaks import load_peaks
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking import utils
from dipy.tracking.local_tracking import (LocalTracking,
                                          ParticleFilteringTracking)
from dipy.tracking.stopping_criterion import (BinaryStoppingCriterion,
                                              CmcStoppingCriterion,
                                              ThresholdStoppingCriterion)
from dipy.workflows.workflow import Workflow


class LocalFiberTrackingPAMFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'track_local'

    def _get_direction_getter(self, strategy_name, pam, pmf_threshold,
                              max_angle):
        """Get Tracking Direction Getter object.

        Parameters
        ----------
        strategy_name : str
            String representing direction getter name.
        pam : instance of PeaksAndMetrics
            An object with ``gfa``, ``peak_directions``, ``peak_values``,
            ``peak_indices``, ``odf``, ``shm_coeffs`` as attributes.
        pmf_threshold : float
            Threshold for ODF functions.
        max_angle : float
            Maximum angle between streamline segments.

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
            msg = "No direction getter defined. Eudx"
            dg = pam

        logging.info('{0} direction getter strategy selected'.format(msg))
        return dg

    def _core_run(self, stopping_path, use_binary_mask, stopping_thr,
                  seeding_path, seed_density, step_size, direction_getter,
                  out_tract, save_seeds):

        stop, affine = load_nifti(stopping_path)
        if use_binary_mask:
            stopping_criterion = BinaryStoppingCriterion(stop > stopping_thr)
        else:
            stopping_criterion = ThresholdStoppingCriterion(stop, stopping_thr)
        logging.info('stopping criterion done')
        seed_mask, _ = load_nifti(seeding_path)
        seeds = \
            utils.seeds_from_mask(
                seed_mask, affine,
                density=[seed_density, seed_density, seed_density])
        logging.info('seeds done')

        tracking_result = LocalTracking(direction_getter,
                                        stopping_criterion,
                                        seeds,
                                        affine,
                                        step_size=step_size,
                                        save_seeds=save_seeds)

        logging.info('LocalTracking initiated')

        if save_seeds:
            streamlines, seeds = zip(*tracking_result)
            seeds = {'seeds': seeds}
        else:
            streamlines = list(tracking_result)
            seeds = {}

        sft = StatefulTractogram(streamlines, seeding_path, Space.RASMM,
                                 data_per_streamline=seeds)
        save_tractogram(sft, out_tract, bbox_valid_check=False)
        logging.info('Saved {0}'.format(out_tract))

    def run(self, pam_files, stopping_files, seeding_files,
            use_binary_mask=False,
            stopping_thr=0.2,
            seed_density=1,
            step_size=0.5,
            tracking_method="eudx",
            pmf_threshold=0.1,
            max_angle=30.,
            out_dir='',
            out_tractogram='tractogram.trk',
            save_seeds=False):
        """Workflow for Local Fiber Tracking.

        This workflow use a saved peaks and metrics (PAM) file as input.

        Parameters
        ----------
        pam_files : string
           Path to the peaks and metrics files. This path may contain
            wildcards to use multiple masks at once.
        stopping_files : string
            Path to images (e.g. FA) used for stopping criterion for tracking.
        seeding_files : string
            A binary image showing where we need to seed for tracking.
        use_binary_mask : bool, optional
            If True, uses a binary stopping criterion. If the provided
            `stopping_files` are not binary, `stopping_thr` will be used to
            binarize the images.
        stopping_thr : float, optional
            Threshold applied to stopping volume's data to identify where
            tracking has to stop.
        seed_density : int, optional
            Number of seeds per dimension inside voxel.
             For example, seed_density of 2 means 8 regularly distributed
             points in the voxel. And seed density of 1 means 1 point at the
             center of the voxel.
        step_size : float, optional
            Step size (in mm) used for tracking.
        tracking_method : string, optional
            Select direction getter strategy :
             - "eudx" (Uses the peaks saved in the pam_files)
             - "deterministic" or "det" for a deterministic tracking
               (Uses the sh saved in the pam_files, default)
             - "probabilistic" or "prob" for a Probabilistic tracking
               (Uses the sh saved in the pam_files)
             - "closestpeaks" or "cp" for a ClosestPeaks tracking
               (Uses the sh saved in the pam_files)
        pmf_threshold : float, optional
            Threshold for ODF functions.
        max_angle : float, optional
            Maximum angle between streamline segments (range [0, 90]).
        out_dir : string, optional
           Output directory. (default current directory)
        out_tractogram : string, optional
           Name of the tractogram file to be saved.
        save_seeds : bool, optional
            If true, save the seeds associated to their streamline
            in the 'data_per_streamline' Tractogram dictionary using
            'seeds' as the key.

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

            self._core_run(stopping_path, use_binary_mask, stopping_thr,
                           seeding_path, seed_density, step_size, dg,
                           out_tract, save_seeds)


class PFTrackingPAMFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'track_pft'

    def run(self, pam_files, wm_files, gm_files, csf_files, seeding_files,
            step_size=0.2,
            seed_density=1,
            pmf_threshold=0.1,
            max_angle=20.,
            pft_back=2,
            pft_front=1,
            pft_count=15,
            out_dir='',
            out_tractogram='tractogram.trk',
            save_seeds=False,
            min_wm_pve_before_stopping=0):
        """Workflow for Particle Filtering Tracking.

        This workflow use a saved peaks and metrics (PAM) file as input.

        Parameters
        ----------
        pam_files : string
           Path to the peaks and metrics files. This path may contain
            wildcards to use multiple masks at once.
        wm_files : string
            Path to white matter partial volume estimate for tracking (CMC).
        gm_files : string
            Path to grey matter partial volume estimate for tracking (CMC).
        csf_files : string
            Path to cerebrospinal fluid partial volume estimate for tracking
            (CMC).
        seeding_files : string
            A binary image showing where we need to seed for tracking.
        step_size : float, optional
            Step size (in mm) used for tracking.
        seed_density : int, optional
            Number of seeds per dimension inside voxel.
             For example, seed_density of 2 means 8 regularly distributed
             points in the voxel. And seed density of 1 means 1 point at the
             center of the voxel.
        pmf_threshold : float, optional
            Threshold for ODF functions.
        max_angle : float, optional
            Maximum angle between streamline segments (range [0, 90]).
        pft_back : float, optional
            Distance in mm to back track before starting the particle filtering
            tractography. The total particle filtering
            tractography distance is equal to back_tracking_dist +
            front_tracking_dist.
        pft_front : float, optional
            Distance in mm to run the particle filtering tractography after the
            the back track distance. The total particle filtering
            tractography distance is equal to back_tracking_dist +
            front_tracking_dist.
        pft_count : int, optional
            Number of particles to use in the particle filter.
        out_dir : string, optional
           Output directory. (default current directory)
        out_tractogram : string, optional
           Name of the tractogram file to be saved.
        save_seeds : bool, optional
            If true, save the seeds associated to their streamline
            in the 'data_per_streamline' Tractogram dictionary using
            'seeds' as the key.
        min_wm_pve_before_stopping : int, optional
            Minimum white matter pve (1 - stopping_criterion.include_map -
            stopping_criterion.exclude_map) to reach before allowing the
            tractography to stop.

        References
        ----------
        Girard, G., Whittingstall, K., Deriche, R., & Descoteaux, M. Towards
        quantitative connectivity analysis: reducing tractography biases.
        NeuroImage, 98, 266-278, 2014.

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
            stopping_criterion = CmcStoppingCriterion.from_pve(
                wm, gm, csf, step_size=step_size, average_voxel_size=avs)
            logging.info('stopping criterion done')
            seed_mask, _ = load_nifti(seeding_path)
            seeds = utils.seeds_from_mask(seed_mask, affine,
                                          density=[seed_density, seed_density,
                                                   seed_density])
            logging.info('seeds done')
            dg = ProbabilisticDirectionGetter

            direction_getter = dg.from_shcoeff(pam.shm_coeff,
                                               max_angle=max_angle,
                                               sphere=pam.sphere,
                                               pmf_threshold=pmf_threshold)

            tracking_result = ParticleFilteringTracking(
                direction_getter,
                stopping_criterion,
                seeds, affine,
                step_size=step_size,
                pft_back_tracking_dist=pft_back,
                pft_front_tracking_dist=pft_front,
                pft_max_trial=20,
                particle_count=pft_count,
                save_seeds=save_seeds,
                min_wm_pve_before_stopping=min_wm_pve_before_stopping)

            logging.info('ParticleFilteringTracking initiated')

            if save_seeds:
                streamlines, seeds = zip(*tracking_result)
                seeds = {'seeds': seeds}
            else:
                streamlines = list(tracking_result)
                seeds = {}

            sft = StatefulTractogram(streamlines, seeding_path, Space.RASMM,
                                     data_per_streamline=seeds)
            save_tractogram(sft, out_tract, bbox_valid_check=False)
            logging.info('Saved {0}'.format(out_tract))
