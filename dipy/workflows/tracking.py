#!/usr/bin/env python3

import logging
import sys

from dipy.data import SPHERE_FILES, get_sphere
from dipy.io.image import load_nifti
from dipy.io.peaks import load_pam
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import (
    BinaryStoppingCriterion,
    CmcStoppingCriterion,
    ThresholdStoppingCriterion,
)
from dipy.tracking.tracker import (
    closestpeak_tracking,
    deterministic_tracking,
    eudx_tracking,
    pft_tracking,
    probabilistic_tracking,
    ptt_tracking,
)
from dipy.workflows.workflow import Workflow


class LocalFiberTrackingPAMFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "track_local"

    def run(
        self,
        pam_files,
        stopping_files,
        seeding_files,
        use_binary_mask=False,
        stopping_thr=0.2,
        seed_density=1,
        minlen=2,
        maxlen=500,
        step_size=0.5,
        tracking_method="deterministic",
        pmf_threshold=0.1,
        max_angle=30.0,
        sphere_name=None,
        save_seeds=False,
        nbr_threads=0,
        random_seed=1,
        seed_buffer_fraction=1.0,
        out_dir="",
        out_tractogram="tractogram.trk",
    ):
        """Workflow for Local Fiber Tracking.

        This workflow use a saved peaks and metrics (PAM) file as input.

        See :footcite:p:`Garyfallidis2012b` and :footcite:p:`Amirbekian2016`
        for further details about the method.

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
        minlen : int, optional
            Minimum length (mm) of the streamlines.
        maxlen : int, optional
            Maximum length (mm) of the streamlines.
        step_size : float, optional
            Step size (in mm) used for tracking.
        tracking_method : string, optional
            Select direction getter strategy:
                - "eudx" (Uses the peaks saved in the pam_files)
                - "deterministic" or "det" for a deterministic tracking
                - "probabilistic" or "prob" for a Probabilistic tracking
                - "closestpeaks" or "cp" for a ClosestPeaks tracking
                - "ptt" for Parallel Transport Tractography

            By default, the sh coefficients saved in the pam_files are used.
        pmf_threshold : float, optional
            Threshold for ODF functions.
        max_angle : float, optional
            Maximum angle between streamline segments (range [0, 90]).
        sphere_name : string, optional
            The sphere used for tracking. If None, the sphere saved in the
            pam_files is used. For faster tracking, use a smaller
            sphere (e.g. 'repulsion200').
        save_seeds : bool, optional
            If true, save the seeds associated to their streamline
            in the 'data_per_streamline' Tractogram dictionary using
            'seeds' as the key.
        nbr_threads : int, optional
            Number of threads to use for the processing. By default, all available
            threads will be used.
        random_seed : int, optional
            Seed for the random number generator, must be >= 0. A value of greater
            than 0 will all produce the same streamline trajectory for a given seed
            coordinate. A value of 0 may produces various streamline tracjectories
            for a given seed coordinate.
        seed_buffer_fraction : float, optional
            Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
            buffer. A value of 0.5 will use half of the seed buffer then the other half.
            a way to reduce memory usage.
        out_dir : string, optional
           Output directory.
        out_tractogram : string, optional
           Name of the tractogram file to be saved.

        References
        ----------
        .. footbibliography::

        """
        io_it = self.get_io_iterator()

        tracking_method = tracking_method.lower()
        sphere_name = SPHERE_FILES.get(sphere_name, None)

        for pams_path, stopping_path, seeding_path, out_tract in io_it:
            if tracking_method == "eudx":
                logging.info(f"EuDX Deterministic tracking on {pams_path}")
            else:
                logging.info(f"{tracking_method.title()} tracking on {pams_path}")

            pam = load_pam(pams_path, verbose=False)
            sphere = pam.sphere if sphere_name is None else get_sphere(name=sphere_name)

            logging.info("Loading stopping criterion")
            stop, affine = load_nifti(stopping_path)
            if use_binary_mask:
                stopping_criterion = BinaryStoppingCriterion(stop > stopping_thr)
            else:
                stopping_criterion = ThresholdStoppingCriterion(stop, stopping_thr)

            logging.info("Loading seeds")
            seed_mask, _ = load_nifti(seeding_path)
            seeds = utils.seeds_from_mask(
                seed_mask, affine, density=[seed_density, seed_density, seed_density]
            )

            logging.info("Starting to track")
            if tracking_method in ["closestpeaks", "cp"]:
                tracking_result = closestpeak_tracking(
                    seeds,
                    stopping_criterion,
                    affine,
                    sh=pam.shm_coeff,
                    random_seed=random_seed,
                    sphere=sphere,
                    max_angle=max_angle,
                    min_len=minlen,
                    max_len=maxlen,
                    step_size=step_size,
                    pmf_threshold=pmf_threshold,
                    save_seeds=save_seeds,
                    nbr_threads=nbr_threads,
                    seed_buffer_fraction=seed_buffer_fraction,
                )
            elif tracking_method in [
                "eudx",
            ]:
                tracking_result = eudx_tracking(
                    seeds,
                    stopping_criterion,
                    affine,
                    sh=pam.shm_coeff,
                    pam=pam,
                    random_seed=random_seed,
                    sphere=sphere,
                    max_angle=max_angle,
                    min_len=minlen,
                    max_len=maxlen,
                    step_size=step_size,
                    pmf_threshold=pmf_threshold,
                    save_seeds=save_seeds,
                    nbr_threads=nbr_threads,
                    seed_buffer_fraction=seed_buffer_fraction,
                )
            elif tracking_method in ["deterministic", "det"]:
                tracking_result = deterministic_tracking(
                    seeds,
                    stopping_criterion,
                    affine,
                    sh=pam.shm_coeff,
                    random_seed=random_seed,
                    sphere=sphere,
                    max_angle=max_angle,
                    min_len=minlen,
                    max_len=maxlen,
                    step_size=step_size,
                    pmf_threshold=pmf_threshold,
                    save_seeds=save_seeds,
                    nbr_threads=nbr_threads,
                    seed_buffer_fraction=seed_buffer_fraction,
                )
            elif tracking_method in ["probabilistic", "prob"]:
                tracking_result = probabilistic_tracking(
                    seeds,
                    stopping_criterion,
                    affine,
                    sh=pam.shm_coeff,
                    random_seed=random_seed,
                    sphere=sphere,
                    max_angle=max_angle,
                    min_len=minlen,
                    max_len=maxlen,
                    step_size=step_size,
                    pmf_threshold=pmf_threshold,
                    save_seeds=save_seeds,
                    nbr_threads=nbr_threads,
                    seed_buffer_fraction=seed_buffer_fraction,
                )
            elif tracking_method in ["ptt"]:
                tracking_result = ptt_tracking(
                    seeds,
                    stopping_criterion,
                    affine,
                    sh=pam.shm_coeff,
                    random_seed=random_seed,
                    sphere=sphere,
                    max_angle=max_angle,
                    min_len=minlen,
                    max_len=maxlen,
                    step_size=step_size,
                    pmf_threshold=pmf_threshold,
                    save_seeds=save_seeds,
                    nbr_threads=nbr_threads,
                    seed_buffer_fraction=seed_buffer_fraction,
                )
            else:
                logging.error(
                    f"Unknown tracking method: {tracking_method}. "
                    f"Please use one of the following: "
                    f"'eudx', 'deterministic', 'probabilistic', 'closestpeaks', 'ptt'"
                )
                sys.exit(1)
            if save_seeds:
                streamlines, seeds = zip(*tracking_result)
                seeds = {"seeds": seeds}
            else:
                streamlines = list(tracking_result)
                seeds = {}

            sft = StatefulTractogram(
                streamlines, seeding_path, Space.RASMM, data_per_streamline=seeds
            )
            save_tractogram(sft, out_tract, bbox_valid_check=False)
            logging.info(f"Saved {out_tract}")


class PFTrackingPAMFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "track_pft"

    def run(
        self,
        pam_files,
        wm_files,
        gm_files,
        csf_files,
        seeding_files,
        step_size=0.2,
        seed_density=1,
        pmf_threshold=0.1,
        max_angle=20.0,
        sphere_name=None,
        pft_back=2,
        pft_front=1,
        pft_count=15,
        pft_max_trial=20,
        save_seeds=False,
        min_wm_pve_before_stopping=0,
        nbr_threads=0,
        random_seed=1,
        seed_buffer_fraction=1.0,
        out_dir="",
        out_tractogram="tractogram.trk",
    ):
        """Workflow for Particle Filtering Tracking.

        This workflow uses a saved peaks and metrics (PAM) file as input.

        See :footcite:p:`Girard2014` for further details about the method.

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
        sphere_name : string, optional
            The sphere used for tracking. If None, the sphere saved in the
            pam_files is used. For faster tracking, use a smaller
            sphere (e.g. 'repulsion200').
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
        pft_max_trial : int, optional
            Maximum number of trials to run the particle filtering tractography.
        save_seeds : bool, optional
            If true, save the seeds associated to their streamline
            in the 'data_per_streamline' Tractogram dictionary using
            'seeds' as the key.
        min_wm_pve_before_stopping : int, optional
            Minimum white matter pve (1 - stopping_criterion.include_map -
            stopping_criterion.exclude_map) to reach before allowing the
            tractography to stop.
        nbr_threads : int, optional
            Number of threads to use for the processing. By default, all available
            threads will be used.
        random_seed : int, optional
            Seed for the random number generator, must be >= 0. A value of greater
            than 0 will all produce the same streamline trajectory for a given seed
            coordinate. A value of 0 may produces various streamline tracjectories
            for a given seed coordinate.
        seed_buffer_fraction : float, optional
            Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
            buffer. A value of 0.5 will use half of the seed buffer then the other half.
            a way to reduce memory usage.
        out_dir : string, optional
           Output directory.
        out_tractogram : string, optional
           Name of the tractogram file to be saved.

        References
        ----------
        .. footbibliography::

        """
        io_it = self.get_io_iterator()
        sphere_name = SPHERE_FILES.get(sphere_name, None)

        for pams_path, wm_path, gm_path, csf_path, seeding_path, out_tract in io_it:
            logging.info(f"Particle Filtering tracking on {pams_path}")

            pam = load_pam(pams_path, verbose=False)
            sphere = pam.sphere if sphere_name is None else get_sphere(name=sphere_name)

            wm, affine, voxel_size = load_nifti(wm_path, return_voxsize=True)
            gm, _ = load_nifti(gm_path)
            csf, _ = load_nifti(csf_path)
            avs = sum(voxel_size) / len(voxel_size)  # average_voxel_size

            logging.info("Preparing stopping criterion")
            stopping_criterion = CmcStoppingCriterion.from_pve(
                wm, gm, csf, step_size=step_size, average_voxel_size=avs
            )

            logging.info("Seeding in mask")
            seed_mask, _ = load_nifti(seeding_path)
            seeds = utils.seeds_from_mask(
                seed_mask, affine, density=[seed_density, seed_density, seed_density]
            )

            logging.info("Start tracking")
            tracking_result = pft_tracking(
                seeds,
                stopping_criterion,
                affine,
                sh=pam.shm_coeff,
                sphere=sphere,
                step_size=step_size,
                pft_back_tracking_dist=pft_back,
                pft_front_tracking_dist=pft_front,
                pft_max_trial=pft_max_trial,
                particle_count=pft_count,
                save_seeds=save_seeds,
                min_wm_pve_before_stopping=min_wm_pve_before_stopping,
                random_seed=random_seed,
                max_angle=max_angle,
                pmf_threshold=pmf_threshold,
                nbr_threads=nbr_threads,
                seed_buffer_fraction=seed_buffer_fraction,
            )

            if save_seeds:
                streamlines, seeds = zip(*tracking_result)
                seeds = {"seeds": seeds}
            else:
                streamlines = list(tracking_result)
                seeds = {}

            sft = StatefulTractogram(
                streamlines, seeding_path, Space.RASMM, data_per_streamline=seeds
            )
            save_tractogram(sft, out_tract, bbox_valid_check=False)
            logging.info(f"Saved {out_tract}")
