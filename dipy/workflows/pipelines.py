from __future__ import division, print_function, absolute_import

import inspect
import logging
import os

from dipy.workflows.denoise import NLMeansFlow
from dipy.workflows.mask import MaskFlow
from dipy.workflows.combined_workflow import CombinedWorkflow
from dipy.workflows.reconst import ReconstODFFlow
from dipy.workflows.reconst import ReconstDtiFlow
from dipy.workflows.segment import MedianOtsuFlow
from dipy.workflows.track_metrics import TrackDensityFlow
from dipy.workflows.tracking import DetTrackFlow


class ClassicFlow(CombinedWorkflow):

    def _get_sub_flows(self):
        return [
            MedianOtsuFlow,
            NLMeansFlow,
            ReconstDtiFlow,
            ReconstODFFlow,
            DetTrackFlow,
            TrackDensityFlow,
            MaskFlow
        ]

    def run(self, input_files, bvalues, bvectors, resume=False, out_dir=''):
        """ A simple dwi processing pipeline with the following steps:
            -Denoising
            -Masking
            -DTI reconstruction
            -HARDI recontruction
            -Deterministic tracking
            -Tracts metrics

        Parameters
        ----------
        input_files : string
            Path to the dwi volumes. This path may contain wildcards to process
            multiple inputs at once.
        bvalues : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        resume : bool, optional
            If enabled, the pipeline will not run tasks if the output exists.
        out_dir : string, optional
            Working directory (default input file directory)
        """
        io_it = self.get_io_iterator(inspect.currentframe())

        flow_base_params = {
            'output_strategy': self._output_strategy,
            'mix_names': self._mix_names,
            'force': self._force_overwrite
        }

        for dwi, bval, bvec in io_it:
            # Masking
            mo_flow = MedianOtsuFlow(**flow_base_params)
            self.run_sub_flow(mo_flow, dwi, out_dir=out_dir)
            dwi_mask, _ = mo_flow.last_generated_outputs[0]

            # Denoising
            skip_denoise = True
            nl_flow = NLMeansFlow(output_strategy=self._output_strategy,
                                  mix_names=self._mix_names,
                                  force=self._force_overwrite,
                                  skip=skip_denoise)

            self.run_sub_flow(nl_flow, dwi, out_dir=out_dir)
            denoised = nl_flow.last_generated_outputs[0][0]

            # DTI reconstruction
            dti_flow = ReconstDtiFlow(output_strategy='append',
                                      mix_names=self._mix_names,
                                      force=self._force_overwrite)

            self.run_sub_flow(dti_flow, denoised, bval, bvec, dwi_mask,
                              out_dir='metrics')
            tensor, fa, ga, rgb, md, ad, rd, mode, evecs, evals = \
                dti_flow.last_generated_outputs[0]

            # CSD Recontruction
            csd_flow = ReconstODFFlow(output_strategy='append',
                                      mix_names=self._mix_names,
                                      force=self._force_overwrite)

            self.run_sub_flow(csd_flow, denoised, bval, bvec,
                              mask_files=dwi_mask, out_dir='peaks_csd',
                              extract_pam_values=True)
            peaks_file = csd_flow.last_generated_outputs[0][0]

            # Create seeding mask
            track_dir = os.path.join(os.path.dirname(denoised), 'tracking')
            mask_flow = MaskFlow(output_strategy='absolute',
                                 mix_names=self._mix_names,
                                 force=self._force_overwrite)

            self.run_sub_flow(mask_flow, fa, out_dir=track_dir)
            fa_seed_mask = mask_flow.last_generated_outputs[0][0]

            # Deterministic Tracking
            tracking_flow = DetTrackFlow(output_strategy='absolute',
                                         mix_names=self._mix_names,
                                         force=self._force_overwrite)

            self.run_sub_flow(tracking_flow, peaks_file, fa, fa_seed_mask,
                              out_dir=track_dir)
            det_tracts = tracking_flow.last_generated_outputs[0][0]

            # Tract density
            logging.warning('TDI disabled for now.')
            #tdi_flow = TrackDensityFlow(output_strategy='absolute',
            #                           mix_names=self._mix_names,
            #                            force=self._force_overwrite)
            #self.run_sub_flow(tdi_flow, det_tracts, fa, out_dir=track_dir)


class QuickFlow(CombinedWorkflow):

    def _get_sub_flows(self):
        return [
            MedianOtsuFlow,
            NLMeansFlow,
            ReconstDtiFlow,
            ReconstODFFlow
        ]

    def run(self, input_files, bvalues, bvectors, out_dir=''):
        """ A simple dwi processing pipeline with the following steps:
            -Denoising
            -Masking
            -DTI reconstruction
            -HARDI recontruction
            -Deterministic tracking
            -Tracts metrics

        Parameters
        ----------
        input_files : string
            Path to the dwi volumes. This path may contain wildcards to process
            multiple inputs at once.
        bvalues : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        out_dir : string, optional
            Working directory (default input file directory)
        """
        io_it = self.get_io_iterator(inspect.currentframe())

        flow_base_params = {
            'output_strategy': self._output_strategy,
            'mix_names': self._mix_names,
            'force': self._force_overwrite
        }

        for dwi, bval, bvec in io_it:
            # Masking
            mo_flow = MedianOtsuFlow(**flow_base_params)
            self.run_sub_flow(mo_flow, dwi, out_dir=out_dir)
            dwi_mask, _ = mo_flow.last_generated_outputs[0]

            # Denoising
            skip_denoise = True
            nl_flow = NLMeansFlow(output_strategy=self._output_strategy,
                                  mix_names=self._mix_names,
                                  force=self._force_overwrite,
                                  skip=skip_denoise)

            self.run_sub_flow(nl_flow, dwi, out_dir=out_dir)
            denoised = nl_flow.last_generated_outputs[0][0]

            # DTI reconstruction
            dti_flow = ReconstDtiFlow(output_strategy='append',
                                      mix_names=self._mix_names,
                                      force=self._force_overwrite)

            self.run_sub_flow(dti_flow, denoised, bval, bvec, dwi_mask,
                              out_dir='metrics')

            # CSD Reconstruction
            csd_flow = ReconstODFFlow(output_strategy='append',
                                      mix_names=self._mix_names,
                                      force=self._force_overwrite)

            self.run_sub_flow(csd_flow, denoised, bval, bvec, dwi_mask,
                              out_dir='peaks_csd', extract_pam_values=True)

            # CSA reconstruction
            self.run_sub_flow(csd_flow, denoised, bval, bvec, dwi_mask,
                              out_dir='peaks_csa', reconst_model='csa',
                              extract_pam_values=True)
