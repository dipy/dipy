from __future__ import division, print_function, absolute_import

import inspect
import logging

from dipy.workflows.workflow import Workflow
from dipy.workflows.segment import MedianOtsuFlow
from dipy.workflows.denoise import NLMeansFlow
from dipy.workflows.reconst import ReconsDtiFlow
from dipy.workflows.reconst import ReconstCSDFlow
from dipy.workflows.tracking import DetTrackFlow
from dipy.workflows.track_metrics import TrackDensityFlow
from dipy.workflows.mask import MaskFlow


class ClassicFlow(Workflow):

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
            mo_flow.run(dwi, out_dir=out_dir)
            dwi_mask, _ = mo_flow.last_generated_outputs[0]

            # Denoising

            logging.warning('Denoising disabled for now.')
            #nl_flow = NLMeansFlow(**flow_base_params)
            #nl_flow.run(dwi, sigma=0.001, out_dir=out_dir)
            #denoised = nl_flow.last_generated_outputs[0][0]
            denoised = dwi #temporary

            # DTI reconstruction
            dti_flow = ReconsDtiFlow(**flow_base_params)
            dti_flow.run(denoised, bval, bvec, dwi_mask, out_dir='metrics')
            tensor, fa, ga, rgb, md, ad, rd, mode, evecs, evals = \
                dti_flow.last_generated_outputs[0]

            # CSD Recontruction
            csd_flow = ReconstCSDFlow(**flow_base_params)
            csd_flow.run(denoised, bval, bvec, mask_files=dwi_mask)
            peaks_file = csd_flow.last_generated_outputs[0][0]

            # Create seeding mask
            mask_flow = MaskFlow(**flow_base_params)
            mask_flow.run(fa, greater_than=0.4)
            fa_seed_mask = mask_flow.last_generated_outputs[0][0]

            # Deterministic Tracking
            tracking_flow = DetTrackFlow(**flow_base_params)
            tracking_flow.run(peaks_file, fa, fa_seed_mask)
            det_tracts = tracking_flow.last_generated_outputs[0][0]

            # Tract density
            logging.warning('TDI disabled for now.')
            #tdi_flow = TrackDensityFlow(**flow_base_params)
            #tdi_flow.run(det_tracts, fa)
