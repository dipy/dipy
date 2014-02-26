import traits.api as traits

from nipype.interfaces.base import (BaseInterfaceInputSpec, BaseInterface,
                                    isdefined, TraitedSpec)
from dipy.workflows.fit_tensor import ft_out as fit_tensor_outputs


class FitTensorInputSpec(BaseInterfaceInputSpec):
    dwi_data = traits.File(exists=True, mandatory=True,
                desc="4d data set of diffusion weighted MRI data.")
    nomask = traits.Bool(desc="If set, no mask will be applied.")
    mask = traits.File(exists=True, mandatory=False,
                desc="Only voxels in mask will be fit. If no mask is given "
                     "the mask will be computed automatically unless "
                     "``nomask`` is set.")
    root = traits.File(exists=False, mandatory=False,
                desc="The root of the filenames for all the output files")
    bvec = traits.File(exits=True, mandatory=True,
                desc="Text file of gradient tables in bvec format.")
    min_signal = traits.Float(1., mandatory=False,
                desc="Minimum valid singal value.")
    scale = traits.Float(1., manditory=False)


class FitTensorOutputSpec(TraitedSpec):
    pass
for key in fit_tensor_outputs:
    FitTensorOutputSpec.add_class_trait(key, traits.File(exists=True))


class DipyFitTensor(BaseInterface):
    input_spec = FitTensorInputSpec
    output_spec = FitTensorOutputSpec

    def _run_interface(self, runtime):
        from dipy.workflows.fit_tensor import fit_tensor
        args = {}
        for key in [k for k, _ in self.inputs.items()
                    if k not in BaseInterfaceInputSpec().trait_names()]:
            value = getattr(self.inputs, key)
            if isdefined(value):
                args[key] = value

        if args.pop('nomask', False):
            args['mask'] = None

        self._output_paths = fit_tensor(**args)
        # self._output_paths = fit_tensor_outputs
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        for key, file in self._output_paths.items():
            if file is not None:
                outputs[key] = file
        return outputs
