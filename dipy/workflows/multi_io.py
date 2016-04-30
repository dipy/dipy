import numpy as np
from glob import glob
import os.path as path
from ipdb import set_trace


def connect_output_paths(inputs, out_dir, out_files):

    outputs = []
    if isinstance(inputs, basestring):
        inputs = [inputs]
    if isinstance(out_files, basestring):
        out_files = [out_files]

    sizes_of_inputs = [len(inp) for inp in inputs]

    if len(inputs) > 1:

        if np.sum(np.diff(sizes_of_inputs)) == 0:
            mixing_names = concatenate_inputs(inputs)

            for (mix_inp, inp) in zip(mixing_names, inputs[0]):
                dname = path.join(out_dir, path.dirname(inp))
                updated_out_files = []
                for out_file in out_files:
                    updated_out_files.append(
                        path.join(dname, mix_inp + '_' + out_file))
                outputs.append(updated_out_files)

        else:
            max_size = np.max(sizes_of_inputs)
            min_size = np.min(sizes_of_inputs)
            if min_size > 1 and min_size != max_size:
                raise ImportError('Size of input issue')
            elif min_size == 1:
                for i, sz in enumerate(sizes_of_inputs):
                    if sz == min_size:
                        inputs[i] = max_size * inputs[i]

            mixing_names = concatenate_inputs(inputs)

            for (mix_inp, inp) in zip(mixing_names, inputs[0]):
                dname = path.join(out_dir, path.dirname(inp))
                updated_out_files = []
                for out_file in out_files:
                    updated_out_files.append(
                        path.join(dname, mix_inp + '_' + out_file))
                outputs.append(updated_out_files)

    elif len(inputs) == 1:

        for inp in inputs[0]:
            dname = path.join(out_dir, path.dirname(inp))
            # base = path.splitext(path.basename(inp))[0]
            base = basename(inp)
            new_out_files = []
            for out_file in out_files:
                new_out_files.append(
                    path.join(dname, base + '_' + out_file))
            outputs.append(new_out_files)

    return inputs, outputs


def concatenate_inputs(multi_inputs):
    """ Concatenate list of inputs
    """

    mixing_names = []
    for inps in zip(*multi_inputs):
        mixing_name = ''
        for i, inp in enumerate(inps):
            # mixing_name += path.splitext(path.basename(inp))[0]
            mixing_name += basename(inp)
            if i < len(inps) - 1:
                mixing_name += '_'
        mixing_names.append(mixing_name)
    return mixing_names


def basename(fname):
    ext = path.splitext(path.basename(fname))[1]
    base = path.splitext(path.basename(fname))[0]
    if ext == '.gz':
        ext = path.splitext(path.basename(base))[1]
        if ext == '.nii':
            base = path.splitext(path.basename(fname))[0]
    return base


class OutputCreator(object):
    """ Create output filenames that work nicely with muiltiple input files

    Use information from input files, out_dir and out_fnames to generate correct outputs which can come from long lists of multiple or single
    inputs
    """

    def __init__(self):
        pass

    def set_inputs(self, *args):
        self.input_args = list(args)
        self.inputs = [sorted(glob(inp)) for inp in self.input_args]

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir

    def set_out_fnames(self, *args):
        self.out_fnames = list(args)

    def create_outputs(self):
        if len(self.inputs) >= 1:
            self.updated_inputs, self.outputs = connect_output_paths(
                self.inputs,
                self.out_dir,
                self.out_fnames)
        else:
            raise ImportError('No inputs')

    def __iter__(self):
        I = np.array(self.inputs).T
        O = np.array(self.outputs)
        IO = np.concatenate([I, O], axis=1)
        for i_o in IO:
            yield i_o
