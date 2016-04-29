import numpy as np
from glob import glob
import os.path as path


def connect_output_paths(inputs, out_dir, out_files):

    outputs = []
    if isinstance(inputs, basestring):
        inputs = [inputs]
    if isinstance(out_files, basestring):
        out_files = [out_files]

    # from ipdb import set_trace
    # set_trace()

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

    elif len(inputs) == 1:

        for inp in inputs:
            dname = path.join(out_dir, path.dirname(inp))
            base = path.splitext(path.basename(inp))[0]
            new_out_files = []
            for out_file in out_files:
                new_out_files.append(
                    path.join(dname, base + '_' + out_file))
            outputs.append(new_out_files)

    return outputs


def concatenate_inputs(multi_inputs):

    mixing_names = []
    for inps in zip(*multi_inputs):
        mixing_name = ''
        for i, inp in enumerate(inps):
            mixing_name += path.splitext(path.basename(inp))[0]
            if i < len(inps) - 1:
                mixing_name += '_'
        mixing_names.append(mixing_name)
    return mixing_names


def split_ext(fname):

    ext = path.splitext(path.basename(fname))[1]
    base = path.splitext(path.basename(fname))[0]
    if ext == '.gz':
        ext = path.splitext(path.basename(base))[1]
        if ext == '.nii':
            base = path.splitext(path.basename(fname))[0]
    return base


def mix_inputs(inputs1, inputs2):

    mixing_names = []
    for inp in inputs1:
        mixing_name = ''
        mixing_name += path.splitext(path.basename(inp))[0]
        mixing_name += '_'
        mixing_name += path.splitext(path.basename(inputs2))[0]
        # print(mixing_name)
        mixing_names.append(mixing_name)

    return mixing_names


def lprint(list_):
    for l in list_:
        print(l)


class OutputGenerator(object):

    def __init__(self, verbose=False):
        self.verbose = verbose

    def set_inputs(self, *args):
        self.input_args = list(args)
        self.inputs = [sorted(glob(inp)) for inp in self.input_args]

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir

    def set_out_fnames(self, *args):
        self.out_fnames = list(args)

    def create_outputs(self):

        if len(self.inputs) == 1:
            self.outputs = connect_output_paths(self.inputs,
                                                self.out_dir,
                                                self.out_files)
        elif len(self.inputs) > 1:
            self.outputs = connect_output_paths(self.inputs,
                                                self.out_dir,
                                                self.out_fnames)
        else:
            raise ImportError('No inputs')
