import os
import numpy as np
from glob import glob
import os.path as path
import inspect


def common_start(sa, sb):
    """ Returns the longest common substring from the beginning of sa and sb """
    def _iter():
        for a, b in zip(sa, sb):
            if a == b:
                yield a
            else:
                return

    return ''.join(_iter())


def slash_to_under(dir_str):
    return ''.join(dir_str.replace('/', '_'))


def connect_output_paths(inputs, out_dir, out_files, output_strategy='append', mix_names=True):
    outputs = []
    if isinstance(inputs, basestring):
        inputs = [inputs]
    if isinstance(out_files, basestring):
        out_files = [out_files]

    sizes_of_inputs = [len(inp) for inp in inputs]

    max_size = np.max(sizes_of_inputs)
    min_size = np.min(sizes_of_inputs)
    if min_size > 1 and min_size != max_size:
        raise ImportError('Size of input issue')

    elif min_size == 1:
        for i, sz in enumerate(sizes_of_inputs):
            if sz == min_size:
                inputs[i] = max_size * inputs[i]

    if mix_names:

        mixing_prefixes = concatenate_inputs(inputs)
    else:
        mixing_prefixes = [''] * len(inputs[0])

    for (mix_pref, inp) in zip(mixing_prefixes, inputs[0]):
        inp_dirname = path.dirname(inp)
        if output_strategy == 'prepend':
            if path.isabs(out_dir):
                dname = out_dir + inp_dirname
            if not path.isabs(out_dir):
                dname = path.join(
                    os.getcwd(), out_dir + inp_dirname)

        elif output_strategy == 'append':
            dname = path.join(inp_dirname, out_dir)

        else: #absolute
            dname = out_dir

        updated_out_files = []
        for out_file in out_files:
            updated_out_files.append(path.join(dname, mix_pref + out_file))

        outputs.append(updated_out_files)

    return inputs, outputs


def concatenate_inputs(multi_inputs):
    """ Concatenate list of inputs
    """

    mixing_names = []
    for inps in zip(*multi_inputs):
        mixing_name = ''
        for i, inp in enumerate(inps):
            mixing_name += basename(inp) + '_'

        mixing_names.append(mixing_name + '_')
    return mixing_names


def basename(fname):
    ext = path.splitext(path.basename(fname))[1]
    base = path.splitext(path.basename(fname))[0]
    if ext == '.gz':
        ext = path.splitext(path.basename(base))[1]
        if ext == '.nii':
            base = path.splitext(path.basename(fname))[0]
            base = path.splitext(path.basename(base))[0]
    return base


def io_iterator(inputs, out_dir, fnames, output_strategy='append', mix_names=True):
    io_it = IOIterator(output_strategy=output_strategy, mix_names=mix_names)
    io_it.set_inputs(*inputs)
    io_it.set_out_dir(out_dir)
    io_it.set_out_fnames(*fnames)
    io_it.create_outputs()

    return io_it


def io_iterator_(frame, fnc, output_strategy='append', mix_names=True):
    args, _, _, values = inspect.getargvalues(frame)
    specs = inspect.getargspec(fnc)
    spargs = specs.args
    defaults = specs.defaults

    len_args = len(spargs)
    len_defaults = len(defaults)
    split_at = len_args - len_defaults

    inputs = []
    outputs = []
    out_dir = ''

    # inputs
    for arv in args[:split_at]:
        inputs.append(values[arv])

    # defaults
    for arv in args[split_at:]:
        if arv == 'out_dir':
            out_dir = values[arv]
        elif 'out_' in arv:
            outputs.append(values[arv])

    return io_iterator(inputs, out_dir, outputs, output_strategy, mix_names)


class IOIterator(object):
    """ Create output filenames that work nicely with muiltiple input files from multiple directories (processing multiple subjects with one command)

    Use information from input files, out_dir and out_fnames to generate correct outputs which can come from long lists of multiple or single
    inputs.
    """

    def __init__(self, output_strategy='append', mix_names=True):
        self.output_strategy = output_strategy
        self.mix_names = mix_names

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
                self.out_fnames,
                self.output_strategy,
                self.mix_names)

            self.create_directories()

        else:
            raise ImportError('No inputs')

    def create_directories(self):
        for outputs in self.outputs:
            for output in outputs:
                directory = path.dirname(output)
                if not (directory == '' or os.path.exists(directory)):
                    os.makedirs(directory)

    def __iter__(self):
        I = np.array(self.inputs).T
        O = np.array(self.outputs)
        IO = np.concatenate([I, O], axis=1)
        for i_o in IO:
            yield i_o
