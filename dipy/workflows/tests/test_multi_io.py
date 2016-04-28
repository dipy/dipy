import os.path as path
import numpy.testing as npt
from glob import glob
from ipdb import set_trace


def common_start(sa, sb):
    """ Returns the longest common substring from the beginning of sa and sb """
    def _iter():
        for a, b in zip(sa, sb):
            if a == b:
                yield a
            else:
                return

    return ''.join(_iter())


def connect_output_paths(inputs, out_dir, out_files):
    outputs = []
    if isinstance(inputs, basestring):
        inputs = [inputs]
    if isinstance(out_files, basestring):
        out_files = [out_files]

    for inp in inputs:
        dname = out_dir + path.dirname(inp)
        base = path.splitext(path.basename(inp))[0]
        new_out_files = []
        for out_file in out_files:
            new_out_files.append(
                path.join(dname, base + '_' + out_file))
        outputs.append(new_out_files)

    return outputs


def test_one_set_of_inputs():

    print('One input to one output')
    inputs = '/home/user/s1.trk'
    out_dir = ''
    out_files = 'comp.txt'

    outputs = connect_output_paths(inputs, out_dir, out_files)

    print(inputs)
    print(outputs)
    print('\n')

    print('One input (with one file) to many outputs')
    inputs = '/home/user/s1.trk'
    out_dir = ''
    out_files = ['comp.txt', 'copy.trk']

    outputs = connect_output_paths(inputs, out_dir, out_files)

    print(inputs)
    print(outputs)
    print('\n')

    print('One input (of many files) to many outputs')
    inputs = ['/home/user/s1.trk', '/home/user/s2.trk']
    out_dir = ''
    out_files = ['comp.txt', 'copy.trk']

    outputs = connect_output_paths(inputs, out_dir, out_files)

    print(inputs)
    print(outputs)
    print('\n')

    print('One input (of many files) to many outputs and out_dir provided')
    inputs = ['/home/user/s1.trk', '/home/user/s2.trk']
    out_dir = '/tmp'
    out_files = ['comp.txt', 'copy.trk']

    outputs = connect_output_paths(inputs, out_dir, out_files)

    print(inputs)
    print(outputs)
    print('\n')


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


def concatenate_many_to_one_inputs(inputs1, inputs2):

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


def test_many_sets_of_inputs():

    print('Concatenating sets of inputs mixing')
    inputs_1 = sorted(glob('../*.py'))
    inputs_2 = [inp.split('.py')[0] + '-fa.tt' for inp in inputs_1]
    inputs_3 = [inp.split('.py')[0] + '-ga.tx' for inp in inputs_1]
    out_dir = ''
    out_files = ['t1.png', 't2.png']

    npt.assert_equal(len(inputs_1), len(inputs_2))

    print('\n>> Inputs_1')
    lprint(inputs_1)
    print('\n>> Inputs_2')
    lprint(inputs_2)
    print('\n>> Inputs_3')
    lprint(inputs_3)

    mixing_names = concatenate_inputs([inputs_1, inputs_2, inputs_3])
    # lprint(mixing_names)

    outputs = connect_output_paths(mixing_names, out_dir, out_files)
    print('\n>> Outputs')
    lprint(outputs)

    print('\nFirst set of inputs connecting with one second input')
    inputs_1 = sorted(glob('../*.py'))
    inputs_2 = 'template_fa.nii.gz'

    print('\n>> Inputs_1')
    lprint(inputs_1)
    print('\n>> Inputs_2')
    print(inputs_2)

    mixing_names = concatenate_many_to_one_inputs(inputs_1, inputs_2)
    # lprint(mixing_names)

    outputs = connect_output_paths(mixing_names, out_dir, out_files)
    print('\n>> Outputs')
    lprint(outputs)



if __name__ == '__main__':

    # test_one_set_of_inputs()
    test_many_sets_of_inputs()
