import os
import numpy as np
import os.path as path
import numpy.testing as npt
from glob import glob
from ipdb import set_trace
from dipy.workflows.multi_io import (OutputGenerator,
                                     concatenate_inputs,
                                     connect_output_paths,
                                     mix_inputs, lprint)


def common_start(sa, sb):
    """ Returns the longest common substring from the beginning of sa and sb """
    def _iter():
        for a, b in zip(sa, sb):
            if a == b:
                yield a
            else:
                return

    return ''.join(_iter())


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

    mixing_names = mix_inputs(inputs_1, inputs_2)
    # lprint(mixing_names)

    outputs = connect_output_paths(mixing_names, out_dir, out_files)
    print('\n>> Outputs')
    lprint(outputs)


def test_output_generator():

    d1 = '/tmp/data/s1/'
    d2 = '/tmp/data/s2/'
    d3 = '/tmp/data/s3/'

    if not os.path.exists(d1):
        os.makedirs(d1)
        os.makedirs(d2)
        os.makedirs(d3)

    in1 = d1 + 'test.txt'
    np.savetxt(in1, np.arange(10))
    in2 = d2 + 'test.txt'
    np.savetxt(in2, 2 * np.arange(10))
    in3 = d3 + 'test.txt'
    np.savetxt(in3, 3 * np.arange(10))

    d1_2 = '/tmp/data/s1/other/'
    d2_2 = '/tmp/data/s2/other/'
    d3_2 = '/tmp/data/s3/other/'

    if not os.path.exists(d1_2):
        os.makedirs(d1_2)
        os.makedirs(d2_2)
        os.makedirs(d3_2)

    in1_2 = d1_2 + 'test2.txt'
    np.savetxt(in1_2, np.arange(10))
    in2_2 = d2_2 + 'test2.txt'
    np.savetxt(in2_2, 2 * np.arange(10))
    in3_2 = d3_2 + 'test2.txt'
    np.savetxt(in3_2, 3 * np.arange(10))

    out_dir = '/tmp/out/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_files = ['out_info.txt', 'out_summary.txt']

    og = OutputGenerator(verbose=True)
    og.set_inputs('/tmp/data/s*/test.txt', '/tmp/data/s*/other/test2.txt')
    og.set_out_dir(out_dir)
    og.set_out_fnames(*out_files)
    og.create_outputs()

    set_trace()


if __name__ == '__main__':

    # test_one_set_of_inputs()
    # test_many_sets_of_inputs()
    test_output_generator()
