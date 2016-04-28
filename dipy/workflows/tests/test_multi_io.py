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

    i1 = '../data/*.bv*'
    i2 = '../data/files/'
    # CREATE TINY FOLDERS AND FILES UNDER DATA

    pass


if __name__ == '__main__':

    test_one_set_of_inputs()
    test_many_sets_of_inputs()
