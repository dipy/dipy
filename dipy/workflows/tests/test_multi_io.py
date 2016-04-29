import os
import numpy as np
import os.path as path
import numpy.testing as npt
from glob import glob
from ipdb import set_trace
from dipy.workflows.multi_io import OutputCreator


def common_start(sa, sb):
    """ Returns the longest common substring from the beginning of sa and sb """
    def _iter():
        for a, b in zip(sa, sb):
            if a == b:
                yield a
            else:
                return

    return ''.join(_iter())


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
    in3_2 = d3_2 + 'test2.txt'
    np.savetxt(in3_2, 3 * np.arange(10))

    template_dir = '/tmp/template/'
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)

    np.savetxt(template_dir + 'avg.txt', 10 * np.arange(10))

    out_dir = '/tmp/out/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_files = ['out_info.txt', 'out_summary.txt']

    print('Two wild inputs')

    og = OutputCreator(verbose=True)
    og.set_inputs('/tmp/data/s*/test.txt', '/tmp/data/s*/other/test2.txt')
    og.set_out_dir(out_dir)
    og.set_out_fnames(*out_files)
    og.create_outputs()

    npt.assert_equal(np.array(og.outputs).shape, (3, 2))
    print(og.outputs)
    print('\n')

    print('One wild input and one single')

    og = OutputCreator(verbose=True)
    og.set_inputs('/tmp/data/s*/test.txt', '/tmp/template/avg.txt')
    og.set_out_dir(out_dir)
    og.set_out_fnames(*out_files)
    og.create_outputs()

    npt.assert_equal(np.array(og.outputs).shape, (3, 2))
    print(og.outputs)
    print('\n')

    print('One single and one single inputs')

    og = OutputCreator(verbose=True)
    og.set_inputs('/tmp/data/s1/test.txt', '/tmp/template/avg.txt')
    og.set_out_dir(out_dir)
    og.set_out_fnames(*out_files)
    og.create_outputs()

    npt.assert_equal(np.array(og.outputs).shape, (1, 2))
    print(og.outputs)
    print('\n')

    print('One single input')

    og = OutputCreator(verbose=True)
    og.set_inputs('/tmp/data/s1/test.txt')
    og.set_out_dir(out_dir)
    og.set_out_fnames(*out_files)
    og.create_outputs()

    npt.assert_equal(np.array(og.outputs).shape, (1, 2))
    print(og.outputs)
    print('\n')


if __name__ == '__main__':

    test_output_generator()
