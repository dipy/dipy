import os
import numpy as np
import os.path as path
import numpy.testing as npt
from glob import glob
from ipdb import set_trace
from dipy.workflows.multi_io import OutputCreator
from nibabel.tmpdirs import TemporaryDirectory


def test_output_generator():

    with TemporaryDirectory() as tmpdir:

        d1 = path.join(tmpdir, 'data', 's1')
        d2 = path.join(tmpdir, 'data', 's2')
        d3 = path.join(tmpdir, 'data', 's3')

        if not os.path.exists(d1):
            os.makedirs(d1)
            os.makedirs(d2)
            os.makedirs(d3)

        in1 = path.join(d1, 'test.txt')
        np.savetxt(in1, np.arange(10))
        in2 = path.join(d2, 'test.txt')
        np.savetxt(in2, 2 * np.arange(10))
        in3 = path.join(d3, 'test.txt')
        np.savetxt(in3, 3 * np.arange(10))

        d1_2 = path.join(tmpdir, 'data', 's1', 'other')
        d2_2 = path.join(tmpdir, 'data', 's2', 'other')
        d3_2 = path.join(tmpdir, 'data', 's3', 'other')

        if not os.path.exists(d1_2):
            os.makedirs(d1_2)
            os.makedirs(d2_2)
            os.makedirs(d3_2)

        in1_2 = path.join(d1_2, 'test2.txt')
        np.savetxt(in1_2, np.arange(10))
        in2_2 = path.join(d2_2, 'test2.txt')
        np.savetxt(in2_2, 2 * np.arange(10))
        in3_2 = path.join(d3_2, 'test2.txt')
        np.savetxt(in3_2, 3 * np.arange(10))

        template_dir = path.join(tmpdir, 'template')
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)

        np.savetxt(path.join(template_dir, 'avg.txt'), 10 * np.arange(10))

        out_dir = path.join(tmpdir, 'out')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_files = ['out_info.txt', 'out_summary.txt']

        print('Two long inputs')

        og = OutputCreator()
        og.set_inputs(path.join(tmpdir, 'data', 's*', 'test.txt'),
                      path.join(tmpdir, 'data', 's*', 'other', 'test2.txt'))

        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)

        og.create_outputs()
        npt.assert_equal(np.array(og.outputs).shape, (3, 2))
        print(og.outputs)
        print('\n')

        for i_o in og:
            print(i_o)
        print('\n')

        print('One long input and one single')

        og = OutputCreator()
        og.set_inputs(path.join(tmpdir, 'data', 's*', 'test.txt'),
                      path.join(tmpdir, 'template', 'avg.txt'))
        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)
        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (3, 2))
        print(og.outputs)
        print('\n')

        for i_o in og:
            print(i_o)
        print('\n')

        print('One single and one single input')

        og = OutputCreator()
        og.set_inputs(path.join(tmpdir, 'data', 's1', 'test.txt'),
                      path.join(tmpdir, 'template', 'avg.txt'))
        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)
        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (1, 2))
        print(og.outputs)
        print('\n')

        for i_o in og:
            print(i_o)
        print('\n')

        print('One single input')

        og = OutputCreator()
        og.set_inputs(path.join(tmpdir, 'data', 's1', 'test.txt'))
        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)
        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (1, 2))
        print(og.outputs)
        print('\n')

        for in1, out1, out2 in og:
            print(in1)
            print(out1)
            print(out2)


if __name__ == '__main__':

    test_output_generator()
