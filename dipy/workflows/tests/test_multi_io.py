import numpy as np
import numpy.testing as npt
import os
import os.path as path
from glob import glob

from nibabel.tmpdirs import TemporaryDirectory

from dipy.workflows.multi_io import IOIterator, io_iterator


def test_output_generator():

    with TemporaryDirectory() as tmpdir:

        # Create dummy file structure.
        d1 = path.join(tmpdir, 'data', 's1')
        d2 = path.join(tmpdir, 'data', 's2')
        d3 = path.join(tmpdir, 'data', 's3')

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
        os.makedirs(template_dir)

        np.savetxt(path.join(template_dir, 'avg.txt'), 10 * np.arange(10))

        out_dir = path.join(tmpdir, 'out')
        os.makedirs(out_dir)

        out_files = ['out_info.txt', 'out_summary.txt']

        input1_wildcard_paths = path.join(tmpdir, 'data', 's*', 'test.txt')
        input2_wildcard_paths = path.join(tmpdir, 'data', 's*', 'test2.txt')

        globbed_input1 = glob(input1_wildcard_paths)
        globbed_input2 = glob(input2_wildcard_paths)


        # Start tests
        print('Testing absolute out_dir with 1 input path globbed for 3 '
              'real inputs')

        og = IOIterator(output_strategy='absolute', mix_names=False)

        og.set_inputs(input1_wildcard_paths)
        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)

        og.create_outputs()
        npt.assert_equal(np.array(og.outputs).shape, (3, 2))

        expected_output_1 = os.path.join(out_dir, out_files[0])
        expected_output_2 = os.path.join(out_dir, out_files[1])

        #set_idx =
        for io_it_vals, expected_input in zip(og, globbed_input1):
            inp1, out1, out2 = io_it_vals
            assert expected_input == expected_input
            assert expected_output_1 == out1
            assert expected_output_2 == out2

        print('Passed \n')

        return
        print('One long input and one single')
        #return
        og = IOIterator()
        og.set_inputs(path.join(tmpdir, 'data', 's*', 'test.txt'),
                      path.join(tmpdir, 'template', 'avg.txt'))
        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)
        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (3, 2))

        for inp1, inp2, out1, out2 in og:
            print(inp1)
            print(inp2)
            print(out1)
            print(out2)
        print('\n')

        print('One single and one single input')

        og = IOIterator()
        og.set_inputs(path.join(tmpdir, 'data', 's1', 'test.txt'),
                      path.join(tmpdir, 'template', 'avg.txt'))
        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)
        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (1, 2))

        for inp1, inp2, out1, out2 in og:
            print(inp1)
            print(inp2)
            print(out1)
            print(out2)
        print('\n')

        print('One single input')

        og = IOIterator()
        og.set_inputs(path.join(tmpdir, 'data', 's1', 'test.txt'))
        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)
        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (1, 2))

        for in1, out1, out2 in og:
            print(in1)
            print(out1)
            print(out2)
        print('\n')

        print('One single output but do not keep input structure')

        og = IOIterator(input_structure=False)
        og.set_inputs(path.join(tmpdir, 'data', 's1', 'test.txt'))
        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)
        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (1, 2))

        for in1, out1, out2 in og:
            print(in1)
            print(out1)
            print(out2)
        print('\n')

        out_dir = 'out'

        print('Do not keep input structure and relative out_dir')

        og = IOIterator(input_structure=False)
        og.set_inputs(path.join(tmpdir, 'data', 's1', 'test.txt'))
        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)
        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (1, 2))

        for in1, out1, out2 in og:
            print(in1)
            print(out1)
            print(out2)
        print('\n')

        print('Two long inputs and input_structure True')

        og = IOIterator(input_structure=True)
        og.set_inputs(path.join(tmpdir, 'data', 's*', 'test.txt'),
                      path.join(tmpdir, 'data', 's*', 'other', 'test2.txt'))

        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)

        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (3, 2))

        for inp1, inp2, out1, out2 in og:
            print(inp1)
            print(inp2)
            print(out1)
            print(out2)

        print('\n')

        print('Two long inputs and input_structure False')

        og = IOIterator(input_structure=False)
        og.set_inputs(path.join(tmpdir, 'data', 's*', 'test.txt'),
                      path.join(tmpdir, 'data', 's*', 'other', 'test2.txt'))

        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)

        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (3, 2))

        for inp1, inp2, out1, out2 in og:
            print(inp1)
            print(inp2)
            print(out1)
            print(out2)

        # set_trace()
        print('\n')

        print('Single line creation')
        io_it = io_iterator(og.input_args, og.out_dir, og.out_fnames,
                            og.input_structure)

        for it1, it2 in zip(io_it, og):
            npt.assert_equal(np.array(it1), np.array(it2))

        out_files = ['out_info_yoga.txt']

        print('Two single inputs, one output and input_structure True')

        og = IOIterator(input_structure=True)
        og.set_inputs(path.join(tmpdir, 'data', 's1', 'test.txt'),
                      path.join(tmpdir, 'data', 's1', 'other', 'test2.txt'))

        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)

        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (1, 1))

        for inp1, inp2, out1 in og:
            print(inp1)
            print(inp2)
            print(out1)

        # set_trace()
        print('\n')

        print('Two single inputs, one output and input_structure False')

        og = IOIterator(input_structure=False)
        og.set_inputs(path.join(tmpdir, 'data', 's1', 'test.txt'),
                      path.join(tmpdir, 'data', 's1', 'other', 'test2.txt'))

        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)

        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (1, 1))

        for inp1, inp2, out1 in og:
            print(inp1)
            print(inp2)
            print(out1)

        # set_trace()
        print('\n')

        print('One single input, one output and input_structure False')

        og = IOIterator(input_structure=False)
        og.set_inputs(path.join(tmpdir, 'data', 's1', 'test.txt'))

        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)

        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (1, 1))

        for inp1, out1 in og:
            print(inp1)
            print(out1)

        # set_trace()
        print('\n')

        print('One single input, one output and input_structure True')

        og = IOIterator(input_structure=True)
        og.set_inputs(path.join(tmpdir, 'data', 's1', 'test.txt'))

        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)

        og.create_outputs()

        npt.assert_equal(np.array(og.outputs).shape, (1, 1))

        for inp1, out1 in og:
            print(inp1)
            print(out1)

        # set_trace()
        print('\n')

if __name__ == '__main__':

    test_output_generator()
