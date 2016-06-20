import numpy as np
import numpy.testing as npt
import os
from os.path import join
from glob import glob

from nibabel.tmpdirs import TemporaryDirectory
from nose.tools import assert_raises

from dipy.workflows.multi_io import IOIterator, io_iterator


def test_output_generator():

    with TemporaryDirectory() as tmpdir:

        # Create dummy file structure.
        d1 = join(tmpdir, 'data', 's1')
        d2 = join(tmpdir, 'data', 's2')
        d3 = join(tmpdir, 'data', 's3')

        os.makedirs(d1)
        os.makedirs(d2)
        os.makedirs(d3)

        in1 = join(d1, 'test.txt')
        np.savetxt(in1, np.arange(10))

        in2 = join(d2, 'test.txt')
        np.savetxt(in2, 2 * np.arange(10))

        in3 = join(d3, 'test.txt')
        np.savetxt(in3, 3 * np.arange(10))

        d1_2 = join(tmpdir, 'data', 's1', 'other')
        d2_2 = join(tmpdir, 'data', 's2', 'other')
        d3_2 = join(tmpdir, 'data', 's3', 'other')

        os.makedirs(d1_2)
        os.makedirs(d2_2)
        os.makedirs(d3_2)

        in1_2 = join(d1_2, 'test2.txt')
        np.savetxt(in1_2, np.arange(10))

        in2_2 = join(d2_2, 'test2.txt')
        np.savetxt(in2_2, 2 * np.arange(10))

        in3_2 = join(d3_2, 'test2.txt')
        np.savetxt(in3_2, 3 * np.arange(10))

        template_dir = join(tmpdir, 'template')
        os.makedirs(template_dir)

        np.savetxt(join(template_dir, 'avg.txt'), 10 * np.arange(10))

        out_dir = join(tmpdir, 'out')
        os.makedirs(out_dir)

        out_files = ['out_info.txt', 'out_summary.txt']

        input1_wildcard_paths = join(tmpdir, 'data', 's*', 'test.txt')
        input2_wildcard_paths = join(tmpdir, 'data', 's*', 'other', 'test2.txt')

        globbed_input1 = glob(input1_wildcard_paths)
        globbed_input2 = glob(input2_wildcard_paths)

        # Testing absolute out_dir with 1 input path globbed for 3 real inputs
        og = IOIterator(output_strategy='absolute', mix_names=False)

        assert_raises(ImportError, og.create_outputs)

        og.set_inputs(input1_wildcard_paths)
        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)

        og.create_outputs()

        # Test if different instanciation methods are equal.
        another_og = io_iterator([input1_wildcard_paths], out_dir, out_files,
                                 output_strategy='absolute', mix_names=False)
        another_og.create_outputs()
        npt.assert_equal(np.array(og.outputs), np.array(another_og.outputs))
        npt.assert_equal(np.array(og.inputs), np.array(another_og.inputs))

        npt.assert_equal(np.array(og.outputs).shape, (3, 2))

        # Test actual outputs
        expected_output_1 = os.path.join(out_dir, out_files[0])
        expected_output_2 = os.path.join(out_dir, out_files[1])

        for io_it_vals, expected_input in zip(og, globbed_input1):
            inp1, out1, out2 = io_it_vals
            assert expected_input == expected_input
            assert expected_output_1 == out1
            assert expected_output_2 == out2

        # Testing append out_dir with 1 input path globbed for 3 real inputs
        out_dir = 'out'
        og = IOIterator(output_strategy='append', mix_names=False)
        og.set_inputs(input1_wildcard_paths)
        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)
        og.create_outputs()

        expected_outputs = []
        for inp_dir in glob(join(tmpdir, 'data', 's*')):
            expected_outputs.append((join(inp_dir, out_dir, out_files[0]),
                                     join(inp_dir, out_dir, out_files[1])))

        for io_it_vals, expected_input, expected_outputs in zip(og, globbed_input1, expected_outputs):
            inp1, out1, out2 = io_it_vals
            assert expected_input == expected_input
            assert expected_outputs[0] == out1
            assert expected_outputs[1] == out2

        # Testing prepend out_dir with 1 input path globbed for 3 real inputs
        out_base = join(tmpdir, 'out_prepend')
        og = IOIterator(output_strategy='prepend', mix_names=False)
        og.set_inputs(input1_wildcard_paths)
        og.set_out_dir(out_base)
        og.set_out_fnames(*out_files)
        og.create_outputs()

        expected_outputs = []
        for inp_dir in glob(join(tmpdir, 'data', 's*')):
            expected_outputs.append((join(out_base + inp_dir, out_files[0]),
                                     join(out_base + inp_dir, out_files[1])))

        for io_it_vals, expected_input, expected_outputs in zip(og, globbed_input1,
                                                                expected_outputs):
            inp1, out1, out2 = io_it_vals
            assert expected_input == expected_input
            assert expected_outputs[0] == out1
            assert expected_outputs[1] == out2

        # Test mixing names
        out_dir = join(tmpdir, 'out')
        og = IOIterator(output_strategy='absolute', mix_names=True)

        og.set_inputs(input1_wildcard_paths, input2_wildcard_paths)
        og.set_out_dir(out_dir)
        og.set_out_fnames(*out_files)

        og.create_outputs()

        expected_output_1 = os.path.join(out_dir, 'test_' + 'test2__' + out_files[0])
        expected_output_2 = os.path.join(out_dir, 'test_' + 'test2__' + out_files[1])

        for io_it_vals, expected_input1, expected_input2 in zip(og, globbed_input1, globbed_input2):
            inp1, inp2, out1, out2 = io_it_vals
            assert inp1 == expected_input1
            assert inp2 == expected_input2
            assert expected_output_1 == out1
            assert expected_output_2 == out2


if __name__ == '__main__':
    test_output_generator()
