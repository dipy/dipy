import os.path as path
import numpy.testing as npt
from glob import glob


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


def test_multi_io():

    print('One input to one output')
    inputs = '/home/user/s1.trk'
    out_dir = ''
    out_files = 'comp.txt'

    outputs = connect_output_paths(inputs, out_dir, out_files)

    print(inputs)
    print(outputs)
    print('\n')

    print('One input to many outputs')
    inputs = '/home/user/s1.trk'
    out_dir = ''
    out_files = ['comp.txt', 'copy.trk']

    outputs = connect_output_paths(inputs, out_dir, out_files)

    print(inputs)
    print(outputs)
    print('\n')

    print('Many inputs to many outputs')
    inputs = ['/home/user/s1.trk', '/home/user/s2.trk']
    out_dir = ''
    out_files = ['comp.txt', 'copy.trk']

    outputs = connect_output_paths(inputs, out_dir, out_files)

    print(inputs)
    print(outputs)
    print('\n')

    print('Out dir')
    inputs = ['/home/user/s1.trk', '/home/user/s2.trk']
    out_dir = '/tmp'
    out_files = ['comp.txt', 'copy.trk']

    outputs = connect_output_paths(inputs, out_dir, out_files)

    print(inputs)
    print(outputs)
    print('\n')


def mix_multi_inputs(multi_inputs):

    for inps in zip(*multi_inputs):
        mixing_name = ''
        for i, inp in enumerate(inps):
            mixing_name += path.splitext(path.basename(inp))[0]
            if i < len(inps)-1:
                mixing_name += '_'
        print(mixing_name)


def test_extra_multi_io():

    from ipdb import set_trace
    inputs_1 = sorted(glob('../*.py'))
    inputs_2 = [inp.split('.py')[0] + '-fa.tt' for inp in inputs_1]
    inputs_3 = [inp.split('.py')[0] + '-ga.tx' for inp in inputs_1]

    print(len(inputs_1))
    print(len(inputs_2))

    print(inputs_1)
    print(inputs_2)

    mix_multi_inputs([inputs_1, inputs_2, inputs_3])

    set_trace()

    pass

if __name__ == '__main__':

    test_multi_io()
    test_extra_multi_io()
