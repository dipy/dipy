from __future__ import division, print_function, absolute_import

import os.path as osp
from nibabel.tmpdirs import InTemporaryDirectory

import numpy as np
import numpy.testing as npt

from dipy.data import get_data
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table


def test_read_bvals_bvecs():
    fimg, fbvals, fbvecs = get_data('small_101D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gt = gradient_table(bvals, bvecs)
    npt.assert_array_equal(bvals, gt.bvals)
    npt.assert_array_equal(bvecs, gt.bvecs)

    # None should also work as an input:
    bvals_none, bvecs_none = read_bvals_bvecs(None, fbvecs)
    npt.assert_array_equal(bvecs_none, gt.bvecs)
    bvals_none, bvecs_none = read_bvals_bvecs(fbvals, None)
    npt.assert_array_equal(bvals_none, gt.bvals)

    # Test for error raising with unknown file formats:
    nan_fbvecs = osp.splitext(fbvecs)[0] + '.nan'  # Nonsense extension
    npt.assert_raises(ValueError, read_bvals_bvecs, fbvals, nan_fbvecs)

    # Test for error raising with incorrect file-contents:

    with InTemporaryDirectory():
        # These bvecs only have two rows/columns:
        new_bvecs1 = bvecs[:, :2]
        # Make a temporary file
        with open('test_bv_file1.txt', 'wt') as bv_file1:
            # And fill it with these 2-columned bvecs:
            for x in range(new_bvecs1.shape[0]):
                bv_file1.write('%s %s\n' % (new_bvecs1[x][0],
                                            new_bvecs1[x][1]))
        npt.assert_raises(IOError, read_bvals_bvecs, fbvals, 'test_bv_file1.txt')

        # These bvecs are saved as one long array:
        new_bvecs2 = np.ravel(bvecs)
        with open('test_bv_file2.npy', 'w') as bv_file2:
            print('FILENAME:', bv_file2.name)
            np.save(bv_file2.name, new_bvecs2)
        npt.assert_raises(IOError, read_bvals_bvecs, fbvals, 'test_bv_file2.npy')

        # There are less bvecs than bvals:
        new_bvecs3 = bvecs[:-1, :]
        with open('test_bv_file3.txt', 'w') as bv_file3:
            np.savetxt(bv_file3.name, new_bvecs3)
        npt.assert_raises(IOError, read_bvals_bvecs, fbvals, 'test_bv_file3.txt')

        # You entered the bvecs on both sides:
        npt.assert_raises(IOError, read_bvals_bvecs, fbvecs, fbvecs)

        # All possible delimiters should work
        bv_file4 = 'test_space.txt'
        with open(bv_file4, 'w') as f:
            f.write("66 55 33")
        bvals_1, _ = read_bvals_bvecs(bv_file4, '')

        bv_file5 = 'test_coma.txt'
        with open(bv_file5, 'w') as f:
            f.write("66, 55, 33")
        bvals_2, _ = read_bvals_bvecs(bv_file5, '')

        bv_file6 = 'test_tabs.txt'
        with open(bv_file6, 'w') as f:
            f.write("66 \t 55 \t 33")
        bvals_3, _ = read_bvals_bvecs(bv_file6, '')

        ans = np.array([66., 55., 33.])
        npt.assert_array_equal(ans, bvals_1)
        npt.assert_array_equal(ans, bvals_2)
        npt.assert_array_equal(ans, bvals_3)

        bv_file7 = 'test_space_2.txt'
        with open(bv_file7, 'w') as f:
            f.write("66 55 33\n45 34 21\n55 32 65\n")
        _, bvecs_1 = read_bvals_bvecs('', bv_file7)

        bv_file8 = 'test_coma_2.txt'
        with open(bv_file8, 'w') as f:
            f.write("66, 55, 33\n45, 34, 21 \n 55, 32, 65\n")
        _, bvecs_2 = read_bvals_bvecs('', bv_file8)

        bv_file9 = 'test_tabs_2.txt'
        with open(bv_file9, 'w') as f:
            f.write("66 \t 55 \t 33\n45 \t 34 \t 21\n55 \t 32 \t 65\n")
        _, bvecs_3 = read_bvals_bvecs('', bv_file9)

        bv_file10 = 'test_multiple_space.txt'
        with open(bv_file10, 'w') as f:
            f.write("66   55   33\n45,   34,   21 \n 55,   32,     65\n")
        _, bvecs_4 = read_bvals_bvecs('', bv_file10)

        ans = np.array([[66., 55., 33.], [45., 34., 21.], [55., 32., 65.]])
        npt.assert_array_equal(ans, bvecs_1)
        npt.assert_array_equal(ans, bvecs_2)
        npt.assert_array_equal(ans, bvecs_3)
        npt.assert_array_equal(ans, bvecs_4)


if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
