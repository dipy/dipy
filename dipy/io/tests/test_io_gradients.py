from __future__ import division, print_function, absolute_import

import os.path as osp
import os
import tempfile

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

    # These bvecs only have two rows/columns:
    new_bvecs1 = bvecs[:, :2]
    # Make a temporary file
    bv_file1 = tempfile.NamedTemporaryFile(mode='wt')
    # And fill it with these 2-columned bvecs:
    for x in range(new_bvecs1.shape[0]):
        bv_file1.file.write('%s %s\n' %
                            (new_bvecs1[x][0], new_bvecs1[x][1]))
    bv_file1.close()
    npt.assert_raises(IOError, read_bvals_bvecs, fbvals, bv_file1.name)

    # These bvecs are saved as one long array:
    new_bvecs2 = np.ravel(bvecs)
    bv_file2 = tempfile.NamedTemporaryFile()
    np.save(bv_file2, new_bvecs2)
    bv_file2.close()
    npt.assert_raises(IOError, read_bvals_bvecs, fbvals, bv_file2.name)

    # There are less bvecs than bvals:
    new_bvecs3 = bvecs[:-1, :]
    bv_file3 = tempfile.NamedTemporaryFile()
    np.save(bv_file3, new_bvecs3)
    bv_file3.close()
    npt.assert_raises(IOError, read_bvals_bvecs, fbvals, bv_file3.name)

    # You entered the bvecs on both sides:
    npt.assert_raises(IOError, read_bvals_bvecs, fbvecs, fbvecs)

    # All possible delimiters should work
    bv_file4 = tempfile.NamedTemporaryFile()
    with open(bv_file4.name, 'w') as f:
        f.write("66 55 33")
    bvals_1, _ = read_bvals_bvecs(bv_file4.name, None)

    bv_file5 = tempfile.NamedTemporaryFile()
    with open(bv_file5.name, 'w') as f:
        f.write("66, 55, 33")
    bvals_2, _ = read_bvals_bvecs(bv_file5.name, None)

    bv_file6 = tempfile.NamedTemporaryFile()
    with open(bv_file6.name, 'w') as f:
        f.write("66 \t 55 \t 33")
    bvals_3, _ = read_bvals_bvecs(bv_file6.name, None)

    ans = np.array([66., 55., 33.])
    npt.assert_array_equal(ans, bvals_1)
    npt.assert_array_equal(ans, bvals_2)
    npt.assert_array_equal(ans, bvals_3)

    bv_file7 = tempfile.NamedTemporaryFile()
    with open(bv_file7.name, 'w') as f:
        f.write("66 55 33 \n 45 34 21 \n 55 32 65")
    _, bvecs_1 = read_bvals_bvecs(None, bv_file7.name)

    bv_file8 = tempfile.NamedTemporaryFile()
    with open(bv_file8.name, 'w') as f:
        f.write("66, 55, 33 \n 45, 34, 21 \n 55, 32, 65")
    _, bvecs_2 = read_bvals_bvecs(None, bv_file8.name)

    bv_file9 = tempfile.NamedTemporaryFile()
    with open(bv_file9.name, 'w') as f:
        f.write("66 \t 55 \t 33 \n 45 \t 34 \t 21 \n 55 \t 32 \t 65")
    _, bvecs_3 = read_bvals_bvecs(None, bv_file9.name)

    ans = np.array([[66., 55., 33.], [45., 34., 21.], [55., 32., 65.]])
    npt.assert_array_equal(ans, bvecs_1)
    npt.assert_array_equal(ans, bvecs_2)
    npt.assert_array_equal(ans, bvecs_3)


if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
