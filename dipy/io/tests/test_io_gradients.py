from __future__ import division, print_function, absolute_import

import os.path as osp
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


if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
