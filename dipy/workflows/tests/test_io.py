from dipy.data import get_fnames
from dipy.workflows.io import IoInfoFlow
import logging

import numpy as np
from tempfile import mkstemp
fname_log = mkstemp()[1]

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(message)s',
                    filename=fname_log,
                    filemode='w')


def test_io_info():
    fimg, fbvals, fbvecs = get_fnames('small_101D')
    io_info_flow = IoInfoFlow()
    io_info_flow.run([fimg, fbvals, fbvecs])

    fimg, fbvals, fvecs = get_fnames('small_25')
    io_info_flow = IoInfoFlow()
    io_info_flow.run([fimg, fbvals, fvecs])

    io_info_flow = IoInfoFlow()
    io_info_flow.run([fimg, fbvals, fvecs], b0_threshold=20, bvecs_tol=0.001) 

    file = open(fname_log, 'r')
    lines = file.readlines()
    try:
        np.testing.assert_equal(
            lines[-3],
            'INFO Total number of unit bvectors 25\n')
    except IndexError:  # logging maybe disabled in IDE setting
        pass
    file.close()


if __name__ == '__main__':
    test_io_info()
