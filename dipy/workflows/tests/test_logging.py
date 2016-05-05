import os
import logging
import filecmp
import subprocess

from nibabel.tmpdirs import TemporaryDirectory
from dipy.data import get_data
from dipy.workflows.segment import median_otsu_flow

from nose.tools import assert_true

def test_logging_consistency():
    with TemporaryDirectory() as tmpdir:
        data_path, _, _ = get_data('small_25')
        prog_log = os.path.join(tmpdir, 'prog_log.txt')
        format = '%(levelname)s:%(message)s'
        level = 'INFO'

        logging.basicConfig(filename=prog_log,
                            format=format,
                            level=level)

        median_otsu_flow(data_path, out_dir=tmpdir)

        cmd_log = os.path.join(tmpdir, 'cmd_log.txt')
        cmd_line = \
            'dipy_median_otsu {0} --out_dir {1} --log_file {2} --log_level {3}'.\
                format(data_path, tmpdir, cmd_log, level)

        subprocess.call(cmd_line.split(' '))
        same_content = filecmp.cmp(prog_log, cmd_log)
        assert_true(same_content)

        logger = logging.getLogger()
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

if __name__ == '__main__':
    test_logging_consistency()