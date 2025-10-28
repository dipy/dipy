import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy.testing as npt

from dipy.data import get_fnames
from dipy.testing import assert_warns
from dipy.utils.deprecator import ArgsDeprecationWarning
from dipy.workflows.segment import MedianOtsuFlow
from dipy.workflows.workflow import Workflow


def test_force_overwrite():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames(name="small_25")
        mo_flow = MedianOtsuFlow(output_strategy="absolute")

        # Generate the first results
        with assert_warns(ArgsDeprecationWarning):
            mo_flow.run(data_path, out_dir=out_dir, vol_idx=[0])
        mask_file = mo_flow.last_generated_outputs["out_mask"]
        first_time = os.path.getmtime(mask_file)

        # re-run with no force overwrite, modified time should not change
        mo_flow.run(data_path, out_dir=out_dir)
        mask_file = mo_flow.last_generated_outputs["out_mask"]
        second_time = os.path.getmtime(mask_file)
        assert first_time == second_time

        # re-run with force overwrite, modified time should change
        mo_flow = MedianOtsuFlow(output_strategy="absolute", force=True)
        # Make sure that at least one second elapsed, so that time-stamp is
        # different (sometimes measured in whole seconds)
        time.sleep(1)
        with assert_warns(ArgsDeprecationWarning):
            mo_flow.run(data_path, out_dir=out_dir, vol_idx=[0])
        mask_file = mo_flow.last_generated_outputs["out_mask"]
        third_time = os.path.getmtime(mask_file)
        assert third_time != second_time


def test_get_sub_runs():
    wf = Workflow()
    assert len(wf.get_sub_runs()) == 0


def test_run():
    wf = Workflow()
    npt.assert_raises(Exception, wf.run, None)


def test_missing_file():
    # The function is invoking a dummy workflow with a non-existent file.
    # So, an OSError will be raised.

    class TestMissingFile(Workflow):
        def run(self, filename, out_dir=""):
            """Dummy Workflow used to test if input file is absent.

            Parameters
            ----------

            filename : string or Path
                path of the first input file.
            out_dir: string or Path, optional
                folder path to save the results.
            """
            _ = self.get_io_iterator()

    dummyflow = TestMissingFile()
    with TemporaryDirectory() as tempdir:
        npt.assert_raises(OSError, dummyflow.run, str(Path(tempdir) / "dummy_file.txt"))
