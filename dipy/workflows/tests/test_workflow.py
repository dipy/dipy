import os
import time

import numpy.testing as npt

from dipy.data import get_fnames
from dipy.workflows.segment import MedianOtsuFlow
from dipy.workflows.tests.workflow_tests_utils import DummyNonPathOutWorkflow
from dipy.workflows.workflow import Workflow


def test_force_overwrite(tmp_path):
    data_path, _, _ = get_fnames(name="small_25")
    mo_flow = MedianOtsuFlow(output_strategy="absolute")

    # Generate the first results
    mo_flow.run(data_path, out_dir=tmp_path, vol_idx=[0])
    mask_file = mo_flow.last_generated_outputs["out_mask"]
    first_time = os.path.getmtime(mask_file)

    # re-run with no force overwrite, modified time should not change
    mo_flow.run(data_path, out_dir=tmp_path)
    mask_file = mo_flow.last_generated_outputs["out_mask"]
    second_time = os.path.getmtime(mask_file)
    assert first_time == second_time

    # re-run with force overwrite, modified time should change
    mo_flow = MedianOtsuFlow(output_strategy="absolute", force=True)
    # Make sure that at least one second elapsed, so that time-stamp is
    # different (sometimes measured in whole seconds)
    time.sleep(1)
    mo_flow.run(data_path, out_dir=tmp_path, vol_idx=[0])
    mask_file = mo_flow.last_generated_outputs["out_mask"]
    third_time = os.path.getmtime(mask_file)
    assert third_time != second_time


def test_get_sub_runs():
    wf = Workflow()
    assert len(wf.get_sub_runs()) == 0


def test_run():
    wf = Workflow()
    npt.assert_raises(Exception, wf.run, None)


def test_missing_file(tmp_path):
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
    npt.assert_raises(OSError, dummyflow.run, str(tmp_path / "dummy_file.txt"))


def test_non_path_outputs_io_rows(tmp_path):
    in_file = tmp_path / "in.txt"
    in_file.write_text("x")
    expected = [(str(in_file), str(tmp_path / "res.txt"))]

    for out_dtype in (None, "float32"):
        flow = DummyNonPathOutWorkflow(force=True)
        rows = flow.run(in_file, out_dtype=out_dtype, out_dir=tmp_path)
        npt.assert_equal(rows, expected)


def test_non_path_outputs_multiple_inputs(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    in_files = [in_dir / "a.txt", in_dir / "b.txt"]
    for in_file in in_files:
        in_file.write_text("x")

    flow = DummyNonPathOutWorkflow(force=True)
    rows = flow.run(in_dir / "*.txt", out_dtype="float32", out_dir=tmp_path)
    npt.assert_equal(
        rows, [(str(in_file), str(tmp_path / "res.txt")) for in_file in in_files]
    )


def test_non_path_outputs_last_generated(tmp_path):
    in_file = tmp_path / "in.txt"
    in_file.write_text("x")

    flow = DummyNonPathOutWorkflow(force=True)
    flow.run(in_file, out_dtype="float32", out_dir=tmp_path)
    npt.assert_equal([str(f) for f in flow.flat_outputs], [str(tmp_path / "res.txt")])
    npt.assert_equal(list(flow.last_generated_outputs), ["out_res"])


def test_non_path_outputs_overwrite(tmp_path):
    in_file = tmp_path / "in.txt"
    in_file.write_text("x")
    (tmp_path / "float32").write_text("x")

    flow = DummyNonPathOutWorkflow()
    rows = flow.run(in_file, out_dtype="float32", out_dir=tmp_path)
    npt.assert_equal(len(rows), 1)

    (tmp_path / "res.txt").write_text("x")
    rows = flow.run(in_file, out_dtype="float32", out_dir=tmp_path)
    npt.assert_equal(rows, [])


def test_non_path_outputs_default():
    npt.assert_equal(Workflow._non_path_outputs, ())
