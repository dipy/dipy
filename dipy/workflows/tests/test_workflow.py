from nose.tools import assert_raises

from nibabel.tmpdirs import TemporaryDirectory

from dipy.data import get_data
from dipy.workflows.segment import MedianOtsuFlow
from dipy.workflows.workflow import Workflow


def test_force_overwrite():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_data('small_25')
        mo_flow = MedianOtsuFlow(output_strategy='absolute')

        # Generate the first results
        mo_flow.run(data_path, out_dir=out_dir, save_masked=True)

        # re-run with no force overwrite, should have any outputs
        mo_flow.run(data_path, out_dir=out_dir, save_masked=True)
        outputs = mo_flow.last_generated_outputs
        assert len(outputs) == 0

        # re-run with force overwrite, should have outputs
        mo_flow._force_overwrite = True
        mo_flow.run(data_path, out_dir=out_dir, save_masked=True)
        outputs = mo_flow.last_generated_outputs
        assert len(outputs) > 0


def test_set_sub_flows_optionals():
    wf = Workflow()
    assert_raises(Exception, wf.set_sub_flows_optionals, None)


def test_get_sub_runs():
    wf = Workflow()
    assert len(wf.get_sub_runs()) == 0


def test_run():
    wf = Workflow()
    assert_raises(Exception, wf.run, None)

if __name__ == '__main__':
    test_force_overwrite()
    test_set_sub_flows_optionals()
    test_get_sub_runs()
    test_run()
