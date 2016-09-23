from nose.tools import assert_raises

import os

from nibabel.tmpdirs import TemporaryDirectory

from dipy.data import get_data
from dipy.workflows.segment import MedianOtsuFlow
from dipy.workflows.workflow import Workflow


def test_force_overwrite():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_data('small_25')
        mo_flow = MedianOtsuFlow(output_strategy='absolute')

        # Generate the first results
        mo_flow.run(data_path, out_dir=out_dir)
        mask_file = mo_flow.last_generated_outputs['out_mask']
        first_time = os.path.getmtime(mask_file)

        # re-run with no force overwrite, modified time should not change
        mo_flow.run(data_path, out_dir=out_dir)
        mask_file = mo_flow.last_generated_outputs['out_mask']
        second_time = os.path.getmtime(mask_file)
        assert first_time == second_time

        # re-run with force overwrite, modified time should change
        mo_flow._force_overwrite = True
        mo_flow.run(data_path, out_dir=out_dir)
        mask_file = mo_flow.last_generated_outputs['out_mask']
        third_time = os.path.getmtime(mask_file)
        assert third_time != second_time


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
