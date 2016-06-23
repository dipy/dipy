import numpy as np
from nose.tools import assert_raises
from os import path

from nibabel.tmpdirs import TemporaryDirectory

from dipy.workflows.combined_workflow import CombinedWorkflow
from dipy.workflows.tests.workflow_tests_utils import DummyCombinedWorkflow,\
    DummyWorkflow1, DummyWorkflow2


def test_get_sub_flows():
    cwf = CombinedWorkflow()
    assert_raises(AttributeError, cwf._get_sub_flows)


def test_combined_run():
    param1_value = 10
    param2_value = 20

    opts = {
        'DummyWorkflow1': {
            'param1': param1_value
        },
        'DummyWorkflow2': {
            'param2': param2_value
        }
    }

    dcwf = DummyCombinedWorkflow()
    dcwf.set_sub_flows_optionals(opts)
    with TemporaryDirectory() as tmpdir:
        inp = path.join(tmpdir, 'test.txt')
        np.savetxt(inp, np.arange(10))

        ret_param1, ret_param2, _ = dcwf.run(inp)
        assert ret_param1 == param1_value
        assert ret_param2 == param2_value


def test_sub_runs():
    dcwf = DummyCombinedWorkflow()
    runs = dcwf.get_sub_runs()

    assert runs[0] == (DummyWorkflow1.__name__,
                       DummyWorkflow1.run,
                       DummyWorkflow1.get_short_name())

    assert runs[1] == (DummyWorkflow2.__name__,
                       DummyWorkflow2.run,
                       DummyWorkflow2.get_short_name())


if __name__ == '__main__':
    test_get_sub_flows()
    test_combined_run()
    test_sub_runs()
