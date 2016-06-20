import numpy as np
import os
from nose.tools import assert_raises
from os import path

from nibabel.tmpdirs import TemporaryDirectory

from dipy.workflows.combined_workflow import CombinedWorkflow
from dipy.workflows.workflow import Workflow


class DummyWorkflow1(Workflow):
    def run(self, inputs, param1=1, out_dir='', output_1='out1.txt'):
        return param1


class DummyWorkflow2(Workflow):
    def run(self, inputs, param2=2, out_dir='', output_1='out2.txt'):
        return param2


class DummyCombinedWorkflow(CombinedWorkflow):
    def _get_sub_flows(self):
        return [DummyWorkflow1, DummyWorkflow2]

    def run(self, inputs, param_combined=3, out_dir='',
            output_combined='out_combined.txt'):

        dwf1 = DummyWorkflow1()
        param1 = self.run_sub_flow(dwf1, inputs)

        dwf2 = DummyWorkflow2()
        param2 = self.run_sub_flow(dwf2, inputs)

        return param1, param2


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

        ret_param1, ret_param2 = dcwf.run(inp)
        assert ret_param1 == param1_value
        assert ret_param2 == param2_value


def test_sub_runs():
    dcwf = DummyCombinedWorkflow()
    runs = dcwf.get_sub_runs()

    assert runs[0] == (DummyWorkflow1.__name__, DummyWorkflow1.run)
    assert runs[1] == (DummyWorkflow2.__name__, DummyWorkflow2.run)


if __name__ == '__main__':
    test_get_sub_flows()
    test_combined_run()
    test_sub_runs()
