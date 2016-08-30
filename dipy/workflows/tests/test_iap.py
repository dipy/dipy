import numpy.testing as npt
import sys

from dipy.workflows.base import IntrospectiveArgumentParser
from dipy.workflows.flow_runner import run_flow
from dipy.workflows.tests.workflow_tests_utils import TestFlow, \
    DummyCombinedWorkflow


def test_iap():
    sys.argv = [sys.argv[0]]
    pos_keys = ['positional_str', 'positional_bool', 'positional_int',
                'positional_float']

    opt_keys = ['optional_str', 'optional_bool', 'optional_int',
                'optional_float']

    pos_results = ['test', 0, 10, 10.2]
    opt_results = ['opt_test', True, 20, 20.2]

    inputs = inputs_from_results(opt_results, opt_keys, optionnal=True)
    inputs.extend(inputs_from_results(pos_results))

    sys.argv.extend(inputs)
    parser = IntrospectiveArgumentParser()
    dummy_flow = TestFlow()
    parser.add_workflow(dummy_flow)
    args = parser.get_flow_args()
    all_keys = pos_keys + opt_keys
    all_results = pos_results + opt_results

    # Test if types and order are respected
    for k, v in zip(all_keys, all_results):
        npt.assert_equal(args[k], v)

    # Test if **args really fits dummy_flow's arguments
    return_values = dummy_flow.run(**args)
    npt.assert_array_equal(return_values, all_results + [2.0])


def test_flow_runner():
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]

    opt_keys = ['param_combined', 'dwf1.param1', 'dwf2.param2', 'force', 'out_strat',
                'mix_names']

    pos_results = ['dipy.txt']
    opt_results = [30, 10, 20, True, 'aboslute', True]

    inputs = inputs_from_results(opt_results, opt_keys, optionnal=True)
    inputs.extend(inputs_from_results(pos_results))

    sys.argv.extend(inputs)

    dcwf = DummyCombinedWorkflow()
    param1, param2, combined = run_flow(dcwf)

    # generic flow params
    assert dcwf._force_overwrite
    assert dcwf._output_strategy == 'aboslute'
    assert dcwf._mix_names

    # sub flow params
    assert param1 == 10
    assert param2 == 20

    # parent flow param
    assert combined == 30

    sys.argv = old_argv


def inputs_from_results(results, keys=None, optionnal=False):
    prefix = '--'
    inputs = []
    for idx, result in enumerate(results):
        if keys is not None:
            inputs.append(prefix+keys[idx])
        if optionnal and str(result) in ['True', 'False']:
            continue
        inputs.append(str(result))

    return inputs

if __name__ == '__main__':
    test_iap()
    test_flow_runner()
