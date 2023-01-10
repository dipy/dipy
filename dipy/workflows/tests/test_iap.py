import sys
from os.path import join as pjoin
from tempfile import TemporaryDirectory

import numpy.testing as npt

from dipy.workflows.base import IntrospectiveArgumentParser, none_or_dtype
from dipy.workflows.flow_runner import run_flow
from dipy.workflows.tests.workflow_tests_utils import (
    DummyFlow, DummyCombinedWorkflow, DummyWorkflow1, DummyWorkflowOptionalStr,
    DummyVariableTypeWorkflow, DummyVariableTypeErrorWorkflow)


def test_none_or_dtype():
    # test None
    for typ in [int, float, str, tuple, list]:
        dec = none_or_dtype(typ)
        npt.assert_equal(dec('None'), 'None')
        npt.assert_equal(dec('none'), 'None')

    dec = none_or_dtype(int)
    npt.assert_raises(ValueError, dec, 'my value')
    npt.assert_equal(dec(4), 4)

    dec = none_or_dtype(str)
    npt.assert_equal(dec(4), '4')
    npt.assert_equal(dec('my value'), 'my value')
    npt.assert_equal(dec([4]), '[4]')

    dec = none_or_dtype(float)
    npt.assert_raises(ValueError, dec, 'my value')
    for val in [(4,), [4, ]]:
        npt.assert_raises(TypeError, dec, val)
    npt.assert_equal(dec(True), 1.0)
    npt.assert_equal(dec(4), 4.0)
    npt.assert_equal(dec(4.0), 4.0)

    dec = none_or_dtype(bool)
    dec = none_or_dtype(tuple)


def test_variable_type():
    with TemporaryDirectory() as out_dir:
        open(pjoin(out_dir, 'test'), 'w').close()
        open(pjoin(out_dir, 'test1'), 'w').close()
        open(pjoin(out_dir, 'test2'), 'w').close()

        sys.argv = [sys.argv[0]]
        pos_results = [pjoin(out_dir, 'test'), pjoin(out_dir, 'test1'),
                       pjoin(out_dir, 'test2'), 12]
        inputs = inputs_from_results(pos_results)
        sys.argv.extend(inputs)
        dcwf = DummyVariableTypeWorkflow()
        _, positional_res, positional_res2 = run_flow(dcwf)
        npt.assert_equal(positional_res2, 12)

        for k, v in zip(positional_res, pos_results[:-1]):
            npt.assert_equal(k, v)

        dcwf = DummyVariableTypeErrorWorkflow()
        npt.assert_raises(ValueError, run_flow, dcwf)


def test_iap():
    sys.argv = [sys.argv[0]]
    pos_keys = ['positional_str', 'positional_bool', 'positional_int',
                'positional_float']

    opt_keys = ['optional_str', 'optional_bool', 'optional_int',
                'optional_float', 'optional_int_2', 'optional_float_2']

    pos_results = ['test', 0, 10, 10.2]
    opt_results = ['opt_test', True, 20, 20.2, None, None]

    inputs = inputs_from_results(opt_results, opt_keys, optional=True)
    inputs.extend(inputs_from_results(pos_results))

    sys.argv.extend(inputs)
    parser = IntrospectiveArgumentParser()
    dummy_flow = DummyFlow()
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


def test_optional_str():
    # Test optional and variable str argument exists but does not have a value
    sys.argv = [sys.argv[0]]
    inputs = ['--optional_str_1']
    sys.argv.extend(inputs)
    parser = IntrospectiveArgumentParser()
    dummy_flow = DummyWorkflowOptionalStr()
    parser.add_workflow(dummy_flow)
    args = parser.get_flow_args()
    all_keys = ['optional_str_1']
    all_results = [[]]
    # Test if types and order are respected
    for k, v in zip(all_keys, all_results):
        npt.assert_equal(args[k], v)
    # Test if **args really fits dummy_flow's arguments
    return_values = dummy_flow.run(**args)
    npt.assert_equal(return_values, all_results + ['default'])

    # Test optional and variable str argument exists and has a value
    sys.argv = [sys.argv[0]]
    inputs = ['--optional_str_1', 'test']
    sys.argv.extend(inputs)
    parser = IntrospectiveArgumentParser()
    dummy_flow = DummyWorkflowOptionalStr()
    parser.add_workflow(dummy_flow)
    args = parser.get_flow_args()
    all_keys = ['optional_str_1']
    all_results = [['test']]
    # Test if types and order are respected
    for k, v in zip(all_keys, all_results):
        npt.assert_equal(args[k], v)
    # Test if **args really fits dummy_flow's arguments
    return_values = dummy_flow.run(**args)
    npt.assert_equal(return_values, all_results + ['default'])

    # Test optional str empty arguments
    sys.argv = [sys.argv[0]]
    inputs = ['--optional_str_2']
    sys.argv.extend(inputs)
    parser = IntrospectiveArgumentParser()
    dummy_flow = DummyWorkflowOptionalStr()
    parser.add_workflow(dummy_flow)
    with npt.assert_raises(SystemExit) as cm:
        parser.get_flow_args()
    npt.assert_equal(cm.exception.code, 2)


def test_iap_epilog_and_description():
    parser = IntrospectiveArgumentParser()
    dummy_flow = DummyWorkflow1()
    parser.add_workflow(dummy_flow)
    assert "dummy references" in parser.epilog
    assert "Workflow used to test combined" in parser.description


def test_flow_runner():
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]

    opt_keys = ['param_combined', 'dwf1.param1', 'dwf2.param2', 'force',
                'out_strat', 'mix_names']

    pos_results = ['dipy.txt']
    opt_results = [30, 10, 20, True, 'absolute', True]

    inputs = inputs_from_results(opt_results, opt_keys, optional=True)
    inputs.extend(inputs_from_results(pos_results))

    sys.argv.extend(inputs)

    dcwf = DummyCombinedWorkflow()
    param1, param2, combined = run_flow(dcwf)

    # generic flow params
    assert dcwf._force_overwrite
    assert dcwf._output_strategy == 'absolute'
    assert dcwf._mix_names

    # sub flow params
    assert param1 == 10
    assert param2 == 20

    # parent flow param
    assert combined == 30

    sys.argv = old_argv


def inputs_from_results(results, keys=None, optional=False):
    prefix = '--'
    inputs = []
    for idx, result in enumerate(results):
        if keys is not None:
            inputs.append(prefix+keys[idx])
        if optional and str(result) in ['True', 'False']:
            continue
        inputs.append(str(result))

    return inputs
