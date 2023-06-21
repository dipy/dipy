import inspect

import pytest
from unittest import mock

from dipy.workflows import base


def test_compile_args_positional():
    param = base.Param("name", "string", "", base.Empty)
    parameters = [("name", "string", [""])]
    arg_defaults = {"name": base.Empty}
    parser = base.IntrospectiveArgumentParser()

    parser._compile_args(parameters, arg_defaults)

    assert parser.positional_parameters == [param]
    assert parser.optional_parameters == []
    assert parser.output_parameters == []


def test_compile_args_optional():
    param = base.Param("name", "string, optional", "", None)
    parameters = [("name", "string, optional", [""])]
    arg_defaults = {"name": None}
    parser = base.IntrospectiveArgumentParser()

    parser._compile_args(parameters, arg_defaults)

    assert parser.positional_parameters == []
    assert parser.optional_parameters == [param]
    assert parser.output_parameters == []


def test_compile_args_output():
    param = base.Param("out_name", "string, optional", "", None)
    parameters = [("out_name", "string, optional", [""])]
    arg_defaults = {"out_name": None}
    parser = base.IntrospectiveArgumentParser()

    parser._compile_args(parameters, arg_defaults)

    assert parser.positional_parameters == []
    assert parser.optional_parameters == []
    assert parser.output_parameters == [param]


def test_compile_args_default_missing():
    parameters = [("name", "string", [""])]
    arg_defaults = {}
    parser = base.IntrospectiveArgumentParser()
    with pytest.raises(ValueError) as e:
        parser._compile_args(parameters, arg_defaults)

    assert "Argument 'name' not present in function signature" in str(e.value)


def test_compile_args_positional_optional_type():
    parameters = [("name", "string, optional", [""])]
    arg_defaults = {"name": base.Empty}
    parser = base.IntrospectiveArgumentParser()
    with pytest.raises(ValueError) as e:
        parser._compile_args(parameters, arg_defaults)

    assert "Required argument 'name' marked as optional" in str(e.value)


def test_compile_args_positional_marked_as_output():
    parameters = [("out_name", "string, optional", [""])]
    arg_defaults = {"out_name": base.Empty}
    parser = base.IntrospectiveArgumentParser()
    with pytest.raises(ValueError) as e:
        parser._compile_args(parameters, arg_defaults)

    assert "Required argument 'out_name' starts with reserved keyword 'out'." in str(e.value)


def test_compile_args_extra_default():
    parameters = []
    arg_defaults = {"extra": None}
    parser = base.IntrospectiveArgumentParser()
    with pytest.raises(ValueError) as e:
        parser._compile_args(parameters, arg_defaults)

    assert "Arguments are missing from docstring: ['extra']" in str(e.value)


@mock.patch("dipy.workflows.base.sys")
def test_get_args_default_py3(mock_sys):
    def _run(self, positional, optional=None, out_output=None):
        pass

    expected = {
        "positional": base.Empty,
        "optional": None,
        "out_output": None
    }
    mock_sys.version_info = (2,)
    assert base.get_args_default(_run) == expected


@mock.patch("dipy.workflows.base.sys")
def test_get_args_default_py2(mock_sys):
    def _run(self, positional, optional=None, out_output=None):
        pass

    expected = {
        "positional": base.Empty,
        "optional": None,
        "out_output": None
    }
    mock_sys.version_info = (2,)
    assert base.get_args_default(_run) == expected


