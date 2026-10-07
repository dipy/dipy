"""Tests for explicitly exported objects in generated API documentation."""

import importlib.util
from pathlib import Path
import sys

import pytest

_spec = importlib.util.spec_from_file_location(
    "apigen", Path(__file__).resolve().parents[2] / "doc" / "tools" / "apigen.py"
)
_apigen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_apigen)


@pytest.fixture
def api_package(tmp_path, monkeypatch):
    """Create a package exposing a small API from a private module."""
    package = tmp_path / "exported_api"
    package.mkdir()
    (package / "_public.py").write_text(
        "from functools import partial\n"
        'def motion_correction():\n    """Correct motion."""\n'
        'class Registration:\n    """Registration result."""\n'
        'def unexported():\n    """Internal helper."""\n'
        "partial_api = partial(motion_correction)\n"
    )
    (package / "__init__.py").write_text(
        "from ._public import motion_correction, Registration, unexported, partial_api\n"
        '__all__ = ["motion_correction", "Registration", "local", "value", "partial_api"]\n'
        'def local():\n    """Local helper."""\n'
        "value = 42\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "exported_api"
    for name in list(sys.modules):
        if name == "exported_api" or name.startswith("exported_api."):
            del sys.modules[name]


def test_explicit_exports(api_package):
    writer = _apigen.ApiDocWriter(api_package, object_skip_patterns=["^ignored$"])
    functions, classes, constants = writer._parse_module_with_import(api_package)
    assert functions == ["local", "motion_correction", "partial_api"]
    assert classes == ["Registration"]
    assert constants == []
    head, body = writer.generate_api_doc(api_package)
    assert "   motion_correction\n" in head
    assert ".. autofunction:: motion_correction" in body
    assert ".. autofunction:: partial_api" in body
    assert ".. autoclass:: Registration" in body
    assert "unexported" not in head + body


def test_exports_respect_skip_patterns(api_package):
    writer = _apigen.ApiDocWriter(
        api_package, object_skip_patterns=["motion_correction|Registration"]
    )
    assert writer._parse_module_with_import(api_package) == (
        ["local", "partial_api"],
        [],
        [],
    )


def test_exports_respect_other_defines(api_package):
    writer = _apigen.ApiDocWriter(
        api_package, object_skip_patterns=["^ignored$"], other_defines=False
    )
    assert writer._parse_module_with_import(api_package) == (
        ["local"],
        [],
        [],
    )


def test_no_explicit_exports(api_package):
    module = sys.modules.get(api_package)
    if module is None:
        module = __import__(api_package)
    del module.__all__
    writer = _apigen.ApiDocWriter(api_package, object_skip_patterns=["^ignored$"])
    assert writer._parse_module_with_import(api_package) == (["local"], [], [])
