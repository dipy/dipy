"""Tests for the PEP 3102 pre-commit checker."""

import ast
import importlib.util
from pathlib import Path
import sys

import pytest

_CHECKER = Path(__file__).resolve().parents[1] / "check_pep3102.py"
_spec = importlib.util.spec_from_file_location("check_pep3102", _CHECKER)
check_pep3102 = importlib.util.module_from_spec(_spec)
sys.modules["check_pep3102"] = check_pep3102
_spec.loader.exec_module(check_pep3102)


def violations(source):
    """Return ``(name, params)`` pairs reported for ``source``, in file order."""
    found = check_pep3102.find_violations(ast.parse(source), source)
    return [(name, params) for _, name, params in found]


def test_plain_default_is_flagged():
    assert violations("def f(a, b=1): pass") == [("f", ["b"])]


def test_keyword_only_default_is_clean():
    assert violations("def f(a, *, b=1): pass") == []
    assert violations("def f(a, *args, b=1, **kw): pass") == []
    assert violations("def f(a, b): pass") == []


def test_dunder_is_exempt():
    assert violations("class C:\n    def __init__(self, x=1): pass") == []


def test_name_mangled_method_is_not_a_dunder():
    assert violations("class C:\n    def __hidden(self, x=1): pass") == [
        ("__hidden", ["x"])
    ]


def test_async_function_is_checked():
    assert violations("async def f(a=1): pass") == [("f", ["a"])]


def test_nested_function_is_checked():
    source = "def outer():\n    def inner(a=1): pass\n"
    assert violations(source) == [("inner", ["a"])]


def test_lambda_is_not_inspected():
    assert violations("f = lambda a, b=1: a + b") == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Positional-only defaults cannot be moved after '*'.
        ("def f(a=1, /): pass", []),
        ("def f(a=1, b=2, /): pass", []),
        # Only the positional-or-keyword tail is actionable.
        ("def f(a=1, /, b=2): pass", [("f", ["b"])]),
        ("def f(a=1, /, b=2, c=3): pass", [("f", ["b", "c"])]),
        ("def f(a, /, b=2): pass", [("f", ["b"])]),
    ],
)
def test_positional_only_parameters(source, expected):
    assert violations(source) == expected


def test_ignore_comment_on_def_line():
    assert violations(f"def f(a, b=1):  {check_pep3102.IGNORE}\n    pass") == []


def test_ignore_comment_on_closing_line_of_wrapped_signature():
    source = f"def f(\n    a,\n    b=1,\n):  {check_pep3102.IGNORE}\n    pass\n"
    assert violations(source) == []


def test_ignore_comment_below_the_signature_is_not_honored():
    source = f"def f(a, b=1):\n    pass  {check_pep3102.IGNORE}\n"
    assert violations(source) == [("f", ["b"])]


def test_violations_are_sorted_by_line_number():
    source = "def outer(a=1):\n    def inner(b=2): pass\n\ndef last(c=3): pass\n"
    found = check_pep3102.find_violations(ast.parse(source), source)
    assert [lineno for lineno, _, _ in found] == [1, 2, 4]


def test_check_file_reports_syntax_errors(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n", encoding="utf-8")
    (message,) = check_pep3102.check_file(str(bad))
    assert "could not parse" in message


def test_check_file_reports_unreadable_files(tmp_path):
    missing = tmp_path / "does_not_exist.py"
    (message,) = check_pep3102.check_file(str(missing))
    assert "could not read" in message


def test_check_file_reports_undecodable_files(tmp_path):
    binary = tmp_path / "binary.py"
    binary.write_bytes(b"def f(a=\xff): pass\n")
    (message,) = check_pep3102.check_file(str(binary))
    assert "could not decode" in message


def test_check_file_message_points_at_the_definition(tmp_path):
    good = tmp_path / "sample.py"
    good.write_text("x = 1\n\n\ndef f(a, b=1, c=2):\n    pass\n", encoding="utf-8")
    (message,) = check_pep3102.check_file(str(good))
    assert message.startswith(f"{good}:4: f() takes defaulted argument(s) b, c")


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    clean = tmp_path / "clean.py"
    clean.write_text("def f(a, *, b=1): pass\n", encoding="utf-8")
    dirty = tmp_path / "dirty.py"
    dirty.write_text("def f(a, b=1): pass\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["check_pep3102.py", str(clean)])
    assert check_pep3102.main() == 0
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(sys, "argv", ["check_pep3102.py", str(dirty)])
    assert check_pep3102.main() == 1
    assert "move them after '*'" in capsys.readouterr().out
