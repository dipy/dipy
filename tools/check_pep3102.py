#!/usr/bin/env python
"""Enforce keyword-only arguments for parameters with defaults (PEP 3102).

Any parameter carrying a default must sit after the bare ``*`` marker, so that
callers always name it. This keeps DIPY's public API free to reorder or extend
optional arguments without silently changing the meaning of positional calls.

Some signatures are dictated by a protocol or a third-party framework that
calls back positionally (the descriptor protocol, VTK observers, SciPy
optimizer callbacks). Those are exempt:

- dunder methods are skipped automatically;
- positional-only parameters (those before ``/``) are skipped, since they
  cannot be made keyword-only;
- ``lambda`` expressions are not inspected, as they carry no place to
  document an exemption;
- anything else can opt out with a ``# pep3102: ignore`` comment placed
  anywhere on the ``def`` line or on the continuation lines of a signature
  spread over several lines.
"""

import argparse
import ast
from pathlib import Path
import sys

IGNORE = "# pep3102: ignore"


def _defaulted_args(args):
    """Return the parameters that carry a default and could be keyword-only.

    ``args.defaults`` covers the tail of ``posonlyargs + args``. Only the part
    landing in ``args.args`` is actionable: a positional-only parameter cannot
    be moved after ``*``.

    Parameters
    ----------
    args : ast.arguments
        Argument node of a function definition.

    Returns
    -------
    list of ast.arg
        Positional-or-keyword parameters carrying a default.
    """
    n_defaults = min(len(args.args), len(args.defaults))
    if not n_defaults:
        return []
    return args.args[len(args.args) - n_defaults :]


def _is_ignored(node, lines):
    """Return True if the signature of ``node`` carries the opt-out comment.

    The comment is honored anywhere between the ``def`` line and the last line
    of the signature, so a signature wrapped over several lines can carry it on
    its closing ``):`` line.

    Parameters
    ----------
    node : ast.FunctionDef or ast.AsyncFunctionDef
        Function definition to inspect.
    lines : list of str
        Source lines of the file.

    Returns
    -------
    bool
        True when the function opts out of the check.
    """
    # ``node.body`` is never empty, and its first statement starts on the line
    # right after the signature (or on the signature line itself for a
    # single-line ``def f(): ...``).
    last = max(node.lineno, node.body[0].lineno - 1)
    return any(IGNORE in line for line in lines[node.lineno - 1 : last])


def find_violations(tree, source):
    """Collect functions whose defaulted parameters are not keyword-only.

    Parameters
    ----------
    tree : ast.Module
        Parsed syntax tree of the file.
    source : str
        Source text of the file, used to detect the opt-out comments.

    Returns
    -------
    list of tuple
        ``(lineno, function_name, offending_parameters)`` triples, sorted by
        line number.
    """
    lines = None
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.args.defaults:
            continue
        defaulted = _defaulted_args(node.args)
        if not defaulted:
            continue
        if node.name.startswith("__") and node.name.endswith("__"):
            continue
        # Splitting the source is only worth it once a candidate shows up.
        if lines is None:
            lines = source.splitlines()
        if _is_ignored(node, lines):
            continue
        violations.append((node.lineno, node.name, [a.arg for a in defaulted]))

    violations.sort()
    return violations


def check_file(path):
    """Report PEP 3102 violations in a single file.

    Parameters
    ----------
    path : str
        Path to the Python file to check.

    Returns
    -------
    list of str
        Human-readable violation messages.
    """
    try:
        source = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as err:
        return [f"{path}: could not decode as UTF-8: {err.reason}"]
    except OSError as err:
        return [f"{path}: could not read: {err.strerror or err}"]

    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        return [f"{path}:{err.lineno}: could not parse: {err.msg}"]

    return [
        f"{path}:{lineno}: {name}() takes defaulted argument(s) "
        f"{', '.join(params)} positionally; move them after '*'"
        for lineno, name, params in find_violations(tree, source)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*", help="files to check")
    args = parser.parse_args()

    messages = [msg for path in args.filenames for msg in check_file(path)]
    if not messages:
        return 0
    print("\n".join(messages))
    return 1


if __name__ == "__main__":
    sys.exit(main())
