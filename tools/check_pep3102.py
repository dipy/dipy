#!/usr/bin/env python
"""Enforce keyword-only arguments for parameters with defaults (PEP 3102).

Any parameter carrying a default must sit after the bare ``*`` marker, so that
callers always name it. This keeps DIPY's public API free to reorder or extend
optional arguments without silently changing the meaning of positional calls.

Some signatures are dictated by a protocol or a third-party framework that
calls back positionally (the descriptor protocol, VTK observers, SciPy
optimizer callbacks). Those are exempt:

- dunder methods the interpreter calls back are skipped automatically, but not
  ``__init__``/``__new__``/``__call__``: those receive what user code wrote at
  the call site, so they are ordinary API surface (see ``CHECKED_DUNDERS``);
- positional-only parameters (those before ``/``) are skipped, since they
  cannot be made keyword-only;
- ``lambda`` expressions are not inspected, as they carry no place to
  document an exemption;
- anything else can opt out with a ``# pep3102: ignore`` comment placed
  anywhere between the first decorator and the last line of the signature.

Only Python is covered. ``ast`` cannot parse Cython, so the ``.pyx`` sources
are outside the reach of this check and their optional arguments stay
positional; that gap is permanent, not a backlog item.
"""

import argparse
import ast
import io
from pathlib import Path
import sys
import tokenize
import warnings

IGNORE = "# pep3102: ignore"

#: Dunders that are not a callback from the interpreter: what they receive is
#: what user code wrote (``C(...)``, ``obj(...)``), so the rule applies to them
#: like it does to any other function.
CHECKED_DUNDERS = frozenset({"__init__", "__new__", "__call__"})


def _is_exempt_dunder(name):
    """Return True for a dunder the interpreter, not user code, calls.

    Parameters
    ----------
    name : str
        Function name.

    Returns
    -------
    bool
        True when the name is a dunder outside :data:`CHECKED_DUNDERS`.
    """
    return (
        len(name) > 4
        and name.startswith("__")
        and name.endswith("__")
        and name not in CHECKED_DUNDERS
    )


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


def _marker_lines(source):
    """Return the line numbers carrying the opt-out marker as a real comment.

    Reading comment tokens rather than raw text keeps a default value that
    merely contains the marker -- a string literal, say -- from opting its
    function out.

    Parameters
    ----------
    source : str
        Source text of the file.

    Returns
    -------
    set of int
        1-based line numbers of the comments carrying the marker.
    """
    marked = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT and IGNORE in token.string:
                marked.add(token.start[0])
    except (tokenize.TokenError, IndentationError):
        # ``ast`` parsed the file already, so this is not expected. Dropping the
        # opt-out is the safe direction: the check reports rather than skips.
        pass
    return marked


def _is_ignored(node, lines, marked):
    """Return True if the signature of ``node`` carries the opt-out comment.

    The comment is honored anywhere between the first decorator and the last
    line of the signature, so a signature wrapped over several lines can carry
    it on its closing ``):`` line, and a stack of decorators can carry it next
    to the one that explains why the exemption is needed.

    Parameters
    ----------
    node : ast.FunctionDef or ast.AsyncFunctionDef
        Function definition to inspect.
    lines : list of str
        Source lines of the file.
    marked : set of int
        Line numbers carrying the marker, from :func:`_marker_lines`.

    Returns
    -------
    bool
        True when the function opts out of the check.
    """
    # ``node.body`` is never empty. Its first statement either opens on the
    # closing line of the signature (``): pass``), in which case that line is
    # still part of the signature and may carry the marker, or it owns its
    # line, in which case the signature stops one line earlier.
    first = node.body[0]
    head = lines[first.lineno - 1][: first.col_offset]
    last = first.lineno if head.rstrip().endswith(":") else first.lineno - 1
    # Decorators sit above the ``def`` line and are part of the declaration.
    start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    end = max(node.lineno, last)
    # ``marked`` holds a handful of lines at most, a declaration can span dozens.
    return any(start <= line <= end for line in marked)


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
    lines = marked = None
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.args.defaults:
            continue
        defaulted = _defaulted_args(node.args)
        if not defaulted:
            continue
        if _is_exempt_dunder(node.name):
            continue
        # Scanning the source is only worth it once a candidate shows up, and
        # tokenizing it only when the marker appears somewhere in the file.
        if marked is None:
            marked = _marker_lines(source) if IGNORE in source else frozenset()
            lines = source.splitlines() if marked else ()
        if marked and _is_ignored(node, lines, marked):
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
        # ``ast.parse`` reports lexical problems of its own (a stray ``\.`` in a
        # non-raw string, say) as warnings pointing at ``<unknown>``. Those are
        # ruff's job, and an unattributed line number here only misleads.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError as err:
        # ``err.lineno`` is None for whole-file failures (a null byte, say);
        # point at the first line rather than printing ``path:None:``.
        return [f"{path}:{err.lineno or 1}: could not parse: {err.msg}"]

    return [
        f"{path}:{lineno}: {name}() takes defaulted argument(s) "
        f"{', '.join(params)} positionally; move them after '*' "
        f"(or add '{IGNORE}' on the def line, or on a decorator line, when a "
        "protocol or a third-party framework calls back positionally)"
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
