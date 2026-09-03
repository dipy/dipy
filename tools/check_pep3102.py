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
  anywhere between the first decorator and the ``:`` closing the signature.

Only Python is covered. ``ast`` cannot parse Cython, so the ``.pyx`` sources
are outside the reach of this check and their optional arguments stay
positional; that gap is permanent, not a backlog item.
"""

import argparse
import ast
import io
from operator import itemgetter
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


def _scan_source(source):
    """Return the marker lines and the last line of every signature.

    Both come from a single pass over the token stream. Reading comment tokens
    rather than raw text keeps a default value that merely contains the marker
    -- a string literal, say -- from opting its function out. Locating the
    ``:`` that closes each ``def`` keeps a comment sitting *inside* the body
    from doing so either: a bare ``# pep3102: ignore`` on the line above the
    first statement is body, not signature.

    Parameters
    ----------
    source : str
        Source text of the file.

    Returns
    -------
    marked : set of int
        1-based line numbers of the comments carrying the marker.
    signature_end : dict
        Maps the 1-based line of each ``def`` to the line of the ``:`` that
        closes its signature.
    """
    marked = set()
    signature_end = {}
    def_line = None
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                if IGNORE in token.string:
                    marked.add(token.start[0])
            elif def_line is None:
                if token.type == tokenize.NAME and token.string == "def":
                    def_line = token.start[0]
                    depth = 0
            elif token.type == tokenize.OP:
                if token.string in "([{":
                    depth += 1
                elif token.string in ")]}":
                    depth -= 1
                elif token.string == ":" and depth == 0:
                    signature_end[def_line] = token.start[0]
                    def_line = None
    except (tokenize.TokenError, IndentationError):
        # Unreachable in principle -- ``ast`` parsed the file already. Losing
        # the opt-out is the safe direction: report rather than skip.
        pass
    return marked, signature_end


def _is_ignored(node, signature_end, marked):
    """Return True if the signature of ``node`` carries the opt-out comment.

    The comment is honored anywhere between the first decorator and the ``:``
    closing the signature, so a signature wrapped over several lines can carry
    it on its closing ``):`` line, and a stack of decorators can carry it next
    to the one that explains why the exemption is needed. Anything below that
    ``:`` is the body and does not exempt the function.

    Parameters
    ----------
    node : ast.FunctionDef or ast.AsyncFunctionDef
        Function definition to inspect.
    signature_end : dict
        Signature end lines keyed by ``def`` line, from :func:`_scan_source`.
    marked : set of int
        Line numbers carrying the marker, from :func:`_scan_source`.

    Returns
    -------
    bool
        True when the function opts out of the check.
    """
    start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    # ``node.lineno`` is the ``def`` line even for a decorated function; the
    # fallback only fires if tokenizing bailed out, and keeps the range empty.
    end = signature_end.get(node.lineno, node.lineno)
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
    marked = None
    signature_end = {}
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
        if marked is None:
            if IGNORE in source:
                marked, signature_end = _scan_source(source)
            else:
                marked = frozenset()
        if marked and _is_ignored(node, signature_end, marked):
            continue
        violations.append((node.lineno, node.name, [a.arg for a in defaulted]))

    violations.sort(key=itemgetter(0))
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
        # ``utf-8-sig`` strips a leading BOM, which ``ast`` rejects outright:
        # a BOM-prefixed file is valid Python, not a parse error to report.
        source = Path(path).read_text(encoding="utf-8-sig")
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
