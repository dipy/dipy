import contextlib

import pytest

from dipy.core.profile import Profiler, have_pyximport


def test_profiler_deprecated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # The deprecation fires before the pyximport guard, so it is covered either
    # way; only the call that follows depends on Cython being installed.
    missing_cython = pytest.raises(ImportError, match="pyximport")
    with pytest.warns(DeprecationWarning, match="Use the `spin profile`"):
        with contextlib.nullcontext() if have_pyximport else missing_cython:
            Profiler(3, call=lambda n: sum(range(n)))
