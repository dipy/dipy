import contextlib
import sys

import pytest

from dipy.core.profile import Profiler, have_pyximport


@pytest.fixture
def restored_import_hooks():
    """Undo the process-wide ``pyximport.install()`` that ``Profiler`` performs."""
    meta_path = list(sys.meta_path)
    path_hooks = list(sys.path_hooks)
    yield
    sys.meta_path[:] = meta_path
    sys.path_hooks[:] = path_hooks
    sys.path_importer_cache.clear()


def test_profiler_deprecated(monkeypatch, tmp_path, restored_import_hooks):
    monkeypatch.chdir(tmp_path)
    # The deprecation fires before the pyximport guard, so it is covered either
    # way; only the call that follows depends on Cython being installed.
    missing_cython = pytest.raises(ImportError, match="pyximport")
    with pytest.warns(DeprecationWarning, match="Use the `spin profile`"):
        with contextlib.nullcontext() if have_pyximport else missing_cython:
            Profiler(3, call=lambda n: sum(range(n)))
