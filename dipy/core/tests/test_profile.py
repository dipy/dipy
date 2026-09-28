import pytest

from dipy.core.profile import Profiler


def test_profiler_deprecated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.warns(DeprecationWarning, match="Use the `spin profile`"):
        Profiler(3, call=lambda n: sum(range(n)))
