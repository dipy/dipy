"""Testing multiproc utilities"""

import os
import sys

from numpy.testing import assert_equal, assert_raises

from dipy.utils.multiproc import determine_num_processes, get_usable_cpu_affinity


def test_determine_num_processes():
    # Test that the correct number of effective num_processes is returned

    # 0 should raise an error
    assert_raises(ValueError, determine_num_processes, 0)

    # A string should raise an error
    assert_raises(TypeError, determine_num_processes, "0")

    # 1 should be 1
    assert_equal(determine_num_processes(1), 1)

    # A positive integer should not change
    assert_equal(determine_num_processes(4), 4)

    # None and -1 should be equal (all cores)
    assert_equal(determine_num_processes(None), determine_num_processes(-1))

    # A big negative number should be 1
    assert_equal(determine_num_processes(-10000), 1)

    # -2 should be one less than -1 (if there are more than 1 cores)
    if determine_num_processes(-1) > 1:
        assert_equal(determine_num_processes(-1), determine_num_processes(-2) + 1)


def test_get_usable_cpu_affinity():
    cpus = get_usable_cpu_affinity()
    assert isinstance(cpus, set)
    assert len(cpus) >= 1
    assert all(isinstance(c, int) for c in cpus)

    # Where the platform reports an affinity mask, it is what we return, and
    # it can be narrower than the machine core count (pinning, cgroups).
    if hasattr(os, "sched_getaffinity"):
        assert_equal(cpus, os.sched_getaffinity(0))
        assert len(cpus) <= (os.cpu_count() or 1)


def test_get_usable_cpu_affinity_fallbacks(monkeypatch):
    """Each fallback is used when the one above it is unavailable."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)

    # Python 3.13+ API.
    monkeypatch.setattr(os, "process_cpu_count", lambda: 3, raising=False)
    assert_equal(get_usable_cpu_affinity(), {0, 1, 2})

    # macOS shells out to sysctl.
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        "dipy.utils.multiproc.subprocess.check_output", lambda *a, **kw: b" 5\n"
    )
    assert_equal(get_usable_cpu_affinity(), {0, 1, 2, 3, 4})

    # Generic fallback when sysctl is unusable.
    monkeypatch.setattr(
        "dipy.utils.multiproc.subprocess.check_output",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("nope")),
    )
    monkeypatch.setattr("dipy.utils.multiproc.cpu_count", lambda: 2)
    assert_equal(get_usable_cpu_affinity(), {0, 1})
