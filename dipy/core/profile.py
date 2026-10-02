"""Deprecated. Class for profiling cython code, removed in DIPY 2.0.

Use the ``spin profile`` developer command instead.
"""

import os
import subprocess

from dipy.utils.deprecator import deprecate_with_version, warning_for_keywords
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package

cProfile, _, _ = optional_package("cProfile")
pstats, _, _ = optional_package(
    "pstats",
    trip_msg="pstats is not installed.  It is "
    "part of the python-profiler package in "
    "Debian/Ubuntu",
)
pyximport, have_pyximport, _ = optional_package("pyximport")


class Profiler:
    """Profile python/cython files or functions

    .. deprecated:: 1.13.0
        ``dipy.core.profile`` is removed in 2.0.0. Use ``spin profile``
        from a source checkout instead.

    If you are profiling cython code you need to add
    # cython: profile=True on the top of your .pyx file

    and for the functions that you do not want to profile you can use
    this decorator in your cython files

    @cython.profile(False)

    Parameters
    ----------
    caller : file or function call
    args : function arguments

    Attributes
    ----------
    stats : function, stats.print_stats(10) will prin the 10 slower functions

    Examples
    --------
    from dipy.core.profile import Profiler
    import numpy as np
    p=Profiler(np.sum,np.random.rand(1000000,3))
    fname='test.py'
    p=Profiler(fname)
    p.print_stats(10)
    p.print_stats('det')

    References
    ----------
    https://docs.cython.org/src/tutorial/profiling_tutorial.html
    https://docs.python.org/library/profile.html
    https://github.com/rkern/line_profiler

    """

    @deprecate_with_version(
        "dipy.core.profile.Profiler is deprecated and is removed in DIPY 2.0.0. "
        "Use the `spin profile` developer command instead.",
        since="1.13.0",
        until="2.0.0",
    )
    def __init__(self, *args, call=None):
        if not have_pyximport:
            raise ImportError(
                "pyximport (Cython) is required by dipy.core.profile.Profiler."
            )

        pyximport.install()

        try:
            ext = os.path.splitext(call)[1].lower()
            logger.info("ext", ext)
            if ext in (".py", ".pyx"):  # python/cython file
                logger.info("profiling python/cython file ...")
                subprocess.call(
                    ["python3", "-m", "cProfile", "-o", "profile.prof", call]
                )
                s = pstats.Stats("profile.prof")
                stats = s.strip_dirs().sort_stats("time")
                self.stats = stats

        except Exception:
            logger.info("profiling function call ...")
            self.args = args
            self.call = call

            cProfile.runctx(
                "self._profile_function()", globals(), locals(), "profile.prof"
            )
            s = pstats.Stats("profile.prof")
            stats = s.strip_dirs().sort_stats("time")
            self.stats = stats

    def _profile_function(self):
        self.call(*self.args)

    @warning_for_keywords()
    def print_stats(self, *, N=10):
        """Print stats for profiling

        You can use it in all different ways developed in pstats
        for example
        print_stats(10) will give you the 10 slowest calls
        or
        print_stats('function_name')
        will give you the stats for all the calls with name 'function_name'

        Parameters
        ----------
        N : stats.print_stats argument

        """
        self.stats.print_stats(N)
