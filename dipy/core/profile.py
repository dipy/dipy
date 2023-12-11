""" Class for profiling cython code
"""

import os
import subprocess

from dipy.utils.optpkg import optional_package

cProfile, _, _ = optional_package('cProfile')
pstats, _, _ = optional_package('pstats',
                                trip_msg='pstats is not installed.  It is '
                                'part of the python-profiler package in '
                                'Debian/Ubuntu')


class Profiler:
    """ Profile python/cython files or functions

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

    def __init__(self, call=None, *args):
        # Delay import until use of class instance.  We were getting some very
        # odd build-as-we-go errors running tests and documentation otherwise
        import pyximport
        pyximport.install()

        try:

            ext = os.path.splitext(call)[1].lower()
            print('ext', ext)
            if ext in ('.py', '.pyx'):  # python/cython file
                print('profiling python/cython file ...')
                subprocess.call(['python', '-m', 'cProfile',
                                 '-o', 'profile.prof', call])
                s = pstats.Stats('profile.prof')
                stats = s.strip_dirs().sort_stats('time')
                self.stats = stats

        except Exception:

            print('profiling function call ...')
            self.args = args
            self.call = call

            cProfile.runctx('self._profile_function()', globals(), locals(),
                            'profile.prof')
            s = pstats.Stats('profile.prof')
            stats = s.strip_dirs().sort_stats('time')
            self.stats = stats

    def _profile_function(self):
        self.call(*self.args)

    def print_stats(self, N=10):
        """ Print stats for profiling

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
