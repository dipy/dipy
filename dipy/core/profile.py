""" Class for profiling cython code
"""

try:
    import cProfile
except ImportError:
    raise ImportError('cProfile is not installed')
try:
    import pstats
except ImportError:
    raise ImportError('pstats is not installed it is part of the python-profiler package')
    print('In Ubuntu you will need to do sudo apt-get install python-profiler')
 
import os
import subprocess

import pyximport
pyximport.install()

class Profiler():
    ''' Profile python/cython files or functions
    
    If you are profiling cython code you need to add    
    # cython: profile=True on the top of your .pyx file
    
    and for the functions that you do not want to profile you can use
    this decorator in your cython files

    @cython.profile(False)

    Parameters
    ----------
    caller: file or function call
    args: function arguments

    Attributes
    ----------
    stats: function, stats.print_stats(10) will prin the 10 slower functions
    
    Examples
    --------
    import dipy.core.profile as p
    import numpy as np
    p.Profiler(np.sum,np.random.rand(1000000,3))
    fname='test.py'
    p.Profiler(fname)
    p.print_stats(10)
    p.print_stats('det')

    References
    ----------
    http://docs.cython.org/src/tutorial/profiling_tutorial.html
    http://docs.python.org/library/profile.html
    http://packages.python.org/line_profiler/
    
    '''

    def __init__(self,call=None,*args):

        try:
            
            ext=os.path.splitext(call)[1].lower()        
            print('ext',ext)               
            if ext == '.py' or ext == '.pyx': #python/cython file
                print('profiling python/cython file ...')
                subprocess.call(['python','-m','cProfile', \
                                 '-o','profile.prof',call])
                s = pstats.Stats('profile.prof')            
                stats=s.strip_dirs().sort_stats('time')
                self.stats=stats
            
        except:

            print('profiling function call ...')   
            self.args=args
            self.call=call

            cProfile.runctx('self._profile_function()',globals(),locals(),\
                                'profile.prof')
            s = pstats.Stats('profile.prof')
            stats=s.strip_dirs().sort_stats('time')
            self.stats=stats


    def _profile_function(self):
        self.call(*self.args)

    def print_stats(self,N=10):
        ''' Print stats for profiling

        You can use it in all different ways developed in pstats
        for example
        print_stats(10) will give you the 10 slowest calls
        or
        print_stats('function_name')
        will give you the stats for all the calls with name 'function_name'
                
        Parameters
        ----------
        N: stats.print_stats argument

        '''
        self.stats.print_stats(N)
