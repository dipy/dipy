import pstats, cProfile
import os
import subprocess

import dipy.core.reconstruction_performance as rp

import pyximport
pyximport.install()

class Profiler():
    ''' Profile python/cython files or functions
    
    If you are profiling cython code you need to add    
    # cython: profile=True on the top of your .pyx file
    
    and for the functions that you do not want to profile you can use this decorator

    @cython.profile(False)

    Parameters
    ----------
    caller: string, file or function call
    rows: int, number of functions to be shown with pstats

    Attributes
    ----------
    stats: function, stats.print_stats(10) will prin the 10 slower functions
    

    Examples
    --------
    p.Profiler('dipy.core.track_metrics.length',
    dipy.core.track_metrics.length,np.random.rand(1000000,3))

    
    '''

    def __init__(self,caller,call=None,*args):

        rows=10

        ext=os.path.splitext(caller)[1].lower()        
        print('ext',ext)        
        
        if ext == '.py': #python file
            print('python file')
            subprocess.call(['python','-m','cProfile', \
                                 '-o','profile.prof',fname])
            s = pstats.Stats('profile.prof')            
            stats=s.strip_dirs().sort_stats('time')
            stats.print_stats(rows)
            self.stats=stats

        elif ext == '.pyx': #cython file

            print('cython file - profiling not yet implemented')

        else :

            print('function call')

            #caller = 'dipy.core.track_metrics.length'

            function=caller.split('.')[-1]
            module=caller.split('.'+function)[0]

            self.function=function
            self.module=module
            self.args=args

            self.call=call

            cProfile.runctx('self.profile_function()',globals(),locals(),\
                                'profile.prof')
            s = pstats.Stats('profile.prof')
            stats=s.strip_dirs().sort_stats('time')
            stats.print_stats(rows)



    def profile_function(self):    
           

        self.call(*self.args)
        

        


            

            


            

        
    

            
            

            

            

            

        

            

            

            


    

        
