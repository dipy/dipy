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
    

    '''

    def __init__(self,caller,args=None,rows=10):

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

            print('cython file - not yet ready')

        else :

            pass
        
            '''

            c=caller.split('.')
            module=caller.split('.'+c[-1])
            #function=c[-1].split('(')
            #function=function.split(')')

            print 'c',c
            print 'c-1',c[-1]
            print 'module',module[0]            
            print 'test',

            m=__import__(module[0])

                       

            print('cython or python function')
            cProfile.runctx(cc,globals(),locals(),\
                                'profile.prof')
            s = pstats.Stats('profile.prof')
            stats=s.strip_dirs().sort_stats('time')
            stats.print_stats(rows)
            self.stats

            '''


            

        
    

            
            

            

            

            

        

            

            

            


    

        
