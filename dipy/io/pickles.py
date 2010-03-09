import cPickle

def save_pickle(fname,dix):
    '''
    Parameters:
    ----------
    fname: filename to save object e.g. a dictionary

    dix: dictionary or other object

    Example:
    --------
    >>> import os
    >>> from tempfile import mkstemp
    >>> fd, fname = mkstemp() # make temporary file
    >>> d={0:{'d':1}}
    >>> save_pickle(fname, d)
    >>> d2=load_pickle(fname)
    >>> os.remove(fname)
    
    '''
    out=open(fname,'wb')
    cPickle.dump(dix,out)
    out.close()

def load_pickle(fname):

    '''
    Parameter:
    -----------
    fname: filename to load dict or other python object 

    Returns:
    --------
    dix: dictionary or other object

    Example:
    --------
    See ``save_pickle``
    '''
    inp=open(fname,'rb')
    dix=cPickle.load(inp)
    inp.close()
    return dix
