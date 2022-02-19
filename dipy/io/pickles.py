""" Load and save pickles """
import pickle


def save_pickle(fname, dix):
    """Save `dix` to `fname` as pickle.

    Parameters
    ----------
    fname : str
       filename to save object e.g. a dictionary
    dix : str
       dictionary or other object

    Examples
    --------
    >>> import os
    >>> from tempfile import mkstemp
    >>> fd, fname = mkstemp() # make temporary file (opened, attached to fh)
    >>> d={0:{'d':1}}
    >>> save_pickle(fname, d)
    >>> d2=load_pickle(fname)

    We remove the temporary file we created for neatness

    >>> os.close(fd) # the file is still open, we need to close the fh
    >>> os.remove(fname)

    See Also
    --------
    dipy.io.pickles.load_pickle

    """
    out = open(fname, 'wb')
    pickle.dump(dix, out, protocol=pickle.HIGHEST_PROTOCOL)
    out.close()


def load_pickle(fname):
    """Load object from pickle file `fname`.

    Parameters
    ----------
    fname : str
       filename to load dict or other python object

    Returns
    -------
    dix : object
       dictionary or other object

    Examples
    --------
    dipy.io.pickles.save_pickle

    """
    inp = open(fname, 'rb')
    dix = pickle.load(inp)
    inp.close()
    return dix
