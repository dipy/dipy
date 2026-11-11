"""Load and save pickles"""

import pickle
import warnings


def save_pickle(fname, dix):
    """Save `dix` to `fname` as pickle.

    Parameters
    ----------
    fname : str
       filename to save object e.g. a dictionary
    dix : object
       dictionary or other object

    Examples
    --------
    >>> import os
    >>> from tempfile import mkstemp
    >>> fd, fname = mkstemp() # make temporary file (opened, attached to fh)
    >>> d={0:{'d':1}}
    >>> save_pickle(fname, d)
    >>> d2=load_pickle(fname, check_safety=False)

    We remove the temporary file we created for neatness

    >>> os.close(fd) # the file is still open, we need to close the fh
    >>> os.remove(fname)

    See Also
    --------
    dipy.io.pickles.load_pickle

    """
    with open(fname, "wb") as out:
        pickle.dump(dix, out, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(fname, *, check_safety=True):
    """Load object from pickle file `fname`.

    Parameters
    ----------
    fname : str
       filename to load dict or other python object
    check_safety : bool, optional
       If True, emit a warning that pickle files can execute arbitrary code.
       Set to False only when loading files from trusted sources.

    Returns
    -------
    dix : object
       dictionary or other object

    See Also
    --------
    dipy.io.pickles.save_pickle

    """
    if check_safety:
        warnings.warn(
            "pickle.load() executes arbitrary code; only load files from trusted sources.",
            UserWarning,
            stacklevel=2,
        )
    with open(fname, "rb") as inp:
        return pickle.load(inp)
