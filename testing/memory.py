import gc
from collections import defaultdict


def get_type_refcount(pattern=None):
    """
    Retrieves refcount of types for which their name matches `pattern`.

    Parameters
    ----------
    pattern : str
        Consider only types that have `pattern` in their name.

    Returns
    -------
    dict
        The key is the type name and the value is the refcount.
    """
    gc.collect()

    refcounts_per_type = defaultdict(int)
    for obj in gc.get_objects():
        obj_type_name = type(obj).__name__
        # If `pattern` is not None, keep only matching types.
        if pattern is None or pattern in obj_type_name:
            refcounts_per_type[obj_type_name] += 1

    return refcounts_per_type
