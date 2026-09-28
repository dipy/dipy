"""Reference-count helpers for leak tests.

The public ``get_type_refcount`` is deprecated and removed in DIPY 2.0. The
private ``_get_type_refcount`` stays for DIPY's own leak tests.
"""

from collections import defaultdict
import gc

from dipy.utils.deprecator import deprecate_with_version, warning_for_keywords


def _get_type_refcount(*, pattern=None):
    """
    Retrieves refcount of types for which their name matches `pattern`.

    Parameters
    ----------
    pattern : str, optional
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


@deprecate_with_version(
    "dipy.testing.memory.get_type_refcount is deprecated and is removed in "
    "DIPY 2.0.0 without replacement.",
    since="1.13.0",
    until="2.0.0",
)
@warning_for_keywords(from_version="1.13.0")
def get_type_refcount(*, pattern=None):
    """
    Retrieves refcount of types for which their name matches `pattern`.

    .. deprecated:: 1.13.0
        Removed in 2.0.0 without replacement.

    Parameters
    ----------
    pattern : str, optional
        Consider only types that have `pattern` in their name.

    Returns
    -------
    dict
        The key is the type name and the value is the refcount.
    """
    return _get_type_refcount(pattern=pattern)
