"""Deprecated. ``histeq`` now lives in :mod:`dipy.utils.histeq`.

This module only re-exports it and is removed in DIPY 2.0.
"""

from dipy.utils.deprecator import deprecate_with_version
from dipy.utils.histeq import histeq as _histeq

__all__ = ["histeq"]

histeq = deprecate_with_version(
    "dipy.core.histeq is deprecated. Import 'histeq' from dipy.utils.histeq instead.",
    since="1.13.0",
    until="2.0.0",
)(_histeq)
