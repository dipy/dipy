"""Module for default handing functions."""

from .convert import expand_range

def handle_vol_idx(vol_idx):
    if vol_idx is not None:
        if isinstance(vol_idx, str):
            vol_idx = expand_range(vol_idx)
        elif isinstance(vol_idx, int):
            vol_idx = [vol_idx]
        elif isinstance(vol_idx, (list, tuple)):
            vol_idx = [int(idx) for idx in vol_idx]
    return vol_idx
