import importlib
import io
from os.path import splitext
import re
import warnings

import numpy as np


def read_bvals_bvecs(fbvals, fbvecs):
    """Read b-values and b-vectors from disk.

    Parameters
    ----------
    fbvals : str
       Full path to file with b-values. None to not read bvals.
    fbvecs : str
       Full path of file with b-vectors. None to not read bvecs.

    Returns
    -------
    bvals : array, (N,) or None
    bvecs : array, (N, 3) or None

    Notes
    -----
    Files can be either '.bvals'/'.bvecs' or '.txt' or '.npy' (containing
    arrays stored with the appropriate values).

    """
    # Loop over the provided inputs, reading each one in turn and adding them
    # to this list:
    vals = []
    for this_fname in [fbvals, fbvecs]:
        # If the input was None or empty string, we don't read anything and
        # move on:
        if this_fname is None or not this_fname:
            vals.append(None)
            continue

        if not isinstance(this_fname, str):
            raise ValueError("String with full path to file is required")

        base, ext = splitext(this_fname)
        if ext in [
            ".bvals",
            ".bval",
            ".bvecs",
            ".bvec",
            ".txt",
            ".eddy_rotated_bvecs",
            "",
        ]:
            with open(this_fname, "r") as f:
                content = f.read()

            munged_content = io.StringIO(re.sub(r"(\t|,)", " ", content))
            vals.append(np.squeeze(np.loadtxt(munged_content)))
        elif ext == ".npy":
            vals.append(np.squeeze(np.load(this_fname)))
        else:
            e_s = f"File type {ext} is not recognized"
            raise ValueError(e_s)

    # Once out of the loop, unpack them:
    bvals, bvecs = vals[0], vals[1]

    # If bvecs is None, you can just return now w/o making more checks:
    if bvecs is None:
        return bvals, bvecs

    if 3 not in bvecs.shape:
        raise OSError("bvec file should have three rows")
    if bvecs.ndim != 2:
        bvecs = bvecs[None, ...]
        bvals = bvals[None, ...]
        msg = "Detected only 1 direction on your bvec file. For diffusion "
        msg += "dataset, it is recommended to have at least 3 directions."
        msg += "You may have problems during the reconstruction step."
        warnings.warn(msg, stacklevel=2)
    if bvecs.shape[1] != 3:
        bvecs = bvecs.T

    # If bvals is None, you don't need to check that they have the same shape:
    if bvals is None:
        return bvals, bvecs

    if len(bvals.shape) > 1:
        raise OSError("bval file should have one row")

    if bvals.shape[0] != bvecs.shape[0]:
        raise OSError("b-values and b-vectors shapes do not correspond")

    return bvals, bvecs


def read_gradient_table(fname):
    """Read gradient table from disk.

    Parameters
    ----------
    fname : str
       Full path to gradient table file

    Returns
    -------
    grad_table : GradientTable

    """
    data = np.genfromtxt(fname, delimiter=",", names=True)

    standard_header = ["bvals", "bvecs_x", "bvecs_y", "bvecs_z", "b0s_mask"]
    if not set(standard_header).issubset(data.dtype.names):
        msg = f"File does not contain the required fields: {standard_header}"
        raise ValueError(msg)

    bvals = data["bvals"]
    bvecs = np.stack((data["bvecs_x"], data["bvecs_y"], data["bvecs_z"])).T
    b0_threshold_idx = np.argmax(data["b0s_mask"] == 0)
    b0_threshold_idx = b0_threshold_idx - 1 if b0_threshold_idx > 0 else 0
    b0_threshold = bvals[b0_threshold_idx]

    small_delta, big_delta = None, None
    if {"small_delta", "big_delta"}.issubset(data.dtype.names):
        small_delta = data["small_delta"][0]
        big_delta = data["big_delta"][0]

    btensor = None
    btensor_header = [
        "btens_xx",
        "btens_xy",
        "btens_xz",
        "btens_yx",
        "btens_yy",
        "btens_yz",
        "btens_zx",
        "btens_zy",
        "btens_zz",
    ]
    if set(btensor_header).issubset(data.dtype.names):
        btensor = np.stack(
            (
                data["btens_xx"],
                data["btens_xy"],
                data["btens_xz"],
                data["btens_yx"],
                data["btens_yy"],
                data["btens_yz"],
                data["btens_zx"],
                data["btens_zy"],
                data["btens_zz"],
            )
        ).T
        btensor = np.reshape(btensor, (bvals.size, 3, 3))

    module = importlib.import_module("dipy.core.gradients")
    gradient_table = module.gradient_table

    return gradient_table(
        bvals,
        bvecs,
        b0_threshold=b0_threshold,
        small_delta=small_delta,
        big_delta=big_delta,
        btens=btensor,
    )


def save_gradient_table(gtab, fname):
    """Save gradient table to disk.

    Parameters
    ----------
    gtab : GradientTable
       Gradient table to save
    fname : str
        Full path to gradient table file

    Returns
    -------
    fgtab : str
       Full path to gradient table file with b-values and b-vectors

    """
    module = importlib.import_module("dipy.core.gradients")
    GradientTable = module.GradientTable
    if not isinstance(gtab, GradientTable):
        raise ValueError("gtab should be a GradientTable instance")

    if not fname.lower().endswith(".gtab.csv"):
        base, ext = splitext(fname)
        fname = f"{base}.gtab.csv"
        warnings.warn(f"File extension not recognized. Saving as {fname}", stacklevel=1)

    header = ",".join(gtab.header)
    np.savetxt(fname, gtab.condensed, delimiter=",", header=header)
    return fname
