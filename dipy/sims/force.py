"""
FORCE: Fast Orientation Reconstruction and Compartment Estimation

This module provides signal simulation for dictionary-based diffusion MRI
reconstruction using multi-compartment tissue models.
"""

import numpy as np

from dipy.sims.voxel import all_tensor_evecs


def bingham_to_sf(f0, k1, k2, major_axis, minor_axis, vertices):
    """
    Evaluate Bingham distribution on a sphere.

    The Bingham distribution models fiber orientation dispersion
    in diffusion MRI.

    Parameters
    ----------
    f0 : float
        Maximum amplitude of the distribution.
    k1 : float
        Concentration parameter along major axis.
    k2 : float
        Concentration parameter along minor axis.
    major_axis : ndarray (3,)
        Major axis of the distribution.
    minor_axis : ndarray (3,)
        Minor axis of the distribution.
    vertices : ndarray (N, 3)
        Unit sphere directions for evaluation.

    Returns
    -------
    sf : ndarray (N,)
        Spherical function values at each vertex.
    """
    sf = f0 * np.exp(
        -k1 * vertices.dot(major_axis) ** 2
        - k2 * vertices.dot(minor_axis) ** 2
    )
    return sf.T


def bingham_dictionary(target_sphere, odi_list):
    """
    Generate dictionary of Bingham spherical functions.

    Pre-computes Bingham distributions for all sphere directions
    and ODI values for efficient signal simulation.

    Parameters
    ----------
    target_sphere : ndarray (N, 3)
        Unit sphere vertices.
    odi_list : ndarray
        List of orientation dispersion index values.

    Returns
    -------
    bingham_sf : dict
        Nested dictionary mapping (vertex_index, odi) to
        spherical function values.
    """
    bingham_sf = {}
    for i in range(len(target_sphere)):
        vertex_key = tuple(target_sphere[i])
        bingham_sf[i] = {}

        for odi in odi_list:
            k = 1 / np.tan(np.pi / 2 * odi)
            evecs = all_tensor_evecs(vertex_key)
            major_axis, minor_axis = evecs[:, 1], evecs[:, 2]

            bingham_sf[i][odi] = bingham_to_sf(
                1, k, k, major_axis, minor_axis, target_sphere
            )

    return bingham_sf


def smallest_shell_bval(bvals, b0_threshold=50, shell_tolerance=50):
    """
    Find the smallest non-zero b-value shell.

    Parameters
    ----------
    bvals : ndarray
        B-values array.
    b0_threshold : float, optional
        Maximum b-value to consider as b0. Default is 50.
    shell_tolerance : float, optional
        Tolerance for shell grouping. Default is 50.

    Returns
    -------
    min_shell : float
        Smallest non-zero shell b-value.
    shell_mask : ndarray
        Boolean mask for volumes in that shell.
    """
    bvals = np.asarray(bvals, dtype=float)
    non_b0 = bvals > b0_threshold
    if not np.any(non_b0):
        raise ValueError("No non-b0 volumes found.")
    rounded = np.round(bvals[non_b0] / shell_tolerance) * shell_tolerance
    min_shell = float(np.min(rounded))
    shell_mask = np.isclose(
        np.round(bvals / shell_tolerance) * shell_tolerance, min_shell
    )
    return min_shell, shell_mask


def _create_memmap(output_dir, name, dtype, shape):
    """
    Create a memory-mapped array file.

    Parameters
    ----------
    output_dir : str
        Directory for memmap files.
    name : str
        Name of the memmap file.
    dtype : dtype
        Data type for the array.
    shape : tuple
        Shape of the array.

    Returns
    -------
    memmap : ndarray
        Memory-mapped array.
    path : str
        Path to the memmap file.
    """
    import os
    path = os.path.join(output_dir, f"{name}.mmap")
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape), path


def _generate_batch_worker(args):
    """
    Worker function for parallel batch generation.

    Parameters
    ----------
    args : tuple
        (start_idx, batch_size, sphere, evecs, bingham, odi,
         bvals, bvecs, wm_thresh, tortuosity, multi_tensor_func,
         diffusivity_cfg, output_arrays)

    Returns
    -------
    batch_size : int
        Number of samples processed.
    """
    (
        start_idx, batch_size, sphere, evecs, bingham, odi,
        bvals, bvecs, wm_thresh, tortuosity, multi_tensor_func,
        diffusivity_cfg, output_arrays
    ) = args

    from dipy.sims._force_core import create_mixed_signal, set_diffusivity_ranges

    # Apply diffusivity config in worker
    set_diffusivity_ranges(**diffusivity_cfg)

    (signals, labels, num_fibers, dispersion, wm_fraction,
     gm_fraction, csf_fraction, nd, odfs, ufa_wm, ufa_voxel,
     fraction_array, wm_disp, wm_d_par, wm_d_perp, gm_d_par,
     csf_d_par, f_ins) = output_arrays

    for i in range(batch_size):
        idx = start_idx + i
        res = create_mixed_signal(
            sphere, evecs, bingham, odi, bvals, bvecs,
            multi_tensor_func, wm_thresh, tortuosity
        )
        signals[idx] = res[0]
        labels[idx] = res[1]
        num_fibers[idx] = res[2]
        dispersion[idx] = res[3]
        wm_fraction[idx] = res[4]
        gm_fraction[idx] = res[5]
        csf_fraction[idx] = res[6]
        nd[idx] = res[7]
        odfs[idx] = res[8]
        ufa_wm[idx] = res[9]
        ufa_voxel[idx] = res[10]
        fraction_array[idx] = res[11]
        wm_disp[idx] = res[12]
        wm_d_par[idx] = res[13]
        wm_d_perp[idx] = res[14]
        gm_d_par[idx] = res[15]
        csf_d_par[idx] = res[16]
        f_ins[idx] = res[17]

    return batch_size
