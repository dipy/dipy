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


def generate_force_dictionary(
    gtab,
    num_simulations=100000,
    output_dir=None,
    num_cpus=1,
    batch_size=1000,
    wm_threshold=0.5,
    tortuosity=False,
    odi_range=(0.01, 0.3),
    num_odi_values=10,
    diffusivity_config=None,
    dtype=np.float32,
    compute_dti=True,
    compute_dki=False,
    verbose=False,
):
    """
    Generate FORCE simulation dictionary.

    Creates a dictionary of simulated diffusion MRI signals with
    corresponding microstructural parameters for dictionary-based
    reconstruction.

    Parameters
    ----------
    gtab : GradientTable
        Gradient table with b-values and b-vectors.
    num_simulations : int, optional
        Number of simulated voxels. Default is 100000.
    output_dir : str, optional
        Directory for output files. If None, uses temporary directory.
    num_cpus : int, optional
        Number of CPU cores for parallel processing. Default is 1.
    batch_size : int, optional
        Batch size for processing. Default is 1000.
    wm_threshold : float, optional
        Minimum WM fraction to include fiber labels. Default is 0.5.
    tortuosity : bool, optional
        Use tortuosity constraint for perpendicular diffusivity.
        Default is False.
    odi_range : tuple, optional
        (min, max) orientation dispersion index range. Default is (0.01, 0.3).
    num_odi_values : int, optional
        Number of ODI values to sample. Default is 10.
    diffusivity_config : dict, optional
        Custom diffusivity ranges. Keys: wm_d_par_range, wm_d_perp_range,
        gm_d_iso_range, csf_d.
    dtype : dtype, optional
        Data type for outputs. Default is np.float32.
    compute_dti : bool, optional
        Compute DTI metrics (FA, MD, RD). Default is True.
    compute_dki : bool, optional
        Compute DKI metrics (AK, RK, MK, KFA). Default is False.
    verbose : bool, optional
        Enable progress output. Default is False.

    Returns
    -------
    dictionary : dict
        Dictionary containing simulated signals and parameters.
    """
    import os
    import tempfile
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from dipy.data import default_sphere
    from dipy.reconst import dti

    from dipy.sims._force_core import set_diffusivity_ranges
    from dipy.sims._multi_tensor_omp import multi_tensor

    # Setup output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="force_sim_")
    os.makedirs(output_dir, exist_ok=True)

    # Setup diffusivity config
    if diffusivity_config is None:
        diffusivity_config = {
            "wm_d_par_range": (2.0e-3, 3.0e-3),
            "wm_d_perp_range": (0.3e-3, 1.5e-3),
            "gm_d_iso_range": (0.7e-3, 1.2e-3),
            "csf_d": 3.0e-3,
        }

    set_diffusivity_ranges(**diffusivity_config)

    # Setup sphere and ODI values
    sphere = default_sphere
    target_sphere = np.ascontiguousarray(sphere.vertices, dtype=np.float64)
    n_dirs = target_sphere.shape[0]

    bvals = np.ascontiguousarray(gtab.bvals.astype(np.float64))
    bvecs = np.ascontiguousarray(gtab.bvecs.astype(np.float64))
    n_bvals = bvals.shape[0]

    # Pre-compute eigenvectors and Bingham dictionary
    evecs = np.array(
        [all_tensor_evecs(tuple(point)) for point in target_sphere],
        dtype=np.float64,
    )
    odi_list = np.linspace(odi_range[0], odi_range[1], num_odi_values).astype(np.float64)
    bingham_sf = bingham_dictionary(target_sphere, odi_list)

    # Allocate output arrays
    signals = np.zeros((num_simulations, n_bvals), dtype=dtype)
    labels = np.zeros((num_simulations, n_dirs), dtype=np.uint8)
    num_fibers = np.zeros(num_simulations, dtype=dtype)
    dispersion = np.zeros(num_simulations, dtype=dtype)
    wm_fraction = np.zeros(num_simulations, dtype=dtype)
    gm_fraction = np.zeros(num_simulations, dtype=dtype)
    csf_fraction = np.zeros(num_simulations, dtype=dtype)
    nd = np.zeros(num_simulations, dtype=dtype)
    odfs = np.zeros((num_simulations, n_dirs), dtype=np.float16)
    ufa_wm = np.zeros(num_simulations, dtype=dtype)
    ufa_voxel = np.zeros(num_simulations, dtype=dtype)
    fraction_array = np.zeros((num_simulations, 3), dtype=dtype)
    wm_disp = np.zeros(num_simulations, dtype=dtype)
    wm_d_par = np.zeros(num_simulations, dtype=dtype)
    wm_d_perp = np.zeros(num_simulations, dtype=dtype)
    gm_d_par = np.zeros(num_simulations, dtype=dtype)
    csf_d_par = np.zeros(num_simulations, dtype=dtype)
    f_ins = np.zeros((num_simulations, 3), dtype=dtype)

    output_arrays = (
        signals, labels, num_fibers, dispersion, wm_fraction,
        gm_fraction, csf_fraction, nd, odfs, ufa_wm, ufa_voxel,
        fraction_array, wm_disp, wm_d_par, wm_d_perp, gm_d_par,
        csf_d_par, f_ins
    )

    # Generate simulations
    from dipy.sims._force_core import create_mixed_signal

    for idx in range(num_simulations):
        res = create_mixed_signal(
            target_sphere, evecs, bingham_sf, odi_list,
            bvals, bvecs, multi_tensor, wm_threshold, tortuosity
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

    # Compute DTI metrics if requested
    fa_dti = np.zeros(num_simulations, dtype=dtype)
    md_dti = np.zeros(num_simulations, dtype=dtype)
    rd_dti = np.zeros(num_simulations, dtype=dtype)

    if compute_dti:
        min_b, shell_mask = smallest_shell_bval(bvals)
        b0_mask = bvals <= 50
        use_mask = shell_mask | b0_mask

        from dipy.core.gradients import gradient_table
        gtab_small = gradient_table(bvals[use_mask], bvecs=bvecs[use_mask])
        dti_model = dti.TensorModel(gtab_small)

        dti_batch_size = 2000
        for start in range(0, num_simulations, dti_batch_size):
            end = min(start + dti_batch_size, num_simulations)
            data_batch = signals[start:end][:, use_mask]
            dti_fit = dti_model.fit(data_batch)
            fa_dti[start:end] = dti_fit.fa.astype(dtype)
            md_dti[start:end] = dti_fit.md.astype(dtype)
            rd_dti[start:end] = dti_fit.rd.astype(dtype)

    # Build output dictionary
    dictionary = {
        "signals": signals,
        "labels": labels,
        "num_fibers": num_fibers,
        "dispersion": dispersion,
        "wm_fraction": wm_fraction,
        "gm_fraction": gm_fraction,
        "csf_fraction": csf_fraction,
        "nd": nd,
        "odfs": odfs,
        "fa": fa_dti,
        "md": md_dti,
        "rd": rd_dti,
        "ufa_wm": ufa_wm,
        "ufa_voxel": ufa_voxel,
        "fraction_array": fraction_array,
    }

    return dictionary


def save_force_dictionary(dictionary, output_path):
    """
    Save FORCE dictionary to compressed NPZ file.

    Parameters
    ----------
    dictionary : dict
        FORCE simulation dictionary.
    output_path : str
        Path for output file (.npz extension).
    """
    np.savez_compressed(output_path, **dictionary)


def load_force_dictionary(input_path):
    """
    Load FORCE dictionary from NPZ file.

    Parameters
    ----------
    input_path : str
        Path to dictionary file.

    Returns
    -------
    dictionary : dict
        FORCE simulation dictionary.
    """
    data = np.load(input_path, allow_pickle=False)
    return {key: data[key] for key in data.files}
