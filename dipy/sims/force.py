"""
FORCE: Forward modeling for complex microstructure estimation

This module provides signal simulation for matching-based diffusion MRI
reconstruction using multi-compartment tissue models.

The FORCE method generates a library of simulated signals representing
various tissue configurations (white matter with 1-3 fibers, gray matter,
CSF) and uses this library to reconstruct microstructural parameters
from real diffusion MRI data.

References
----------
:footcite:p:`Shah2025`
"""

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import multiprocessing as mp
import os
import sys
import tempfile

import numpy as np

from dipy.data import default_sphere
from dipy.reconst.bingham import _single_bingham_to_sf as bingham_to_sf
from dipy.sims.voxel import all_tensor_evecs
from dipy.utils.logging import logger
from dipy.utils.multiproc import determine_num_processes


def dispersion_lut(target_sphere, odi_list):
    """Generate spherical functions for all directions and ODI values.
    Currently uses Bingham distribution, but can be extended to other
    dispersion models in the future.

    Parameters
    ----------
    target_sphere : ndarray (N, 3)
        Unit sphere vertices.
    odi_list : ndarray
        List of orientation dispersion index values.

    Returns
    -------
    SF : dict
        Nested dictionary mapping (vertex_index, odi) to spherical function (SF)
        values.
    """
    dispersion_sf_lut = {}
    for i in range(len(target_sphere)):
        vertex_key = tuple(target_sphere[i])
        dispersion_sf_lut[i] = {}

        for odi in odi_list:
            k = 1 / np.tan(np.pi / 2 * odi)
            evecs = all_tensor_evecs(vertex_key)
            major_axis, minor_axis = evecs[:, 1], evecs[:, 2]

            dispersion_sf_lut[i][odi] = bingham_to_sf(
                1, k, k, major_axis, minor_axis, target_sphere
            )

    return dispersion_sf_lut


def smallest_shell_bval(bvals, *, b0_threshold=50, shell_tolerance=50, n=1):
    """Find the smallest non-zero b-value shell.

    Parameters
    ----------
    bvals : ndarray
        B-values array.
    b0_threshold : float, optional
        Maximum b-value for b0 volumes.
    shell_tolerance : float, optional
        Tolerance for grouping shells.
    n : int, optional
        Number of smallest shells to return.

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
    unique_shells = np.unique(rounded)
    if len(unique_shells) < n:
        raise ValueError(
            f"Only {len(unique_shells)} unique shells found, but n={n} requested."
        )
    else:
        min_shells = unique_shells[:n]

    rounded_all = np.round(bvals / shell_tolerance) * shell_tolerance
    shell_mask = np.isin(rounded_all, min_shells)
    return min_shells, shell_mask


def init_worker(base_seed=None):
    """Initialize a ProcessPoolExecutor worker with a unique RNG state.

    With ``base_seed=None``, seed = PID + high-resolution time so every
    worker has a different stream. With ``initargs=(base_seed,)`` for
    reproducibility, seed = base_seed + PID so workers differ but the run
    is reproducible for a fixed worker count.

    Parameters
    ----------
    base_seed : int, optional
        Base seed for reproducible worker initialization.
    """
    import random
    import time

    if base_seed is not None:
        seed = (int(base_seed) + os.getpid()) % (2**32)
    else:
        # Unique per process and run: PID + high-resolution time
        seed = (os.getpid() * (2**16) + int(time.perf_counter_ns())) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _create_memmap(output_dir, name, dtype, shape):
    """Create a memory-mapped file."""
    path = os.path.join(output_dir, f"{name}.mmap")
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape), path


def _generate_batch_worker(
    start_idx,
    batch_size,
    sphere,
    evecs,
    bingham,
    odi,
    bval,
    bvec,
    wm_thresh,
    tort,
    memmap_info,
    diffusivity_cfg,
):
    """Worker function for parallel batch generation.

    Opens memmaps by path in each process and writes directly.
    """
    from dipy.sims._force_core import create_mixed_signal, set_diffusivity_ranges
    from dipy.sims._multi_tensor_omp import multi_tensor

    # Unpack memmap info
    (
        signals_path,
        signals_shape,
        signals_dtype,
        labels_path,
        labels_shape,
        labels_dtype,
        num_fibers_path,
        num_fibers_shape,
        num_fibers_dtype,
        dispersion_path,
        dispersion_shape,
        dispersion_dtype,
        wm_fraction_path,
        wm_fraction_shape,
        wm_fraction_dtype,
        gm_fraction_path,
        gm_fraction_shape,
        gm_fraction_dtype,
        csf_fraction_path,
        csf_fraction_shape,
        csf_fraction_dtype,
        nd_path,
        nd_shape,
        nd_dtype,
        odfs_path,
        odfs_shape,
        odfs_dtype,
        ufa_wm_path,
        ufa_wm_shape,
        ufa_wm_dtype,
        ufa_voxel_path,
        ufa_voxel_shape,
        ufa_voxel_dtype,
        fraction_array_path,
        fraction_array_shape,
        fraction_array_dtype,
        wm_disp_path,
        wm_disp_shape,
        wm_disp_dtype,
        wm_d_par_path,
        wm_d_par_shape,
        wm_d_par_dtype,
        wm_d_perp_path,
        wm_d_perp_shape,
        wm_d_perp_dtype,
        gm_d_par_path,
        gm_d_par_shape,
        gm_d_par_dtype,
        csf_d_par_path,
        csf_d_par_shape,
        csf_d_par_dtype,
        f_ins_path,
        f_ins_shape,
        f_ins_dtype,
    ) = memmap_info

    # Apply diffusivity ranges in worker
    set_diffusivity_ranges(**diffusivity_cfg)

    # Open memmaps in read-write mode
    signals_mm = np.memmap(
        signals_path, mode="r+", dtype=signals_dtype, shape=signals_shape
    )
    labels_mm = np.memmap(
        labels_path, mode="r+", dtype=labels_dtype, shape=labels_shape
    )
    num_fibers_mm = np.memmap(
        num_fibers_path, mode="r+", dtype=num_fibers_dtype, shape=num_fibers_shape
    )
    dispersion_mm = np.memmap(
        dispersion_path, mode="r+", dtype=dispersion_dtype, shape=dispersion_shape
    )
    wm_fraction_mm = np.memmap(
        wm_fraction_path, mode="r+", dtype=wm_fraction_dtype, shape=wm_fraction_shape
    )
    gm_fraction_mm = np.memmap(
        gm_fraction_path, mode="r+", dtype=gm_fraction_dtype, shape=gm_fraction_shape
    )
    csf_fraction_mm = np.memmap(
        csf_fraction_path, mode="r+", dtype=csf_fraction_dtype, shape=csf_fraction_shape
    )
    nd_mm = np.memmap(nd_path, mode="r+", dtype=nd_dtype, shape=nd_shape)
    odfs_mm = np.memmap(odfs_path, mode="r+", dtype=odfs_dtype, shape=odfs_shape)
    ufa_wm_mm = np.memmap(
        ufa_wm_path, mode="r+", dtype=ufa_wm_dtype, shape=ufa_wm_shape
    )
    ufa_voxel_mm = np.memmap(
        ufa_voxel_path, mode="r+", dtype=ufa_voxel_dtype, shape=ufa_voxel_shape
    )
    fraction_array_mm = np.memmap(
        fraction_array_path,
        mode="r+",
        dtype=fraction_array_dtype,
        shape=fraction_array_shape,
    )
    wm_disp_mm = np.memmap(
        wm_disp_path, mode="r+", dtype=wm_disp_dtype, shape=wm_disp_shape
    )
    wm_d_par_mm = np.memmap(
        wm_d_par_path, mode="r+", dtype=wm_d_par_dtype, shape=wm_d_par_shape
    )
    wm_d_perp_mm = np.memmap(
        wm_d_perp_path, mode="r+", dtype=wm_d_perp_dtype, shape=wm_d_perp_shape
    )
    gm_d_par_mm = np.memmap(
        gm_d_par_path, mode="r+", dtype=gm_d_par_dtype, shape=gm_d_par_shape
    )
    csf_d_par_mm = np.memmap(
        csf_d_par_path, mode="r+", dtype=csf_d_par_dtype, shape=csf_d_par_shape
    )
    f_ins_mm = np.memmap(f_ins_path, mode="r+", dtype=f_ins_dtype, shape=f_ins_shape)

    for i in range(batch_size):
        idx = start_idx + i
        res = create_mixed_signal(
            sphere, evecs, bingham, odi, bval, bvec, multi_tensor, wm_thresh, tort
        )
        (
            signals_mm[idx],
            labels_mm[idx],
            num_fibers_mm[idx],
            dispersion_mm[idx],
            wm_fraction_mm[idx],
            gm_fraction_mm[idx],
            csf_fraction_mm[idx],
            nd_mm[idx],
            odfs_mm[idx],
            ufa_wm_mm[idx],
            ufa_voxel_mm[idx],
            fraction_array_mm[idx],
            wm_disp_mm[idx],
            wm_d_par_mm[idx],
            wm_d_perp_mm[idx],
            gm_d_par_mm[idx],
            csf_d_par_mm[idx],
            f_ins_mm[idx],
        ) = res

    # Flush memmaps
    for mm in [
        signals_mm,
        labels_mm,
        num_fibers_mm,
        dispersion_mm,
        wm_fraction_mm,
        gm_fraction_mm,
        csf_fraction_mm,
        nd_mm,
        odfs_mm,
        ufa_wm_mm,
        ufa_voxel_mm,
        fraction_array_mm,
        wm_disp_mm,
        wm_d_par_mm,
        wm_d_perp_mm,
        gm_d_par_mm,
        csf_d_par_mm,
        f_ins_mm,
    ]:
        mm.flush()

    return batch_size


def _main_is_guarded():
    """Return True if ``__main__`` has a top-level ``if __name__ == '__main__':`` guard.

    Parses the source of the ``__main__`` module with :mod:`ast` and inspects
    only the direct children of the module node (top-level statements).  Returns
    ``True`` also when the main module was invoked with ``python -m pkg.mod``
    (``__spec__`` is not None), or from an interactive session (no ``__file__``),
    because those cases do not cause spawn re-execution problems.

    Returns
    -------
    guarded : bool
        ``True`` if spawning workers is safe without causing recursive
        re-execution of the caller's script.
    """
    main = sys.modules.get("__main__")
    if main is None:
        return True
    if getattr(main, "__spec__", None) is not None:
        # Invoked as `python -m pkg.mod` — spawn imports the module, no re-run
        return True
    main_file = getattr(main, "__file__", None)
    if main_file is None:
        # Interactive session or frozen executable
        return True
    try:
        with open(main_file) as fh:
            source = fh.read()
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return False
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            return True
    return False


def generate_force_simulations(
    gtab,
    *,
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
    verbose=True,
):
    """Generate FORCE simulations.

    Creates a library of simulated diffusion MRI signals with
    corresponding microstructural parameters for matching-based
    reconstruction.

    Parameters
    ----------
    gtab : GradientTable
        Gradient table with b-values and b-vectors.
    num_simulations : int, optional
        Number of simulated voxels.
    output_dir : str, optional
        Directory for output files. If None, uses a temporary directory.
    num_cpus : int or None, optional
        Number of CPU cores for parallel processing.
        ``1`` runs in-process (no subprocess, safe everywhere).
        ``None`` or ``-1`` uses all available cores.
        Values ``< -1`` leave that many cores idle
        (e.g. ``-2`` = all minus one). ``0`` raises ``ValueError``.
    batch_size : int, optional
        Batch size for processing.
    wm_threshold : float, optional
        Minimum WM fraction to include fiber labels.
    tortuosity : bool, optional
        Use tortuosity constraint for perpendicular diffusivity.
    odi_range : tuple, optional
        (min, max) orientation dispersion index range.
    num_odi_values : int, optional
        Number of ODI values to sample.
    diffusivity_config : dict, optional
        Custom diffusivity ranges.
    dtype : dtype, optional
        Data type for outputs.
    compute_dti : bool, optional
        Compute DTI metrics (FA, MD, RD).
    compute_dki : bool, optional
        Compute DKI metrics (AK, RK, MK, KFA).
    verbose : bool, optional
        Enable progress output.

    Returns
    -------
    simulations : dict
        Simulations containing signals and parameters.
    """
    from tqdm import tqdm

    from dipy.reconst import dti
    from dipy.sims._force_core import set_diffusivity_ranges

    num_cpus = determine_num_processes(num_cpus)

    # Setup output directory
    _tmpdir_ctx = None
    if output_dir is None:
        _tmpdir_ctx = tempfile.TemporaryDirectory(prefix="force_sim_")
        output_dir = _tmpdir_ctx.name
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

    # Pre-compute eigenvectors and dispersion lookup table
    evecs = np.array(
        [all_tensor_evecs(tuple(point)) for point in target_sphere],
        dtype=np.float64,
    )
    odi_list = np.linspace(odi_range[0], odi_range[1], num_odi_values).astype(
        np.float64
    )
    dispersion_sf = dispersion_lut(target_sphere, odi_list)

    label_dtype = np.uint8

    # Create memmaps for large arrays
    memmap_paths = {}

    def create_mm(name, dt, shape):
        mm, path = _create_memmap(output_dir, name, dt, shape)
        memmap_paths[name] = path
        return mm

    signals_mm = create_mm("signals", dtype, (num_simulations, n_bvals))
    labels_mm = create_mm("labels", label_dtype, (num_simulations, n_dirs))
    num_fibers_mm = create_mm("num_fibers", dtype, (num_simulations,))
    dispersion_mm = create_mm("dispersion", dtype, (num_simulations,))
    wm_fraction_mm = create_mm("wm_fraction", dtype, (num_simulations,))
    gm_fraction_mm = create_mm("gm_fraction", dtype, (num_simulations,))
    csf_fraction_mm = create_mm("csf_fraction", dtype, (num_simulations,))
    nd_mm = create_mm("nd", dtype, (num_simulations,))
    odfs_mm = create_mm("odfs", np.float16, (num_simulations, n_dirs))
    ufa_wm_mm = create_mm("ufa_wm", dtype, (num_simulations,))
    ufa_voxel_mm = create_mm("ufa_voxel", dtype, (num_simulations,))
    fraction_array_mm = create_mm("fraction_array", dtype, (num_simulations, 3))
    wm_disp_mm = create_mm("wm_disp", dtype, (num_simulations,))
    wm_d_par_mm = create_mm("wm_d_par", dtype, (num_simulations,))
    wm_d_perp_mm = create_mm("wm_d_perp", dtype, (num_simulations,))
    gm_d_par_mm = create_mm("gm_d_par", dtype, (num_simulations,))
    csf_d_par_mm = create_mm("csf_d_par", dtype, (num_simulations,))
    f_ins_mm = create_mm("f_ins", dtype, (num_simulations, 3))

    memmaps = [
        signals_mm,
        labels_mm,
        num_fibers_mm,
        dispersion_mm,
        wm_fraction_mm,
        gm_fraction_mm,
        csf_fraction_mm,
        nd_mm,
        odfs_mm,
        ufa_wm_mm,
        ufa_voxel_mm,
        fraction_array_mm,
        wm_disp_mm,
        wm_d_par_mm,
        wm_d_perp_mm,
        gm_d_par_mm,
        csf_d_par_mm,
        f_ins_mm,
    ]

    # Build batch specs
    num_batches_full = num_simulations // batch_size
    remainder = num_simulations % batch_size
    total_batches = num_batches_full + (1 if remainder > 0 else 0)

    batch_specs = []
    current_start = 0
    for batch_idx in range(total_batches):
        bs = batch_size if batch_idx < num_batches_full else remainder
        batch_specs.append((current_start, bs))
        current_start += bs

    # Pack memmap info for workers
    memmap_info = (
        memmap_paths["signals"],
        (num_simulations, n_bvals),
        dtype,
        memmap_paths["labels"],
        (num_simulations, n_dirs),
        label_dtype,
        memmap_paths["num_fibers"],
        (num_simulations,),
        dtype,
        memmap_paths["dispersion"],
        (num_simulations,),
        dtype,
        memmap_paths["wm_fraction"],
        (num_simulations,),
        dtype,
        memmap_paths["gm_fraction"],
        (num_simulations,),
        dtype,
        memmap_paths["csf_fraction"],
        (num_simulations,),
        dtype,
        memmap_paths["nd"],
        (num_simulations,),
        dtype,
        memmap_paths["odfs"],
        (num_simulations, n_dirs),
        np.float16,
        memmap_paths["ufa_wm"],
        (num_simulations,),
        dtype,
        memmap_paths["ufa_voxel"],
        (num_simulations,),
        dtype,
        memmap_paths["fraction_array"],
        (num_simulations, 3),
        dtype,
        memmap_paths["wm_disp"],
        (num_simulations,),
        dtype,
        memmap_paths["wm_d_par"],
        (num_simulations,),
        dtype,
        memmap_paths["wm_d_perp"],
        (num_simulations,),
        dtype,
        memmap_paths["gm_d_par"],
        (num_simulations,),
        dtype,
        memmap_paths["csf_d_par"],
        (num_simulations,),
        dtype,
        memmap_paths["f_ins"],
        (num_simulations, 3),
        dtype,
    )

    # Run simulations with progress bar
    pbar = tqdm(total=num_simulations, desc="Simulating", disable=not verbose)

    if num_cpus == 1:
        # Serial in-process path — no subprocess spawned, safe on all platforms
        init_worker()
        for start_idx, bs in batch_specs:
            batch_done = _generate_batch_worker(
                start_idx,
                bs,
                target_sphere,
                evecs,
                dispersion_sf,
                odi_list,
                bvals,
                bvecs,
                wm_threshold,
                tortuosity,
                memmap_info,
                diffusivity_config,
            )
            pbar.update(batch_done)
    elif sys.platform == "win32":
        if _main_is_guarded():
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=num_cpus, initializer=init_worker, mp_context=ctx
            ) as executor:
                futures = {
                    executor.submit(
                        _generate_batch_worker,
                        start_idx,
                        bs,
                        target_sphere,
                        evecs,
                        dispersion_sf,
                        odi_list,
                        bvals,
                        bvecs,
                        wm_threshold,
                        tortuosity,
                        memmap_info,
                        diffusivity_config,
                    ): (start_idx, bs)
                    for start_idx, bs in batch_specs
                }
                for future in as_completed(futures):
                    batch_done = future.result()
                    pbar.update(batch_done)
        else:
            logger.warning(
                "Parallel simulation on Windows requires your script to be "
                "protected with `if __name__ == '__main__':`. "
                "Falling back to serial execution."
            )
            init_worker()
            for start_idx, bs in batch_specs:
                batch_done = _generate_batch_worker(
                    start_idx,
                    bs,
                    target_sphere,
                    evecs,
                    dispersion_sf,
                    odi_list,
                    bvals,
                    bvecs,
                    wm_threshold,
                    tortuosity,
                    memmap_info,
                    diffusivity_config,
                )
                pbar.update(batch_done)
    else:
        # fork: workers are copies of the parent — __main__ is NOT re-run,
        # so no `if __name__ == '__main__':` guard is needed in the caller's
        # script. Safe for CLI scientific-computing scripts on macOS/Linux
        # (no GUI/ObjC).
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=num_cpus, initializer=init_worker, mp_context=ctx
        ) as executor:
            futures = {
                executor.submit(
                    _generate_batch_worker,
                    start_idx,
                    bs,
                    target_sphere,
                    evecs,
                    dispersion_sf,
                    odi_list,
                    bvals,
                    bvecs,
                    wm_threshold,
                    tortuosity,
                    memmap_info,
                    diffusivity_config,
                ): (start_idx, bs)
                for start_idx, bs in batch_specs
            }

            for future in as_completed(futures):
                batch_done = future.result()
                pbar.update(batch_done)

    pbar.close()

    # Flush memmaps
    for mm in memmaps:
        mm.flush()

    # Compute DTI metrics
    fa_dti = np.zeros(num_simulations, dtype=dtype)
    md_dti = np.zeros(num_simulations, dtype=dtype)
    rd_dti = np.zeros(num_simulations, dtype=dtype)

    # Compute DKI metrics
    ak_arr = np.zeros(num_simulations, dtype=dtype)
    rk_arr = np.zeros(num_simulations, dtype=dtype)
    mk_arr = np.zeros(num_simulations, dtype=dtype)
    kfa_arr = np.zeros(num_simulations, dtype=dtype)

    if compute_dti:
        from dipy.core.gradients import gradient_table as gt_func

        min_b, shell_mask = smallest_shell_bval(bvals)
        b0_mask = bvals <= gtab.b0_threshold
        use_mask = shell_mask | b0_mask

        gtab_small = gt_func(bvals[use_mask], bvecs=bvecs[use_mask])
        dti_model = dti.TensorModel(gtab_small)

        dti_batch_size = 2000
        pbar = tqdm(total=num_simulations, desc="DTI fitting", disable=not verbose)

        for start in range(0, num_simulations, dti_batch_size):
            end = min(start + dti_batch_size, num_simulations)
            data_batch = signals_mm[start:end][:, use_mask]
            dti_fit = dti_model.fit(data_batch)
            fa_dti[start:end] = dti_fit.fa.astype(dtype)
            md_dti[start:end] = dti_fit.md.astype(dtype)
            rd_dti[start:end] = dti_fit.rd.astype(dtype)
            pbar.update(end - start)

        pbar.close()

    if compute_dki:
        from dipy.core.gradients import gradient_table as gt_func
        from dipy.reconst.dki import DiffusionKurtosisModel

        min_bs, shell_mask = smallest_shell_bval(bvals, n=2)
        b0_mask = bvals <= gtab.b0_threshold
        use_mask = shell_mask | b0_mask

        gtab_dki = gt_func(bvals[use_mask], bvecs=bvecs[use_mask])
        dki_model = DiffusionKurtosisModel(gtab_dki)

        dki_batch_size = 2000
        pbar = tqdm(
            total=num_simulations, desc="Kurtosis Estimation", disable=not verbose
        )

        for start in range(0, num_simulations, dki_batch_size):
            end = min(start + dki_batch_size, num_simulations)
            data_batch = signals_mm[start:end][:, use_mask]
            dki_fit = dki_model.multi_fit(data_batch)[0]
            ak_arr[start:end] = dki_fit.ak().astype(dtype)
            rk_arr[start:end] = dki_fit.rk().astype(dtype)
            mk_arr[start:end] = dki_fit.mk().astype(dtype)
            kfa_arr[start:end] = dki_fit.kfa.astype(dtype)
            pbar.update(end - start)

        pbar.close()

    # Build output simulations dict
    simulations = {
        "signals": np.array(signals_mm),
        "labels": np.array(labels_mm),
        "num_fibers": np.array(num_fibers_mm),
        "dispersion": np.array(dispersion_mm),
        "wm_fraction": np.array(wm_fraction_mm),
        "gm_fraction": np.array(gm_fraction_mm),
        "csf_fraction": np.array(csf_fraction_mm),
        "nd": np.array(nd_mm),
        "odfs": np.array(odfs_mm),
        "fa": fa_dti,
        "md": md_dti,
        "rd": rd_dti,
        "ufa_wm": np.array(ufa_wm_mm),
        "ufa_voxel": np.array(ufa_voxel_mm),
        "fraction_array": np.array(fraction_array_mm),
    }

    if compute_dki:
        simulations.update(
            {
                "ak": ak_arr,
                "rk": rk_arr,
                "mk": mk_arr,
                "kfa": kfa_arr,
            }
        )

    # Cleanup memmaps
    for mm in memmaps:
        if hasattr(mm, "base") and hasattr(mm.base, "close"):
            mm.base.close()
        del mm
    gc.collect()

    for path in memmap_paths.values():
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    gc.collect()

    if _tmpdir_ctx is not None:
        _tmpdir_ctx.cleanup()

    return simulations


def save_force_simulations(simulations, output_path):
    """Save FORCE simulations to compressed NPZ file.

    Parameters
    ----------
    simulations : dict
        FORCE simulations.
    output_path : str
        Path for output file (.npz extension).
    """
    np.savez_compressed(output_path, **simulations)


def load_force_simulations(input_path):
    """Load FORCE simulations from NPZ file.

    Parameters
    ----------
    input_path : str
        Path to simulations file.

    Returns
    -------
    simulations : dict
        FORCE simulations.
    """
    data = np.load(input_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def validate_diffusivity_config(config):
    """Validate diffusivity configuration dictionary.

    Parameters
    ----------
    config : dict
        Diffusivity configuration with keys:
        wm_d_par_range, wm_d_perp_range, gm_d_iso_range, csf_d.

    Returns
    -------
    valid : bool
        True if configuration is valid.

    Raises
    ------
    ValueError
        If configuration is invalid.
    """
    required_keys = ["wm_d_par_range", "wm_d_perp_range", "gm_d_iso_range", "csf_d"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")

    for key in ["wm_d_par_range", "wm_d_perp_range", "gm_d_iso_range"]:
        val = config[key]
        if hasattr(val, "__len__"):
            if len(val) != 2:
                raise ValueError(f"{key} must be scalar or (min, max) tuple")
            if val[0] > val[1]:
                raise ValueError(f"{key} min must be <= max")

    csf_d = config["csf_d"]
    if not isinstance(csf_d, (int, float)) or csf_d <= 0:
        raise ValueError("csf_d must be a positive number")

    return True


def get_default_diffusivity_config():
    """Get default diffusivity configuration.

    Returns
    -------
    config : dict
        Default diffusivity ranges for FORCE simulation.
    """
    return {
        "wm_d_par_range": (2.0e-3, 3.0e-3),
        "wm_d_perp_range": (0.3e-3, 1.5e-3),
        "gm_d_iso_range": (0.7e-3, 1.2e-3),
        "csf_d": 3.0e-3,
    }
