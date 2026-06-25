import numba
import numpy as np

from dipy.tracking.generic_jit_tracker import StreamlineChunk, streamline_generator
from dipy.tracking.numba.generate_streamlines import (
    gen_streamlines_prob_generator,
)
from dipy.tracking.numba.num_streamlines import (
    get_num_streamlines_prob_generator,
)


def numba_sl_generator(jit_tracker_data, seeds, seed_directions=None, nbr_threads=0):
    if nbr_threads != 0:
        old_numba_n_threads = numba.get_num_threads()
        numba.set_num_threads(nbr_threads)

    count_kernel = get_num_streamlines_prob_generator(
        jit_tracker_data.dimx,
        jit_tracker_data.dimy,
        jit_tracker_data.dimz,
        jit_tracker_data.dimt,
        float(jit_tracker_data.relative_peak_thresh),
        float(jit_tracker_data.min_separation_angle),
        jit_tracker_data.nedges,
        jit_tracker_data.sphere_symm,
        float(jit_tracker_data.pmf_threshold),
        jit_tracker_data.real_dtype,
    )

    gen_kernel = gen_streamlines_prob_generator(
        jit_tracker_data.dimx,
        jit_tracker_data.dimy,
        jit_tracker_data.dimz,
        jit_tracker_data.dimt,
        jit_tracker_data.sphere_symm,
        float(jit_tracker_data.step_size),
        float(jit_tracker_data.max_angle),
        float(jit_tracker_data.stop_threshold),
        jit_tracker_data.max_sline_len,
        float(jit_tracker_data.pmf_threshold),
        jit_tracker_data.real_dtype,
    )

    chunk_offset = 0

    def propagate(seeds):
        nonlocal chunk_offset
        seeds = np.ascontiguousarray(seeds, dtype=jit_tracker_data.real_dtype)

        nseed = len(seeds)

        peak_dirs = np.zeros(
            (nseed, jit_tracker_data.dimt, 3), dtype=jit_tracker_data.real_dtype
        )
        sline_offsets = np.zeros(nseed + 1, dtype=np.int32)

        if seed_directions is not None:
            start = chunk_offset
            chunk_dirs = np.ascontiguousarray(
                seed_directions[start : start + nseed],
                dtype=jit_tracker_data.real_dtype,
            )
            peak_dirs[:, 0, :] = chunk_dirs
            sline_offsets[:nseed] = 1
            chunk_offset += nseed
        else:
            count_kernel(
                seeds,
                jit_tracker_data.dataf,
                jit_tracker_data.sphere_vertices,
                jit_tracker_data.sphere_edges,
                peak_dirs,
                sline_offsets,
            )

        counts = sline_offsets[:nseed].copy()
        sline_offsets[0] = 0
        np.cumsum(counts, out=sline_offsets[1:])

        nSlines = int(sline_offsets[-1])

        slineSeed = np.full(nSlines, -1, dtype=np.int32)
        sline_len = np.zeros(nSlines, dtype=np.int32)
        sline = np.zeros(
            (nSlines * jit_tracker_data.max_sline_len * 2, 3),
            dtype=jit_tracker_data.real_dtype,
        )

        gen_kernel(
            seeds,
            jit_tracker_data.dataf,
            jit_tracker_data.metric_map,
            jit_tracker_data.sphere_vertices,
            sline_offsets,
            peak_dirs,
            slineSeed,
            sline_len,
            sline,
        )

        return StreamlineChunk(
            n_slines=nSlines,
            slines=sline,
            sline_lens=sline_len,
            step=jit_tracker_data.max_sline_len * 2,
            min_steps=jit_tracker_data.min_steps,
            max_steps=jit_tracker_data.max_steps,
            real_dtype=jit_tracker_data.real_dtype,
        )

    def close():
        if nbr_threads != 0:
            numba.set_num_threads(old_numba_n_threads)

    return streamline_generator(
        propagate=propagate,
        chunk_size=jit_tracker_data.chunk_size,
        n_procs=jit_tracker_data.n_procs,
        seeds=seeds,
        close=close,
    )
