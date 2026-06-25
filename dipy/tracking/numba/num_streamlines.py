import math

import numpy as np

from dipy.tracking.jit.numba.tracking_helpers import trilinear_interp_generator


def get_num_streamlines_prob_generator(
    DIMX,
    DIMY,
    DIMZ,
    DIMT,
    RELATIVE_PEAK_THRESH,
    MIN_SEPARATION_ANGLE,
    NUM_EDGES,
    SPHERE_SYMM,
    PMF_THRESHOLD_P,
    REAL_DTYPE,
):
    from numba import njit, prange

    trilinear_interp = trilinear_interp_generator(DIMX, DIMY, DIMZ, DIMT)

    @njit
    def peak_directions(odf, sphere_vertices, sphere_edges, dirs_out):
        shInd = np.zeros(DIMT, dtype=np.int32)

        odf_min = np.inf
        for i in range(DIMT):
            if odf[i] < odf_min:
                odf_min = odf[i]
        if odf_min < 0.0:
            odf_min = 0.0

        #  local_maxima: mark via edges
        #  For each edge (u,v): the smaller-valued vertex is marked -1 (not a max),
        #  the larger is marked >=1 (candidate).
        for e in range(NUM_EDGES):
            u = sphere_edges[e, 0]
            v = sphere_edges[e, 1]
            u_val = odf[u]
            v_val = odf[v]
            if u_val < v_val:
                shInd[u] = -1
                if shInd[v] != -1:
                    shInd[v] = shInd[v] | 1
            elif v_val < u_val:
                shInd[v] = -1
                if shInd[u] != -1:
                    shInd[u] = shInd[u] | 1

        masked_max = -np.inf
        for i in range(DIMT):
            if shInd[i] > 0:
                val = odf[i] - odf_min
                if val > masked_max:
                    masked_max = val
        comp_thres = RELATIVE_PEAK_THRESH * masked_max

        #  compact: keep indices where shInd[i]>0
        #  AND (odf[i]-odf_min) >= compThres
        n = 0
        for i in range(DIMT):
            if shInd[i] > 0 and (odf[i] - odf_min) >= comp_thres:
                shInd[n] = i
                n += 1

        if n == 0:
            return 0

        #  sort compacted indices by descending odf value
        for i in range(1, n):
            key_i = odf[shInd[i]]
            idx_i = shInd[i]
            j = i - 1
            while j >= 0 and odf[shInd[j]] < key_i:
                shInd[j + 1] = shInd[j]
                j -= 1
            shInd[j + 1] = idx_i

        #  remove_similar_vertices
        cos_sim = math.cos(MIN_SEPARATION_ANGLE)

        dirs_out[0, 0] = sphere_vertices[shInd[0], 0]
        dirs_out[0, 1] = sphere_vertices[shInd[0], 1]
        dirs_out[0, 2] = sphere_vertices[shInd[0], 2]
        k = 1

        for i in range(1, n):
            ax = sphere_vertices[shInd[i], 0]
            ay = sphere_vertices[shInd[i], 1]
            az = sphere_vertices[shInd[i], 2]

            too_close = False
            for j in range(k):
                dot = ax * dirs_out[j, 0] + ay * dirs_out[j, 1] + az * dirs_out[j, 2]
                if SPHERE_SYMM:
                    if dot < 0.0:
                        dot = -dot
                if dot > cos_sim:
                    too_close = True
                    break

            if not too_close:
                dirs_out[k, 0] = ax
                dirs_out[k, 1] = ay
                dirs_out[k, 2] = az
                k += 1

        return k

    @njit
    def get_direction_prob_start(
        pmf_volume,  # 4-D float array  [nz, ny, nx, DIMT]
        point,  # float[3]  fractional voxel coords
        sphere_vertices,  # float[DIMT, 3]
        sphere_edges,  # int[E, 2]
        dirs_out,  # float[DIMT, 3]  output buffer (pre-allocated)
        pmf_scratch,  # float[DIMT]     scratch buffer (pre-allocated)
    ):
        # Step 1  interpolate PMF
        rv = trilinear_interp(pmf_volume, point, pmf_scratch)
        if rv != 0:
            return 0

        # Step 2  threshold
        max_pmf = -np.inf
        for i in range(DIMT):
            if pmf_scratch[i] > max_pmf:
                max_pmf = pmf_scratch[i]
        abs_thresh = PMF_THRESHOLD_P * max_pmf

        for i in range(DIMT):
            if pmf_scratch[i] < abs_thresh:
                pmf_scratch[i] = 0.0

        # Step 3  peak directions
        ndir = peak_directions(pmf_scratch, sphere_vertices, sphere_edges, dirs_out)

        return ndir

    @njit(parallel=True)
    def get_num_streamlines_prob(
        seeds,  # float[nseed, 3]
        pmf_volume,  # float[nz, ny, nx, DIMT]
        sphere_vertices,  # float[DIMT, 3]
        sphere_edges,  # int[E, 2]
        peak_dirs,  # float[nseed * DIMT, 3]   output: peak dirs per seed
        sline_offsets,  # int[nseed + 1]      output: prefix-sum offsets
    ):
        nseed = seeds.shape[0]

        for slid in prange(nseed):
            pmf_scratch = np.empty(DIMT, dtype=REAL_DTYPE)

            #  get peak directions at this seed
            sline_offsets[slid] = get_direction_prob_start(
                pmf_volume,
                seeds[slid],
                sphere_vertices,
                sphere_edges,
                peak_dirs[slid],
                pmf_scratch,
            )

    return get_num_streamlines_prob
