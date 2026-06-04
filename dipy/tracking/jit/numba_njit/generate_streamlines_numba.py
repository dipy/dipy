import math
import random

import numpy as np

from dipy.tracking.jit.numba_njit.tracking_helpers import trilinear_interp_generator


def genStreamlinesMergeProb_generator(
    DIMX,
    DIMY,
    DIMZ,
    DIMT,
    SPHERE_SYMM,
    STEP_SIZE,
    MAX_ANGLE,
    TC_THRESHOLD,
    MAX_SLINE_LEN,
    PMF_THRESHOLD_P,
):
    from numba import njit, prange

    trilinear_interp = trilinear_interp_generator(DIMX, DIMY, DIMZ, DIMT)

    @njit
    def check_point(point, metric_map):
        px, py, pz = point[0], point[1], point[2]

        if px < -0.5 or px + 0.5 >= DIMX:
            return 0
        if py < -0.5 or py + 0.5 >= DIMY:
            return 0
        if pz < -0.5 or pz + 0.5 >= DIMZ:
            return 0

        flx = math.floor(px)
        fly = math.floor(py)
        flz = math.floor(pz)

        wx1 = px - flx
        wx0 = 1.0 - wx1
        wy1 = py - fly
        wy0 = 1.0 - wy1
        wz1 = pz - flz
        wz0 = 1.0 - wz1

        ix0 = max(0, int(flx))
        ix1 = min(DIMX - 1, ix0 + 1)
        iy0 = max(0, int(fly))
        iy1 = min(DIMY - 1, iy0 + 1)
        iz0 = max(0, int(flz))
        iz1 = min(DIMZ - 1, iz0 + 1)

        val = (
            wx0 * wy0 * wz0 * metric_map[ix0, iy0, iz0]
            + wx0 * wy0 * wz1 * metric_map[ix0, iy0, iz1]
            + wx0 * wy1 * wz0 * metric_map[ix0, iy1, iz0]
            + wx0 * wy1 * wz1 * metric_map[ix0, iy1, iz1]
            + wx1 * wy0 * wz0 * metric_map[ix1, iy0, iz0]
            + wx1 * wy0 * wz1 * metric_map[ix1, iy0, iz1]
            + wx1 * wy1 * wz0 * metric_map[ix1, iy1, iz0]
            + wx1 * wy1 * wz1 * metric_map[ix1, iy1, iz1]
        )

        return 2 if val > TC_THRESHOLD else 3

    @njit
    def get_direction_prob_step(
        pmf_volume,  # float64[DIMX, DIMY, DIMZ, DIMT]
        direction,  # float64[3]  current propagation direction
        point,  # float64[3]                   current position
        sphere_vertices,  # float64[DIMT, 3]
        new_dir,  # float64[3]    output: chosen next direction
        pmf_scratch,  # float64[DIMT]            scratch buffer
    ):
        # interpolate PMF
        rv = trilinear_interp(pmf_volume, point, pmf_scratch)
        if rv != 0:
            return 0

        # absolute PMF threshold
        max_pmf = -np.inf
        for i in range(DIMT):
            if pmf_scratch[i] > max_pmf:
                max_pmf = pmf_scratch[i]
        abs_thresh = PMF_THRESHOLD_P * max_pmf
        for i in range(DIMT):
            if pmf_scratch[i] < abs_thresh:
                pmf_scratch[i] = 0.0

        # angle filtering
        cos_sim = math.cos(MAX_ANGLE)
        for i in range(DIMT):
            dot = (
                direction[0] * sphere_vertices[i, 0]
                + direction[1] * sphere_vertices[i, 1]
                + direction[2] * sphere_vertices[i, 2]
            )
            if SPHERE_SYMM:
                if dot < 0.0:
                    dot = -dot
            if dot < cos_sim:
                pmf_scratch[i] = 0.0

        # cumulative sum
        for i in range(1, DIMT):
            pmf_scratch[i] += pmf_scratch[i - 1]

        last_cdf = pmf_scratch[DIMT - 1]
        if last_cdf == 0.0:
            return 0

        u = random.random() * last_cdf

        # Binary search for first CDF entry >= u
        low = 0
        high = DIMT - 1
        while low < high:
            mid = (low + high) // 2
            if pmf_scratch[mid] <= u:
                low = mid + 1
            else:
                high = mid
        ind_prob = low

        # Flip if needed
        if SPHERE_SYMM:
            dot = (
                direction[0] * sphere_vertices[ind_prob, 0]
                + direction[1] * sphere_vertices[ind_prob, 1]
                + direction[2] * sphere_vertices[ind_prob, 2]
            )
            if dot > 0.0:
                new_dir[0] = sphere_vertices[ind_prob, 0]
                new_dir[1] = sphere_vertices[ind_prob, 1]
                new_dir[2] = sphere_vertices[ind_prob, 2]
            else:
                new_dir[0] = -sphere_vertices[ind_prob, 0]
                new_dir[1] = -sphere_vertices[ind_prob, 1]
                new_dir[2] = -sphere_vertices[ind_prob, 2]
        else:
            new_dir[0] = sphere_vertices[ind_prob, 0]
            new_dir[1] = sphere_vertices[ind_prob, 1]
            new_dir[2] = sphere_vertices[ind_prob, 2]

        return 1

    @njit
    def tracker(
        seed,  # float64[3]  starting position
        first_step,  # float64[3]  initial direction
        pmf_volume,  # float64[DIMX, DIMY, DIMZ, DIMT]
        metric_map,  # float64[DIMX, DIMY, DIMZ]
        sphere_vertices,  # float64[DIMT, 3]
        streamline,  # float64[MAX_SLINE_LEN*2, 3]
        pmf_scratch,  # float64[DIMT]  scratch buffer
    ):
        point = np.empty(3, dtype=np.float64)
        direction = np.empty(3, dtype=np.float64)
        new_dir = np.empty(3, dtype=np.float64)

        point[0] = seed[0]
        point[1] = seed[1]
        point[2] = seed[2]
        direction[0] = first_step[0]
        direction[1] = first_step[1]
        direction[2] = first_step[2]

        streamline[0, 0] = point[0]
        streamline[0, 1] = point[1]
        streamline[0, 2] = point[2]

        tissue_class = 2

        i = 1
        while i < MAX_SLINE_LEN:
            ndir = get_direction_prob_step(
                pmf_volume,
                direction,
                point,
                sphere_vertices,
                new_dir,
                pmf_scratch,
            )

            if ndir == 0:
                break

            direction[0] = new_dir[0]
            direction[1] = new_dir[1]
            direction[2] = new_dir[2]

            point[0] += direction[0] * STEP_SIZE
            point[1] += direction[1] * STEP_SIZE
            point[2] += direction[2] * STEP_SIZE

            streamline[i, 0] = point[0]
            streamline[i, 1] = point[1]
            streamline[i, 2] = point[2]

            tissue_class = check_point(point, metric_map)

            if tissue_class == 0 or tissue_class == 1 or tissue_class == 3:
                break

            i += 1

        return i, tissue_class

    @njit(parallel=True)
    def genStreamlinesMergeProb(
        seeds,  # float64[nseed, 3]
        pmf_volume,  # float64[DIMX, DIMY, DIMZ, DIMT]
        metric_map,  # float64[DIMX, DIMY, DIMZ]
        sphere_vertices,  # float64[DIMT, 3]
        slineOutOff,  # int32[nseed+1]   prefix-sum offsets from getNumStreamlinesProb
        shDir0,  # float64[nseed*DIMT, 3]   peak directions from getNumStreamlinesProb
        slineSeed,  # int32[total_slines]            output: seed index per streamline
        slineLen,  # int32[total_slines]             output: length of each streamline
        sline,  # float64[total_slines * MAX_SLINE_LEN*2, 3] output: streamline points
    ):
        nseed = seeds.shape[0]

        for slid in prange(nseed):
            ndir = slineOutOff[slid + 1] - slineOutOff[slid]
            slineOff = slineOutOff[slid]

            pmf_scratch = np.empty(DIMT, dtype=np.float64)

            seed = seeds[slid]

            for i in range(ndir):
                first_step = shDir0[slid, i]

                sline_start = slineOff * MAX_SLINE_LEN * 2

                slineSeed[slineOff] = slid

                # Backward pass: start from seed, direction = -first_step
                neg_first = np.empty(3, dtype=np.float64)
                neg_first[0] = -first_step[0]
                neg_first[1] = -first_step[1]
                neg_first[2] = -first_step[2]

                curr_sline = sline[sline_start : sline_start + MAX_SLINE_LEN * 2]

                stepsB, _ = tracker(
                    seed,
                    neg_first,
                    pmf_volume,
                    metric_map,
                    sphere_vertices,
                    curr_sline,
                    pmf_scratch,
                )

                # Reverse the backward segment in-place
                lo = 0
                hi = stepsB - 1
                while lo < hi:
                    tmp0 = curr_sline[lo, 0]
                    tmp1 = curr_sline[lo, 1]
                    tmp2 = curr_sline[lo, 2]
                    curr_sline[lo, 0] = curr_sline[hi, 0]
                    curr_sline[lo, 1] = curr_sline[hi, 1]
                    curr_sline[lo, 2] = curr_sline[hi, 2]
                    curr_sline[hi, 0] = tmp0
                    curr_sline[hi, 1] = tmp1
                    curr_sline[hi, 2] = tmp2
                    lo += 1
                    hi -= 1

                # Forward pass: append at the junction (currSline + stepsB-1)
                fwd_sline = curr_sline[stepsB - 1 :]

                stepsF, _ = tracker(
                    seed,
                    first_step,
                    pmf_volume,
                    metric_map,
                    sphere_vertices,
                    fwd_sline,
                    pmf_scratch,
                )

                # Total length: backward points + forward points, junction counted once
                slineLen[slineOff] = stepsB - 1 + stepsF

                slineOff += 1

    return genStreamlinesMergeProb
