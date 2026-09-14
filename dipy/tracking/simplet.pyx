# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False
import numpy as np
from cython.parallel cimport prange
cimport openmp
from libc.math cimport floor, cos, INFINITY
from libc.stdint cimport uint64_t, int32_t


cdef int trilinear_interp(const real_t[:, :, :, ::1] pmf, const real_t *pt,
                          real_t *out, const Params *P) noexcept nogil:
    cdef real_t px = pt[0], py = pt[1], pz = pt[2]
    if px < -0.5 or px + 0.5 >= P.dimx:
        return -1
    if py < -0.5 or py + 0.5 >= P.dimy:
        return -1
    if pz < -0.5 or pz + 0.5 >= P.dimz:
        return -1

    cdef real_t flx = floor(px), fly = floor(py), flz = floor(pz)
    cdef real_t wx1 = px - flx, wx0 = 1.0 - wx1
    cdef real_t wy1 = py - fly, wy0 = 1.0 - wy1
    cdef real_t wz1 = pz - flz, wz0 = 1.0 - wz1

    cdef int ix0 = max(0, <int>flx), ix1 = min(P.dimx - 1, ix0 + 1)
    cdef int iy0 = max(0, <int>fly), iy1 = min(P.dimy - 1, iy0 + 1)
    cdef int iz0 = max(0, <int>flz), iz1 = min(P.dimz - 1, iz0 + 1)

    cdef real_t w000 = wx0*wy0*wz0, w001 = wx0*wy0*wz1
    cdef real_t w010 = wx0*wy1*wz0, w011 = wx0*wy1*wz1
    cdef real_t w100 = wx1*wy0*wz0, w101 = wx1*wy0*wz1
    cdef real_t w110 = wx1*wy1*wz0, w111 = wx1*wy1*wz1

    cdef const real_t *c000 = &pmf[ix0, iy0, iz0, 0]
    cdef const real_t *c001 = &pmf[ix0, iy0, iz1, 0]
    cdef const real_t *c010 = &pmf[ix0, iy1, iz0, 0]
    cdef const real_t *c011 = &pmf[ix0, iy1, iz1, 0]
    cdef const real_t *c100 = &pmf[ix1, iy0, iz0, 0]
    cdef const real_t *c101 = &pmf[ix1, iy0, iz1, 0]
    cdef const real_t *c110 = &pmf[ix1, iy1, iz0, 0]
    cdef const real_t *c111 = &pmf[ix1, iy1, iz1, 0]

    cdef int t
    for t in range(P.dimt):
        out[t] = (w000*c000[t] + w001*c001[t] + w010*c010[t] + w011*c011[t] +
                  w100*c100[t] + w101*c101[t] + w110*c110[t] + w111*c111[t])
    return 0


cdef int check_point(const real_t[:, :, ::1] metric, const real_t *pt,
                     const Params *P) noexcept nogil:
    cdef real_t px = pt[0], py = pt[1], pz = pt[2]
    if px < -0.5 or px + 0.5 >= P.dimx:
        return 0
    if py < -0.5 or py + 0.5 >= P.dimy:
        return 0
    if pz < -0.5 or pz + 0.5 >= P.dimz:
        return 0

    cdef real_t flx = floor(px), fly = floor(py), flz = floor(pz)
    cdef real_t wx1 = px - flx, wx0 = 1.0 - wx1
    cdef real_t wy1 = py - fly, wy0 = 1.0 - wy1
    cdef real_t wz1 = pz - flz, wz0 = 1.0 - wz1
    cdef int ix0 = max(0, <int>flx), ix1 = min(P.dimx - 1, ix0 + 1)
    cdef int iy0 = max(0, <int>fly), iy1 = min(P.dimy - 1, iy0 + 1)
    cdef int iz0 = max(0, <int>flz), iz1 = min(P.dimz - 1, iz0 + 1)

    cdef real_t val = (
        wx0*wy0*wz0*metric[ix0, iy0, iz0] + wx0*wy0*wz1*metric[ix0, iy0, iz1] +
        wx0*wy1*wz0*metric[ix0, iy1, iz0] + wx0*wy1*wz1*metric[ix0, iy1, iz1] +
        wx1*wy0*wz0*metric[ix1, iy0, iz0] + wx1*wy0*wz1*metric[ix1, iy0, iz1] +
        wx1*wy1*wz0*metric[ix1, iy1, iz0] + wx1*wy1*wz1*metric[ix1, iy1, iz1])
    return 2 if val > P.tc_threshold else 3


cdef void apply_pmf_threshold(real_t *pmf, int n, double frac) noexcept nogil:
    cdef int i
    cdef real_t mx = -INFINITY
    for i in range(n):
        if pmf[i] > mx:
            mx = pmf[i]
    cdef real_t th = <real_t>(frac * mx)
    for i in range(n):
        if pmf[i] < th:
            pmf[i] = 0.0


cdef int peak_directions(const real_t *odf, const real_t[:, ::1] verts,
                         const int32_t[:, ::1] edges, real_t[:, ::1] dirs_out,
                         int32_t *shInd, const Params *P) noexcept nogil:
    cdef int i, j, e, u, v, n, k, idx_i
    cdef real_t odf_min = INFINITY, masked_max = -INFINITY, val, key_i, ax, ay, az, dot
    cdef bint too_close

    for i in range(P.dimt):
        shInd[i] = 0
        if odf[i] < odf_min:
            odf_min = odf[i]
    if odf_min < 0.0:
        odf_min = 0.0

    for e in range(P.nedges):
        u = edges[e, 0]
        v = edges[e, 1]
        if odf[u] < odf[v]:
            shInd[u] = -1
            if shInd[v] != -1:
                shInd[v] |= 1
        elif odf[v] < odf[u]:
            shInd[v] = -1
            if shInd[u] != -1:
                shInd[u] |= 1

    for i in range(P.dimt):
        if shInd[i] > 0:
            val = odf[i] - odf_min
            if val > masked_max:
                masked_max = val
    cdef real_t comp_thres = <real_t>(P.relative_peak_thresh * masked_max)

    n = 0
    for i in range(P.dimt):
        if shInd[i] > 0 and (odf[i] - odf_min) >= comp_thres:
            shInd[n] = i
            n += 1
    if n == 0:
        return 0

    for i in range(1, n):
        idx_i = shInd[i]
        key_i = odf[idx_i]
        j = i - 1
        while j >= 0 and odf[shInd[j]] < key_i:
            shInd[j + 1] = shInd[j]
            j -= 1
        shInd[j + 1] = idx_i

    dirs_out[0, 0] = verts[shInd[0], 0]
    dirs_out[0, 1] = verts[shInd[0], 1]
    dirs_out[0, 2] = verts[shInd[0], 2]
    k = 1
    for i in range(1, n):
        ax = verts[shInd[i], 0]
        ay = verts[shInd[i], 1]
        az = verts[shInd[i], 2]
        too_close = False
        for j in range(k):
            dot = ax*dirs_out[j, 0] + ay*dirs_out[j, 1] + az*dirs_out[j, 2]
            if P.sphere_symm and dot < 0.0:
                dot = -dot
            if dot > P.cos_min_sep:
                too_close = True
                break
        if not too_close:
            dirs_out[k, 0] = ax
            dirs_out[k, 1] = ay
            dirs_out[k, 2] = az
            k += 1
    return k


cdef int get_direction_prob_step(const real_t[:, :, :, ::1] pmf, real_t *direction,
                                 const real_t *pt, const real_t[:, ::1] verts,
                                 real_t *new_dir, real_t *scratch,
                                 uint64_t *rng, const Params *P) noexcept nogil:
    cdef int i, lo, hi, mid
    cdef real_t dot, last, u

    if trilinear_interp(pmf, pt, scratch, P) != 0:
        return 0
    apply_pmf_threshold(scratch, P.dimt, P.pmf_threshold)

    for i in range(P.dimt):
        dot = (direction[0]*verts[i, 0] + direction[1]*verts[i, 1] +
               direction[2]*verts[i, 2])
        if P.sphere_symm and dot < 0.0:
            dot = -dot
        if dot < P.cos_max_angle:
            scratch[i] = 0.0

    for i in range(1, P.dimt):
        scratch[i] += scratch[i - 1]
    last = scratch[P.dimt - 1]
    if last == 0.0:
        return 0

    u = <real_t>(rng_uniform(rng) * last)
    lo = 0
    hi = P.dimt - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if scratch[mid] <= u:
            lo = mid + 1
        else:
            hi = mid

    if P.sphere_symm:
        dot = (direction[0]*verts[lo, 0] + direction[1]*verts[lo, 1] +
               direction[2]*verts[lo, 2])
        if dot > 0.0:
            new_dir[0] = verts[lo, 0]
            new_dir[1] = verts[lo, 1]
            new_dir[2] = verts[lo, 2]
        else:
            new_dir[0] = -verts[lo, 0]
            new_dir[1] = -verts[lo, 1]
            new_dir[2] = -verts[lo, 2]
    else:
        new_dir[0] = verts[lo, 0]
        new_dir[1] = verts[lo, 1]
        new_dir[2] = verts[lo, 2]
    return 1


cdef int tracker(const real_t *seed, const real_t *first_step,
                 const real_t[:, :, :, ::1] pmf, const real_t[:, :, ::1] metric,
                 const real_t[:, ::1] verts, real_t[:, ::1] sline,
                 real_t *scratch, uint64_t *rng, const Params *P) noexcept nogil:
    cdef real_t pt[3]
    cdef real_t direction[3]
    cdef real_t new_dir[3]
    cdef int i = 1, tissue

    pt[0] = seed[0]
    pt[1] = seed[1]
    pt[2] = seed[2]
    direction[0] = first_step[0]
    direction[1] = first_step[1]
    direction[2] = first_step[2]
    sline[0, 0] = pt[0]
    sline[0, 1] = pt[1]
    sline[0, 2] = pt[2]

    while i < P.max_sline_len:
        if get_direction_prob_step(pmf, direction, pt, verts, new_dir,
                                   scratch, rng, P) == 0:
            break
        direction[0] = new_dir[0]
        direction[1] = new_dir[1]
        direction[2] = new_dir[2]
        pt[0] += direction[0] * P.step_size
        pt[1] += direction[1] * P.step_size
        pt[2] += direction[2] * P.step_size
        sline[i, 0] = pt[0]
        sline[i, 1] = pt[1]
        sline[i, 2] = pt[2]
        tissue = check_point(metric, pt, P)
        if tissue != 2:
            break
        i += 1
    return i


# ---------------------------------------------------------------- python API
cdef Params make_params(dict kw):
    cdef Params P
    P.dimx = kw["dimx"]
    P.dimy = kw["dimy"]
    P.dimz = kw["dimz"]
    P.dimt = kw["dimt"]
    P.nedges = kw["nedges"]
    P.sphere_symm = 1 if kw["sphere_symm"] else 0
    P.step_size = kw["step_size"]
    P.cos_max_angle = cos(kw["max_angle"])
    P.cos_min_sep = cos(kw["min_separation_angle"])
    P.relative_peak_thresh = kw["relative_peak_thresh"]
    P.pmf_threshold = kw["pmf_threshold"]
    P.max_sline_len = kw["max_sline_len"]
    P.tc_threshold = kw["tc_threshold"]
    return P


def get_num_streamlines_prob(const real_t[:, ::1] seeds, const real_t[:, :, :, ::1] pmf,
                             const real_t[:, ::1] verts, const int32_t[:, ::1] edges,
                             real_t[:, ::1] shDir0, int32_t[::1] slineOutOff,
                             dict params, int num_threads=0):
    cdef Params P = make_params(params)
    cdef int nseed = seeds.shape[0], dimt = P.dimt
    cdef int slid, d, ndir, base
    if num_threads <= 0:
        num_threads = openmp.omp_get_max_threads()

    # thread-local scratch, sliced by seed index
    if real_t is float:
        np_real = np.float32
    else:
        np_real = np.float64
    cdef real_t[:, ::1] scratch = np.empty((nseed, dimt), dtype=np_real)
    cdef int32_t[:, ::1] shInd = np.empty((nseed, dimt), dtype=np.int32)
    cdef real_t[:, :, ::1] dirs = np.empty((nseed, dimt, 3), dtype=np_real)

    with nogil:
        for slid in prange(nseed, schedule="dynamic", chunksize=64,
                           num_threads=num_threads):
            ndir = 0
            if trilinear_interp(pmf, &seeds[slid, 0], &scratch[slid, 0], &P) == 0:
                apply_pmf_threshold(&scratch[slid, 0], dimt, P.pmf_threshold)
                ndir = peak_directions(&scratch[slid, 0], verts, edges, dirs[slid],
                                       &shInd[slid, 0], &P)
            slineOutOff[slid] = ndir
            base = slid * dimt
            for d in range(ndir):
                shDir0[base + d, 0] = dirs[slid, d, 0]
                shDir0[base + d, 1] = dirs[slid, d, 1]
                shDir0[base + d, 2] = dirs[slid, d, 2]


def gen_streamlines_prob(const real_t[:, ::1] seeds, const real_t[:, :, :, ::1] pmf,
                         const real_t[:, :, ::1] metric, const real_t[:, ::1] verts,
                         const int32_t[::1] slineOutOff, const real_t[:, ::1] shDir0,
                         int32_t[::1] slineSeed, int32_t[::1] slineLen,
                         real_t[:, ::1] sline, dict params,
                         unsigned long long random_seed=0, int num_threads=0):
    cdef Params P = make_params(params)
    cdef int nseed = seeds.shape[0], dimt = P.dimt
    cdef int slid, i, ndir, off, start, stepsB, stepsF, lo, hi
    if num_threads <= 0:
        num_threads = openmp.omp_get_max_threads()
    cdef real_t t0, t1, t2
    cdef uint64_t rng
    if real_t is float:
        np_real = np.float32
    else:
        np_real = np.float64
    cdef real_t[:, ::1] scratch = np.empty((nseed, dimt), dtype=np_real)
    cdef real_t[:, ::1] neg = np.empty((nseed, 3), dtype=np_real)

    with nogil:
        for slid in prange(nseed, schedule="dynamic", chunksize=16,
                           num_threads=num_threads):
            ndir = slineOutOff[slid + 1] - slineOutOff[slid]
            off = slineOutOff[slid]
            for i in range(ndir):
                rng = splitmix64(<uint64_t>random_seed + <uint64_t>(off + i) + 1)
                start = (off + i) * P.max_sline_len * 2
                slineSeed[off + i] = slid
                neg[slid, 0] = -shDir0[slid*dimt + i, 0]
                neg[slid, 1] = -shDir0[slid*dimt + i, 1]
                neg[slid, 2] = -shDir0[slid*dimt + i, 2]

                stepsB = tracker(&seeds[slid, 0], &neg[slid, 0], pmf, metric, verts,
                                 sline[start:start + P.max_sline_len*2],
                                 &scratch[slid, 0], &rng, &P)
                lo = 0
                hi = stepsB - 1
                while lo < hi:
                    t0 = sline[start+lo, 0]
                    t1 = sline[start+lo, 1]
                    t2 = sline[start+lo, 2]
                    sline[start+lo, 0] = sline[start+hi, 0]
                    sline[start+lo, 1] = sline[start+hi, 1]
                    sline[start+lo, 2] = sline[start+hi, 2]
                    sline[start+hi, 0] = t0
                    sline[start+hi, 1] = t1
                    sline[start+hi, 2] = t2
                    lo = lo + 1
                    hi = hi - 1

                stepsF = tracker(&seeds[slid, 0], &shDir0[slid*dimt + i, 0],
                                 pmf, metric, verts,
                                 sline[start + stepsB - 1:start + P.max_sline_len*2],
                                 &scratch[slid, 0], &rng, &P)
                slineLen[off + i] = stepsB - 1 + stepsF
