/* Metal backend for dipy.tracking.accel
 *
 * Compile-time defines prepended by dipy/tracking/accel/metal.py:
 *   SPHERE_SYMM   1 if sphere vertices are antipodally symmetric
 *   N32DIMT       dimt rounded up to a multiple of 32 (threadgroup memory size)
 *
 * Conventions:
 *   - device buffers use packed_float3 (12 bytes, matches numpy (N, 3) float32)
 *   - one SIMD group (32 threads) tracks one streamline, BLOCK_Y streamlines
 *     per threadgroup
 *   - Metal only supports float32
 *   - the tracking loop, random number generator and output layout match
 *     kernel.cu (see the comments there)
 */

#include <metal_stdlib>
using namespace metal;

#define THR_X_SL 32
#define BLOCK_Y  2

#define REAL_MAX FLT_MAX
#define REAL_MIN (-FLT_MAX)

#if SPHERE_SYMM == 0
  #define APPLY_ABS_IF_SYM(x) (x)
#else
  #define APPLY_ABS_IF_SYM(x) abs(x)
#endif

#define SIMD_BALLOT_MASK(pred) uint(ulong(simd_ballot(pred)))

enum { OUTSIDEIMAGE, INVALIDPOINT, TRACKPOINT, ENDPOINT };

struct ProbTrackingParams {
    float max_angle;
    float tc_threshold;
    float step_size;
    float pmf_threshold;
    uint  rng_seed_lo;
    uint  rng_seed_hi;
    uint  sl_base_lo;     // index in the tractogram of the first streamline
    uint  sl_base_hi;     // of this dispatch
    int   nseed;
    int   dimx;
    int   dimy;
    int   dimz;
    int   dimt;
    int   max_sline_len;
};

// ── packed_float3 helpers ────────────────────────────────────────────

inline float3 load_f3(const device packed_float3* p, uint idx) {
    return float3(p[idx]);
}

inline void store_f3(device packed_float3* p, uint idx, float3 v) {
    p[idx] = packed_float3(v);
}

// ── Philox4x32-10 RNG (same generator as kernel.cu) ─────────────────
// Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"
//            (SC '11).  DOI 10.1145/2063384.2063405

constant uint PHILOX_M4x32_0 = 0xD2511F53u;
constant uint PHILOX_M4x32_1 = 0xCD9E8D57u;
constant uint PHILOX_W32_0   = 0x9E3779B9u;
constant uint PHILOX_W32_1   = 0xBB67AE85u;

struct PhiloxState {
    uint4 counter;
    uint2 key;
    uint4 output;
    uint  idx;
};

inline uint mulhi32(uint a, uint b) {
    return uint((ulong(a) * ulong(b)) >> 32);
}

inline uint4 philox4x32_single_round(uint4 ctr, uint2 key) {
    uint lo0 = ctr.x * PHILOX_M4x32_0;
    uint hi0 = mulhi32(ctr.x, PHILOX_M4x32_0);
    uint lo1 = ctr.z * PHILOX_M4x32_1;
    uint hi1 = mulhi32(ctr.z, PHILOX_M4x32_1);
    return uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

inline uint4 philox4x32_10(uint4 ctr, uint2 key) {
    const uint2 bump = uint2(PHILOX_W32_0, PHILOX_W32_1);
    for (int r = 0; r < 9; r++) {
        ctr = philox4x32_single_round(ctr, key);
        key += bump;
    }
    return philox4x32_single_round(ctr, key);
}

// random stream of the streamline with index sl_index in the tractogram
inline PhiloxState philox_init(uint2 seed, ulong sl_index) {
    PhiloxState s;
    s.key = seed;
    s.counter = uint4(0u, uint(sl_index), uint(sl_index >> 32), 0u);
    s.output = uint4(0u);
    s.idx = 4;
    return s;
}

// uniform float in [0, 1) from the 24 high bits: exact in float32 and float64
inline float philox_uniform(thread PhiloxState& s) {
    if (s.idx == 4) {
        s.output = philox4x32_10(s.counter, s.key);
        s.counter.x += 1;
        s.idx = 0;
    }
    uint bits;
    switch (s.idx) {
        case 0: bits = s.output.x; break;
        case 1: bits = s.output.y; break;
        case 2: bits = s.output.z; break;
        default: bits = s.output.w; break;
    }
    s.idx++;
    return float(bits >> 8) * (1.0f / 16777216.0f);
}

// ── SIMD reductions ──────────────────────────────────────────────────

inline float simd_max_reduce(int n, const threadgroup float* src, float minVal,
                             uint tidx) {
    float m = minVal;
    for (int i = tidx; i < n; i += THR_X_SL) {
        m = max(m, src[i]);
    }
    for (int i = THR_X_SL / 2; i > 0; i /= 2) {
        m = max(m, simd_shuffle_xor(m, ushort(i)));
    }
    return m;
}

inline void prefix_sum_sh(threadgroup float* num_sh, int len, uint tidx) {
    for (int j = 0; j < len; j += THR_X_SL) {
        if ((tidx == 0) && (j != 0)) {
            num_sh[j] += num_sh[j - 1];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float t_pmf = 0.0f;
        if (j + int(tidx) < len) {
            t_pmf = num_sh[j + tidx];
        }
        for (int i = 1; i < THR_X_SL; i *= 2) {
            float tmp = simd_shuffle_up(t_pmf, ushort(i));
            if ((int(tidx) >= i) && (j + int(tidx) < len)) {
                t_pmf += tmp;
            }
        }
        if (j + int(tidx) < len) {
            num_sh[j + tidx] = t_pmf;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── trilinear interpolation ──────────────────────────────────────────

struct TrilinearSetup {
    int   status;
    float wgh[3][2];
    long  coo[3][2];
};

inline TrilinearSetup trilinear_setup(const int dimx, const int dimy, const int dimz,
                                      const float3 point) {
    TrilinearSetup r;
    const float HALF = 0.5f;

    if (point.x < -HALF || point.x + HALF >= float(dimx) ||
        point.y < -HALF || point.y + HALF >= float(dimy) ||
        point.z < -HALF || point.z + HALF >= float(dimz)) {
        r.status = -1;
        return r;
    }

    // as in dipy.core.interpolation: within half a voxel of the border,
    // only the border voxel is used
    const float3 fl = floor(point);
    const int3 fli = int3(fl);

    r.wgh[0][1] = point.x - fl.x;
    r.wgh[0][0] = 1.0f - r.wgh[0][1];
    r.coo[0][0] = max(0, fli.x);
    r.coo[0][1] = min(dimx - 1, fli.x + 1);

    r.wgh[1][1] = point.y - fl.y;
    r.wgh[1][0] = 1.0f - r.wgh[1][1];
    r.coo[1][0] = max(0, fli.y);
    r.coo[1][1] = min(dimy - 1, fli.y + 1);

    r.wgh[2][1] = point.z - fl.z;
    r.wgh[2][0] = 1.0f - r.wgh[2][1];
    r.coo[2][0] = max(0, fli.z);
    r.coo[2][1] = min(dimz - 1, fli.z + 1);

    r.status = 0;
    return r;
}

inline float interpolation_helper(const device float* dataf,
                                  thread const TrilinearSetup& s,
                                  int dimy, int dimz, int dimt, int t) {
    float tmp = 0.0f;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                tmp += s.wgh[0][i] * s.wgh[1][j] * s.wgh[2][k] *
                       dataf[s.coo[0][i] * dimy * dimz * dimt +
                             s.coo[1][j] * dimz * dimt +
                             s.coo[2][k] * dimt +
                             t];
            }
        }
    }
    return tmp;
}

// Interpolate all dimt channels of dataf at point into vox_data.
inline int trilinear_interp(const int dimx, const int dimy, const int dimz,
                            const int dimt,
                            const device float* dataf,
                            const float3 point,
                            threadgroup float* vox_data,
                            uint tidx) {
    const TrilinearSetup s = trilinear_setup(dimx, dimy, dimz, point);
    if (s.status != 0) {
        return -1;
    }
    for (int t = int(tidx); t < dimt; t += THR_X_SL) {
        vox_data[t] = interpolation_helper(dataf, s, dimy, dimz, dimt, t);
    }
    return 0;
}

inline int check_point(const float tc_threshold,
                       const float3 point,
                       const int dimx, const int dimy, const int dimz,
                       const device float* metric_map) {
    const TrilinearSetup s = trilinear_setup(dimx, dimy, dimz, point);
    if (s.status != 0) {
        return OUTSIDEIMAGE;
    }
    const float val = interpolation_helper(metric_map, s, dimy, dimz, 1, 0);
    return (val > tc_threshold) ? TRACKPOINT : ENDPOINT;
}

// ── probabilistic direction getter ───────────────────────────────────

inline int get_direction_prob(thread PhiloxState& st,
                              constant ProbTrackingParams& params,
                              const device float* pmf,
                              float3 dir,
                              const float3 point,
                              const device packed_float3* sphere_vertices,
                              threadgroup float3* out_dir,
                              threadgroup float* pmf_data_sh,
                              uint tidx) {
    const int dimt = params.dimt;

    simdgroup_barrier(mem_flags::mem_threadgroup);
    const int rv = trilinear_interp(params.dimx, params.dimy, params.dimz, dimt,
                                    pmf, point, pmf_data_sh, tidx);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (rv != 0) {
        return 0;
    }

    const float absolpmf_thresh =
        params.pmf_threshold * simd_max_reduce(dimt, pmf_data_sh, REAL_MIN, tidx);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const float cos_similarity = cos(params.max_angle);
    for (int i = int(tidx); i < dimt; i += THR_X_SL) {
        const float3 sv = load_f3(sphere_vertices, uint(i));
        const float dot = dir.x * sv.x + dir.y * sv.y + dir.z * sv.z;
        if (pmf_data_sh[i] < absolpmf_thresh ||
            APPLY_ABS_IF_SYM(dot) < cos_similarity) {
            pmf_data_sh[i] = 0.0f;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    prefix_sum_sh(pmf_data_sh, dimt, tidx);

    const float last_cdf = pmf_data_sh[dimt - 1];
    if (last_cdf == 0.0f) {
        return 0;
    }

    float tmp = 0.0f;
    if (tidx == 0) {
        tmp = philox_uniform(st) * last_cdf;
    }
    const float selected_cdf = simd_broadcast_first(tmp);

    // first index with cdf > selected_cdf (see kernel.cu)
    int low = 0;
    int high = dimt - 1;
    while ((high - low) >= THR_X_SL) {
        const int mid = (low + high) / 2;
        if (pmf_data_sh[mid] <= selected_cdf) {
            low = mid;
        } else {
            high = mid;
        }
    }
    const bool ballot_pred =
        (low + int(tidx) <= high) ? (selected_cdf < pmf_data_sh[low + tidx]) : false;
    const uint msk = SIMD_BALLOT_MASK(ballot_pred);
    const int indProb = (msk != 0) ? (low + int(ctz(msk))) : high;

    if (tidx == 0) {
        const float3 sv = load_f3(sphere_vertices, uint(indProb));
        if ((dir.x * sv.x + dir.y * sv.y + dir.z * sv.z) > 0) {
            *out_dir = sv;
        } else {
            *out_dir = -sv;
        }
    }
    return 1;
}

// ── tracker: follow one direction from a seed ────────────────────────
// Point i is written to streamline[i * stride]; returns the number of points
// kept, counting the seed (see tracker_d in kernel.cu).

inline int tracker_prob(thread PhiloxState& st,
                        constant ProbTrackingParams& params,
                        float3 seed,
                        float3 first_step,
                        const device float* dataf,
                        const device float* metric_map,
                        const device packed_float3* sphere_vertices,
                        device packed_float3* streamline,
                        int stride,
                        thread float3& first_dir,
                        threadgroup float3* sh_new_dir,
                        threadgroup float* pmf_data_sh,
                        uint tidx, uint tidy) {
    float3 point = seed;
    float3 direction = first_step;

    int i;
    for (i = 1; i < params.max_sline_len; i++) {
        const int ndir = get_direction_prob(st, params, dataf, direction, point,
                                            sphere_vertices, sh_new_dir + tidy,
                                            pmf_data_sh, tidx);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        direction = sh_new_dir[tidy];
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (ndir == 0) {
            break;
        }
        if (i == 1) {
            first_dir = direction;
        }

        point += direction * params.step_size;

        if (tidx == 0) {
            streamline[i * stride] = packed_float3(point);
        }
        simdgroup_barrier(mem_flags::mem_device);

        if (check_point(params.tc_threshold, point, params.dimx, params.dimy,
                        params.dimz, metric_map) != TRACKPOINT) {
            break;
        }
    }
    return min(i, params.max_sline_len - 1);
}

// ── kernel: generate streamlines from precomputed seed directions ────
// Same buffers and layout as genStreamlinesProb_k in kernel.cu.

kernel void genStreamlinesProb_k(
    constant ProbTrackingParams& params          [[buffer(0)]],
    const device packed_float3* seeds            [[buffer(1)]],
    const device float* dataf                    [[buffer(2)]],
    const device float* metric_map               [[buffer(3)]],
    const device packed_float3* sphere_vertices  [[buffer(4)]],
    const device int* slineOutOff                [[buffer(5)]],
    const device packed_float3* shDir0           [[buffer(6)]],
    device int* slineStart                       [[buffer(7)]],
    device int* slineLen                         [[buffer(8)]],
    device packed_float3* sline                  [[buffer(9)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint tidx = tid.x;
    const uint tidy = tid.y;
    const uint slid = gid.x * BLOCK_Y + tidy;

    if (int(slid) >= params.nseed) return;

    threadgroup float sh_mem[BLOCK_Y * N32DIMT];
    threadgroup float3 sh_new_dir[BLOCK_Y];
    threadgroup float* pmf_data_sh = sh_mem + tidy * N32DIMT;

    const uint2 rng_seed = uint2(params.rng_seed_lo, params.rng_seed_hi);
    const ulong sl_base = (ulong(params.sl_base_hi) << 32) | ulong(params.sl_base_lo);
    const float3 seed = load_f3(seeds, slid);
    const int first_sl = slineOutOff[slid];
    const int ndir = slineOutOff[slid + 1] - first_sl;

    for (int i = 0; i < ndir; i++) {
        const int sl = first_sl + i;
        PhiloxState st = philox_init(rng_seed, sl_base + ulong(sl));

        const float3 first_step = load_f3(shDir0, uint(sl));
        device packed_float3* center =
            sline + (ulong(sl) * ulong(params.max_sline_len) * 2 + ulong(params.max_sline_len));
        if (tidx == 0) {
            center[0] = packed_float3(seed);
        }
        simdgroup_barrier(mem_flags::mem_device);

        // forward, then backward from the direction of the first forward step
        float3 first_dir = first_step;
        const int stepsF = tracker_prob(st, params, seed, first_step,
                                        dataf, metric_map, sphere_vertices,
                                        center, 1, first_dir,
                                        sh_new_dir, pmf_data_sh, tidx, tidy);
        const float3 back = (stepsF > 1) ? first_dir : first_step;
        float3 unused = back;
        const int stepsB = tracker_prob(st, params, seed, -back,
                                        dataf, metric_map, sphere_vertices,
                                        center, -1, unused,
                                        sh_new_dir, pmf_data_sh, tidx, tidy);

        if (tidx == 0) {
            slineStart[sl] = params.max_sline_len - (stepsB - 1);
            slineLen[sl] = stepsF + stepsB - 1;
        }
    }
}
