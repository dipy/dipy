/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* CUDA backend for dipy.tracking.accel
 *
 * Compile-time macros (set by dipy/tracking/accel/cuda.py):
 *   DIMX, DIMY, DIMZ, DIMT    volume dimensions
 *   STEP_SIZE                 step size in voxels
 *   MAX_ANGLE                 max turning angle (radians)
 *   TC_THRESHOLD              stopping threshold on the metric map
 *   PMF_THRESHOLD_P           relative PMF threshold
 *   MAX_SLINE_LEN             max number of steps in one direction
 *   SPHERE_SYMM               1 if sphere vertices are antipodally symmetric
 *   THR_X_SL, THR_X_BL        threads per streamline, threads per block
 *
 * The tracking loop, the random number generator (Philox4x32-10 keyed by the
 * random seed and the streamline index, see seed_philox in
 * dipy/utils/fast_numpy.pyx) and the output layout follow the CPU tracker
 * (dipy/tracking/tractogen.pyx with rng="philox"), so all backends draw the
 * same random numbers.
 */

#include <cuda_runtime.h>

#define REAL       float
#define REAL3      float3
#define MAKE_REAL3 make_float3
#define FLOOR      floorf
#define COS        __cosf
#define FABS       fabsf
#define REAL_MAX   __int_as_float(0x7f7fffffU)
#define REAL_MIN   (-REAL_MAX)

#if SPHERE_SYMM == 0
  #define APPLY_ABS_IF_SYM(x) (x)
#else
  #define APPLY_ABS_IF_SYM(x) FABS(x)
#endif

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

enum {OUTSIDEIMAGE, INVALIDPOINT, TRACKPOINT, ENDPOINT};

#define WARP_MASK(BDIM_X) \
        ((((1ull << (BDIM_X))-1)) << ((((threadIdx.y*(BDIM_X) + threadIdx.x) % 32)) & (~((BDIM_X)-1))))

// ── Philox4x32-10 RNG ────────────────────────────────────────────────
// Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"
//            (SC '11).  DOI 10.1145/2063384.2063405
// Same generator in every backend: key = random seed, counter =
// (block, streamline index lo, streamline index hi, 0).

#define PHILOX_M4x32_0 0xD2511F53u
#define PHILOX_M4x32_1 0xCD9E8D57u
#define PHILOX_W32_0   0x9E3779B9u
#define PHILOX_W32_1   0xBB67AE85u

struct PhiloxState {
        uint4 ctr;
        uint2 key;
        uint4 out;
        int idx;
};

__device__ __forceinline__ uint4 philox4x32_round(const uint4 ctr, const uint2 key) {
        const unsigned int lo0 = ctr.x * PHILOX_M4x32_0;
        const unsigned int hi0 = __umulhi(ctr.x, PHILOX_M4x32_0);
        const unsigned int lo1 = ctr.z * PHILOX_M4x32_1;
        const unsigned int hi1 = __umulhi(ctr.z, PHILOX_M4x32_1);
        return make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

__device__ uint4 philox4x32_10(uint4 ctr, uint2 key) {
        #pragma unroll
        for (int r = 0; r < 9; r++) {
                ctr = philox4x32_round(ctr, key);
                key.x += PHILOX_W32_0;
                key.y += PHILOX_W32_1;
        }
        return philox4x32_round(ctr, key);
}

// random stream of the streamline with index sl_index in the tractogram
__device__ PhiloxState philox_init(const unsigned long long seed,
                                   const unsigned long long sl_index) {
        PhiloxState s;
        s.key = make_uint2((unsigned int) seed, (unsigned int) (seed >> 32));
        s.ctr = make_uint4(0u, (unsigned int) sl_index, (unsigned int) (sl_index >> 32), 0u);
        s.out = make_uint4(0u, 0u, 0u, 0u);
        s.idx = 4;
        return s;
}

// uniform in [0, 1) from the 24 high bits: exact in float32 and float64
__device__ REAL philox_uniform(PhiloxState *s) {
        if (s->idx == 4) {
                s->out = philox4x32_10(s->ctr, s->key);
                s->ctr.x += 1;
                s->idx = 0;
        }
        unsigned int w;
        switch (s->idx) {
                case 0: w = s->out.x; break;
                case 1: w = s->out.y; break;
                case 2: w = s->out.z; break;
                default: w = s->out.w; break;
        }
        s->idx++;
        return (REAL) (w >> 8) * (1.0f / 16777216.0f);
}

template<int BDIM_X>
__device__ REAL max_d(const int n, const REAL *__restrict__ src, const REAL minVal) {
        const int tidx = threadIdx.x;
        const unsigned int WMASK = WARP_MASK(BDIM_X);

        REAL __m = minVal;
        for(int i = tidx; i < n; i += BDIM_X) {
                __m = MAX(__m, src[i]);
        }
        #pragma unroll
        for(int i = BDIM_X/2; i; i /= 2) {
                const REAL __tmp = __shfl_xor_sync(WMASK, __m, i, BDIM_X);
                __m = MAX(__m, __tmp);
        }
        return __m;
}

template<int BDIM_X>
__device__ void prefix_sum_sh_d(REAL *num_sh, int __len) {
        const int tidx = threadIdx.x;
        const unsigned int WMASK = WARP_MASK(BDIM_X);

        for (int j = 0; j < __len; j += BDIM_X) {
                if ((tidx == 0) && (j != 0)) {
                        num_sh[j] += num_sh[j-1];
                }
                __syncwarp(WMASK);

                REAL __t_pmf;
                if (j+tidx < __len) {
                        __t_pmf = num_sh[j+tidx];
                }
                for (int i = 1; i < BDIM_X; i*=2) {
                        REAL __tmp = __shfl_up_sync(WMASK, __t_pmf, i, BDIM_X);
                        if ((tidx >= i) && (j+tidx < __len)) {
                                __t_pmf += __tmp;
                        }
                }
                if (j+tidx < __len) {
                        num_sh[j+tidx] = __t_pmf;
                }
                __syncwarp(WMASK);
        }
}

// Trilinear weights and voxel indices of point, as in
// dipy.core.interpolation: a point within half a voxel of the border uses the
// border voxel only. Returns -1 outside the image.
__device__ int trilinear_setup_d(const REAL3 point,
                                 long long coo[3][2],
                                 REAL wgh[3][2]) {
        const REAL HALF = 0.5f;

        if (point.x < -HALF || point.x+HALF >= DIMX ||
            point.y < -HALF || point.y+HALF >= DIMY ||
            point.z < -HALF || point.z+HALF >= DIMZ) {
                return -1;
        }

        const REAL ONE = 1.0f;
        const REAL3 fl = MAKE_REAL3(FLOOR(point.x), FLOOR(point.y), FLOOR(point.z));
        const long long flx = (long long) fl.x;
        const long long fly = (long long) fl.y;
        const long long flz = (long long) fl.z;

        wgh[0][1] = point.x - fl.x;
        wgh[0][0] = ONE-wgh[0][1];
        coo[0][0] = MAX(0, flx);
        coo[0][1] = MIN(DIMX-1, flx+1);

        wgh[1][1] = point.y - fl.y;
        wgh[1][0] = ONE-wgh[1][1];
        coo[1][0] = MAX(0, fly);
        coo[1][1] = MIN(DIMY-1, fly+1);

        wgh[2][1] = point.z - fl.z;
        wgh[2][0] = ONE-wgh[2][1];
        coo[2][0] = MAX(0, flz);
        coo[2][1] = MIN(DIMZ-1, flz+1);
        return 0;
}

template<int BDIM_X>
__device__ int trilinear_interp_d(const REAL *__restrict__ dataf,
                                  const REAL3 point,
                                  REAL *__restrict__ __vox_data) {
        long long coo[3][2];
        REAL wgh[3][2];
        if (trilinear_setup_d(point, coo, wgh) != 0) {
                return -1;
        }

        for (int t = threadIdx.x; t < DIMT; t += BDIM_X) {
                REAL __tmp = 0;
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                        #pragma unroll
                        for (int j = 0; j < 2; j++) {
                                #pragma unroll
                                for (int k = 0; k < 2; k++) {
                                        __tmp += wgh[0][i] * wgh[1][j] * wgh[2][k] *
                                                dataf[coo[0][i] * DIMY * DIMZ * DIMT +
                                                      coo[1][j] * DIMZ * DIMT +
                                                      coo[2][k] * DIMT +
                                                      t];
                                }
                        }
                }
                __vox_data[t] = __tmp;
        }
        return 0;
}

__device__ int check_point_d(const REAL3 point,
                             const REAL *__restrict__ metric_map) {
        long long coo[3][2];
        REAL wgh[3][2];
        if (trilinear_setup_d(point, coo, wgh) != 0) {
                return OUTSIDEIMAGE;
        }
        REAL val = 0;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                        #pragma unroll
                        for (int k = 0; k < 2; k++) {
                                val += wgh[0][i] * wgh[1][j] * wgh[2][k] *
                                       metric_map[coo[0][i] * DIMY * DIMZ +
                                                  coo[1][j] * DIMZ +
                                                  coo[2][k]];
                        }
                }
        }
        return (val > TC_THRESHOLD) ? TRACKPOINT : ENDPOINT;
}


template<int BDIM_X, int BDIM_Y>
__device__ int get_direction_prob_d(PhiloxState *st,
                                    const REAL *__restrict__ pmf,
                                    REAL3 dir,
                                    const REAL3 point,
                                    const REAL3 *__restrict__ sphere_vertices,
                                    REAL3 *__restrict__ new_dir) {
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
        const unsigned int WMASK = WARP_MASK(BDIM_X);

        __shared__ REAL pmf_data_sh[BDIM_Y][DIMT];
        REAL* __pmf_data_sh = pmf_data_sh[tidy];

        __syncwarp(WMASK);
        const int rv = trilinear_interp_d<BDIM_X>(pmf, point, __pmf_data_sh);
        __syncwarp(WMASK);
        if (rv != 0) {
                return 0;
        }

        const REAL absolpmf_thresh = PMF_THRESHOLD_P * max_d<BDIM_X>(DIMT, __pmf_data_sh, REAL_MIN);
        __syncwarp(WMASK);

        const REAL cos_similarity = COS(MAX_ANGLE);

        #pragma unroll
        for(int i = tidx; i < DIMT; i += BDIM_X) {
                const REAL dot = dir.x*sphere_vertices[i].x+
                                 dir.y*sphere_vertices[i].y+
                                 dir.z*sphere_vertices[i].z;
                if (__pmf_data_sh[i] < absolpmf_thresh ||
                    APPLY_ABS_IF_SYM(dot) < cos_similarity) {
                        __pmf_data_sh[i] = 0.0;
                }
        }
        __syncwarp(WMASK);

        prefix_sum_sh_d<BDIM_X>(__pmf_data_sh, DIMT);

        REAL last_cdf = __pmf_data_sh[DIMT - 1];
        if (last_cdf == 0) {
                return 0;
        }

        REAL __tmp;
        if (tidx == 0) {
                __tmp = philox_uniform(st) * last_cdf;
        }
        const REAL selected_cdf = __shfl_sync(WMASK, __tmp, 0, BDIM_X);

        // first index with cdf > selected_cdf (numpy.searchsorted "right",
        // as on the CPU). selected_cdf < last_cdf, and the search keeps
        // selected_cdf < cdf[high], so the ballot below is never empty.
        int low = 0;
        int high = DIMT - 1;
        while ((high - low) >= BDIM_X) {
                const int mid = (low + high) / 2;
                if (__pmf_data_sh[mid] <= selected_cdf) {
                        low = mid;
                } else {
                        high = mid;
                }
        }
        const bool __ballot = (low+tidx <= high) ? (selected_cdf < __pmf_data_sh[low+tidx]) : 0;
        const unsigned int __msk = __ballot_sync(WMASK, __ballot);
        // guard: never index sphere_vertices[-1] (__ffs(0) == 0)
        const int indProb = (__msk != 0) ? (low + __ffs(__msk) - 1) : high;

        if (tidx == 0) {
                const REAL3 v = sphere_vertices[indProb];
                if ((dir.x * v.x + dir.y * v.y + dir.z * v.z) > 0) {
                        *new_dir = v;
                } else {
                        *new_dir = MAKE_REAL3(-v.x, -v.y, -v.z);
                }
        }
        return 1;
}


// Track from seed along first_step. Point i is written to streamline[i * stride]
// (the seed, point 0, is written by the caller). Returns the number of points
// kept, counting the seed, with the same convention as the CPU tracker: the
// point that stopped the streamline is not kept. *first_dir receives the
// direction of the first step.
template<int BDIM_X, int BDIM_Y>
__device__ int tracker_d(PhiloxState *st,
                         const REAL3 seed,
                         const REAL3 first_step,
                         const REAL *__restrict__ dataf,
                         const REAL *__restrict__ metric_map,
                         const REAL3 *__restrict__ sphere_vertices,
                         REAL3 *__restrict__ streamline,
                         const int stride,
                         REAL3 *first_dir) {
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
        const unsigned int WMASK = WARP_MASK(BDIM_X);

        REAL3 point = seed;
        REAL3 direction = first_step;
        __shared__ REAL3 __sh_new_dir[BDIM_Y];

        int i;
        for(i = 1; i < MAX_SLINE_LEN; i++) {
                const int ndir = get_direction_prob_d<BDIM_X, BDIM_Y>(
                        st, dataf, direction, point, sphere_vertices, __sh_new_dir + tidy);
                __syncwarp(WMASK);
                direction = __sh_new_dir[tidy];
                __syncwarp(WMASK);

                if (ndir == 0) {
                        break;
                }
                if (i == 1) {
                        *first_dir = direction;
                }

                point.x += direction.x * STEP_SIZE;
                point.y += direction.y * STEP_SIZE;
                point.z += direction.z * STEP_SIZE;

                if (tidx == 0) {
                        streamline[i * stride] = point;
                }
                __syncwarp(WMASK);

                if (check_point_d(point, metric_map) != TRACKPOINT) {
                        break;
                }
        }
        // the CPU loop variable stops at MAX_SLINE_LEN - 1
        return MIN(i, MAX_SLINE_LEN - 1);
}

// ── kernel: generate streamlines from precomputed seed directions ────
//
// rng_seed    : random seed (Philox key)
// sl_base     : index in the tractogram of the first streamline of this launch
// slineOutOff : (nseed+1,) exclusive prefix sum of the number of
//               directions per seed
// shDir0      : (sum(ndir),) initial direction of each streamline
// slineStart  : (sum(ndir),) first point of each streamline in its block
// slineLen    : (sum(ndir),) number of points of each streamline
// sline       : (sum(ndir) * MAX_SLINE_LEN * 2,) output points; streamline s
//               occupies the block starting at s * MAX_SLINE_LEN * 2, with
//               its seed at offset MAX_SLINE_LEN

template<int BDIM_X, int BDIM_Y>
__global__ void genStreamlinesProb_k(const int nseed,
                                     const unsigned long long rng_seed,
                                     const unsigned long long sl_base,
                                     const REAL3 *__restrict__ seeds,
                                     const REAL *__restrict__ dataf,
                                     const REAL *__restrict__ metric_map,
                                     const REAL3 *__restrict__ sphere_vertices,
                                     const int *__restrict__ slineOutOff,
                                     const REAL3 *__restrict__ shDir0,
                                     int *__restrict__ slineStart,
                                     int *__restrict__ slineLen,
                                     REAL3 *__restrict__ sline) {
        const int tidx = threadIdx.x;
        const int slid = blockIdx.x*blockDim.y + threadIdx.y;
        const unsigned int WMASK = WARP_MASK(BDIM_X);

        if (slid >= nseed) {
                return;
        }

        const REAL3 seed = seeds[slid];
        const int firstSl = slineOutOff[slid];
        const int ndir = slineOutOff[slid+1] - firstSl;

        for(int i = 0; i < ndir; i++) {
                const int sl = firstSl + i;
                // one random stream per streamline, keyed by its index in the
                // tractogram: independent of chunks, devices and launch shape
                PhiloxState st = philox_init(rng_seed, sl_base + (unsigned long long) sl);

                const REAL3 first_step = shDir0[sl];
                REAL3 *__restrict__ center = sline + ((size_t) sl * MAX_SLINE_LEN * 2 + MAX_SLINE_LEN);
                if (tidx == 0) {
                        center[0] = seed;
                }
                __syncwarp(WMASK);

                // forward, then backward from the direction of the first
                // forward step (as on the CPU)
                REAL3 first_dir = first_step;
                const int stepsF = tracker_d<BDIM_X, BDIM_Y>(&st, seed, first_step,
                                                             dataf, metric_map, sphere_vertices,
                                                             center, 1, &first_dir);
                const REAL3 back = (stepsF > 1) ? first_dir : first_step;
                REAL3 unused;
                const int stepsB = tracker_d<BDIM_X, BDIM_Y>(&st, seed,
                                                             MAKE_REAL3(-back.x, -back.y, -back.z),
                                                             dataf, metric_map, sphere_vertices,
                                                             center, -1, &unused);

                if (tidx == 0) {
                        slineStart[sl] = MAX_SLINE_LEN - (stepsB - 1);
                        slineLen[sl] = stepsF + stepsB - 1;
                }
        }
}
