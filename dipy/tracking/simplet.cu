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

/* CUDA backend for dipy.tracking.simpletracker
 *
 * Compile-time macros (set by dipy/tracking/cudasimplet.py):
 *   DIMX, DIMY, DIMZ, DIMT    volume dimensions
 *   STEP_SIZE                 step size in voxels
 *   MAX_ANGLE                 max turning angle (radians)
 *   TC_THRESHOLD              stopping threshold on the metric map
 *   PMF_THRESHOLD_P           relative PMF threshold
 *   MAX_SLINE_LEN             max number of steps in one direction
 *   RNG_SEED                  random seed
 *   SPHERE_SYMM               1 if sphere vertices are antipodally symmetric
 *   THR_X_SL, THR_X_BL        threads per streamline, threads per block
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

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

template<int BDIM_X>
__device__ int trilinear_interp_d(const REAL *__restrict__ dataf,
                                  const REAL3 point,
                                  REAL *__restrict__ __vox_data) {
        const REAL HALF = 0.5f;

        if (point.x < -HALF || point.x+HALF >= DIMX ||
            point.y < -HALF || point.y+HALF >= DIMY ||
            point.z < -HALF || point.z+HALF >= DIMZ) {
                return -1;
        }

        long long coo[3][2];
        REAL wgh[3][2];
        const REAL ONE = 1.0f;

        const REAL3 fl = MAKE_REAL3(FLOOR(point.x), FLOOR(point.y), FLOOR(point.z));

        wgh[0][1] = point.x - fl.x;
        wgh[0][0] = ONE-wgh[0][1];
        coo[0][0] = MAX(0, fl.x);
        coo[0][1] = MIN(DIMX-1, coo[0][0]+1);

        wgh[1][1] = point.y - fl.y;
        wgh[1][0] = ONE-wgh[1][1];
        coo[1][0] = MAX(0, fl.y);
        coo[1][1] = MIN(DIMY-1, coo[1][0]+1);

        wgh[2][1] = point.z - fl.z;
        wgh[2][0] = ONE-wgh[2][1];
        coo[2][0] = MAX(0, fl.z);
        coo[2][1] = MIN(DIMZ-1, coo[2][0]+1);

        for (int t = threadIdx.x; t < DIMT; t += BDIM_X) {
                __vox_data[t] = 0;
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                        #pragma unroll
                        for (int j = 0; j < 2; j++) {
                                #pragma unroll
                                for (int k = 0; k < 2; k++) {
                                        __vox_data[t] += wgh[0][i] * wgh[1][j] * wgh[2][k] *
                                                dataf[coo[0][i] * DIMY * DIMZ * DIMT +
                                                      coo[1][j] * DIMZ * DIMT +
                                                      coo[2][k] * DIMT +
                                                      t];
                                }
                        }
                }
        }
        return 0;
}

__device__ int check_point_d(const REAL3 point,
                             const cudaTextureObject_t *__restrict__ metric_map) {
        // linear-filtered textures put texel i at coordinate i + 0.5
        float val = tex3D<float>(*metric_map, (float) point.z + 0.5f,
                                 (float) point.y + 0.5f, (float) point.x + 0.5f);
        if (val == -1.0f) {
                return OUTSIDEIMAGE;
        }
        return (val > TC_THRESHOLD) ? TRACKPOINT : ENDPOINT;
}


template<int BDIM_X, int BDIM_Y>
__device__ int get_direction_prob_d(curandStatePhilox4_32_10_t *st,
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
                __tmp = curand_uniform(st) * last_cdf;
        }
        REAL selected_cdf = __shfl_sync(WMASK, __tmp, 0, BDIM_X);

        int low = 0;
        int high = DIMT - 1;
        while ((high - low) >= BDIM_X) {
                const int mid = (low + high) / 2;
                if (__pmf_data_sh[mid] < selected_cdf) {
                        low = mid;
                } else {
                        high = mid;
                }
        }
        const bool __ballot = (low+tidx <= high) ? (selected_cdf < __pmf_data_sh[low+tidx]) : 0;
        const int __msk = __ballot_sync(WMASK, __ballot);
        const int indProb = low + __ffs(__msk) - 1;

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


template<int BDIM_X, int BDIM_Y>
__device__ int tracker_d(curandStatePhilox4_32_10_t *st,
                         REAL3 seed,
                         REAL3 first_step,
                         const REAL *__restrict__ dataf,
                         const cudaTextureObject_t *__restrict__ metric_map,
                         const REAL3 *__restrict__ sphere_vertices,
                         int *__restrict__ nsteps,
                         REAL3 *__restrict__ streamline) {
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
        const unsigned int WMASK = WARP_MASK(BDIM_X);

        int tissue_class = TRACKPOINT;

        REAL3 point = seed;
        REAL3 direction = first_step;
        __shared__ REAL3 __sh_new_dir[BDIM_Y];

        if (tidx == 0) {
                streamline[0] = point;
        }
        __syncwarp(WMASK);

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

                point.x += direction.x * STEP_SIZE;
                point.y += direction.y * STEP_SIZE;
                point.z += direction.z * STEP_SIZE;

                if (tidx == 0) {
                        streamline[i] = point;
                }
                __syncwarp(WMASK);

                tissue_class = check_point_d(point, metric_map);
                if (tissue_class != TRACKPOINT) {
                        break;
                }
        }
        nsteps[0] = i;
        return tissue_class;
}

// ── kernel: generate streamlines from precomputed seed directions ────
//
// slineOutOff : (nseed+1,) exclusive prefix sum of the number of
//               directions per seed (from get_num_streamlines_prob)
// shDir0      : (sum(ndir),) initial direction of each streamline
// sline       : (sum(ndir) * MAX_SLINE_LEN * 2,) output points

template<int BDIM_X, int BDIM_Y>
__global__ void genStreamlinesProb_k(const int nseed,
                                     const REAL3 *__restrict__ seeds,
                                     const REAL *__restrict__ dataf,
                                     const cudaTextureObject_t *__restrict__ metric_map,
                                     const REAL3 *__restrict__ sphere_vertices,
                                     const int *__restrict__ slineOutOff,
                                     const REAL3 *__restrict__ shDir0,
                                     int *__restrict__ slineSeed,
                                     int *__restrict__ slineLen,
                                     REAL3 *__restrict__ sline) {
        const int tidx = threadIdx.x;
        const int slid = blockIdx.x*blockDim.y + threadIdx.y;
        const unsigned int WMASK = WARP_MASK(BDIM_X);

        curandStatePhilox4_32_10_t st;
        const size_t gid = blockIdx.x * blockDim.y * blockDim.x + blockDim.x * threadIdx.y + threadIdx.x;
        curand_init(RNG_SEED, gid+1, 0, &st);

        if (slid >= nseed) {
                return;
        }

        const REAL3 seed = seeds[slid];
        const int ndir = slineOutOff[slid+1]-slineOutOff[slid];
        __syncwarp(WMASK);

        int slineOff = slineOutOff[slid];

        for(int i = 0; i < ndir; i++) {
                const REAL3 first_step = shDir0[slineOff];
                REAL3 *__restrict__ currSline = sline + slineOff*MAX_SLINE_LEN*2;

                if (tidx == 0) {
                        slineSeed[slineOff] = slid;
                }

                int stepsB;
                tracker_d<BDIM_X, BDIM_Y>(&st, seed,
                                          MAKE_REAL3(-first_step.x, -first_step.y, -first_step.z),
                                          dataf, metric_map, sphere_vertices,
                                          &stepsB, currSline);

                // reverse backward sline
                for(int j = 0; j < stepsB/2; j += BDIM_X) {
                        if (j+tidx < stepsB/2) {
                                const REAL3 __p = currSline[j+tidx];
                                currSline[j+tidx] = currSline[stepsB-1 - (j+tidx)];
                                currSline[stepsB-1 - (j+tidx)] = __p;
                        }
                }

                int stepsF;
                tracker_d<BDIM_X, BDIM_Y>(&st, seed, first_step,
                                          dataf, metric_map, sphere_vertices,
                                          &stepsF, currSline + stepsB-1);

                if (tidx == 0) {
                        slineLen[slineOff] = stepsB-1+stepsF;
                }
                slineOff += 1;
        }
}
