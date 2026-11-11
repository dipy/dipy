/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <distances.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__AVX__) || defined(_MSC_VER)
#include <immintrin.h>
#endif

namespace faiss {

/*********************************************************
 * Reference implementations (no SIMD)
 *********************************************************/

float fvec_inner_product_ref(const float* x, const float* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        res += x[i] * y[i];
    }
    return res;
}

float fvec_L2sqr_ref(const float* x, const float* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

/*********************************************************
 * SSE/AVX implementations
 *********************************************************/

#ifdef __SSE3__

float fvec_inner_product_sse(const float* x, const float* y, size_t d) {
    __m128 sum = _mm_setzero_ps();
    size_t i;

    for (i = 0; i + 3 < d; i += 4) {
        __m128 xi = _mm_loadu_ps(x + i);
        __m128 yi = _mm_loadu_ps(y + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(xi, yi));
    }

    // Horizontal sum
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    float result;
    _mm_store_ss(&result, sum);

    // Scalar tail
    for (; i < d; i++) {
        result += x[i] * y[i];
    }

    return result;
}

#endif // __SSE3__

#ifdef __AVX__

float fvec_inner_product_avx(const float* x, const float* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i;

    for (i = 0; i + 7 < d; i += 8) {
        __m256 xi = _mm256_loadu_ps(x + i);
        __m256 yi = _mm256_loadu_ps(y + i);

#ifdef __FMA__
        sum = _mm256_fmadd_ps(xi, yi, sum);  // Fused multiply-add
#else
        sum = _mm256_add_ps(sum, _mm256_mul_ps(xi, yi));
#endif
    }

    // Horizontal reduction
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    sum_low = _mm_add_ps(sum_low, sum_high);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum_low = _mm_hadd_ps(sum_low, sum_low);

    float result;
    _mm_store_ss(&result, sum_low);

    // Scalar tail
    for (; i < d; i++) {
        result += x[i] * y[i];
    }

    return result;
}

float fvec_L2sqr_avx(const float* x, const float* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i;

    for (i = 0; i + 7 < d; i += 8) {
        __m256 xi = _mm256_loadu_ps(x + i);
        __m256 yi = _mm256_loadu_ps(y + i);
        __m256 diff = _mm256_sub_ps(xi, yi);

#ifdef __FMA__
        sum = _mm256_fmadd_ps(diff, diff, sum);
#else
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
#endif
    }

    // Horizontal reduction
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    sum_low = _mm_add_ps(sum_low, sum_high);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum_low = _mm_hadd_ps(sum_low, sum_low);

    float result;
    _mm_store_ss(&result, sum_low);

    // Scalar tail
    for (; i < d; i++) {
        const float tmp = x[i] - y[i];
        result += tmp * tmp;
    }

    return result;
}

#endif // __AVX__

/*********************************************************
 * Main entry points (dispatch to best available version)
 *********************************************************/

float fvec_inner_product(const float* x, const float* y, size_t d) {
#ifdef __AVX__
    return fvec_inner_product_avx(x, y, d);
#elif defined(__SSE3__)
    return fvec_inner_product_sse(x, y, d);
#else
    return fvec_inner_product_ref(x, y, d);
#endif
}

float fvec_L2sqr(const float* x, const float* y, size_t d) {
#ifdef __AVX__
    return fvec_L2sqr_avx(x, y, d);
#else
    return fvec_L2sqr_ref(x, y, d);
#endif
}

float fvec_norm_L2sqr(const float* x, size_t d) {
    float sum = 0;
#ifdef __AVX__
    __m256 acc = _mm256_setzero_ps();
    size_t i;

    for (i = 0; i + 7 < d; i += 8) {
        __m256 xi = _mm256_loadu_ps(x + i);
#ifdef __FMA__
        acc = _mm256_fmadd_ps(xi, xi, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(xi, xi));
#endif
    }

    __m128 sum_high = _mm256_extractf128_ps(acc, 1);
    __m128 sum_low = _mm256_castps256_ps128(acc);
    sum_low = _mm_add_ps(sum_low, sum_high);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    _mm_store_ss(&sum, sum_low);

    for (; i < d; i++) {
        sum += x[i] * x[i];
    }
#else
    for (size_t i = 0; i < d; i++) {
        sum += x[i] * x[i];
    }
#endif
    return sum;
}

} // namespace faiss
