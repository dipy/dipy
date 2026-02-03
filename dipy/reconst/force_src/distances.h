/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <platform_macros.h>

namespace faiss {

/*********************************************************
 * Optimized SIMD distance computations
 *********************************************************/

/// Inner product between two vectors (SIMD optimized: AVX2/FMA)
FAISS_API float fvec_inner_product(const float* x, const float* y, size_t d);

/// Squared L2 distance between two vectors (SIMD optimized)
FAISS_API float fvec_L2sqr(const float* x, const float* y, size_t d);

/// L2 norm of a vector (SIMD optimized)
FAISS_API float fvec_norm_L2sqr(const float* x, size_t d);

} // namespace faiss
