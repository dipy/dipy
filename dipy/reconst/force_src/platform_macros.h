/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// API visibility macros
#define FAISS_API

namespace faiss {

// Platform detection
#ifdef _MSC_VER
    #define FAISS_PRAGMA(X) __pragma(X)
#else
    #define FAISS_PRAGMA(X) _Pragma(#X)
#endif

} // namespace faiss
