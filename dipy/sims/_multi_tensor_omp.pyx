# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
"""
OpenMP-accelerated Multi-Tensor Signal Generation

High-performance implementation of multi-tensor diffusion signal
computation with optional OpenMP parallelization.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp
