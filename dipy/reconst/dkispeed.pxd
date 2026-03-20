# Header file for dkispeed.pyx
# Provides declarations for Cython-optimized DKI functions

cimport numpy as cnp

# Single-value helper functions (nogil compatible)
cdef bint positive_evals_single(double L1, double L2, double L3, 
                                 double er=*) noexcept nogil

cdef double carlson_rf_single(double x, double y, double z, 
                               double errtol=*) noexcept nogil

cdef double carlson_rd_single(double x, double y, double z, 
                               double errtol=*) noexcept nogil

cdef double F1m_single(double a, double b, double c, 
                        double er=*) noexcept nogil

cdef double F2m_single(double a, double b, double c, 
                        double er=*) noexcept nogil

cdef double G1m_single(double a, double b, double c,
                        double er=*) noexcept nogil

cdef double G2m_single(double a, double b, double c,
                        double er=*) noexcept nogil

cdef double Wrotate_element_single(double[:] kt, int indi, int indj, 
                                    int indk, int indl, 
                                    double[:, :] B) noexcept nogil
