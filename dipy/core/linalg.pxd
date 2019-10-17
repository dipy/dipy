cimport numpy as cnp

cdef void fast_matvec(char ta, double[:,::1] A, double[:] b,
                      double[:] y, double alpha=*, double beta=*,
                      int incx=*) nogil

cdef void fast_eig(double[:,::1] arr, double[::1] out_w, double[::1] out_work,
                   int lwork, int[::1] out_iwork, int liwork) nogil

cdef void fast_dgemm(double[:,::1] in_arr, double[:,::1] out_arr) nogil