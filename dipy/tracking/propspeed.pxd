cimport numpy as cnp

cdef cnp.npy_intp _propagation_direction(double *point, double* prev, double* qa,
                                double *ind, double *odf_vertices,
                                double qa_thr, double ang_thr,
                                cnp.npy_intp *qa_shape,cnp.npy_intp* strides,
                                double *direction,double total_weight) nogil
