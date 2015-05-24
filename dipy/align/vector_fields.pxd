cdef inline double _apply_affine_3d_x0(double x0, double x1, double x2,
                                       double h, double[:, :] aff) nogil
cdef inline double _apply_affine_3d_x1(double x0, double x1, double x2,
                                       double h, double[:, :] aff) nogil
cdef inline double _apply_affine_3d_x2(double x0, double x1, double x2,
                                       double h, double[:, :] aff) nogil
cdef inline double _apply_affine_2d_x0(double x0, double x1, double h,
                                       double[:, :] aff) nogil
cdef inline double _apply_affine_2d_x1(double x0, double x1, double h,
                                       double[:, :] aff) nogil
