cdef extern from *:
    """
    /* np.finfo('f8').max */
    #define __Pyx_BIGGEST_DOUBLE 1.7976931348623157e+308
    /* < FLT_MAX (3.4028235e+38); avoids double-to-float overflow */
    #define __Pyx_BIGGEST_FLOAT 3.402823e+38f
    /* > -FLT_MAX; avoids double-to-float overflow */
    #define __Pyx_SMALLEST_FLOAT (-3.402823e+38f)
    """
    const double BIGGEST_DOUBLE "__Pyx_BIGGEST_DOUBLE"
    const float BIGGEST_FLOAT "__Pyx_BIGGEST_FLOAT"
    const float SMALLEST_FLOAT "__Pyx_SMALLEST_FLOAT"
