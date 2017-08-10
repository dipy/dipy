cdef enum TissueClass:
    PYERROR = -2
    OUTSIDEIMAGE = -1
    INVALIDPOINT = 0
    TRACKPOINT = 1
    ENDPOINT = 2


cdef class TissueClassifier:
    # Please update doc-string in tissue_classifer.pyx if you change these
    # declarations
    cdef:
        double interp_out_double[1]
        double[::1] interp_out_view
    cpdef TissueClass check_point(self, double[::1] point) except PYERROR

cdef class BinaryTissueClassifier(TissueClassifier):
    cdef:
        unsigned char [:, :, :] mask
    pass

cdef class ThresholdTissueClassifier(TissueClassifier):
    cdef:
        double threshold
        double[:, :, :] metric_map
    pass

cdef class ConstrainedTissueClassifier(TissueClassifier):
    cdef:
        double[:, :, :] include_map, exclude_map
    cpdef double get_exclude(self, double[::1] point)
    cpdef double get_include(self, double[::1] point)
    pass

cdef class ActTissueClassifier(ConstrainedTissueClassifier):
    pass

cdef class CmcTissueClassifier(ConstrainedTissueClassifier):
    cdef:
        double step_size
        double average_voxel_size
        double correction_factor
    pass
