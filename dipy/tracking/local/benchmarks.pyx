cimport cython
from libc.stdlib cimport RAND_MAX, rand
cimport numpy as np
import numpy as np

from .tissue_classifier cimport (TissueClassifier, TissueClass,
                                 ThresholdTissueClassifier)



cdef double drand():
    return (<double> rand()) / (<double> RAND_MAX + 1)


cdef inline void _setpoint(double[:] point, np.npy_intp[:] shape):
    cdef:
        np.npy_intp i
    for i in range(3):
        point[i] = drand() * shape[i] - .5 


# def benchmark_ThresholdClassifier(np.npy_intp N=10**6):
cdef class ThreshouldClassifierBenchmarks:
    cdef:
        np.npy_intp[:] shape
        ThresholdTissueClassifier classifier
        double[::1] point

    def __cinit__(self):
        self.point = np.zeros(3)
        m = np.random.random((100, 100, 50))
        self.classifier = ThresholdTissueClassifier(m, .5)
        self.shape = np.zeros(3, dtype='int')
        for i in range(3):
            self.shape[i] = m.shape[i]


    def runGoodPoint(self, np.npy_intp N=10**6):
        cdef:
            np.npy_intp i, 
            TissueClass tsucls
        for i in range(N):
            _setpoint(self.point, self.shape)
            tsucls = self.classifier.check_point(self.point)

    def runBadPoint(self, np.npy_intp N=10**6):
        cdef:
            np.npy_intp i, 
            TissueClass tsucls

        for i in range(3):
            self.point[i] = -2
        for i in range(N):
            tsucls = self.classifier.check_point(self.point)

            

