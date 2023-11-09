cimport numpy as cnp

from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.direction.pmf cimport PmfGen

cpdef generate_tractogram(cnp.ndarray[cnp.float_t, ndim=2] seed_positons,
                          cnp.ndarray[cnp.float_t, ndim=2] seed_directions,
                          StoppingCriterion sc,
                          PmfGen pmf_gen):
    cdef:
        cnp.npy_intp _len=seed_positons.shape[0]

    return generate_tractogram_c(<double[:_len,:3]> seed_positons,
                                 <double[:_len,:3]> seed_directions,
                                 sc, pmf_gen)

cdef double[:, :] generate_tractogram_c(double[:, :] seed_positons,
                                        double[:, :] seed_directions,
                                        StoppingCriterion sc,
                                        PmfGen pmf_gen)