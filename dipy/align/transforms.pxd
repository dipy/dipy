cdef enum:
    TRANSLATION = 1
cdef enum:
    ROTATION=2
cdef enum:
    RIGID=3
cdef enum:
    SCALING=4
cdef enum:
    AFFINE=5

ctypedef int (*jacobian_function)(double[:], double[:], double[:,:]) nogil
r""" Type of a function that computes the Jacobian of a transform.
Jacobian functions receive a vector containing the current parameters
of the transformation, the coordinates of a point to compute the
Jacobian at, and the Jacobian matrix to write the result in. The
shape of the resulting Jacobian must be a dxn matrix, where d is the
dimension of the transform, and n is the number of parameters of the
transformation.

If the Jacobian is CONSTANT along its domain, the corresponding
jacobian_function must RETURN 1. Otherwise it must RETURN 0. This
information is used by the optimizer to avoid making unnecessary
function calls

Note: we need jacobian functions to be declared as nogil because they may
be called once for each sample static/moving voxel pair. This way we
can make this call inside a nogil loop
"""

ctypedef void (*param_to_matrix_function)(double[:], double[:,:]) nogil
r""" Type of a function that computes the matrix associated to an
affine transform in canonical coordinates.

Note: this function should be called only O(1) times per iteration of the
optimization method, as opposed to once per static/moving voxel pair
that jacobian_functions are called. So, it is not crucial for this type
to be declared as nogil.
"""

cdef jacobian_function get_jacobian_function(int transform_type, int dim) nogil
cdef param_to_matrix_function get_param_to_matrix_function(int transform_type,
                                                           int dim) nogil
