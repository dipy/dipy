# cython: language_level=3
from libc.stdint cimport uint64_t, int32_t


ctypedef fused real_t:
    float
    double


cdef struct Params:
    int dimx, dimy, dimz, dimt, nedges
    int sphere_symm
    int max_sline_len
    double pmf_threshold
    double step_size
    double cos_max_angle
    double cos_min_sep
    double relative_peak_thresh
    double tc_threshold


cdef inline uint64_t splitmix64(uint64_t x) noexcept nogil:
    x += <uint64_t>0x9E3779B97F4A7C15
    x = (x ^ (x >> 30)) * <uint64_t>0xBF58476D1CE4E5B9
    x = (x ^ (x >> 27)) * <uint64_t>0x94D049BB133111EB
    return x ^ (x >> 31)


cdef inline double rng_uniform(uint64_t *s) noexcept nogil:
    cdef uint64_t x = s[0]
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    s[0] = x
    return (<double>((x * <uint64_t>0x2545F4914F6CDD1D) >> 11) *
            (1.0 / 9007199254740992.0))


cdef Params make_params(dict kw)

cdef int trilinear_interp(const real_t[:, :, :, ::1] pmf, const real_t *pt,
                          real_t *out, const Params *P) noexcept nogil

cdef int check_point(const real_t[:, :, ::1] metric, const real_t *pt,
                     const Params *P) noexcept nogil

cdef void apply_pmf_threshold(real_t *pmf, int n, double frac) noexcept nogil

cdef int peak_directions(const real_t *odf, const real_t[:, ::1] verts,
                         const int32_t[:, ::1] edges, real_t[:, ::1] dirs_out,
                         int32_t *shInd, const Params *P) noexcept nogil

cdef int get_direction_prob_step(const real_t[:, :, :, ::1] pmf, real_t *direction,
                                 const real_t *pt, const real_t[:, ::1] verts,
                                 real_t *new_dir, real_t *scratch,
                                 uint64_t *rng, const Params *P) noexcept nogil

cdef int tracker(const real_t *seed, const real_t *first_step,
                 const real_t[:, :, :, ::1] pmf, const real_t[:, :, ::1] metric,
                 const real_t[:, ::1] verts, real_t[:, ::1] sline,
                 real_t *scratch, uint64_t *rng, const Params *P) noexcept nogil
