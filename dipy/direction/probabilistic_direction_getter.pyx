# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False

"""
Implementation of a probabilistic direction getter based on sampling from
discrete distribution (pmf) at each step of the tracking.
"""
from random import random

import numpy as np
cimport numpy as cnp

from dipy.direction.closest_peak_direction_getter cimport PmfGenDirectionGetter
from dipy.utils.fast_numpy cimport (copy_point, cumsum, norm, normalize,
                                     where_to_insert)

from dipy.tracking.stopping_criterion cimport (StreamlineStatus,
                                               StoppingCriterion,
                                               TRACKPOINT,
                                               ENDPOINT,
                                               OUTSIDEIMAGE,
                                               INVALIDPOINT,)

cdef class ProbabilisticDirectionGetter(PmfGenDirectionGetter):
    """Randomly samples direction of a sphere based on probability mass
    function (pmf).

    The main constructors for this class are current from_pmf and from_shcoeff.
    The pmf gives the probability that each direction on the sphere should be
    chosen as the next direction. To get the true pmf from the "raw pmf"
    directions more than ``max_angle`` degrees from the incoming direction are
    set to 0 and the result is normalized.
    """
    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=.1, **kwargs):
        """Direction getter from a pmf generator.

        Parameters
        ----------
        pmf_gen : PmfGen
            Used to get probability mass function for selecting tracking
            directions.
        max_angle : float, [0, 90]
            The maximum allowed angle between incoming direction and new
            direction.
        sphere : Sphere
            The set of directions to be used for tracking.
        pmf_threshold : float [0., 1.]
            Used to remove direction from the probability mass function for
            selecting the tracking direction.
        relative_peak_threshold : float in [0., 1.]
            Used for extracting initial tracking directions. Passed to
            peak_directions.
        min_separation_angle : float in [0, 90]
            Used for extracting initial tracking directions. Passed to
            peak_directions.

        See Also
        --------
        dipy.direction.peaks.peak_directions

        """
        PmfGenDirectionGetter.__init__(self, pmf_gen, max_angle, sphere,
                                       pmf_threshold=pmf_threshold, **kwargs)
        # The vertices need to be in a contiguous array
        self.vertices = self.sphere.vertices.copy()


    cdef int get_direction_c(self, double[::1] point, double[::1] direction):
        """Samples a pmf to updates ``direction`` array with a new direction.

        Parameters
        ----------
        point : memory-view (or ndarray), shape (3,)
            The point in an image at which to lookup tracking directions.
        direction : memory-view (or ndarray), shape (3,)
            Previous tracking direction.

        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.

        """
        cdef:
            cnp.npy_intp i, idx, _len
            double[:] newdir
            double* pmf
            double last_cdf, cos_sim

        _len = self.len_pmf
        pmf = self._get_pmf(point)


        if norm(&direction[0]) == 0:
            return 1
        normalize(&direction[0])

        with nogil:
            for i in range(_len):
                cos_sim = self.vertices[i][0] * direction[0] \
                        + self.vertices[i][1] * direction[1] \
                        + self.vertices[i][2] * direction[2]
                if cos_sim < 0:
                    cos_sim = cos_sim * -1
                if cos_sim < self.cos_similarity:
                    pmf[i] = 0

            cumsum(pmf, pmf, _len)
            last_cdf = pmf[_len - 1]
            if last_cdf == 0:
                return 1

        idx = where_to_insert(pmf, random() * last_cdf, _len)

        newdir = self.vertices[idx]
        # Update direction and return 0 for error
        if (direction[0] * newdir[0]
            + direction[1] * newdir[1]
            + direction[2] * newdir[2] > 0):
            copy_point(&newdir[0], &direction[0])
        else:
            newdir[0] = newdir[0] * -1
            newdir[1] = newdir[1] * -1
            newdir[2] = newdir[2] * -1
            copy_point(&newdir[0], &direction[0])
        return 0


cdef class DeterministicMaximumDirectionGetter(ProbabilisticDirectionGetter):
    """Return direction of a sphere with the highest probability mass
    function (pmf).
    """
    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=.1, **kwargs):
        ProbabilisticDirectionGetter.__init__(self, pmf_gen, max_angle, sphere,
                                              pmf_threshold=pmf_threshold, **kwargs)

    cdef int get_direction_c(self, double[::1] point, double[::1] direction):
        """Find direction with the highest pmf to updates ``direction`` array
        with a new direction.
        Parameters
        ----------
        point : memory-view (or ndarray), shape (3,)
            The point in an image at which to lookup tracking directions.
        direction : memory-view (or ndarray), shape (3,)
            Previous tracking direction.
        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.
        """
        cdef:
            cnp.npy_intp _len, max_idx
            double[:] newdir
            double* pmf
            double max_value, cos_sim

        pmf = self._get_pmf(point)
        _len = self.len_pmf
        max_idx = 0
        max_value = 0.0

        if norm(&direction[0]) == 0:
            return 1
        normalize(&direction[0])

        with nogil:
            for i in range(_len):
                cos_sim = self.vertices[i][0] * direction[0] \
                        + self.vertices[i][1] * direction[1] \
                        + self.vertices[i][2] * direction[2]
                if cos_sim < 0:
                    cos_sim = cos_sim * -1
                if cos_sim > self.cos_similarity and pmf[i] > max_value:
                    max_idx = i
                    max_value = pmf[i]

            if max_value <= 0:
                return 1

            newdir = self.vertices[max_idx]
            # Update direction and return 0 for error
            if (direction[0] * newdir[0]
                + direction[1] * newdir[1]
                + direction[2] * newdir[2] > 0):
                copy_point(&newdir[0], &direction[0])
            else:
                newdir[0] = newdir[0] * -1
                newdir[1] = newdir[1] * -1
                newdir[2] = newdir[2] * -1
                copy_point(&newdir[0], &direction[0])
        return 0

cdef class FlockingDirectionGetter(PmfGenDirectionGetter):
    """
    """
    
    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=.1, particle_count = 16, r_min=0.1, r_max=1.0, delta=0.995, alpha=1.0, **kwargs):
        """Direction getter from a pmf generator.

        Parameters
        ----------
        pmf_gen : PmfGen
            Used to get probability mass function for selecting tracking
            directions.
        max_angle : float, [0, 90]
            The maximum allowed angle between incoming direction and new
            direction.
        sphere : Sphere
            The set of directions to be used for tracking.
        pmf_threshold : float [0., 1.]
            Used to remove direction from the probability mass function for
            selecting the tracking direction.
        particle_count
            Number of particles in a flock
        r_min
            Minimum distance for gravitational attraction
        r_max
            Maximum distance for gravitational attraction
        delta
            Weight parameter between probabilistic direction and gravitational attraction
        alpha
            Radius of the flock seeding ball at the start
        relative_peak_threshold : float in [0., 1.]
            Used for extracting initial tracking directions. Passed to
            peak_directions.
        min_separation_angle : float in [0, 90]
            Used for extracting initial tracking directions. Passed to
            peak_directions.

        See Also
        --------
        dipy.direction.peaks.peak_directions

        """
        PmfGenDirectionGetter.__init__(self, pmf_gen, max_angle, sphere,
                                       pmf_threshold, **kwargs)
        self.vertices = self.sphere.vertices.copy()
        self.particle_count = particle_count
        self.r_min = r_min
        self.r_max = r_max
        self.delta = delta
        self.alpha= alpha
        


    cdef int get_direction_c(self, double[::1] point, double[::1] direction):
        """Samples a pmf to updates ``direction`` array with a new direction.

        Parameters
        ----------
        point : memory-view (or ndarray), shape (3,)
            The point in an image at which to lookup tracking directions.
        direction : memory-view (or ndarray), shape (3,)
            Previous tracking direction.

        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.

        """
        cdef:
            cnp.npy_intp i, idx, _len
            double[:] newdir
            double* pmf
            double last_cdf, cos_sim

        _len = self.len_pmf
        pmf = self._get_pmf(point)


        if norm(&direction[0]) == 0:
            return 1
        normalize(&direction[0])

        with nogil:
            for i in range(_len):
                cos_sim = self.vertices[i][0] * direction[0] \
                        + self.vertices[i][1] * direction[1] \
                        + self.vertices[i][2] * direction[2]
                if cos_sim < 0:
                    cos_sim = cos_sim * -1
                if cos_sim < self.cos_similarity:
                    pmf[i] = 0

            cumsum(pmf, pmf, _len)
            last_cdf = pmf[_len - 1]
            if last_cdf == 0:
                return 1

        idx = where_to_insert(pmf, random() * last_cdf, _len)

        newdir = self.vertices[idx]
        # Update direction and return 0 for error
        if (direction[0] * newdir[0]
            + direction[1] * newdir[1]
            + direction[2] * newdir[2] > 0):
            copy_point(&newdir[0], &direction[0])
        else:
            newdir[0] = newdir[0] * -1
            newdir[1] = newdir[1] * -1
            newdir[2] = newdir[2] * -1
            copy_point(&newdir[0], &direction[0])
        return 0
        
    cpdef tuple generate_streamline(self,
                                    double[::1] seed,
                                    double[::1] direction,
                                    double[::1] voxel_size,
                                    double step_size,
                                    StoppingCriterion stopping_criterion,
                                    cnp.float_t[:, :] streamline,
                                    StreamlineStatus stream_status,
                                    int fixedstep
                                    ):
        cdef:
            cnp.npy_intp i
            cnp.npy_intp len_streamlines = streamline.shape[0]
            cnp.npy_intp particle_count = self.particle_count
            double point[3]
            double point_par[3]
            double voxdir[3]
            double dir[3]
            double N, G, r_min, r_max, delta_1, delta_2, alpha, norma_vec
            cnp.float_t[:, :] particles = np.empty((particle_count, 3), dtype=np.float64)
            cnp.float_t[:, :] particles_dir = np.empty((particle_count, 3), dtype=np.float64)
            cnp.float_t[:, :] u_m = np.empty((particle_count, 3), dtype=np.float64)
            cnp.uint8_t[:] particle_values = np.ones(particle_count, dtype=np.uint8)
            cnp.float_t[:, :] F_m = np.empty((particle_count, 3), dtype=np.float64)
            double voxel_min[3]
            double voxel_max[3]

        copy_point(&seed[0], point)
       
        terminate = False
        r_min = self.r_min
        r_max = self.r_max
        delta_1 = self.delta
        delta_2 = 1-delta_1
        alpha = self.alpha
        
    
        for k in range(1, particle_count):
            copy_point(&direction[0], &particles_dir[k, 0])
            vec = np.random.normal(0, 1, 3)
            norma_vec = np.linalg.norm(vec)
            while norma_vec == 0:
                vec = np.random.normal(0, 1, 3)
                norma_vec = np.linalg.norm(vec)
            vec /= norma_vec
            r = np.cbrt(np.random.uniform(0, 1)) * alpha
            for j in range(3):
                particles[k, j] = seed[j] + r*vec[j]
               
        copy_point(&direction[0], &particles_dir[0, 0])
        copy_point(&seed[0], &particles[0, 0])
        copy_point(&seed[0], &streamline[0,0])

        stream_status = TRACKPOINT
        for i in range(1, len_streamlines):
            if terminate:
                break
            # Particles
            for m in range(particle_count):
                copy_point(&particles[m, 0], point_par)
                copy_point(&particles_dir[m, 0], dir)
                if self.get_direction_c(point_par, dir):
                    particle_values[m] = 0
                    if m==0:
                        terminate = True
                        continue
                copy_point(dir, &particles_dir[m, 0])
               
            for m in range(particle_count):
                for j in range(3):
                    F_m[m, j] = 0
                if particle_values[m] == 0:
                    continue
                for k in range(particle_count):
                    if k != m:
                        for j in range(3):
                            u_m[k, j] = particles[m, j] - particles[k, j]
                        N = norm(&u_m[k,0])
                        N = N*N
                        if N < r_min:
                            for j in range(3):
                                u_m[k, j] /= r_min
                                F_m[m, j] += -u_m[k, j]
                        elif r_min <= N and N <r_max:
                            for j in range(3):
                                u_m[k, j] /= N
                                F_m[m, j] += -u_m[k, j]
                       
                copy_point(&particles[m, 0], point_par)
                for j in range(3):
                    particles_dir[m, j] *= delta_1
                    particles_dir[m, j] += delta_2 * F_m[m, j]
                normalize(&particles_dir[m, 0])
                for j in range(3):
                    voxdir[j] = particles_dir[m, j] / voxel_size[j]
                for k in range(3):
                    point_par[k] += voxdir[k] * step_size
                copy_point(point_par, &particles[m, 0])
           
            copy_point(&particles[0, 0], &streamline[i, 0])
            stream_status = stopping_criterion.check_point_c(&point[0])
            copy_point(&streamline[i, 0], point) 
            copy_point(&particles_dir[0, 0], &direction[0])
           
            if stream_status == TRACKPOINT:
                continue
            elif (stream_status == ENDPOINT or
                 stream_status == INVALIDPOINT or
                 stream_status == OUTSIDEIMAGE):
                break
        else:
            # maximum length of streamline has been reached, return everything
            i = streamline.shape[0]
           
        return i, stream_status
