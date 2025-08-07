# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

"""
Implementation of the Parallel Transport Tractography (PTT) algorithm by
:footcite:t:`Aydogan2021`. PTT Default parameter values are slightly different
than in Trekker to optimise performances. The rejection sampling
algorithm also uses fewer samples to estimate the maximum of the posterior, and
fewer tries to obtain a suitable propagation candidate. Moreover, the initial
tangent direction in this implementation is always obtained from the voxel-wise
peaks.

References
----------
.. footbibliography::
"""

cimport numpy as cnp
from libc.math cimport M_PI, pow, sin, cos, fabs
from libc.stdlib cimport malloc, free

from dipy.direction.probabilistic_direction_getter cimport \
        ProbabilisticDirectionGetter
from dipy.utils.fast_numpy cimport (copy_point, cross, normalize, random,
                                    random_perpendicular_vector,
                                    random_point_within_circle)
from dipy.tracking.stopping_criterion cimport (StreamlineStatus,
                                               StoppingCriterion,
                                               TRACKPOINT,
                                               ENDPOINT,
                                               OUTSIDEIMAGE,
                                               INVALIDPOINT)
from dipy.tracking.utils import min_radius_curvature_from_angle


cdef class PTTDirectionGetter(ProbabilisticDirectionGetter):
    """Parallel Transport Tractography (PTT) direction getter.
    """
    cdef double       angular_separation
    cdef double       data_support_exponent
    cdef double[3][3] frame
    cdef double       k1
    cdef double       k2
    cdef double       k_small
    cdef double       last_val
    cdef double       last_val_cand
    cdef double       max_angle
    cdef double       max_curvature
    cdef double[3]    position
    cdef int          probe_count
    cdef double       probe_length
    cdef double       probe_normalizer
    cdef int          probe_quality
    cdef double       probe_radius
    cdef double       probe_step_size
    cdef double[9]    propagator
    cdef double       step_size
    cdef int          rejection_sampling_max_try
    cdef int          rejection_sampling_nbr_sample
    cdef double[3]    voxel_size
    cdef double[3]    inv_voxel_size


    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=None,
                 double probe_length=0.5, double probe_radius=0,
                 int probe_quality=3, int probe_count=1,
                 double data_support_exponent=1, **kwargs):
        """PTT used probe for estimating future propagation steps. A probe is a
        short, cylindrical model of the connecting segment.

        Parameters
        ----------
        pmf_gen : PmfGen
            Used to get probability mass function for selecting tracking
            directions.
        max_angle : float, [0, 90]
            Is used to set the upper limits for the k1 and k2 parameters
            of parallel transport frame (max_curvature).
        sphere : Sphere
            The set of directions to be used for tracking.
        pmf_threshold : float, [0., 1.]
            Used to remove direction from the probability mass function for
            selecting the tracking direction.
        probe_length : double
            The length of the probes. Shorter probe_length yields more
            dispersed fibers.
        probe_radius : double
            The radius of the probe. A large probe_radius helps mitigate noise
            in the pmf but it might make it harder to sample thin and intricate
            connections, also the boundary of fiber bundles might be eroded.
        probe_quality : integer,
            The quality of the probe. This parameter sets the number of
            segments to split the cylinder along the length of the
            probe (minimum=2).
        probe_count : integer
            The number of probes. This parameter sets the number of parallel
            lines used to model the cylinder (minimum=1).
        data_support_exponent : double
            Data support to the power dataSupportExponent is used for
            rejection sampling.

         """
        if not probe_length > 0:
            raise ValueError("probe_length must be greater than 0.")
        if not probe_radius >= 0:
            raise ValueError("probe_radius must be greater or equal to 0.")
        if not probe_quality >= 2:
            raise ValueError("probe_quality must be greater or equal than 2.")
        if not probe_count >= 1:
            raise ValueError("probe_count must be greater or equal than 1.")

        self.max_angle = max_angle
        self.probe_length = probe_length
        self.probe_radius = probe_radius
        self.probe_quality = probe_quality
        self.probe_count = probe_count
        self.probe_step_size = self.probe_length / (self.probe_quality - 1)
        self.probe_normalizer = 1.0 / float(self.probe_quality
                                            * self.probe_count)
        self.angular_separation = 2.0 * M_PI / float(self.probe_count)
        self.data_support_exponent = data_support_exponent

        self.k_small = 0.0001
        self.rejection_sampling_max_try = 100
        self.rejection_sampling_nbr_sample = 10 # Adaptively set in Trekker.

        ProbabilisticDirectionGetter.__init__(self, pmf_gen, max_angle, sphere,
                                       pmf_threshold=pmf_threshold, **kwargs)


    cdef void initialize_candidate(self, double[:] init_dir):
        """"Initialize the parallel transport frame.

        After initial position is set, a parallel transport frame is set using
        the initial direction (a walking frame, i.e., 3 orthonormal vectors,
        plus 2 scalars, i.e. k1 and k2).

        A point and parallel transport frame parametrizes a curve that is named
        the "probe". Using probe parameters (probe_length, probe_radius,
        probe_quality, probe_count), a short fiber bundle segment is modelled.

        Parameters
        ----------
        init_dir : np.array
            Initial tracking direction (tangent)

        """
        cdef double[3] position
        cdef int count
        cdef cnp.npy_intp i

        # Initialize Frame
        self.frame[0][0] = init_dir[0]
        self.frame[0][1] = init_dir[1]
        self.frame[0][2] = init_dir[2]
        random_perpendicular_vector(&self.frame[2][0], &self.frame[0][0])
        cross(&self.frame[1][0], &self.frame[2][0], &self.frame[0][0])
        self.k1, self.k2 = random_point_within_circle(self.max_curvature)

        self.last_val = 0

        if self.probe_count == 1:
            self.last_val = self.pmf_gen.get_pmf_value_c(self.position,
                                                         self.frame[0])
        else:
            for count in range(self.probe_count):
                for i in range(3):
                    position[i] = (self.position[i]
                                   + self.frame[1][i] * self.probe_radius
                                   * cos(count * self.angular_separation)
                                   * self.inv_voxel_size[i]
                                   +
                                   self.frame[2][i] * self.probe_radius
                                   * sin(count * self.angular_separation)
                                   * self.inv_voxel_size[i])

                self.last_val += self.pmf_gen.get_pmf_value_c(position,
                                                              self.frame[0])


    cdef void prepare_propagator(self, double arclength) nogil:
        """Prepare the propagator.

        The propagator used for transporting the moving frame forward.

        Parameters
        ----------
        arclength : double
            Arclenth, which is equivalent to step size along the arc

        """
        cdef double tmp_arclength

        if (fabs(self.k1) < self.k_small
            and fabs(self.k2) < self.k_small):
            self.propagator[0] = arclength
            self.propagator[1] = 0
            self.propagator[2] = 0
            self.propagator[3] = 1
            self.propagator[4] = 0
            self.propagator[5] = 0
            self.propagator[6] = 0
            self.propagator[7] = 0
            self.propagator[8] = 1
        else:
            if fabs(self.k1) < self.k_small:
                self.k1 = self.k_small
            if fabs(self.k2) < self.k_small:
                self.k2 = self.k_small

            tmp_arclength  = arclength * arclength / 2.0

            self.propagator[0] = arclength
            self.propagator[1] = self.k1 * tmp_arclength
            self.propagator[2] = self.k2 * tmp_arclength
            self.propagator[3] = (1
                                  - self.k2 * self.k2 * tmp_arclength
                                  - self.k1 * self.k1 * tmp_arclength)
            self.propagator[4] = self.k1 * arclength
            self.propagator[5] = self.k2 * arclength
            self.propagator[6] = -self.k2 * arclength
            self.propagator[7] = -self.k1 * self.k2 * tmp_arclength
            self.propagator[8] = (1 - self.k2 * self.k2 * tmp_arclength)


    cdef double calculate_data_support(self):
        """Calculates data support for the candidate probe."""

        cdef double fod_amp
        cdef double[3] position
        cdef double[3][3] frame
        cdef double[3] tangent
        cdef double[3] normal
        cdef double[3] binormal
        cdef double[3] new_position
        cdef double likelihood
        cdef int c, i, j, q

        self.prepare_propagator(self.probe_step_size)

        for i in range(3):
            position[i] = self.position[i]
            for j in range(3):
                frame[i][j] = self.frame[i][j]

        likelihood = self.last_val

        for q in range(1, self.probe_quality):
            for i in range(3):
                position[i] = \
                    (self.propagator[0] * frame[0][i] * self.voxel_size[i]
                     + self.propagator[1] * frame[1][i] * self.voxel_size[i]
                     + self.propagator[2] * frame[2][i] * self.voxel_size[i]
                     + position[i])
                tangent[i] = (self.propagator[3] * frame[0][i]
                              + self.propagator[4] * frame[1][i]
                              + self.propagator[5] * frame[2][i])
            normalize(&tangent[0])

            if q < (self.probe_quality - 1):
                for i in range(3):
                    binormal[i] = (self.propagator[6] * frame[0][i]
                                   + self.propagator[7] * frame[1][i]
                                   + self.propagator[8] * frame[2][i])
                cross(&normal[0], &binormal[0], &tangent[0])

                copy_point(&tangent[0], &frame[0][0])
                copy_point(&normal[0], &frame[1][0])
                copy_point(&binormal[0], &frame[2][0])

            if self.probe_count == 1:
                fod_amp = self.pmf_gen.get_pmf_value_c(position, tangent)
                fod_amp = fod_amp if fod_amp > self.pmf_threshold else 0
                self.last_val_cand = fod_amp
                likelihood += self.last_val_cand
            else:
                self.last_val_cand = 0
                if q == self.probe_quality-1:
                    for i in range(3):
                        binormal[i] = (self.propagator[6] * frame[0][i]
                                      + self.propagator[7] * frame[1][i]
                                      + self.propagator[8] * frame[2][i])
                    cross(&normal[0], &binormal[0], &tangent[0])

                for c in range(self.probe_count):
                    for i in range(3):
                        new_position[i] = (position[i]
                                           + normal[i] * self.probe_radius
                                           * cos(c * self.angular_separation)
                                           * self.inv_voxel_size[i]
                                           + binormal[i] * self.probe_radius
                                           * sin(c * self.angular_separation)
                                           * self.inv_voxel_size[i])
                    fod_amp = self.pmf_gen.get_pmf_value_c(new_position, tangent)
                    fod_amp = fod_amp if fod_amp > self.pmf_threshold else 0
                    self.last_val_cand += fod_amp

                likelihood += self.last_val_cand

        likelihood *= self.probe_normalizer
        if self.data_support_exponent != 1:
            likelihood = pow(likelihood, self.data_support_exponent)

        return likelihood


    cdef int initialize(self, double[:] seed_point, double[:] seed_direction):
        """Sample an initial curve by rejection sampling.

        Parameters
        ----------
        seed_point : double[3]
            Initial point
        seed_direction : double[3]
            Initial direction

        Returns
        -------
        status : int
            Returns 0 if the initialization was successful, or
            1 otherwise.
        """
        cdef double data_support = 0
        cdef double max_posterior = 0
        cdef int tries

        self.position[0] = seed_point[0]
        self.position[1] = seed_point[1]
        self.position[2] = seed_point[2]

        for tries in range(self.rejection_sampling_nbr_sample):
            self.initialize_candidate(seed_direction)
            data_support = self.calculate_data_support()
            if data_support > max_posterior:
                max_posterior = data_support

        # Compensation for underestimation of max posterior estimate
        max_posterior = pow(2.0 * max_posterior, self.data_support_exponent)

        # Initialization is successful if a suitable candidate can be sampled
        # within the trial limit
        for tries in range(self.rejection_sampling_max_try):
            self.initialize_candidate(seed_direction)
            if (random() * max_posterior <= self.calculate_data_support()):
                self.last_val = self.last_val_cand
                return 0

        return 1

    cdef int propagate(self):
        """Propagates the position by step_size amount. The propagation is using
        the parameters of the last candidate curve. Then, randomly generate
        curve parametrization from the current position. The walking frame
        is the same, only the k1 and k2 parameters are randomly picked.
        Rejection sampling is used to pick the next curve using the data
        support (likelihood).

        Returns
        -------
        status : int
            Returns 0 if the propagation was successful, or
            1 otherwise.
        """
        cdef double max_posterior = 0
        cdef double data_support = 0
        cdef double[3] tangent
        cdef int tries
        cdef cnp.npy_intp i

        self.prepare_propagator(self.step_size)

        for i in range(3):
            self.position[i] = \
                (self.propagator[0] * self.frame[0][i] * self.inv_voxel_size[i]
                + self.propagator[1] * self.frame[1][i] * self.inv_voxel_size[i]
                + self.propagator[2] * self.frame[2][i] * self.inv_voxel_size[i]
                + self.position[i])
            tangent[i] = (self.propagator[3] * self.frame[0][i]
                          + self.propagator[4] * self.frame[1][i]
                          + self.propagator[5] * self.frame[2][i])
            self.frame[2][i] = (self.propagator[6] * self.frame[0][i]
                                + self.propagator[7] * self.frame[1][i]
                                + self.propagator[8] * self.frame[2][i])
        normalize(&tangent[0])
        cross(&self.frame[1][0], &self.frame[2][0], &tangent[0])
        normalize(&self.frame[1][0])
        cross(&self.frame[2][0], &tangent[0], &self.frame[1][0])
        self.frame[0][0] = tangent[0]
        self.frame[0][1] = tangent[1]
        self.frame[0][2] = tangent[2]

        for tries in range(self.rejection_sampling_nbr_sample):
            self.k1, self.k2 = random_point_within_circle(self.max_curvature)
            data_support = self.calculate_data_support()
            if data_support > max_posterior:
                max_posterior = data_support

        # Compensation for underestimation of max posterior estimate
        max_posterior = pow(2.0 * max_posterior, self.data_support_exponent)

        # Propagation is successful if a suitable candidate can be sampled
        # within the trial limit
        for tries in range(self.rejection_sampling_max_try):
            self.k1, self.k2 = random_point_within_circle(self.max_curvature)
            if random() * max_posterior <= self.calculate_data_support():
                self.last_val = self.last_val_cand
                return 0

        return 1


    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    cpdef tuple generate_streamline(self,
                                    double[::1] seed,
                                    double[::1] dir,
                                    double[::1] voxel_size,
                                    double step_size,
                                    StoppingCriterion stopping_criterion,
                                    cnp.float_t[:, :] streamline,
                                    StreamlineStatus stream_status,
                                    int fixedstep):
        cdef:
            cnp.npy_intp i
            cnp.npy_intp len_streamlines = streamline.shape[0]
            double average_voxel_size = 0

        if not fixedstep > 0:
           raise ValueError("PTT only supports fixed step size.")

        self.step_size = step_size
        for i in range(3):
            self.voxel_size[i] = voxel_size[i]
            self.inv_voxel_size[i] = 1 / voxel_size[i]
            average_voxel_size += voxel_size[i] / 3

        # convert max_angle from degrees to radians
        self.max_curvature = 1 / min_radius_curvature_from_angle(
            self.max_angle * M_PI / 180.0, self.step_size / average_voxel_size)

        copy_point(&seed[0], &streamline[0,0])
        i = 0
        stream_status = TRACKPOINT

        if not self.initialize(seed, dir):
            # the initialization was successful
            for i in range(1, len_streamlines):
                if self.propagate():
                    # the propagation failed
                    break
                copy_point(<double *>&self.position[0], &streamline[i, 0])
                stream_status = stopping_criterion\
                    .check_point_c(<double * > &self.position[0])
                if stream_status == TRACKPOINT:
                    continue
                elif (stream_status == ENDPOINT or
                    stream_status == INVALIDPOINT or
                    stream_status == OUTSIDEIMAGE):
                    break
            else:
                # maximum length has been reached, return everything
                i = streamline.shape[0]
        return i, stream_status

