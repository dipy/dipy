# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

"""
Implementation of parallel transport tractography (PTT).
"""

import numpy as np
cimport numpy as cnp

from dipy.direction.probabilistic_direction_getter cimport ProbabilisticDirectionGetter
from dipy.direction.peaks import peak_directions, default_sphere
from dipy.direction.pmf cimport PmfGen, SimplePmfGen, SHCoeffPmfGen
from dipy.utils.fast_numpy cimport cumsum, where_to_insert, copy_point
from dipy.tracking.stopping_criterion cimport (StreamlineStatus,
                                               StoppingCriterion,
                                               TRACKPOINT,
                                               ENDPOINT,
                                               OUTSIDEIMAGE,
                                               INVALIDPOINT,
                                               PYERROR,
                                               NODATASUPPORT)
from dipy.tracking.direction_getter cimport _fixed_step, _step_to_boundary
from dipy.utils.fast_numpy cimport random, norm, normalize, dot, cross
from dipy.tracking.utils import min_radius_curvature_from_angle

from libc.math cimport M_PI, pow, sin, cos


cdef double unidis_m1_p1():
    """Picks a random number between -1 and 1

    Returns
    -------
    double
        random number
    """
    return 2.0 * random() - 1.0


cdef void sample_unit_random_vector(double[:] out):
    """Generate a unit random vector

    Parameters
    ----------
    out : double[3]
        input vector

    Notes
    -----
    Overwrites the input
    """
    out[0] = unidis_m1_p1()
    out[1] = unidis_m1_p1()
    out[2] = unidis_m1_p1()
    normalize(out)


cdef void sample_unit_random_perpendicular_vector(double[:] out,double[:] inp):
    """Generate a unit random perpendicular vector

    Parameters
    ----------
    out : double[3]
        input vector

    inp : double[3]
        input vector

    Notes
    -----
    Overwrites the first argument
    """
    cdef double[3] tmp
    sample_unit_random_vector(tmp)
    cross(out,inp,tmp)
    normalize(out)


cdef (double, double) sample_random_point_within_disk(double r):
    """Generate a random point within a disk

    Parameters
    ----------
    r : double
        The radius of the disk

    Returns
    -------
    x : double
        x coordinate of the random point

    y : double
        y coordinate of the random point

    """
    cdef double x = 1
    cdef double y = 1
    while ((x * x + y * y) > 1):
        x = unidis_m1_p1()
        y = unidis_m1_p1()
    return (r * x, r * y)


cdef class PTTDirectionGetter(ProbabilisticDirectionGetter):
    """Parallel Transport Tractography direction getter.
    """

    cdef double[3]    position              # Last position
    cdef double[3][3] frame                 
    cdef double       k1                    # k1 value of the current frame
    cdef double       k2                    # k2 value of the current frame
    cdef double       k1_candidate          
    cdef double       k2_candidate          
    cdef double       likelihood            # Likelihood of the next candidate frame constructed with k1_candidate and k2_candidate

    cdef double[3]    init_position         
    cdef double[3][3] init_frame            
    cdef double       init_k1               
    cdef double       init_k2               

    # The following variables are mainly used for code optimization
    cdef double[9]    propagator
    cdef double       angular_separation
    cdef double       probe_step_size
    cdef double       probe_normalizer
    cdef double       last_val
    cdef double       last_val_cand
    cdef double       init_last_val
    cdef double       max_angle
    cdef double       min_radius_curvature
    cdef double       probe_length
    cdef double       probe_radius
    cdef double       probe_quality
    cdef double       probe_count
    cdef double       data_support_exponent
    cdef double       step_size

    # For each streamline, create a new PTF object with tracking parameters
    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=None,
                 probe_length=1/2, probe_radius=0, probe_quality=3, 
                 probe_count=1, data_support_exponent=1,
                 **kwargs):
        """Direction getter from a pmf generator.

        Parameters
        ----------
        pmf_gen : PmfGen
            Used to get probability mass function for selecting tracking
            directions.
        max_angle : float, [0, 90]
            Is used to set the upper limits for the k1 and k2 parameters
            of parallel transport frame (min_radius_curvature)
        sphere : Sphere
            The set of directions to be used for tracking.
        pmf_threshold : None
            Not used for PTT
        probe_length : double
            ptt uses probes for estimating future propagation steps.
            A probe is a short, cylinderical model of the connecting segment.
            Shorter probe_length yields more dispersed fibers.
        probe_radius : double
            ptt uses probes for estimating future propagation steps.
            A probe is a short, cylinderical model of the connecting segment.
            A large probe_radius helps mitigate noise in the fODF
            but it might make it harder to sample thin and intricate connections,
            also the boundary of fiber bundles might be eroded.
        probe_quality : integer
            ptt uses probes for estimating future propagation steps.
            A probe is a short, cylinderical model of the connecting segment.
            This parameter sets the number of segments to split the cylinder
            along the length of the probe.
        probe_count : integer
            ptt uses probes for estimating future propagation steps.
            A probe is a short, cylinderical model of the connecting segment.
            This parameter sets the number of parallel lines used to model the cylinder.
        data_support_exponent : double
            Data support to the power dataSupportExponent is used for rejection sampling.

        See also
        --------
        dipy.direction.peaks.peak_directions

        """

        # TODO: review max_angle vs min_radius_curvature.
        self.max_angle = max_angle
        self.min_radius_curvature = 0
        self.probe_length = probe_length
        self.probe_radius = probe_radius
        self.probe_quality = probe_quality
        self.probe_count = probe_count
        self.data_support_exponent = data_support_exponent
        ProbabilisticDirectionGetter.__init__(self, pmf_gen, max_angle, sphere,
                                       pmf_threshold, **kwargs)

        # Initialize this PTF's internal tracking parameters
        self.angular_separation = 2.0 * M_PI / float(self.probe_count)
        self.probe_step_size = self.probe_length / (self.probe_quality - 1)
        self.probe_normalizer = 1.0 / float(self.probe_quality * self.probe_count)


    cdef double set_initial_candidate(self, double[:] init_dir):
        """"Return the likelihood value for a randomly picked candidate.

        After initial position is set, a random PTF (a walking frame, i.e.,
        3 orthonormal vectors (frame), plus 2 scalars, i.e., k1 and k2) is set
        with this function. Optionally, the tangential component of PTF can be
        user provided with the input initDir parameter.
        A point + PTF parametrizes a curve that is named the "probe". Using
        probe parameters (probe_length, probe_radius, probe_quality,
        probe_count), a short fiber bundle segment is modelled.

        Parameters
        ----------
        init_dir : np.array
            Use initdir=[0,0,0] if initDir is not available.

        Notes
        -----
        This function does NOT pick the initial curve. It only returns the
        datasupport (likelihood) value for a randomly picked candidate.

        """
        cdef double[3] position

        self.get_random_frame(init_dir)
        (self.k1_candidate, self.k2_candidate)\
                = sample_random_point_within_disk(self.min_radius_curvature)

        self.k1 = self.k1_candidate
        self.k2 = self.k2_candidate

        self.last_val = 0

        if self.probe_count == 1:
            self.last_val = self.pmf_gen.get_pmf_value(self.position,
                                                       self.frame[0])
        else:
            for count in range(self.probe_count):
                for i in range(3):
                    position[i] = (self.position[i] +
                                   self.frame[1][i] * self.probe_radius *
                                   cos(count * self.angular_separation) +
                                   self.frame[2][i] * self.probe_radius *
                                   sin(count * self.angular_separation))

                self.last_val += self.pmf_gen.get_pmf_value(position, 
                                                            self.frame[0])

        self.init_last_val = self.last_val

        return self.calculate_data_support()


    cdef void walk(self):
        """Propagates the last position (p) by step_size amount.
        The progation is using the parameters of the last candidate.
        """
        self.k1 = self.k1_candidate
        self.k2 = self.k2_candidate

        cdef double[3] T

        for i in range(3):
            self.position[i] = (self.propagator[0] * self.frame[0][i] +
                                self.propagator[1] * self.frame[1][i] +
                                self.propagator[2] * self.frame[2][i] + 
                                self.position[i])
            T[i] = (self.propagator[3] * self.frame[0][i] 
                    + self.propagator[4] * self.frame[1][i]
                    + self.propagator[5] * self.frame[2][i])
            self.frame[2][i] = (self.propagator[6] * self.frame[0][i] +
                                self.propagator[7] * self.frame[1][i] +
                                self.propagator[8] * self.frame[2][i])

        normalize(T)
        cross(self.frame[1], self.frame[2], T)
        normalize(self.frame[1])
        cross(self.frame[2], T, self.frame[1])

        self.frame[0][0] = T[0]
        self.frame[0][1] = T[1]
        self.frame[0][2] = T[2]

        self.likelihood = 0.0


    cdef double set_candidate(self):
        """Pick a random curve parametrization by using current position.

        The walking frame (frame) is same, only the k1 and k2 are randomly 
        picked.This function does NOT pick the
        next curve. It only returns the datasupport (likelihood) value for the
        randomly picked candidate.

        """
        (self.k1_candidate, self.k2_candidate)\
                = sample_random_point_within_disk(self.min_radius_curvature)
        return self.calculate_data_support()


    cdef void flip(self):
        """"Copy PTF parameters then flips the curve.

        This function can be used after the initial curve is picked in order
        to save a copy of the curve for tracking towards the other side.

        """

        for i in range(3):
            self.position[i] = self.init_position[i]
            for j in range(3):
                self.frame[i][j] = self.init_frame[i][j]

        self.k1_candidate = self.init_k1
        self.k2_candidate = self.init_k2
        self.last_val = self.init_last_val

        return


    cdef void get_random_frame(self,double[:] _dir):
        """Randomly generate 3 unit vectors that are orthogonal to each other.

        This is used for initializing the moving frame of the tracker
        Optionally, the initial direction, i.e., tangent, can also be provided
        if norm(_dir.size)==0, then the tangent will also be a random vector

        Parameters
        ----------
        _dir : double[3]
            The optional initial direction (tangent) of the parallel transport frame

        """
        if norm(_dir) == 0:
            sample_unit_random_vector(self.frame[0])
        else:
            self.frame[0][0] = _dir[0]
            self.frame[0][1] = _dir[1]
            self.frame[0][2] = _dir[2]

        sample_unit_random_perpendicular_vector(self.frame[2],self.frame[0])
        cross(self.frame[1],self.frame[2],self.frame[0])


    cdef void prepare_propagator(self, double arclength):
        """Prepare the propagator.

        The propagator used for transporting the moving frame forward.

        Parameters
        ----------
        arclength : double
            Arclenth, which is equivalent to step size along the arc

        """
        cdef double tmp_arclength

        if abs(self.k1_candidate) < 0.0001 and abs(self.k2_candidate) < 0.0001:
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
            if abs(self.k1_candidate) < 0.0001:
                self.k1_candidate = 0.0001
            if abs(self.k2_candidate) < 0.0001:
                self.k2_candidate = 0.0001

            tmp_arclength  = arclength * arclength / 2.0

            self.propagator[0] = arclength
            self.propagator[1] = self.k1_candidate * tmp_arclength
            self.propagator[2] = self.k2_candidate * tmp_arclength
            self.propagator[3] = (1 
                                  - self.k2_candidate * self.k2_candidate 
                                  * tmp_arclength 
                                  - self.k1_candidate * self.k1_candidate 
                                  * tmp_arclength)
            self.propagator[4] = self.k1_candidate * arclength
            self.propagator[5] = self.k2_candidate * arclength
            self.propagator[6] = -self.k2_candidate * arclength
            self.propagator[7] = -self.k1_candidate * self.k2_candidate * tmp_arclength
            self.propagator[8] = 1 - self.k2_candidate * self.k2_candidate * tmp_arclength

    cdef double calculate_data_support(self):
        """Calculates data support for the candidate probe"""

        cdef double fod_amp
        cdef double[3] position
        cdef double[3][3] frame
        cdef double[3] _T = [0,0,0]
        cdef double[3] _N1 = [0,0,0]
        cdef double[3] _N2 = [0,0,0]
        cdef double[3] pp

        self.prepare_propagator(self.probe_step_size)

        for i in range(3):
            position[i] = self.position[i]
            for j in range(3):
                frame[i][j] = self.frame[i][j]

        self.likelihood = self.last_val

        for q in range(1, int(self.probe_quality)):

            for i in range(3):
                position[i] = (self.propagator[0] * frame[0][i] 
                               + self.propagator[1] * frame[1][i] 
                               + self.propagator[2] * frame[2][i] + position[i])
                _T[i] = (self.propagator[3] * frame[0][i]
                         + self.propagator[4] * frame[1][i]
                         + self.propagator[5] * frame[2][i])

            normalize(_T)

            if q < (self.probe_quality - 1):
                for i in range(3):
                    _N2[i] = (self.propagator[6] * frame[0][i] +  
                              self.propagator[7] * frame[1][i] +
                              self.propagator[8] * frame[2][i])

                cross(_N1,_N2,_T)

                for i in range(3):
                    frame[0][i] =  _T[i]
                    frame[1][i] = _N1[i]
                    frame[2][i] = _N2[i]

            if self.probe_count == 1:
                fod_amp = self.pmf_gen.get_pmf_value(position, _T)
                self.last_val_cand = fod_amp
                self.likelihood += self.last_val_cand
            else:
                self.last_val_cand = 0

                if q == self.probe_quality-1:
                    for i in range(3):
                        _N2[i] = (self.propagator[6] * frame[0][i] 
                                  + self.propagator[7] * frame[1][i]
                                  + self.propagator[8] * frame[2][i])
                    cross(_N1,_N2,_T)

                for c in range(int(self.probe_count)):

                    for i in range(3):
                        pp[i] = (position[i] 
                                 + _N1[i] * self.probe_radius
                                 * cos(c * self.angular_separation)
                                 + _N2[i] * self.probe_radius
                                 * sin(c * self.angular_separation))

                    fod_amp = self.pmf_gen.get_pmf_value(pp, _T)
                    self.last_val_cand += fod_amp

                self.likelihood += self.last_val_cand

        self.likelihood *= self.probe_normalizer
        if self.data_support_exponent != 1:
            self.likelihood  = pow(self.likelihood, self.data_support_exponent)

        return self.likelihood


    cdef StreamlineStatus initialize(self, 
                                       double[:] seed_point, 
                                       double[:] seed_direction):
        """Sample an initial curve by rejection sampling.

        Parameters
        ----------
        seed_point : double[3]
            Initial point
        seed_direction : double[3]
            Initial direction

        """

        # Set initial position
        self.position[0] = seed_point[0]
        self.position[1] = seed_point[1]
        self.position[2] = seed_point[2]

        # Initial max estimate
        cdef double data_support = 0
        cdef double max_posterior = 0

        cdef int tries
        for tries in range(1000):
            data_support = self.set_initial_candidate(seed_direction)
            if data_support > max_posterior:
                posteriorMax = data_support

        # Compensation for underestimation of max posterior estimate
        max_posterior = pow(2.0 * max_posterior, self.data_support_exponent)

        # Initialization is successful if a suitable candidate can be sampled
        # within the trial limit
        for tries in range(1000):
            if (random() * max_posterior 
                    <= self.set_initial_candidate(seed_direction)):
                self.last_val = self.last_val_cand

                for i in range(3):
                    self.init_position[i] = self.position[i]
                    self.init_frame[0][i] = -self.frame[0][i]
                    self.init_frame[1][i] = -self.frame[1][i]
                    self.init_frame[2][i] = self.frame[2][i]

                self.init_k1 = -self.k1
                self.init_k2 = self.k2

                return TRACKPOINT
        return NODATASUPPORT

    cdef StreamlineStatus propagate(self):
        """Takes a step forward along the chosen candidate"""

        self.prepare_propagator(self.step_size)

        self.walk()

        # Initial max estimate
        cdef double data_support = 0
        cdef double posteriorMax = 0

        cdef int tries
        # This is adaptively set in Trekker. But let's ignore it for now since
        # implementation of that is challenging.
        for tries in range(20): 
            data_support = self.set_candidate()
            if data_support > posteriorMax:
                posteriorMax = data_support

        # Compensation for underestimation of max posterior estimate
        posteriorMax = pow(2.0 * posteriorMax, self.data_support_exponent)

        # Propagation is successful if a suitable candidate can be sampled 
        # within the trial limit
        for tries in range(1000):
            if random() * posteriorMax <= self.set_candidate():
                self.last_val = self.last_val_cand
                return TRACKPOINT

        return NODATASUPPORT


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
                                    int fixedstep
                                    ):
        cdef:
            cnp.npy_intp i
            cnp.npy_intp len_streamlines = streamline.shape[0]
            double point[3]
            double voxdir[3]
            void (*step)(double*, double*, double) nogil

        self.step_size = step_size
        self.min_radius_curvature = min_radius_curvature_from_angle(
            np.deg2rad(self.max_angle), self.step_size)

        copy_point(&seed[0], &streamline[0,0]) 
        i = 0
        stream_status = self.initialize(seed, dir)
        # If initialization is successful than the tracker propagates.
        # Propagation first pushes the moving frame forward.
        # Then the propagator sets the frame for the next propagation step 
        # (here data support for the next frame is checked)
        # i.e. self.positon is always supported by data (to be be added to the 
        # streamline unless there is another reason such a termination ROI etc.)
        # If propagator returns NODATASUPPORT, further propagation is impossible
        # If propagator returns TRACKPOINT, self.position can be pushed forward
        # Propagator works like this because it does not check termination 
        # conditions. If self.position reaches a termination ROI, 
        # then it should not be appended to the streamline.
        if stream_status == TRACKPOINT:
            for i in range(1, len_streamlines):
                stream_status = self.propagate()
                if stream_status == NODATASUPPORT:
                    break
                copy_point(<double *>&self.position[0], &streamline[i, 0])
                stream_status = stopping_criterion.check_point_c(<double * > &self.position[0])
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
