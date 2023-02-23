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

    cdef bint         initialized           # True is initialization was done. This is used for flipping.
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


    # First set the (initial) position of the parallel transport frame (PTF), i.e. set the seed point
    cdef void set_position(self, double[:] pos):
        self.position[0] = pos[0]
        self.position[1] = pos[1]
        self.position[2] = pos[2]


    cdef double get_initial_candidate(self, double[:] init_dir):
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
        cdef double fod_amp
        cdef double[3] pp

        self.get_random_frame(init_dir)
        (self.k1_candidate, self.k2_candidate) = sample_random_point_within_disk(self.min_radius_curvature)

        self.k1 = self.k1_candidate
        self.k2 = self.k2_candidate

        self.last_val = 0

        if self.probe_count == 1:
            fod_amp = self.pmf_gen.get_pmf_value(self.position, self.frame[0])
            self.last_val = fod_amp
        else:
            for c in range(self.probe_count):
                for i in range(3):
                    pp[i] = (self.position[i] +
                             self.frame[1][i] * self.probe_radius *
                             cos(c * self.angular_separation) +
                             self.frame[2][i] * self.probe_radius *
                             sin(c * self.angular_separation))

                fod_amp = self.pmf_gen.get_pmf_value(pp, self.frame[0])
                self.last_val += fod_amp

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
            T[i] = (self.propagator[3] * self.frame[0][i] + self.propagator[4] * self.frame[1][i]
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


    cdef double get_candidate(self):
        """Pick a random curve parametrization by using current position.

        The walking frame (frame) is same, only the k1 and k2 are randomly picked.
        This was a smooth curve is sampled. This function does NOT pick the
        next curve. It only returns the datasupport (likelihood) value for the
        randomly picked candidate.

        """
        self.k1_candidate, self.k2_candidate = sample_random_point_within_disk(self.min_radius_curvature)
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


    cdef void prepare_propagator(self, double t):
        """Prepare the propagator.

        The propagator used for transporting the moving frame forward.

        Parameters
        ----------
        t : double
            Arclenth, which is equivalent to step size along the arc

        """
        cdef double tto2

        if (abs(self.k1_candidate) < 0.0001) & (abs(self.k2_candidate) < 0.0001):
            self.propagator[0] = t
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

            tto2  = t * t / 2.0

            self.propagator[0] = t
            self.propagator[1] = self.k1_candidate * tto2
            self.propagator[2] = self.k2_candidate * tto2
            self.propagator[3] = (1 - self.k2_candidate * self.k2_candidate * tto2 -
                          self.k1_candidate * self.k1_candidate * tto2)
            self.propagator[4] = self.k1_candidate * t
            self.propagator[5] = self.k2_candidate * t
            self.propagator[6] = -self.k2_candidate * t
            self.propagator[7] = -self.k1_candidate * self.k2_candidate * tto2
            self.propagator[8] = 1 - self.k2_candidate * self.k2_candidate * tto2

    cdef double calculate_data_support(self):
        """Calculates data support for the candidate probe"""

        cdef double fod_amp
        cdef double[3] _p
        cdef double[3][3] _F
        cdef double[3] _T = [0,0,0]
        cdef double[3] _N1 = [0,0,0]
        cdef double[3] _N2 = [0,0,0]
        cdef double[3] pp

        self.prepare_propagator(self.probe_step_size)

        for i in range(3):
            _p[i] = self.position[i]
            for j in range(3):
                _F[i][j] = self.frame[i][j]

        self.likelihood = self.last_val

        for q in range(1, int(self.probe_quality)):

            for i in range(3):
                _p[i] = (self.propagator[0] * _F[0][i] +  self.propagator[1] * _F[1][i] +
                        self.propagator[2] * _F[2][i] + _p[i])
                _T[i] = (self.propagator[3] * _F[0][i] + self.propagator[4]*_F[1][i]  +
                         self.propagator[5] * _F[2][i])

            normalize(_T)

            if q < (self.probe_quality - 1):

                for i in range(3):
                    _N2[i] = (self.propagator[6] * _F[0][i] +  
                              self.propagator[7]*_F[1][i] +
                              self.propagator[8]*_F[2][i])

                cross(_N1,_N2,_T)

                for i in range(3):
                    _F[0][i] =  _T[i]
                    _F[1][i] = _N1[i]
                    _F[2][i] = _N2[i]

            if self.probe_count == 1:
                fod_amp = self.pmf_gen.get_pmf_value(_p, _T)
                self.last_val_cand = fod_amp
                self.likelihood += self.last_val_cand
            else:
                self.last_val_cand = 0

                if q == self.probe_quality-1:
                    for i in range(3):
                        _N2[i] = (self.propagator[6] * _F[0][i] +
                                  self.propagator[7] * _F[1][i] + self.propagator[8] * _F[2][i])
                    cross(_N1,_N2,_T)

                for c in range(int(self.probe_count)):

                    for i in range(3):
                        pp[i] = (_p[i] + _N1[i] * self.probe_radius *
                                 cos(c * self.angular_separation) +
                                 _N2[i] * self.probe_radius *
                                 sin(c * self.angular_separation))

                    fod_amp = self.pmf_gen.get_pmf_value(pp, _T)
                    self.last_val_cand += fod_amp

                self.likelihood += self.last_val_cand

        self.likelihood *= self.probe_normalizer
        if self.data_support_exponent != 1:
            self.likelihood  = pow(self.likelihood, self.data_support_exponent)

        return self.likelihood


    cdef StreamlineStatus reinitialize(self, double[:] _seed_point, double[:] _seed_direction):
        """Sample an initial curve by rejection sampling.

        Parameters
        ----------
        _seed_point : double[3]
            Initial point

        _seed_direction : double[3]
            Initial direction

        """

        # Reset initialization
        self.initialized = False

        # Set initial position
        self.set_position(_seed_point)

        # Initial max estimate
        cdef double dataSupport  = 0
        cdef double posteriorMax = 0

        cdef int tries
        for tries in range(1000):
            dataSupport = self.get_initial_candidate(_seed_direction)
            if dataSupport > posteriorMax:
                posteriorMax = dataSupport

        # Compensation for underestimation of max posterior estimate
        posteriorMax = pow(2.0*posteriorMax, self.data_support_exponent)

        # Initialization is successful if a suitable candidate can be sampled
        # within the trial limit
        for tries in range(1000):
            if random() * posteriorMax <= self.get_initial_candidate(_seed_direction):
                self.last_val = self.last_val_cand

                for i in range(3):
                    self.init_position[i] = self.position[i]
                    self.init_frame[0][i] = -self.frame[0][i]
                    self.init_frame[1][i] = -self.frame[1][i]
                    self.init_frame[2][i] = self.frame[2][i]

                self.init_k1 = -self.k1
                self.init_k2 = self.k2
                self.initialized = True

                return TRACKPOINT
        return NODATASUPPORT

    cdef StreamlineStatus propagate(self):
        """Takes a step forward along the chosen candidate"""

        self.prepare_propagator(self.step_size)

        self.walk()

        # Initial max estimate
        cdef double dataSupport  = 0
        cdef double posteriorMax = 0

        cdef int tries
        # This is adaptively set in Trekker. But let's ignore it for now since
        # implementation of that is challenging.
        for tries in range(20): 
            dataSupport = self.get_candidate()
            if dataSupport > posteriorMax:
                posteriorMax = dataSupport

        # Compensation for underestimation of max posterior estimate
        posteriorMax = pow(2.0 * posteriorMax,
                           self.data_support_exponent)

        # Propagation is successful if a suitable candidate can be sampled 
        # within the trial limit
        for tries in range(1000):
            if random() * posteriorMax <= self.get_candidate():
                self.last_val = self.last_val_cand
                return TRACKPOINT

        return NODATASUPPORT


    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    cpdef tuple generate_streamline(self,
                                    double[::1] seed,
                                    double[::1] dir,
                                    # TODO: Move step_size, voxel_size, 
                                    # fixed_Step variable
                                    # should be outside of generate_streamline
                                    # use it inside the init
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

        copy_point(&seed[0], point)
        copy_point(&seed[0], &streamline[0,0])

        stream_status = TRACKPOINT

        # An initial direction can be provided to the tracker for initialization
        # But for now let's assume no initialiation is done, i.e., [0,0,0]
        cdef double[3] seed_direction = [0,0,0]
        cdef double[3] seed_point = [seed[0], seed[1], seed[2]]

        # This step only initializes or flips the frame at the seed point.
        # It does not do any propagation, i.e., tracker is at the seed point 
        #and it is ready to propagate
        # Initialization basically checks whether the tracker has data support 
        #to move or propagate forward (but it does not move it)
        # TO DO - temporarely disable: foward/backward stremline segments are 
        #generated independently
        ###if self.initialized:
        ###    self.flip()
        ###else:

        stream_status = self.reinitialize(seed_point, seed_direction)

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
