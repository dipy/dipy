import numpy as np
cimport numpy as cnp
cimport cython

from safe_openmp cimport have_openmp
from cython.parallel import parallel, prange, threadid

from scipy.spatial import KDTree
from scipy.interpolate import interp1d

from dipy.utils.omp import determine_num_threads
from dipy.utils.omp cimport set_num_threads, restore_default_num_threads

cdef class FBCMeasures:

    cdef int [:] streamline_length
    cdef double [:, :, :] streamline_points
    cdef double [:, :] streamlines_lfbc
    cdef double [:] streamlines_rfbc

    def __init__(self,
                 streamlines,
                 kernel,
                 min_fiberlength=10,
                 max_windowsize=7,
                 num_threads=None,
                 verbose=False):
        """ Compute the fiber to bundle coherence measures for a set of
        streamlines.

        Parameters
        ----------
        streamlines : list
            A collection of streamlines, each n by 3, with n being the number of
            nodes in the fiber.
        kernel : Kernel object
            A diffusion kernel object created from EnhancementKernel.
        min_fiberlength : int
            Fibers with fewer points than minimum_length are excluded from FBC
            computation.
        max_windowsize : int
            The maximal window size used to calculate the average LFBC region
        num_threads : int, optional
            Number of threads to be used for OpenMP parallelization. If None
            (default) the value of OMP_NUM_THREADS environment variable is used
            if it is set, otherwise all available threads are used. If < 0 the
            maximal number of threads minus |num_threads + 1| is used (enter -1
            to use as many threads as possible). 0 raises an error.
        verbose : boolean
            Enable verbose mode.

        References
        ----------
        [Meesters2016_HBM] S. Meesters, G. Sanguinetti, E. Garyfallidis,
                           J. Portegies, P. Ossenblok, R. Duits. (2016) Cleaning
                           output of tractography via fiber to bundle coherence,
                           a new open source implementation. Human Brain Mapping
                           conference 2016.
        [Portegies2015b] J. Portegies, R. Fick, G. Sanguinetti, S. Meesters,
                         G.Girard, and R. Duits. (2015) Improving Fiber Alignment
                         in HARDI by Combining Contextual PDE flow with
                         Constrained Spherical Deconvolution. PLoS One.
        """
        self.compute(streamlines,
                     kernel,
                     min_fiberlength,
                     max_windowsize,
                     num_threads,
                     verbose)

    def get_points_rfbc_thresholded(self, threshold, emphasis=.5, verbose=False):
        """ Set a threshold on the RFBC to remove spurious fibers.

        Parameters
        ----------
        threshold : float
            The threshold to set on the RFBC, should be within 0 and 1.
        emphasis : float
            Enhances the coloring of the fibers by LFBC. Increasing emphasis
            will stress spurious fibers by logarithmic weighting.
        verbose : boolean
            Prints info about the found RFBC for the set of fibers such as
            median, mean, min and max values.

        Returns
        -------
        output : tuple with 3 elements
            The output contains:
            1) a collection of streamlines, each n by 3, with n
            being the number of nodes in the fiber that remain after filtering
            2) the r,g,b values of the local fiber to bundle coherence (LFBC)
            3) the relative fiber to bundle coherence (RFBC)
        """
        if verbose:
            print("median RFBC: " + str(np.median(self.streamlines_rfbc)))
            print("mean RFBC: " + str(np.mean(self.streamlines_rfbc)))
            print("min RFBC: " + str(np.min(self.streamlines_rfbc)))
            print("max RFBC: " + str(np.max(self.streamlines_rfbc)))

        # logarithmic transform of color values to emphasize spurious fibers
        minval = np.nanmin(self.streamlines_lfbc)
        maxval = np.nanmax(self.streamlines_lfbc)
        lfbc_log = np.log((self.streamlines_lfbc - minval) /
                          (maxval - minval + 10e-10) + emphasis)
        minval = np.nanmin(lfbc_log)
        maxval = np.nanmax(lfbc_log)
        lfbc_log = (lfbc_log - minval) / (maxval - minval)

        # define color interpolation functions
        x = np.linspace(0, 1, num=4, endpoint=True)
        r = np.array([1, 1, 0, 0])
        g = np.array([1, 0, 0, 1])
        b = np.array([0, 0, 1, 1])
        fr = interp1d(x, r, bounds_error=False, fill_value=0)
        fg = interp1d(x, g, bounds_error=False, fill_value=0)
        fb = interp1d(x, b, bounds_error=False, fill_value=0)

        # select fibers above the RFBC threshold
        streamline_out = []
        color_out = []
        rfbc_out = []
        for i in range((self.streamlines_rfbc).shape[0]):
            rfbc = self.streamlines_rfbc[i]
            lfbc = lfbc_log[i]
            if rfbc > threshold:
                fiber = np.array(self.streamline_points[i])
                fiber = fiber[0:self.streamline_length[i] - 1]
                streamline_out.append(fiber)

                rfbc_out.append(rfbc)

                lfbc = lfbc[0:self.streamline_length[i] - 1]
                lfbc_colors = np.transpose([fr(lfbc), fg(lfbc), fb(lfbc)])
                color_out.append(lfbc_colors.tolist())

        return streamline_out, color_out, rfbc_out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void compute(self,
                      py_streamlines,
                      kernel,
                      min_fiberlength,
                      max_windowsize,
                      num_threads=None,
                      verbose=False):
        """ Compute the fiber to bundle coherence measures for a set of
        streamlines.

        Parameters
        ----------
        py_streamlines : list
            A collection of streamlines, each n by 3, with n being the number of
            nodes in the fiber.
        kernel : Kernel object
            A diffusion kernel object created from EnhancementKernel.
        min_fiberlength : int
            Fibers with fewer points than minimum_length are excluded from FBC
            computation.
        max_windowsize : int
            The maximal window size used to calculate the average LFBC region
        num_threads : int, optional
            Number of threads to be used for OpenMP parallelization. If None
            (default) the value of OMP_NUM_THREADS environment variable is used
            if it is set, otherwise all available threads are used. If < 0 the
            maximal number of threads minus |num_threads + 1| is used (enter -1
            to use as many threads as possible). 0 raises an error.
        verbose : boolean
            Enable verbose mode.
        """
        cdef:
            cnp.npy_intp num_fibers, max_length, dim
            double [:, :, :] streamlines
            int [:] streamlines_length
            double [:, :, :] streamlines_tangent
            int [:, :] streamlines_nearestp
            double [:, :] streamline_scores
            cnp.npy_intp line_id = 0
            cnp.npy_intp point_id = 0
            cnp.npy_intp line_id2 = 0
            cnp.npy_intp point_id2 = 0
            cnp.npy_intp dims
            double score
            double [:] score_mp
            int [:] xd_mp, yd_mp, zd_mp
            cnp.npy_intp xd, yd, zd, N, hn
            double [:, :, :, :, ::1] lut
            cnp.npy_intp threads_to_use = -1

        threads_to_use = determine_num_threads(num_threads)
        set_num_threads(threads_to_use)

        # if the fibers are too short FBC measures cannot be applied,
        # remove these.
        streamlines_length = np.array([x.shape[0] for x in py_streamlines],
                                      dtype=np.int32)
        min_length = min(streamlines_length)
        if min_length < min_fiberlength:
            print("The minimum fiber length is 10 points. \
                    Shorter fibers were found and removed.")
            py_streamlines = [x for x in py_streamlines
                              if x.shape[0] >= min_fiberlength]
            streamlines_length = np.array([x.shape[0] for x in py_streamlines],
                                          dtype=np.int32)
        num_fibers = len(py_streamlines)
        self.streamline_length = streamlines_length
        max_length = max(streamlines_length)

        dim = 3

        # get lookup table info
        lut = kernel.get_lookup_table()
        N = lut.shape[2]
        hn = (N-1) / 2

        # prepare numpy arrays for speed
        streamlines = np.zeros((num_fibers, max_length, dim),
                               dtype=np.float64) * np.nan
        streamlines_tangents = np.zeros((num_fibers, max_length, dim),
                                        dtype=np.float64)
        streamlines_nearestp = np.zeros((num_fibers, max_length),
                                        dtype=np.int32)
        streamline_scores = np.zeros((num_fibers, max_length),
                                     dtype=np.float64) * np.nan

        # copy python streamlines into numpy array
        for line_id in range(num_fibers):
            for point_id in range(streamlines_length[line_id]):
                for dims in range(3):
                    streamlines[line_id, point_id, dims] = \
                        py_streamlines[line_id][point_id][dims]
        self.streamline_points = streamlines

        # compute tangents
        for line_id in range(num_fibers):
            for point_id in range(streamlines_length[line_id] - 1):
                tangent = np.subtract(streamlines[line_id, point_id + 1],
                                      streamlines[line_id, point_id])
                streamlines_tangents[line_id, point_id] = \
                    np.divide(tangent,
                              np.sqrt(np.dot(tangent, tangent)))

        # estimate which kernel LUT index corresponds to angles
        tree = KDTree(kernel.get_orientations())
        for line_id in range(num_fibers):
            for point_id in range(streamlines_length[line_id] - 1):
                streamlines_nearestp[line_id, point_id] = \
                    tree.query(streamlines[line_id, point_id])[1]

        # arrays for parallel computing
        score_mp = np.zeros(num_fibers)
        xd_mp = np.zeros(num_fibers, dtype=np.int32)
        yd_mp = np.zeros(num_fibers, dtype=np.int32)
        zd_mp = np.zeros(num_fibers, dtype=np.int32)

        if verbose:
            if have_openmp:
                print("Running in parallel!")
            else:
                print("No OpenMP...")

        # compute fiber LFBC measures
        with nogil:

            for line_id in prange(num_fibers, schedule='guided'):
                for point_id in range(streamlines_length[line_id] - 1):
                    score_mp[line_id] = 0.0
                    for line_id2 in range(num_fibers):

                        # skip lfbc computation with itself
                        if line_id == line_id2:
                            continue

                        for point_id2 in range(streamlines_length[line_id2] - 1):
                            # compute displacement
                            xd_mp[line_id] = \
                                int(streamlines[line_id, point_id, 0] -
                                    streamlines[line_id2, point_id2, 0] + 0.5)
                            yd_mp[line_id] = \
                                int(streamlines[line_id, point_id, 1] -
                                    streamlines[line_id2, point_id2, 1] + 0.5)
                            zd_mp[line_id] = \
                                int(streamlines[line_id, point_id, 2] -
                                    streamlines[line_id2, point_id2, 2] + 0.5)

                            # if position is outside the kernel bounds, skip
                            if xd_mp[line_id] > hn or -xd_mp[line_id] > hn or \
                               yd_mp[line_id] > hn or -yd_mp[line_id] > hn or \
                               zd_mp[line_id] > hn or -zd_mp[line_id] > hn:
                                continue

                            # grab kernel value from LUT
                            score_mp[line_id] += \
                                lut[streamlines_nearestp[line_id, point_id],
                                    streamlines_nearestp[line_id2, point_id2],
                                    hn+xd_mp[line_id],
                                    hn+yd_mp[line_id],
                                    hn+zd_mp[line_id]]  # ang_v, ang_r, x, y, z

                    streamline_scores[line_id, point_id] = score_mp[line_id]

        # Reset number of OpenMP cores to default
        if num_threads is not None:
            restore_default_num_threads()

        # Save LFBC as class member
        self.streamlines_lfbc = streamline_scores

        # compute RFBC for each fiber
        self.streamlines_rfbc = compute_rfbc(streamlines_length,
                                             streamline_scores,
                                             max_windowsize)


def compute_rfbc(streamlines_length, streamline_scores, max_windowsize=7):
    """ Compute the relative fiber to bundle coherence (RFBC)

    Parameters
    ----------
    streamlines_length : 1D int array
        Contains the length of each streamline
    streamlines_scores : 2D double array
        Contains the local fiber to bundle coherence (LFBC) for each streamline
        element.
    max_windowsize : int
        The maximal window size used to calculate the average LFBC region

    Returns
    -------
    output: normalized lowest average LFBC region along the fiber
    """

    # finds the region of the fiber with maximal length of max_windowsize in
    # which the LFBC is the lowest
    int_length = min(np.amin(streamlines_length), max_windowsize)
    int_value = np.apply_along_axis(lambda x: min_moving_average(x[~np.isnan(x)],
                                    int_length), 1, streamline_scores)
    avg_total = np.mean(
                    np.apply_along_axis(
                        lambda x: np.mean(np.extract(x[~np.isnan(x)] >= 0,
                                          x[~np.isnan(x)])),
                        1, streamline_scores))
    if not avg_total == 0:
        return int_value / avg_total
    else:
        return int_value


def min_moving_average(a, n):
    """ Return the lowest cumulative sum for the score of a streamline segment

    Parameters
    ----------
    a : array
        Input array

    n : int
        Length of the segment

    Returns
    -------
    output: normalized lowest average LFBC region along the fiber
    """
    ret = np.cumsum(np.extract(a >= 0, a))
    ret[n:] = ret[n:] - ret[:-n]
    return np.amin(ret[n - 1:] / n)
