import numpy as np
cimport cython

cimport safe_openmp as openmp
from safe_openmp cimport have_openmp
from cython.parallel import parallel, prange, threadid

from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from math import sqrt, log

from dipy.data import get_sphere
from dipy.denoise.enhancement_kernel import EnhancementKernel
from dipy.core.ndindex import ndindex

cdef class FBCMeasures:

    cdef int [:] streamline_length
    cdef double [:, :, :] streamline_points
    cdef double [:, :] streamlines_lfbc
    cdef double [:] streamlines_rfbc

    ## Python functions
    
    def __init__(self, streamlines, kernel, num_threads=None):
        """ Compute the fiber to bundle coherence measures for a set of streamlines.

        Parameters
        ----------
        streamlines : list
            A collection of streamlines, each n by 3, with n being the number of
            nodes in the fiber.
        kernel : Kernel object
            A diffusion kernel object created from EnhancementKernel.
        num_threads : int
            Number of threads to use for OpenMP.
            
        References
        -------
        [Portegies2015b] J. Portegies, R. Fick, G. Sanguinetti, S. Meesters, G.Girard,
                     and R. Duits. (2015) Improving Fiber Alignment in HARDI by 
                     Combining Contextual PDE flow with Constrained Spherical 
                     Deconvolution. PLoS One.
        """
        self.compute(streamlines, kernel)
        
    def get_points_rfbc_thresholded(self, threshold, showInfo=False, emphasis=.5):
        """ Set a threshold on the RFBC to remove spurious fibers.

        Parameters
        ----------
        threshold : float
            The threshold to set on the RFBC, should be within 0 and 1.
        showInfo : boolean
            Prints info about the found RFBC for the set of fibers such as median,
            mean, min and max values.
        emphasis : float
            Enhances the coloring of the fibers by LFBC. Increasing emphasis will
            stress spurious fibers by logarithmic weighting.
            
        Returns
        -------
        output : tuple with 3 elements
            The output contains:
            1) a collection of streamlines, each n by 3, with n 
            being the number of nodes in the fiber that remain after filtering 
            2) the r,g,b values of the local fiber to bundle coherence (LFBC) 
            3) the relative fiber to bundel coherence (RFBC)
        """
        if showInfo:
            print "median RFBC: " + str(np.median(self.streamlines_rfbc))
            print "mean RFBC: " + str(np.mean(self.streamlines_rfbc))
            print "min RFBC: " + str(np.min(self.streamlines_rfbc))
            print "max RFBC: " + str(np.max(self.streamlines_rfbc))

        # logarithmic transform of color values to emphasize spurious fibers
        minval = np.nanmin(self.streamlines_lfbc)
        maxval = np.nanmax(self.streamlines_lfbc)
        lfbc_log = np.log((self.streamlines_lfbc - minval)/
                        (maxval - minval + 10e-10) + emphasis )
        minval = np.nanmin(lfbc_log)
        maxval = np.nanmax(lfbc_log)
        lfbc_log = (lfbc_log - minval)/(maxval - minval)

        # define color interpolation functions
        x = np.linspace(0, 1, num=4, endpoint=True)
        r = np.array([1,1,0,0])
        g = np.array([1,0,0,1])
        b = np.array([0,0,1,1])
        fr = interp1d(x, r, bounds_error=False, fill_value=0)
        fg = interp1d(x, g, bounds_error=False, fill_value=0)
        fb = interp1d(x, b, bounds_error=False, fill_value=0)

        # select fibers above the RFBC threshold
        streamlinelist = []
        rfbclist = []
        lfbclist = []
        for i in range(len(self.streamlines_rfbc)):
            rfbc = self.streamlines_rfbc[i]
            lfbc = lfbc_log[i]
            if rfbc > threshold:
                fiber = np.array(self.streamline_points[i])
                fiber = fiber[0:self.streamline_length[i]-1]
                streamlinelist.append(fiber)

                rfbclist.append(rfbc)

                lfbc = lfbc[0:self.streamline_length[i]-1]
                lfbcfiberlist = np.transpose([fr(lfbc),fg(lfbc),fb(lfbc)])
                lfbclist.append(lfbcfiberlist.tolist())

        return streamlinelist, lfbclist, rfbclist
    
    ## Cython functions
    
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void compute(self, 
                        py_streamlines,
                        kernel,
                        num_threads=None):
        """ Compute the fiber to bundle coherence measures for a set of streamlines.

        Parameters
        ----------
        py_streamlines : list
            A collection of streamlines, each n by 3, with n being the number of
            nodes in the fiber.
        kernel : Kernel object
            A diffusion kernel object created from EnhancementKernel.
        num_threads : int
            Number of threads to use for OpenMP.
        """
        cdef:
            int numberOfFibers
            int maxLength
            int dim
            double [:, :, :] streamlines
            int [:] streamlines_length
            double [:, :, :] streamlines_tangent
            int [:, :] streamlines_nearestp
            double [:, :] streamline_scores
            double [:] tangent
            int lineId, pointId
            int lineId2, pointId2
            double score
            int xd, yd, zd
            double [:, :, :, :, ::1] lut
            int N
            int hn
            int threads_to_use = -1
            int all_cores = openmp.omp_get_num_procs()

            double [:] score_mp
            int [:] xd_mp
            int [:] yd_mp
            int [:] zd_mp

        if num_threads is not None:
            threads_to_use = num_threads
        else:
            threads_to_use = all_cores

        if have_openmp:
            openmp.omp_set_dynamic(0)
            openmp.omp_set_num_threads(threads_to_use)
        
        # if the fibers are too short FBC measures cannot be applied, remove these.
        streamlines_length = np.array([len(x) for x in py_streamlines], 
                                dtype=np.intc)
        minLength = min(streamlines_length)
        if minLength < 10:
            print("The minimum fiber length is 10 points. \
                    Shorter fibers were found and removed.")
            py_streamlines = [x for x in py_streamlines if len(x) >= 10]
            streamlines_length = np.array([len(x) for x in py_streamlines], 
                                    dtype=np.intc)
            minLength = min(streamlines_length)
        numberOfFibers = len(py_streamlines)
        self.streamline_length = streamlines_length
        maxLength = max(streamlines_length)
        dim = 3

        # get lookup table info
        lut = kernel.get_lookup_table()
        N = lut.shape[2]
        hn = (N-1)/2
        
        # prepare numpy arrays for speed
        streamlines = np.zeros((numberOfFibers, maxLength, dim), 
                                dtype=np.float64)*np.nan
        streamlines_tangents = np.zeros((numberOfFibers, maxLength, dim), 
                                dtype=np.float64)
        streamlines_nearestp = np.zeros((numberOfFibers, maxLength), 
                                dtype=np.intc)
        streamline_scores = np.zeros((numberOfFibers, maxLength), 
                                dtype=np.float64)*np.nan
        
        # copy python streamlines into numpy array
        for lineId in range(numberOfFibers):
            for pointId in range(streamlines_length[lineId]):
                for dim in range(3):
                    streamlines[lineId, pointId, dim] = \
                        py_streamlines[lineId][pointId][dim]
        self.streamline_points = streamlines
        
        # compute tangents
        for lineId in range(numberOfFibers):
            for pointId in range(streamlines_length[lineId]-1):
                tangent = np.subtract(streamlines[lineId, pointId+1], 
                                        streamlines[lineId, pointId])
                streamlines_tangents[lineId, pointId] = np.divide(tangent, 
                                            np.sqrt(np.dot(tangent, tangent)))
        
        # estimate which kernel LUT index corresponds to angles
        tree = KDTree(kernel.get_orientations())
        for lineId in range(numberOfFibers):
            for pointId in range(streamlines_length[lineId]-1):
                streamlines_nearestp[lineId, pointId] = \
                    tree.query(streamlines[lineId, pointId])[1]
        
        # arrays for parallel computing
        score_mp = np.zeros(numberOfFibers)
        xd_mp = np.zeros(numberOfFibers, dtype=np.int32)
        yd_mp = np.zeros(numberOfFibers, dtype=np.int32)
        zd_mp = np.zeros(numberOfFibers, dtype=np.int32)

        if have_openmp:
            print("Running in parallel!")
        else:
            print("No OpenMP...")

        # compute fiber LFBC measures
        with nogil:
            for lineId in prange(numberOfFibers, schedule='guided'):
                for pointId in range(streamlines_length[lineId]-1):
                    score_mp[lineId] = 0.0
                    for lineId2 in range(numberOfFibers):
                    
                        # skip lfbc computation with itself
                        if lineId == lineId2:
                            continue
                            
                        for pointId2 in range(streamlines_length[lineId2]-1):
                            # compute displacement
                            xd_mp[lineId] = int(streamlines[lineId, pointId, 0] 
                            - streamlines[lineId2, pointId2, 0] + 0.5)
                            yd_mp[lineId] = int(streamlines[lineId, pointId, 1] 
                            - streamlines[lineId2, pointId2, 1] + 0.5)
                            zd_mp[lineId] = int(streamlines[lineId, pointId, 2] 
                            - streamlines[lineId2, pointId2, 2] + 0.5)
                            
                            # if position is outside the kernel bounds, skip
                            if xd_mp[lineId] > hn or -xd_mp[lineId] > hn or \
                               yd_mp[lineId] > hn or -yd_mp[lineId] > hn or \
                               zd_mp[lineId] > hn or -zd_mp[lineId] > hn:
                                continue
                            
                            # grab kernel value from LUT
                            score_mp[lineId] += \
                                        lut[streamlines_nearestp[lineId, pointId], 
                                        streamlines_nearestp[lineId2, pointId2], 
                                        hn+xd_mp[lineId], 
                                        hn+yd_mp[lineId], 
                                        hn+zd_mp[lineId]]  # ang_v, ang_r, x, y, z
                                    
                    streamline_scores[lineId, pointId] = score_mp[lineId]

        if have_openmp and num_threads is not None:
            openmp.omp_set_num_threads(all_cores)
        
        # Save LFBC as class member
        self.streamlines_lfbc = streamline_scores
        
        # compute RFBC for each fiber
        self.streamlines_rfbc = compute_rfbc(streamlines_length, 
                                              streamline_scores)
        
def compute_rfbc(streamlines_length, streamline_scores):
    """ Compute the relative fiber to bundle coherence (RFBC)

    Parameters
    ----------
    streamlines_length : 1D int array
        Contains the length of each streamline
    streamlines_scores : 2D double array
        Constains the local fiber to bundle coherence (LFBC) for each streamline 
        element.

    Returns
    ----------
    output: normalized lowest average LFBC region along the fiber
    """

    # finds the region of the fiber with minimal length if 7 points in which the
    # LFBC is the lowest
    intLength = min(np.amin(streamlines_length), 7)
    intValue = np.apply_along_axis(lambda x: min_moving_average(x, intLength), 
                                    1, streamline_scores)
    averageTotal = np.mean(np.apply_along_axis(
                lambda x:np.mean(np.extract(x>=0, x)), 1, streamline_scores))
    if not averageTotal == 0:
        return intValue/averageTotal
    else:
        return intValue
            
def min_moving_average(a, n):
    """ Return the lowest cumulative sum for the score of a streamline segment

    Parameters
    ----------
    a : array
        Input array

    n : int
        Length of the segment

    Returns
    ----------
    output: normalized lowest average LFBC region along the fiber
    """
    ret = np.cumsum(np.extract(a>=0, a))
    ret[n:] = ret[n:] - ret[:-n]
    return np.amin(ret[n - 1:] / n)
        

    
    
    
    
    