cimport numpy as np
import numpy as np

from dipy.core.interpolation import nearestneighbor_interpolate
from dipy.direction.closest_peak_direction_getter cimport closest_peak
from dipy.tracking.direction_getter cimport DirectionGetter

cdef class PeakDirectionGetter(DirectionGetter):

    def __init__(self, peaks, max_angle, **kwargs):
        self.peaks = peaks
        self.nbr_peaks = peaks.shape[-2]
        self.cos_similarity = np.cos(np.deg2rad(max_angle))


    @classmethod
    def from_peaks(klass, peaks, max_angle, **kwargs):
        """Create a PeakDirectionGetter using peak data

        Parameters
        ----------
        peaks : ndarray, float, (..., N, 3)
            Peaks data with N peaks per voxel.
        max_angle : float (0, 90)
            Maximum angle between tract segments. This angle can be more
            generous (larger) than values typically used with probabilistic
            direction getters.

        """
        return klass(peaks, max_angle, **kwargs)


    def _get_peak_directions(self, point):
        """Gets directions associated to the point.
        """
        nbr = 0
        peaks = np.asarray(nearestneighbor_interpolate(self.peaks, point))
        for peak in peaks:
                if not (peak[0] == 0 and peak[1] == 0 and peak[2] == 0):
                        nbr = nbr + 1
        return peaks, nbr


    cpdef np.ndarray[np.float_t, ndim=2] initial_direction(self,
                                                           double[::1] point):
        """Returns best directions at seed location to start tracking.

        Parameters
        ----------
        point : ndarray, shape (3,)
            The point in an image at which to lookup tracking directions.

        Returns
        -------
        directions : ndarray, shape (N, 3)
            Possible tracking directions from point. ``N`` may be 0, all
            directions should be unique.

        """
        cdef:
            double p[3]
            int nbr
            np.ndarray[np.float_t, ndim=2] peaks

        for i in range(3):
            p[i] = point[i]
        peaks, nbr = self._get_peak_directions(p)
        return peaks[:nbr,:]


    cdef int get_direction_c(self, double* point, double* direction):
        """Closest direction from peaks data.

        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.
        """
        cdef:
            double p[3]
            int nbr
            np.ndarray[np.float_t, ndim=2] peaks

        for i in range(3):
            p[i] = point[i]

        peaks, nbr = self._get_peak_directions(p)
        if nbr > 0:
            return closest_peak(peaks[:nbr,:], direction, self.cos_similarity)
        return 1

cdef class AxtractDirectionGetter(PeakDirectionGetter):

    def __init__(self, peaks, max_angle, peak_values, **kwargs):
        PeakDirectionGetter.__init__(self, peaks, max_angle, **kwargs)
        self.peak_values = peak_values


    @classmethod
    def from_peaks(klass, peaks, max_angle, peak_values, **kwargs):
        """Create a PeakDirectionGetter using peak data

        Parameters
        ----------
        peaks : ndarray, float, (..., N, 3)
            Peaks data with N peaks per voxel.
        max_angle : float (0, 90)
            Maximum angle between tract segments. This angle can be more
            generous (larger) than values typically used with probabilistic
            direction getters.
        peak_values : ndarray, float, (..., N)
            Peak values data with N peaks per voxel. Peak values will be used
            in the selection of the direction.
        """
        return klass(peaks, max_angle, peak_values, **kwargs)

    cpdef int get_axtract_direction(self,
                            double[::1] point,
                            double[::1] direction,
                            double streamline_value) except -1:
        return self.get_axtract_direction_c(&point[0],
                                            &direction[0],
                                            streamline_value)

    cdef int get_axtract_direction_c(self, double* point, double* direction,
                                     double streamline_value):
        """Directions from peaks data with peak value the closest to
           the streamline value.

        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.
        """
        cdef:
            double p[3]
            int nbr
            np.ndarray[np.float_t, ndim=2] peaks

        for i in range(3):
            p[i] = point[i]

        ####
        # instead of return the peak with the closest angle, it should return
        # the peak with the peak value the most similar to streamline_value
        ####
        peaks, nbr = self._get_peak_directions(p)
        if nbr > 0:
            return closest_peak(peaks[:nbr,:], direction, self.cos_similarity)
        return 1
