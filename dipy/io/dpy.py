"""A class for handling large tractography datasets.

It is built using the h5py which in turn implement
key features of the HDF5 (hierarchical data format) API [1]_.

References
----------
.. [1] https://www.hdfgroup.org/HDF5/doc/H5.intro.html
"""

import h5py
from nibabel.streamlines import ArraySequence as Streamlines
import numpy as np

from dipy.testing.decorators import warning_for_keywords

# Make sure not to carry across setup module from * import
__all__ = ["Dpy"]


class Dpy:
    @warning_for_keywords()
    def __init__(self, fname, *, mode="r", compression=0):
        """Advanced storage system for tractography based on HDF5

        Parameters
        ----------
        fname : str or Path
            Full filename
        mode : str, optional
            Use 'r' to read, 'w' to write, and 'r+' to read and write (only if
            file already exists).
        compression : int, optional
            0 no compression to 9 maximum compression.

        Examples
        --------
        >>> import os
        >>> from tempfile import mkstemp #temp file
        >>> from dipy.io.dpy import Dpy
        >>> def dpy_example():
        ...     fd,fname = mkstemp()
        ...     fname += '.dpy'#add correct extension
        ...     dpw = Dpy(fname, mode='w')
        ...     A=np.ones((5,3))
        ...     B=2*A.copy()
        ...     C=3*A.copy()
        ...     dpw.write_track(A)
        ...     dpw.write_track(B)
        ...     dpw.write_track(C)
        ...     dpw.close()
        ...     dpr = Dpy(fname, mode='r')
        ...     dpr.read_track()
        ...     dpr.read_track()
        ...     dpr.read_tracksi([0, 1, 2, 0, 0, 2])
        ...     dpr.close()
        ...     os.remove(fname) #delete file from disk
        >>> dpy_example()
        """

        self.mode = mode
        self.f = h5py.File(fname, mode=self.mode)
        self.compression = compression

        if self.mode == "w":
            self.f.attrs["version"] = "0.0.1"

            self.streamlines = self.f.create_group("streamlines")

            self.tracks = self.streamlines.create_dataset(
                "tracks", shape=(0, 3), dtype="f4", maxshape=(None, 3), chunks=True
            )

            self.offsets = self.streamlines.create_dataset(
                "offsets", shape=(1,), dtype="i8", maxshape=(None,), chunks=True
            )

            self.curr_pos = 0
            self.offsets[:] = np.array([self.curr_pos]).astype(np.int64)

        if self.mode == "r":
            self.tracks = self.f["streamlines"]["tracks"]
            self.offsets = self.f["streamlines"]["offsets"]
            self.track_no = len(self.offsets) - 1
            self.offs_pos = 0

    def version(self):
        """Return the version of the Dpy file.

        Returns
        -------
        version : str
            The version string stored in the file.
        """
        return self.f.attrs["version"]

    def write_track(self, track):
        """Write a single track to the Dpy file.

        Parameters
        ----------
        track : array-like (N, 3)
            The streamline to be written.
        """
        self.tracks.resize(self.tracks.shape[0] + track.shape[0], axis=0)
        self.tracks[-track.shape[0] :] = track.astype(np.float32)
        self.curr_pos += track.shape[0]

        self.offsets.resize(self.offsets.shape[0] + 1, axis=0)
        self.offsets[-1] = self.curr_pos

    def write_tracks(self, tracks):
        """Write a multiple tracks to the Dpy file.

        Parameters
        ----------
        tracks : Streamlines or list of array-like
            The tractography dataset to be written.
        """

        self.tracks.resize(self.tracks.shape[0] + tracks._data.shape[0], axis=0)
        self.tracks[-tracks._data.shape[0] :] = tracks._data

        self.offsets.resize(self.offsets.shape[0] + tracks._offsets.shape[0], axis=0)
        self.offsets[-tracks._offsets.shape[0] :] = (
            self.offsets[-tracks._offsets.shape[0] - 1]
            + tracks._offsets
            + tracks._lengths
        )

    def read_track(self):
        """Read one track from the Dpy file at the current position.

        Returns
        -------
        track : ndarray (N, 3)
            A single streamline.
        """
        off0, off1 = self.offsets[self.offs_pos : self.offs_pos + 2]
        self.offs_pos += 1
        return self.tracks[off0:off1]

    def read_tracksi(self, indices):
        """Read tracks with specific indices from the Dpy file.

        Parameters
        ----------
        indices : list or array-like
            The indices of the tracks to read.

        Returns
        -------
        tracks : Streamlines
            The streamlines corresponding to the given indices.
        """
        tracks = Streamlines()
        for i in indices:
            off0, off1 = self.offsets[i : i + 2]
            tracks.append(self.tracks[off0:off1])
        return tracks

    def read_tracks(self):
        """Read the entire tractography dataset from the Dpy file.

        Returns
        -------
        tracks : Streamlines
            The entire set of streamlines in the file.
        """
        offsets = self.offsets[:]
        TR = self.tracks[:]
        tracks = Streamlines()
        for i in range(len(offsets) - 1):
            off0, off1 = offsets[i : i + 2]
            tracks.append(TR[off0:off1])
        return tracks

    def close(self):
        """Close the Dpy file descriptor."""
        self.f.close()
