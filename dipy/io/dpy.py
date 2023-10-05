""" A class for handling large tractography datasets.

    It is built using the h5py which in turn implement
    key features of the HDF5 (hierarchical data format) API [1]_.

    References
    ----------
    .. [1] http://www.hdfgroup.org/HDF5/doc/H5.intro.html
"""

import numpy as np
import h5py

from nibabel.streamlines import ArraySequence as Streamlines

# Make sure not to carry across setup module from * import
__all__ = ['Dpy']


class Dpy:
    def __init__(self, fname, mode='r', compression=0):
        """ Advanced storage system for tractography based on HDF5

        Parameters
        ----------
        fname : str, full filename
        mode : 'r' read
         'w' write
         'r+' read and write only if file already exists
        compression : 0 no compression to 9 maximum compression

        Examples
        --------
        >>> import os
        >>> from tempfile import mkstemp #temp file
        >>> from dipy.io.dpy import Dpy
        >>> def dpy_example():
        ...     fd,fname = mkstemp()
        ...     fname += '.dpy'#add correct extension
        ...     dpw = Dpy(fname,'w')
        ...     A=np.ones((5,3))
        ...     B=2*A.copy()
        ...     C=3*A.copy()
        ...     dpw.write_track(A)
        ...     dpw.write_track(B)
        ...     dpw.write_track(C)
        ...     dpw.close()
        ...     dpr = Dpy(fname,'r')
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

        if self.mode == 'w':

            self.f.attrs['version'] = '0.0.1'

            self.streamlines = self.f.create_group('streamlines')

            self.tracks = self.streamlines.create_dataset(
                    'tracks',
                    shape=(0, 3),
                    dtype='f4',
                    maxshape=(None, 3), chunks=True)

            self.offsets = self.streamlines.create_dataset(
                    'offsets',
                    shape=(1,),
                    dtype='i8',
                    maxshape=(None,), chunks=True)

            self.curr_pos = 0
            self.offsets[:] = np.array([self.curr_pos]).astype(np.int64)

        if self.mode == 'r':
            self.tracks = self.f['streamlines']['tracks']
            self.offsets = self.f['streamlines']['offsets']
            self.track_no = len(self.offsets) - 1
            self.offs_pos = 0

    def version(self):

        return self.f.attrs['version']

    def write_track(self, track):
        """ write on track each time
        """
        self.tracks.resize(self.tracks.shape[0] + track.shape[0], axis=0)
        self.tracks[-track.shape[0]:] = track.astype(np.float32)
        self.curr_pos += track.shape[0]

        self.offsets.resize(self.offsets.shape[0] + 1, axis=0)
        self.offsets[-1] = self.curr_pos

    def write_tracks(self, tracks):
        """ write many tracks together
        """

        self.tracks.resize(self.tracks.shape[0] + tracks._data.shape[0],
                           axis=0)
        self.tracks[-tracks._data.shape[0]:] = tracks._data

        self.offsets.resize(self.offsets.shape[0] + tracks._offsets.shape[0],
                            axis=0)
        self.offsets[-tracks._offsets.shape[0]:] = \
            self.offsets[-tracks._offsets.shape[0] - 1] + \
            tracks._offsets + tracks._lengths

    def read_track(self):
        """ read one track each time
        """
        off0, off1 = self.offsets[self.offs_pos:self.offs_pos + 2]
        self.offs_pos += 1
        return self.tracks[off0:off1]

    def read_tracksi(self, indices):
        """ read tracks with specific indices
        """
        tracks = Streamlines()
        for i in indices:
            off0, off1 = self.offsets[i:i + 2]
            tracks.append(self.tracks[off0:off1])
        return tracks

    def read_tracks(self):
        """ read the entire tractography
        """
        I = self.offsets[:]
        TR = self.tracks[:]
        tracks = Streamlines()
        for i in range(len(I) - 1):
            off0, off1 = I[i:i + 2]
            tracks.append(TR[off0:off1])
        return tracks

    def close(self):
        self.f.close()


if __name__ == '__main__':
    pass
