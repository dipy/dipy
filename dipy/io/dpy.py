""" A class for handling large tractography datasets.

    It is built using the pytables tools which in turn implement
    key features of the HDF5 (hierachical data format) API [1]_.

    References
    ----------
    .. [1] http://www.hdfgroup.org/HDF5/doc/H5.intro.html
"""

import numpy as np
import h5py

# Make sure not to carry across setup module from * import
__all__ = ['Dpy']


class Dpy(object):
    def __init__(self, fname, mode='r', compression=0):
        """ Advanced storage system for tractography based on HDF5

        Parameters
        ------------
        fname : str, full filename
        mode : 'r' read
         'w' write
         'r+' read and write only if file already exists
        compression : 0 no compression to 9 maximum compression

        Examples
        ----------
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
        ...     A=dpr.read_track()
        ...     B=dpr.read_track()
        ...     T=dpr.read_tracksi([0,1,2,0,0,2])
        ...     dpr.close()
        ...     os.remove(fname) #delete file from disk
        >>> dpy_example()  # skip if not have_tables

        """

        self.mode = mode
        self.f = h5py.File(fname, mode=self.mode)
        self.N = 5 * 10**9
        self.compression = compression

        if self.mode == 'w':

            self.streamlines = f.create_group('streamlines')

            # create a version number
            self.version = self.streamlines.create_dataset(
                    'version',
                    [b"0.0.2"])

            self.tracks = self.streamlines.create_dataset(
                    'tracks',
                    shape=(0, 3),
                    dtype='f4',
                    maxshape=(None, 3))

            self.offsets = self.streamlines.create_dataset(
                    'offsets',
                    shape=(0,),
                    dtype='i8',
                    maxshape=(None))

            self.curr_pos = 0
            self.offsets.append(np.array([self.curr_pos]).astype(np.int64))

        if self.mode == 'r':
            self.tracks = self.f['streamlines']['tracks']
            self.offsets = self.f['streamlines']['offsets']
            self.track_no = len(self.offsets) - 1
            self.offs_pos = 0

    def version(self):
        ver = self.f.root.version[:]
        return ver[0].decode()

    def write_track(self, track):
        """ write on track each time
        """
        self.tracks.append(track.astype(np.float32))
        self.curr_pos += track.shape[0]
        self.offsets.append(np.array([self.curr_pos]).astype(np.int64))

    def write_tracks(self, T):
        """ write many tracks together
        """
        for track in T:
            self.tracks.append(track.astype(np.float32))
            self.curr_pos += track.shape[0]
            self.offsets.append(np.array([self.curr_pos]).astype(np.int64))

    def read_track(self):
        """ read one track each time
        """
        off0, off1 = self.offsets[self.offs_pos:self.offs_pos + 2]
        self.offs_pos += 1
        return self.tracks[off0:off1]

    def read_tracksi(self, indices):
        """ read tracks with specific indices
        """
        T = []
        for i in indices:
            # print(self.offsets[i:i+2])
            off0, off1 = self.offsets[i:i + 2]
            T.append(self.tracks[off0:off1])
        return T

    def read_tracks(self):
        """ read the entire tractography
        """
        I = self.offsets[:]
        TR = self.tracks[:]
        T = []
        for i in range(len(I) - 1):
            off0, off1 = I[i:i + 2]
            T.append(TR[off0:off1])
        return T

    def close(self):
        self.f.close()


if __name__ == '__main__':
    pass
