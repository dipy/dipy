''' Stream-like reader for packed data '''

from struct import Struct

_ENDIAN_CODES = '@=<>!'


class Unpacker(object):
    ''' Class to unpack values from buffer

    The buffer is usually a string. 

    Examples
    --------
    >>> a = '1234567890'
    >>> upk = Unpacker(a)
    >>> upk.unpack('2s')
    ('12',)
    >>> upk.unpack('2s')
    ('34',)
    >>> upk.ptr
    4
    >>> upk.read(3)
    '567'
    >>> upk.ptr
    7
    '''
    def __init__(self, buf, ptr=0, endian=None):
        ''' Initialize unpacker

        Parameters
        ----------
        buf : buffer
           object implementing buffer protocol (e.g. str)
        ptr : int, optional
           offset at which to begin reads from `buf`
        endian : None or str, optional
           endian code to prepend to format, as for ``unpack`` endian
           codes. 
        '''
        self.buf = buf
        self.ptr = ptr
        self.endian = endian
        self._cache = {}

    def unpack(self, fmt):
        ''' Unpack values from contained buffer

        Parameters
        ----------
        fmt : str
           format string as for ``unpack``

        Returns
        -------
        values : tuple
           values as unpacked from ``self.buf`` according to `fmt`
        '''
        pkst = self._cache.get(fmt)
        if pkst is None: # struct not in cache
            if self.endian is None or fmt[0] in _ENDIAN_CODES:
                pkst = Struct(fmt)
            else:
                endian_fmt = self.endian + fmt
                pkst = Struct(endian_fmt)
                self._cache[endian_fmt] = pkst
            self._cache[fmt] = pkst
        values = pkst.unpack_from(self.buf, self.ptr)
        self.ptr += pkst.size
        return values

    def read(self, n_bytes):
        ''' Read, return byte string, updating pointer'''
        start = self.ptr
        end = start + n_bytes
        self.ptr = end
        return self.buf[start:end]
        

