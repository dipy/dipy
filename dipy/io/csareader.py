''' CSA header reader from SPM spec

'''
from struct import unpack


class CSAReadError(Exception):
    pass


def read(fileobj):
    ''' Read CSA header from fileobj

    Parameters
    ----------
    fileobj : file-like
       file-like object implementing ``read, seek, tell`.  The data is
       taken to start at position 0 and end at the end of the file. 

    Returns
    -------
    header : dict
       header information
    tags : sequence
       sequence of tags from header
    '''
    fileobj.seek(0)
    hdr_id = fileobj.read(4)
    hdr = {}
    tags = []
    if hdr_id != 'SV10':
        hdr['type'] = 1
        hdr['n_tags'], = unpack('<i', hdr_id)
    else:
        hdr['type'] = 2
        hdr['unused1'] = fileobj.read(4)
        hdr['n_tags'], = unpack('<I', fileobj.read(4))
    hdr['unused3'], = unpack('<I', fileobj.read(4))
    return hdr, tags
