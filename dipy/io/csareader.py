''' CSA header reader from SPM spec

'''
from struct import unpack_from, calcsize


class CSAReadError(Exception):
    pass


def read(csa_str):
    ''' Read CSA header from string `csa_str`

    Parameters
    ----------
    csa_str : str
       byte string containing CSA header information

    Returns
    -------
    header : dict
       header information as dict, where `header` has fields (at least)
       ``type, n_tags, tags``.  ``tags`` is also a dictionary with one
       key, value pair for each tag in the header.
    '''
    hdr_id = csa_str[:4]
    csa_len = len(csa_str)
    csa_dict = {'tags': {}}
    ptr = 0
    if hdr_id != 'SV10':
        csa_dict['type'] = 1
        ptr, csa_dict['n_tags'] = _ptr_unpack('<I', hdr_id, ptr)
        raise CSAReadError("We cannot read CSA1 headers")
    else:
        ptr, _ = _ptr_unpack('4s', csa_str, ptr)
        csa_dict['type'] = 2
        ptr, csa_dict['unused0'] = _ptr_unpack('4s', csa_str, ptr)
        ptr, csa_dict['n_tags'] = _ptr_unpack('<I', csa_str, ptr)
    if csa_dict['n_tags'] < 1 or csa_dict['n_tags'] > 128:
        raise CSAReadError('Number of tags should be '
                           '> 0, <= 128')
    ptr, csa_dict['check'] = _ptr_unpack('<I', csa_str, ptr)
    for tag_no in range(csa_dict['n_tags']):
        ptr, name, vm, vr, syngodt, n_items, xx = \
            _ptr_unpack('<64si4s3i',csa_str, ptr)
        vr = nt_str(vr)
        name = nt_str(name)
        tag = {'n_items': n_items,
               'vm': vm,
               'vr': vr,
               'syngodt': syngodt,
               'last3': xx}
        assert n_items < 100
        items = []
        for item_no in range(n_items):
            ptr, x0,x1,x2,x3 = _ptr_unpack('<4i',csa_str, ptr)
            item_len = x1
            if (ptr + item_len) > csa_len:
                raise CSAReadError('Item is too long, aborting read')
            item = nt_str(csa_str[ptr:ptr+item_len])
            plus4 = item_len % 4
            if plus4 != 0:
                ptr+=item_len + (4-plus4)
            else:
                ptr+=item_len
            items.append(item)
        tag['items'] = items
        csa_dict['tags'][name] = tag
    return csa_dict


def nt_str(s):
    ''' Strip string to first null

    Parameters
    ----------
    s : str

    Returns
    -------
    sdash : str
       s stripped to first occurence of null (0)
    '''
    zero_pos = s.find(chr(0))
    if zero_pos == -1:
        return s
    return s[:zero_pos]


def _ptr_unpack(fmt, input_str, ptr):
    fmt_len = calcsize(fmt)
    return (ptr + fmt_len,) + unpack_from(
        fmt, input_str, ptr)
