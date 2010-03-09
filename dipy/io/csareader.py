''' CSA header reader from SPM spec

'''
from struct import unpack_from, calcsize


_converters = {
    'FL': float, # float
    'FD': float, # double
    'DS': float, # decimal string
    'SS': int, # signed short
    'US': int, # unsigned short
    'SL': int, # signed long
    'UL': int, # unsigned long
    'IS': int, # integer string
    }


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
       ``type, n_tags, tags``.  ``header['tags']`` is also a dictionary
       with one key, value pair for each tag in the header.
    '''
    csa_len = len(csa_str)
    csa_dict = {'tags': {}}
    hdr_id = csa_str[:4]
    ptr = 0
    if hdr_id != 'SV10': # CSA1
        hdr_type = 1
        csa_dict['type'] = 1
    else:
        hdr_type = 2
        ptr, _ = _ptr_unpack('4s', csa_str, ptr)
        ptr, csa_dict['unused0'] = _ptr_unpack('4s', csa_str, ptr)
    ptr, csa_dict['n_tags'] = _ptr_unpack('<I', csa_str, ptr)
    csa_dict['type'] = hdr_type
    if not 0 < csa_dict['n_tags'] <= 128:
        raise CSAReadError('Number of tags `t` should be '
                           '0 < t <= 128')
    ptr, csa_dict['check'] = _ptr_unpack('<I', csa_str, ptr)
    for tag_no in range(csa_dict['n_tags']):
        ptr, name, vm, vr, syngodt, n_items, last3 = \
            _ptr_unpack('<64si4s3i',csa_str, ptr)
        vr = nt_str(vr)
        name = nt_str(name)
        tag = {'n_items': n_items,
               'vm': vm, # value multiplicity
               'vr': vr, # value representation
               'syngodt': syngodt,
               'last3': last3,
               'tag_no': tag_no}
        if vm == 0:
            n_values = n_items
        else:
            n_values = vm
        # data converter
        converter = _converters.get(vr)
        # CSA1 specific length modifier
        if tag_no == 1:
            tag0_n_items = n_items
        assert n_items < 100
        items = []
        for item_no in range(n_items):
            ptr, x0,x1,x2,x3 = _ptr_unpack('<4i', csa_str, ptr)
            if hdr_type == 1:  # CSA1 - odd length calculation
                item_len = x0 - tag0_n_items
                if item_len < 0 or (ptr + item_len) > csa_len:
                    if item_no < vm:
                        items.append('')
                    break
            else: # CSA2
                item_len = x1
                if (ptr + item_len) > csa_len:
                    raise CSAReadError('Item is too long, '
                                       'aborting read')
            if item_no >= n_values:
                assert item_len == 0
                continue
            item = nt_str(csa_str[ptr:ptr+item_len])
            if converter:
                item = converter(item)
            items.append(item)
            ptr += item_len
            # go to 4 byte boundary
            plus4 = item_len % 4
            if plus4 != 0:
                ptr += (4-plus4)
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
