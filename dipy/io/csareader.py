''' CSA header reader from SPM spec

'''
from struct import unpack


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
       header information
    '''
    hdr_id = csa_str[:4]
    csa_len = len(csa_str)
    csa_dict = {'tags': {}}
    if hdr_id != 'SV10':
        csa_dict['type'] = 1
        csa_dict['n_tags'], = unpack('<i', hdr_id)
        ptr = 4
    else:
        csa_dict['type'] = 2
        csa_dict['unused0'] = csa_str[4:8]
        csa_dict['n_tags'], = unpack('<I', csa_str[8:12])
        ptr = 12
    csa_dict['check'], = unpack('<I', csa_str[ptr:ptr+4])
    ptr +=4
    for tag_no in range(csa_dict['n_tags']):
        name, vm, vr, syngodt, n_items, xx = \
            unpack('<64sI4sIII',csa_str[ptr:ptr+84])
        name = nt_str(name)
        tag = {'pointer': ptr,
               'n_items': n_items,
               'last3': xx}
        ptr+=84
        assert n_items < 100
        for item_no in range(n_items):
            tag['pointer'] = ptr
            x0,x1,x2,x3 = unpack('<4i',csa_str[ptr:ptr+16])
            tag['xx'] = [x0,x1,x2,x3]
            ptr+=16
            item_len = x1
            if (ptr + item_len) > csa_len:
                raise CSAReadError('Item is too long, aborting read')
            format = '<%ds' % item_len
            tag['value'] = unpack(format,csa_str[ptr:ptr+item_len])
            plus4 = item_len % 4
            if plus4 != 0:
                ptr+=item_len + (4-plus4)
            else:
                ptr+=item_len
            tag['end_pointer'] = ptr
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
