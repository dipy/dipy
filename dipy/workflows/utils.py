from os.path import join, dirname, isabs, exists
from os import makedirs


def choose_create_out_dir(out_dir, root_path):
    if out_dir == '':
        result_path = dirname(root_path)
    elif not isabs(out_dir):
        result_path = join(dirname(root_path), out_dir)
        if not exists(result_path):
            makedirs(result_path)
    else:
        result_path = out_dir

    return result_path

def int_list_param(val, sep=' '):
    if val == 'None':
        return None

    return val.split(sep)
