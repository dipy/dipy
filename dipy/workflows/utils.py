from os.path import join, dirname, isabs, exists
from os import makedirs
from glob import glob


def choose_create_out_dir(out_dir, root_path):
    """Analyses the parameters and returns the appropriate output path.
    It creates the directory if it does not exists.

    Parameters:
    -----------
    out_dir : string
        The directory where you want your output saved.
    root_path : string
        Directory where input data is located.
    """
    if out_dir == '':
        result_path = dirname(root_path)
    elif not isabs(out_dir):
        result_path = join(dirname(root_path), out_dir)
        if not exists(result_path):
            makedirs(result_path)
    else:
        result_path = out_dir

    return result_path

def int_param(val):
    return int(val) if val != 'None' else None

def bool_param(val):
    return (val == 'True') if val != 'None' else None

def int_list_param(val, sep=' '):
    if val == 'None':
        return None

    return val.split(sep)

def all_files_exist(file_path):
    dirs = glob(dirname(file_path))
    files = glob(file_path)

    return len(files) > 0 and (len(files) == len(dirs))
