from os.path import join, dirname, isabs, exists
from os import makedirs
from glob import glob

# FIND NEW PARAM NAMES
def glob_or_value(globable, maybe_globable, default=None):
    globed_globable = glob(globable)
    try:
        globed_maybe = glob(maybe_globable)
    except:
        globed_maybe = None

    if globed_maybe is None or len(globed_maybe) != len(globed_globable):
        globed_maybe = [default] * len(globed_globable)

    return globed_globable, globed_maybe



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
    else:
        result_path = out_dir

    if not exists(result_path):
        makedirs(result_path)

    return result_path
