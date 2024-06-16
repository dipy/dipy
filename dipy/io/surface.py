from warnings import warn

import nibabel as nib

from dipy.testing.decorators import warning_for_keywords


@warning_for_keywords()
def load_pial(fname, *, return_meta=False):
    """Load pial file.

    Parameters
    ----------
    fname : str
        Absolute path of the file.
    return_meta : bool, optional
        Whether to read the metadata of the file or not, by default False.

    Returns
    -------
    tuple
        (vertices, faces) if return_meta=False. Otherwise, (vertices, faces,
        metadata).
    """
    try:
        return nib.freesurfer.read_geometry(fname, read_metadata=return_meta)
    except ValueError:
        warn(f"The file {fname} provided does not have geometry data.", stacklevel=2)


def load_gifti(fname):
    """Load gifti file.

    Parameters
    ----------
    fname : str
        Absolute path of the file.

    Returns
    -------
    tuple
        (vertices, faces)
    """
    surf_img = nib.load(fname)
    return surf_img.agg_data(("pointset", "triangle"))
