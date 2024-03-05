from warnings import warn
import nibabel as nib


def load_pial(fname, return_meta=False):
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
        data = nib.freesurfer.read_geometry(fname, read_metadata=return_meta)

        if data[0].shape[-1] != 3:
            raise ValueError('Vertices do not have correct shape:' +
                             {data[0].shape})
        if data[1].shape[-1] != 3:
            raise ValueError('Faces do not have correct shape:' +
                             {data[1].shape})
        return data
    except ValueError as e:
        if e.message:
            warn(e.message)
        else:
            warn(f'The file {fname} provided does not have correct data.')
