import os
import tempfile

import numpy as np


def count_sketch(matrixa_name, matrixa_dtype, matrixa_shape, sketch_rows, tmp_dir):
    """Count Sketching algorithm to reduce the size of the matrix.

    Parameters
    ----------
    matrixa_name : str
        The name of the memmap file containing the matrix A.
    matrixa_dtype : dtype
        The dtype of the matrix A.
    matrixa_shape : tuple
        The shape of the matrix A.
    sketch_rows : int
        The number of rows in the sketch matrix.
    tmp_dir : str
        The directory to save the temporary files.

    Returns
    -------
    matrixc_file.name : str
        The name of the memmap file containing the sketch matrix.
    matrixc.dtype : dtype
        The dtype of the sketch matrix.
    matrixc.shape : tuple
        The shape of the sketch matrix.

    """
    matrixa = np.squeeze(
        np.memmap(matrixa_name, dtype=matrixa_dtype, mode="r+", shape=matrixa_shape)
    ).reshape(np.prod(matrixa_shape[:-1]), matrixa_shape[-1])

    with tempfile.NamedTemporaryFile(
        delete=False, dir=tmp_dir, suffix="matrix_t"
    ) as matrixt_file:
        matrixt = np.memmap(
            matrixt_file.name, dtype=matrixa_dtype, mode="w+", shape=matrixa.shape
        )
        hashed_indices = np.random.choice(sketch_rows, matrixa.shape[0], replace=True)
        rand_signs = np.random.choice(2, matrixa.shape[0], replace=True) * 2 - 1
        for i in range(0, matrixa.shape[0], matrixa.shape[0] // 20):
            end_index = min(i + matrixa.shape[0] // 20, matrixa.shape[0])
            matrixt[i:end_index, :] = (
                matrixa[i:end_index, :] * rand_signs[i:end_index, np.newaxis]
            )

    with tempfile.NamedTemporaryFile(
        delete=False, dir=tmp_dir, suffix="matrix_C"
    ) as matrixc_file:
        matrixc = np.memmap(
            matrixc_file.name,
            dtype=matrixa_dtype,
            mode="w+",
            shape=(sketch_rows, matrixa.shape[1]),
        )
        np.add.at(matrixc, hashed_indices, matrixt)
        matrixc.flush()
        matrixt.flush()
    del matrixt
    os.unlink(matrixt_file.name)
    return matrixc_file.name, matrixc.dtype, matrixc.shape
