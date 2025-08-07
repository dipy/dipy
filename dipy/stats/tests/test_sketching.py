import os
import tempfile

import numpy as np
import pytest

from dipy.stats.sketching import count_sketch


@pytest.fixture
def setup_matrices():
    """Fixture to set up matrix A as a memmap file and return its details."""
    # Matrix A details
    matrixa_dtype = np.float64
    matrixa_shape = (100, 50)
    matrixa = np.random.rand(*matrixa_shape)

    # Create a temporary file to store matrix A
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mmap") as matrixa_file:
        np.memmap(
            matrixa_file.name, dtype=matrixa_dtype, mode="w+", shape=matrixa_shape
        )[:] = matrixa
        matrixa_file_name = matrixa_file.name

    yield matrixa_file_name, matrixa_dtype, matrixa_shape

    # Cleanup: Remove the matrix file after the test
    os.unlink(matrixa_file_name)


def test_count_sketch_basic_functionality(setup_matrices):
    """Test the basic functionality of the count_sketch function."""
    matrixa_name, matrixa_dtype, matrixa_shape = setup_matrices
    sketch_rows = 10

    # Create a temporary directory for the output files
    with tempfile.TemporaryDirectory() as tmp_dir:
        matrixc_file_name, matrixc_dtype, matrixc_shape = count_sketch(
            matrixa_name, matrixa_dtype, matrixa_shape, sketch_rows, tmp_dir
        )

        # Load the resulting sketch matrix to check properties
        matrixc = np.memmap(
            matrixc_file_name, dtype=matrixc_dtype, mode="r+", shape=matrixc_shape
        )

        # Verify the output sketch matrix shape and dtype
        assert matrixc.shape == (sketch_rows, matrixa_shape[1]), "Output shape mismatch"
        assert matrixc.dtype == matrixa_dtype, "Output dtype mismatch"

        # Clean up
        del matrixc
        os.unlink(matrixc_file_name)


def test_count_sketch_randomness(setup_matrices):
    """Test if count_sketch produces different results with different random seeds."""
    matrixa_name, matrixa_dtype, matrixa_shape = setup_matrices
    sketch_rows = 10

    with tempfile.TemporaryDirectory() as tmp_dir:
        np.random.seed(42)
        matrixc_file_name_1, _, _ = count_sketch(
            matrixa_name, matrixa_dtype, matrixa_shape, sketch_rows, tmp_dir
        )
        matrixc_1 = np.memmap(matrixc_file_name_1, dtype=matrixa_dtype, mode="r+")

        np.random.seed(43)
        matrixc_file_name_2, _, _ = count_sketch(
            matrixa_name, matrixa_dtype, matrixa_shape, sketch_rows, tmp_dir
        )
        matrixc_2 = np.memmap(matrixc_file_name_2, dtype=matrixa_dtype, mode="r+")

        # Verify that the two resulting sketches are different
        assert not np.allclose(
            matrixc_1, matrixc_2
        ), "Sketches should differ with different random seeds"

        # Clean up
        del matrixc_1
        del matrixc_2
        os.unlink(matrixc_file_name_1)
        os.unlink(matrixc_file_name_2)


def test_count_sketch_preserves_shape_and_dtype(setup_matrices):
    """Test if the count_sketch preserves the shape and dtype of the input matrix A."""
    matrixa_name, matrixa_dtype, matrixa_shape = setup_matrices
    sketch_rows = 20

    with tempfile.TemporaryDirectory() as tmp_dir:
        matrixc_file_name, matrixc_dtype, matrixc_shape = count_sketch(
            matrixa_name, matrixa_dtype, matrixa_shape, sketch_rows, tmp_dir
        )

        # Check the resulting matrix properties
        assert matrixc_shape == (
            sketch_rows,
            matrixa_shape[1],
        ), "Sketch matrix shape is incorrect"
        assert matrixc_dtype == matrixa_dtype, "Sketch matrix dtype is incorrect"

        # Clean up
        os.unlink(matrixc_file_name)


def test_count_sketch_with_large_matrix():
    """Test count_sketch function with a large matrix to check performance."""
    matrixa_dtype = np.float64
    matrixa_shape = (1000, 500)
    matrixa = np.random.rand(*matrixa_shape)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mmap") as matrixa_file:
        np.memmap(
            matrixa_file.name, dtype=matrixa_dtype, mode="w+", shape=matrixa_shape
        )[:] = matrixa
        matrixa_name = matrixa_file.name

    sketch_rows = 50
    with tempfile.TemporaryDirectory() as tmp_dir:
        matrixc_file_name, matrixc_dtype, matrixc_shape = count_sketch(
            matrixa_name, matrixa_dtype, matrixa_shape, sketch_rows, tmp_dir
        )

        # Load the resulting sketch matrix
        matrixc = np.memmap(
            matrixc_file_name, dtype=matrixc_dtype, mode="r+", shape=matrixc_shape
        )

        # Verify that the sketch was successfully computed
        assert matrixc.shape == (
            sketch_rows,
            matrixa_shape[1],
        ), "Output shape mismatch for large matrix"

        # Clean up
        del matrixc
        os.unlink(matrixa_name)
        os.unlink(matrixc_file_name)
