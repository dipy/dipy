import logging

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import default_sphere
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.image import save_nifti
from dipy.io.peaks import save_pam
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.viz.skyline.io import EMERGENCY_REF, load_files, load_npy

AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, -10.0],
        [0.0, 2.0, 0.0, -20.0],
        [0.0, 0.0, 2.0, -30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _volume(tmp_path, name="vol.nii.gz", shape=(4, 5, 6)):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    path = tmp_path / name
    save_nifti(str(path), data, AFFINE)
    return str(path), data


def _pam5(tmp_path, name="peaks.pam5"):
    n_vertices = default_sphere.vertices.shape[0]
    pam = PeaksAndMetrics()
    pam.affine = AFFINE
    pam.peak_dirs = np.zeros((2, 2, 2, 5, 3), dtype=np.float32)
    pam.peak_dirs[..., 0, 0] = 1.0
    pam.peak_values = np.ones((2, 2, 2, 5), dtype=np.float32)
    pam.peak_indices = np.zeros((2, 2, 2, 5), dtype=np.int32)
    pam.shm_coeff = np.zeros((2, 2, 2, 45), dtype=np.float32)
    pam.shm_coeff[..., 0] = 1.0
    pam.sphere = default_sphere
    pam.B = np.zeros((45, n_vertices), dtype=np.float32)
    pam.total_weight = 0.5
    pam.ang_thr = 45.0
    pam.gfa = np.zeros((2, 2, 2), dtype=np.float32)
    pam.qa = np.zeros((2, 2, 2, 5), dtype=np.float32)
    pam.odf = np.zeros((2, 2, 2, n_vertices), dtype=np.float32)
    path = tmp_path / name
    save_pam(str(path), pam, affine=AFFINE)
    return str(path)


def _streamlines():
    return [
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    ]


def _tractogram(tmp_path, name, shape=(4, 5, 6)):
    reference = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), AFFINE)
    sft = StatefulTractogram(_streamlines(), reference, Space.RASMM)
    path = tmp_path / name
    save_tractogram(sft, str(path), bbox_valid_check=False)
    return str(path)


def _mesh():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return vertices, faces


def _pial(tmp_path, name="lh.pial"):
    vertices, faces = _mesh()
    path = tmp_path / name
    nib.freesurfer.io.write_geometry(str(path), vertices, faces)
    return str(path), vertices, faces


def _gifti(tmp_path, name="surface.gii", vertices=None, faces=None):
    if vertices is None:
        vertices, faces = _mesh()
    image = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(vertices, intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE"),
        ]
    )
    path = tmp_path / name
    nib.save(image, str(path))
    return str(path)


def test_load_files_with_no_input_returns_empty_lists():
    loaded = load_files(None)

    assert loaded == {
        "images": [],
        "peaks": [],
        "rois": [],
        "surfaces": [],
        "tractograms": [],
        "shm_coeffs": [],
    }


@pytest.mark.parametrize("name", ["vol.nii.gz", "vol.nii"])
def test_load_files_reads_nifti_images(tmp_path, name):
    path, data = _volume(tmp_path, name=name)

    loaded = load_files([path])

    assert len(loaded["images"]) == 1
    image_data, image_affine, image_path = loaded["images"][0]
    npt.assert_allclose(image_data, data)
    npt.assert_allclose(image_affine, AFFINE)
    assert image_path == path


def test_load_files_reads_pam5_peaks(tmp_path):
    path = _pam5(tmp_path)

    loaded = load_files([path])

    assert len(loaded["peaks"]) == 1
    peak_dirs, affine, peak_path, peak_values = loaded["peaks"][0]
    assert peak_path == path
    assert peak_dirs.shape == (2, 2, 2, 5, 3)
    assert peak_values.shape == (2, 2, 2, 5)
    npt.assert_allclose(affine, AFFINE)


@pytest.mark.parametrize("reshape", [False, True])
def test_load_files_reads_nifti_peaks(tmp_path, reshape):
    arr = np.zeros((2, 2, 2, 2, 3), dtype=np.float32)
    arr[..., 0, :] = [3.0, 0.0, 0.0]
    arr[..., 1, :] = [0.0, 1.0, 0.0]
    arr[0, 0, 0, 1, :] = np.nan
    saved = arr.reshape(2, 2, 2, 6) if reshape else arr
    path = tmp_path / "peaks.nii.gz"
    save_nifti(str(path), saved, AFFINE)

    loaded = load_files([], peaks=[str(path)])

    assert len(loaded["peaks"]) == 1
    peak_dirs, affine, path_out, peak_values = loaded["peaks"][0]
    assert peak_dirs.shape == (2, 2, 2, 2, 3)
    npt.assert_allclose(peak_dirs, np.nan_to_num(arr))
    assert peak_values is None
    npt.assert_allclose(affine, AFFINE)
    assert path_out == str(path)
    assert loaded["images"] == []


def test_load_files_reads_pam5_through_peaks_option(tmp_path):
    path = _pam5(tmp_path)

    loaded = load_files([], peaks=[path])

    assert len(loaded["peaks"]) == 1
    peak_dirs, _affine, path_out, peak_values = loaded["peaks"][0]
    assert peak_dirs.shape == (2, 2, 2, 5, 3)
    assert peak_values is not None
    assert path_out == path


def test_load_files_rejects_non_peak_inputs(tmp_path, caplog):
    volume_path, _ = _volume(tmp_path, name="not_peaks.nii.gz")
    trk_path = _tractogram(tmp_path, "tracts.trk")

    with caplog.at_level(logging.ERROR):
        loaded = load_files([], peaks=[volume_path])
    assert loaded["peaks"] == []
    assert "is not a peaks volume" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        loaded = load_files([], peaks=[trk_path])
    assert loaded["peaks"] == []
    assert "is not supported for peaks in Skyline" in caplog.text


@pytest.mark.parametrize("sh_basis", ["descoteaux07", "tournier07"])
def test_load_files_reads_nifti_shm_coefficients(tmp_path, sh_basis):
    data = np.zeros((2, 2, 2, 6), dtype=np.float32)
    data[...] = np.arange(6, dtype=np.float32)
    path = tmp_path / "odf.nii.gz"
    save_nifti(str(path), data, AFFINE)

    loaded = load_files([], shm_coeffs=[str(path)], sh_basis=sh_basis)

    assert len(loaded["shm_coeffs"]) == 1
    coeffs, affine, coeff_path, basis = loaded["shm_coeffs"][0]
    if sh_basis == "tournier07":
        npt.assert_allclose(coeffs[0, 0, 0], [0, 5, 4, 3, 2, 1])
    else:
        npt.assert_allclose(coeffs[0, 0, 0], np.arange(6))
    assert basis == "descoteaux"
    assert coeff_path == str(path)
    npt.assert_allclose(affine, AFFINE)


def test_load_files_reads_pial_surfaces(tmp_path):
    path, vertices, faces = _pial(tmp_path)

    loaded = load_files([path])

    assert len(loaded["surfaces"]) == 1
    loaded_vertices, loaded_faces, surface_path = loaded["surfaces"][0]
    npt.assert_allclose(loaded_vertices, vertices)
    npt.assert_array_equal(loaded_faces, faces)
    assert surface_path == path


def test_load_files_reads_gifti_surfaces(tmp_path):
    vertices, faces = _mesh()
    path = _gifti(tmp_path)

    loaded = load_files([path])

    assert len(loaded["surfaces"]) == 1
    loaded_vertices, loaded_faces, _ = loaded["surfaces"][0]
    npt.assert_allclose(np.asarray(loaded_vertices), vertices)
    npt.assert_array_equal(np.asarray(loaded_faces), faces)


def test_load_files_warns_on_gifti_without_geometry(tmp_path, caplog):
    path = _gifti(
        tmp_path,
        name="empty.gii",
        vertices=np.zeros((0, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
    )

    with caplog.at_level(logging.WARNING):
        loaded = load_files([path])

    assert loaded["surfaces"] == []
    assert "does not have any surface geometry" in caplog.text


def test_load_files_reads_trk_tractograms(tmp_path):
    path = _tractogram(tmp_path, "tracts.trk")

    loaded = load_files([path])

    assert len(loaded["tractograms"]) == 1
    sft, tractogram_path = loaded["tractograms"][0]
    assert tractogram_path == path
    assert len(sft.streamlines) == 2
    npt.assert_allclose(sft.streamlines[0], _streamlines()[0], atol=1e-4)


def test_load_files_uses_the_first_image_as_reference_for_tck(tmp_path):
    image_path, _ = _volume(tmp_path)
    tck_path = _tractogram(tmp_path, "tracts.tck")

    loaded = load_files([image_path, tck_path])

    sft, _ = loaded["tractograms"][0]
    npt.assert_allclose(sft.affine, AFFINE)
    npt.assert_allclose(sft.streamlines[0], _streamlines()[0], atol=1e-4)


def test_load_files_falls_back_to_emergency_reference_for_tck(tmp_path):
    tck_path = _tractogram(tmp_path, "tracts.tck")

    loaded = load_files([tck_path])

    sft, _ = loaded["tractograms"][0]
    npt.assert_allclose(sft.dimensions, EMERGENCY_REF["dim"][1:4])
    npt.assert_allclose(sft.affine, EMERGENCY_REF.get_best_affine())


def test_emergency_reference_matches_mni_2009c_geometry():
    npt.assert_allclose(
        EMERGENCY_REF.get_best_affine(),
        np.array(
            [
                [1.0, 0.0, 0.0, -96.0],
                [0.0, 1.0, 0.0, -132.0],
                [0.0, 0.0, 1.0, -78.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    npt.assert_array_equal(EMERGENCY_REF["dim"][1:4], (193, 229, 193))
    npt.assert_allclose(EMERGENCY_REF["pixdim"][1:4], (1.0, 1.0, 1.0))


def test_load_files_ignores_npy_inputs(tmp_path):
    path = tmp_path / "pvals.npy"
    np.save(str(path), np.arange(5))

    loaded = load_files([str(path)])

    assert all(value == [] for value in loaded.values())


def test_load_files_logs_unsupported_extensions(tmp_path, caplog):
    path = tmp_path / "notes.txt"
    path.write_text("not a supported file")

    with caplog.at_level(logging.ERROR):
        loaded = load_files([str(path)])

    assert all(value == [] for value in loaded.values())
    assert "'.txt' is not supported in Skyline" in caplog.text


def test_load_files_reads_rois(tmp_path):
    path, data = _volume(tmp_path, name="roi.nii.gz")

    loaded = load_files([], rois=[path])

    assert loaded["images"] == []
    assert len(loaded["rois"]) == 1
    roi_data, roi_affine, roi_path = loaded["rois"][0]
    npt.assert_allclose(roi_data, data)
    npt.assert_allclose(roi_affine, AFFINE)
    assert roi_path == path


def test_load_files_logs_unsupported_roi_extensions(tmp_path, caplog):
    path = _tractogram(tmp_path, "tracts.trk")

    with caplog.at_level(logging.ERROR):
        loaded = load_files([], rois=[path])

    assert loaded["rois"] == []
    assert "is not supported for ROIs in Skyline" in caplog.text


def test_load_files_reads_shm_coefficients(tmp_path):
    path = _pam5(tmp_path, name="odf.pam5")

    loaded = load_files([], shm_coeffs=[path])

    assert loaded["peaks"] == []
    assert len(loaded["shm_coeffs"]) == 1
    coeffs, affine, coeff_path, basis = loaded["shm_coeffs"][0]
    assert coeffs.shape == (2, 2, 2, 45)
    npt.assert_allclose(affine, AFFINE)
    assert coeff_path == path
    assert basis == "descoteaux"


@pytest.mark.parametrize("shape", [(4, 5, 6), (2, 2, 2, 7)])
def test_load_files_rejects_nifti_without_sh_coefficients(tmp_path, caplog, shape):
    data = np.zeros(shape, dtype=np.float32)
    path = tmp_path / "not_shm.nii.gz"
    save_nifti(str(path), data, AFFINE)

    with caplog.at_level(logging.ERROR):
        loaded = load_files([], shm_coeffs=[str(path)])

    assert loaded["shm_coeffs"] == []
    assert "does not contain SH coefficients" in caplog.text


def test_load_files_combines_every_supported_input(tmp_path):
    image_path, _ = _volume(tmp_path)
    roi_path, _ = _volume(tmp_path, name="roi.nii.gz")
    pam_path = _pam5(tmp_path)
    pial_path, _, _ = _pial(tmp_path)
    trk_path = _tractogram(tmp_path, "tracts.trk")

    loaded = load_files(
        [image_path, pam_path, pial_path, trk_path],
        rois=[roi_path],
        shm_coeffs=[pam_path],
    )

    assert len(loaded["images"]) == 1
    assert len(loaded["peaks"]) == 1
    assert len(loaded["surfaces"]) == 1
    assert len(loaded["tractograms"]) == 1
    assert len(loaded["rois"]) == 1
    assert len(loaded["shm_coeffs"]) == 1


def test_load_npy_returns_the_stored_array(tmp_path):
    path = tmp_path / "pvals.npy"
    values = np.array([0.01, 0.2, 0.5])
    np.save(str(path), values)

    npt.assert_allclose(load_npy(str(path)), values)


def test_load_npy_returns_none_and_logs_on_failure(tmp_path, caplog):
    path = tmp_path / "broken.npy"
    path.write_bytes(b"definitely not a numpy file")

    with caplog.at_level(logging.ERROR):
        result = load_npy(str(path))

    assert result is None
    assert "Error loading numpy file" in caplog.text
