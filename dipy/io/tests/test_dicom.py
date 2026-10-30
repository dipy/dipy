import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from dipy.io import dicom
from dipy.io.dicom import _dcm2niix_executable, convert_dicom
from dipy.utils.optpkg import optional_package
from dipy.workflows import cli
from dipy.workflows.io import ConvertDicomFlow

pydicom, have_pydicom, _ = optional_package("pydicom", min_version="3")
pytestmark = pytest.mark.skipif(
    not have_pydicom or _dcm2niix_executable() is None,
    reason="Requires pydicom and dcm2niix",
)


def _write_series(directory, *, series=1, offset=0, diffusion=False):
    """Write synthetic classic MR slices, in deliberately reversed file order."""
    directory.mkdir(parents=True, exist_ok=True)
    pixels = np.arange(36, dtype=np.uint16).reshape(3, 3, 4) + offset
    paths = []
    for volume in range(2 if diffusion else 1):
        for plane in range(3):
            instance = volume * 3 + plane + 1
            sop_uid = f"2.25.123456789.{series}.{instance}"
            meta = pydicom.dataset.FileMetaDataset()
            meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
            meta.MediaStorageSOPInstanceUID = sop_uid
            meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            meta.ImplementationClassUID = "2.25.123456789.999"
            path = directory / f"slice {volume}-{3 - plane}.dcm"
            ds = pydicom.dataset.FileDataset(
                str(path), {}, file_meta=meta, preamble=b"\0" * 128
            )
            ds.SOPClassUID = pydicom.uid.MRImageStorage
            ds.SOPInstanceUID = sop_uid
            ds.StudyInstanceUID = "2.25.123456789"
            ds.SeriesInstanceUID = f"2.25.123456789.{series}"
            ds.FrameOfReferenceUID = "2.25.123456789.1000"
            ds.StudyID = "SYNTHETIC"
            ds.SeriesNumber = series
            ds.InstanceNumber = instance
            ds.AcquisitionNumber = 1
            ds.Modality = "MR"
            ds.Manufacturer = "SIEMENS"
            ds.ManufacturerModelName = "Synthetic"
            ds.PatientName = "Synthetic^Only"
            ds.PatientID = "DIPY-SYNTHETIC"
            ds.PatientBirthDate = "20000101"
            ds.PatientSex = "O"
            ds.StudyDate = "20200101"
            ds.StudyTime = "120000"
            ds.SeriesDate = ds.StudyDate
            ds.SeriesTime = ds.StudyTime
            ds.AcquisitionDate = ds.StudyDate
            ds.AcquisitionTime = f"12000{volume}"
            ds.ContentDate = ds.StudyDate
            ds.ContentTime = ds.AcquisitionTime
            ds.SeriesDescription = (
                "Synthetic diffusion" if diffusion else "Synthetic MR"
            )
            ds.ProtocolName = "synthetic_dwi" if diffusion else "synthetic_mr"
            ds.ImageType = ["ORIGINAL", "PRIMARY", "DIFFUSION" if diffusion else "M"]
            ds.ScanningSequence = "EP" if diffusion else "SE"
            ds.SequenceVariant = "NONE"
            ds.ScanOptions = ""
            ds.MRAcquisitionType = "2D"
            ds.RepetitionTime = 1000
            ds.EchoTime = 10
            ds.EchoNumbers = 1
            ds.FlipAngle = 90
            ds.MagneticFieldStrength = 3
            ds.PatientPosition = "HFS"
            ds.Rows = 3
            ds.Columns = 4
            ds.PixelSpacing = [2, 3]
            ds.SliceThickness = 4
            ds.SpacingBetweenSlices = 4
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [10, 20, 30 + plane * 4]
            ds.SliceLocation = 30 + plane * 4
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            if diffusion:
                # Standard diffusion attributes, without vendor-private headers.
                ds.TemporalPositionIdentifier = volume + 1
                ds.NumberOfTemporalPositions = 2
                ds.DiffusionBValue = float(volume * 1000)
                ds.DiffusionDirectionality = "DIRECTIONAL" if volume else "NONE"
                ds.DiffusionGradientOrientation = [float(volume), 0.0, 0.0]
            ds.PixelData = (pixels[plane] + volume * 100).astype("<u2").tobytes()
            ds.save_as(path, enforce_file_format=True)
            paths.append(path)
    return paths, pixels


@pytest.fixture
def dicom_series(tmp_path):
    source = tmp_path / "input with spaces"
    paths, pixels = _write_series(source)
    return source, paths, pixels


def _assert_image(path, pixels, *, first_slice=0):
    """Check both intensities and physical coordinates, independent of storage axes."""
    image = nib.as_closest_canonical(nib.load(path))
    expected = pixels.transpose(2, 1, 0)[::-1, ::-1, :]
    assert_array_equal(np.asanyarray(image.dataobj), expected)
    assert_allclose(
        image.affine,
        [[3, 0, 0, -19], [0, 2, 0, -24], [0, 0, 4, 30 + first_slice * 4], [0, 0, 0, 1]],
        atol=1e-6,
    )


def _assert_outputs(outputs, out_dir):
    assert not list(out_dir.glob(".dipy-dcm2niix-*"))
    assert set(outputs) == {"nifti", "json", "bval", "bvec"}
    for paths in outputs.values():
        assert isinstance(paths, list)
        assert paths == sorted(paths)
        for path in paths:
            assert isinstance(path, Path)
            assert path.is_absolute()
            assert path.parent == out_dir.resolve()
            assert path.is_file()


@pytest.mark.parametrize("compress", [False, True])
def test_convert_image_and_geometry(dicom_series, tmp_path, compress):
    source, _, pixels = dicom_series
    out_dir = tmp_path / "output with spaces"
    outputs = convert_dicom(source, out_dir=out_dir, compress=compress)
    _assert_outputs(outputs, out_dir)
    assert len(outputs["nifti"]) == len(outputs["json"]) == 1
    assert outputs["bval"] == outputs["bvec"] == []
    assert outputs["nifti"][0].name == (
        "001_synthetic_mr.nii.gz" if compress else "001_synthetic_mr.nii"
    )
    assert outputs["json"][0].name == "001_synthetic_mr.json"
    _assert_image(outputs["nifti"][0], pixels)
    metadata = json.loads(outputs["json"][0].read_text())
    assert metadata["Modality"] == "MR"
    assert_allclose(metadata["EchoTime"], 0.01)
    assert "PatientName" not in metadata
    assert "PatientID" not in metadata


def test_metadata_options(dicom_series, tmp_path):
    source, _, pixels = dicom_series
    outputs = convert_dicom(source, out_dir=tmp_path / "identified", anonymize=False)
    metadata = json.loads(outputs["json"][0].read_text())
    assert metadata["PatientID"] == "DIPY-SYNTHETIC"
    _assert_image(outputs["nifti"][0], pixels)

    out_dir = tmp_path / "without metadata"
    outputs = convert_dicom(source, out_dir=out_dir, bids=False)
    assert outputs["json"] == []
    assert list(out_dir.glob("*.json")) == []
    _assert_image(outputs["nifti"][0], pixels)


def test_recursive_multiple_series(tmp_path):
    source = tmp_path / "input"
    _, first = _write_series(source / "one" / "nested", series=1)
    _, second = _write_series(source / "two" / "nested", series=2, offset=1000)
    out_dir = tmp_path / "output"
    outputs = convert_dicom(source, out_dir=out_dir, filename="series_%s")
    _assert_outputs(outputs, out_dir)
    assert [path.name for path in outputs["nifti"]] == [
        "series_1.nii.gz",
        "series_2.nii.gz",
    ]
    assert [path.name for path in outputs["json"]] == ["series_1.json", "series_2.json"]
    for path, pixels in zip(outputs["nifti"], [first, second]):
        _assert_image(path, pixels)


def test_collision_preserves_existing_outputs(dicom_series, tmp_path):
    source, _, pixels = dicom_series
    out_dir = tmp_path / "output"
    first = convert_dicom(source, out_dir=out_dir, filename="same name")
    # Existing sidecars not associated with this conversion must not be reported.
    for suffix in ["nii", "nii.gz", "json", "bval", "bvec"]:
        (out_dir / f"unrelated.{suffix}").write_bytes(b"preexisting sentinel")
    original = {path: path.read_bytes() for path in out_dir.iterdir()}
    second = convert_dicom(source, out_dir=out_dir, filename="same name")
    _assert_outputs(second, out_dir)
    assert len(second["nifti"]) == len(second["json"]) == 1
    assert second["bval"] == second["bvec"] == []
    for paths in second.values():
        assert not set(paths).intersection(original)
    for path, content in original.items():
        assert path.read_bytes() == content
    _assert_image(first["nifti"][0], pixels)
    _assert_image(second["nifti"][0], pixels)


def test_orphan_sidecar_is_not_overwritten(dicom_series, tmp_path):
    source, _, pixels = dicom_series
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    orphan = out_dir / "scan.json"
    orphan.write_bytes(b"existing metadata without an image")
    outputs = convert_dicom(source, out_dir=out_dir, filename="scan")
    assert orphan.read_bytes() == b"existing metadata without an image"
    assert len(outputs["json"]) == len(outputs["nifti"]) == 1
    assert outputs["json"][0] != orphan
    assert outputs["nifti"][0].name == outputs["json"][0].stem + ".nii.gz"
    _assert_image(outputs["nifti"][0], pixels)


def test_single_file_does_not_convert_neighboring_slices(dicom_series, tmp_path):
    _, paths, pixels = dicom_series
    outputs = convert_dicom(paths[1], out_dir=tmp_path / "output")
    assert len(outputs["nifti"]) == 1
    _assert_image(outputs["nifti"][0], pixels[1:2], first_slice=1)


def test_resolves_relative_paths(dicom_series, tmp_path, monkeypatch):
    source, _, pixels = dicom_series
    monkeypatch.chdir(tmp_path)
    outputs = convert_dicom(source.name, out_dir="relative output")
    _assert_outputs(outputs, tmp_path / "relative output")
    _assert_image(outputs["nifti"][0], pixels)


def test_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_dicom(tmp_path / "missing", out_dir=tmp_path / "output")


@pytest.mark.parametrize("kind", ["empty", "malformed_directory", "malformed_file"])
def test_input_without_images_fails(tmp_path, kind):
    source = tmp_path / "input"
    source.mkdir()
    if kind != "empty":
        path = source / "bad.dcm"
        path.write_bytes(b"not a DICOM image\n" * 64)
        if kind == "malformed_file":
            source = path
    out_dir = tmp_path / "output"
    with pytest.raises(RuntimeError, match="exit status") as exc_info:
        convert_dicom(source, out_dir=out_dir)
    assert "dcm2niiX version" in str(exc_info.value)
    assert list(out_dir.iterdir()) == []


def test_partial_failure_keeps_converted_series(tmp_path):
    source = tmp_path / "input"
    _, pixels = _write_series(source / "good", series=1)
    paths, _ = _write_series(source / "truncated", series=2)
    paths[0].write_bytes(paths[0].read_bytes()[:-10])
    out_dir = tmp_path / "output"
    outputs = convert_dicom(source, out_dir=out_dir, filename="series_%s")
    _assert_outputs(outputs, out_dir)
    assert [path.name for path in outputs["nifti"]] == ["series_1.nii.gz"]
    _assert_image(outputs["nifti"][0], pixels)


def test_collision_suffix_is_shared_by_all_outputs(tmp_path):
    source = tmp_path / "input"
    _write_series(source / "one", series=1)
    _write_series(source / "two", series=2, offset=1000)
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    # dcm2niix names the second series "scana"; only that name is taken.
    (out_dir / "scana.json").write_text("{}")
    outputs = convert_dicom(source, out_dir=out_dir, filename="scan")
    _assert_outputs(outputs, out_dir)
    assert [path.name for path in outputs["nifti"]] == [
        "scan_1.nii.gz",
        "scana_1.nii.gz",
    ]
    assert [path.name for path in outputs["json"]] == ["scan_1.json", "scana_1.json"]


def test_converter_lookup(dicom_series, tmp_path, monkeypatch):
    source, _, pixels = dicom_series
    executable = _dcm2niix_executable()
    monkeypatch.setattr(dicom, "have_dcm2niix", False)
    monkeypatch.setattr(dicom.shutil, "which", lambda name: executable)
    outputs = convert_dicom(source, out_dir=tmp_path / "output")
    _assert_image(outputs["nifti"][0], pixels)
    monkeypatch.setattr(dicom.shutil, "which", lambda name: None)
    with pytest.raises(ImportError, match="pip install dcm2niix"):
        convert_dicom(source, out_dir=tmp_path / "output")


@pytest.mark.parametrize("filename", ["", ".", "..", "../escape", r"a\b", "a\0b"])
def test_rejects_filename_paths(dicom_series, tmp_path, filename):
    source, _, _ = dicom_series
    with pytest.raises(ValueError):
        convert_dicom(source, out_dir=tmp_path / "output", filename=filename)


def test_diffusion_sidecars(tmp_path):
    source = tmp_path / "diffusion"
    _, pixels = _write_series(source, diffusion=True)
    outputs = convert_dicom(source, out_dir=tmp_path / "output")
    assert len(outputs["nifti"]) == len(outputs["bval"]) == len(outputs["bvec"]) == 1
    bvals = np.loadtxt(outputs["bval"][0])
    bvecs = np.loadtxt(outputs["bvec"][0])
    assert_array_equal(bvals, [0, 1000])
    assert bvecs.shape == (3, 2)
    assert_allclose(bvecs[:, 0], 0, atol=1e-6)
    # Axial LPS x points along the image x axis; its sign is immaterial to DWI.
    assert_allclose(np.abs(bvecs[:, 1]), [1, 0, 0], atol=1e-6)
    image = nib.as_closest_canonical(nib.load(outputs["nifti"][0]))
    expected = pixels.transpose(2, 1, 0)[::-1, ::-1, :]
    assert_array_equal(
        np.asanyarray(image.dataobj), np.stack([expected, expected + 100], axis=-1)
    )


def test_workflow_tracks_only_latest_outputs(dicom_series, tmp_path):
    source, _, pixels = dicom_series
    flow = ConvertDicomFlow()
    out_dir = tmp_path / "workflow output"
    flow.run(str(source), out_dir=str(out_dir))
    first = {key: list(paths) for key, paths in flow.last_generated_outputs.items()}
    assert set(first) == {"out_nifti", "out_json", "out_bval", "out_bvec"}
    assert len(first["out_nifti"]) == len(first["out_json"]) == 1
    assert first["out_bval"] == first["out_bvec"] == []
    _assert_image(first["out_nifti"][0], pixels)
    flow.run(str(source), compress=False, bids=False, out_dir=str(out_dir))
    second = flow.last_generated_outputs
    assert second["out_json"] == second["out_bval"] == second["out_bvec"] == []
    assert len(second["out_nifti"]) == 1
    assert second["out_nifti"][0].endswith(".nii")
    for paths in second.values():
        assert isinstance(paths, list)
        assert all(isinstance(path, str) for path in paths)
    assert Path(first["out_nifti"][0]).is_file()
    _assert_image(second["out_nifti"][0], pixels)


@pytest.mark.parametrize(
    "options, compress, bids, anonymize",
    [
        ([], True, True, True),
        (["--no-compress", "--no-bids", "--no-anonymize"], False, False, False),
        (["--no-anonymize"], True, True, False),
    ],
)
def test_cli_conversion(
    dicom_series, tmp_path, monkeypatch, options, compress, bids, anonymize
):
    source, _, pixels = dicom_series
    out_dir = tmp_path / "CLI output with spaces"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dipy_convert_dicom",
            str(source),
            "--out_dir",
            str(out_dir),
            "--filename",
            "cli_%s",
            *options,
        ],
    )
    cli.run()
    nifti = out_dir / ("cli_1.nii.gz" if compress else "cli_1.nii")
    _assert_image(nifti, pixels)
    sidecar = out_dir / "cli_1.json"
    if not bids:
        assert not sidecar.exists()
    else:
        metadata = json.loads(sidecar.read_text())
        assert metadata["Modality"] == "MR"
        if anonymize:
            assert "PatientID" not in metadata
            assert "PatientName" not in metadata
        else:
            assert metadata["PatientID"] == "DIPY-SYNTHETIC"


def test_existing_nifti_does_not_mask_empty_input(dicom_series, tmp_path):
    source, _, pixels = dicom_series
    out_dir = tmp_path / "output"
    outputs = convert_dicom(source, out_dir=out_dir)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError):
        convert_dicom(empty, out_dir=out_dir)
    _assert_image(outputs["nifti"][0], pixels)
