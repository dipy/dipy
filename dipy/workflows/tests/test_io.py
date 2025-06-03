import importlib
from inspect import getmembers, isfunction
import logging
import os
from os.path import join as pjoin
import shutil
import sys
from tempfile import TemporaryDirectory, mkstemp

import numpy as np
import numpy.testing as npt
import pytest

import dipy.core.gradients as grad
from dipy.data import default_sphere, get_fnames
from dipy.data.fetcher import dipy_home
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.image import load_nifti, save_nifti
from dipy.io.peaks import load_pam, save_pam
from dipy.io.streamline import load_tractogram
from dipy.io.utils import nifti1_symmat
from dipy.reconst import dti, utils as reconst_utils
from dipy.reconst.shm import convert_sh_descoteaux_tournier
from dipy.testing import assert_true
from dipy.utils.optpkg import optional_package
from dipy.utils.tripwire import TripWireError
from dipy.workflows.io import (
    ConcatenateTractogramFlow,
    ConvertSHFlow,
    ConvertTensorsFlow,
    ConvertTractogramFlow,
    ExtractB0Flow,
    ExtractShellFlow,
    ExtractVolumeFlow,
    FetchFlow,
    IoInfoFlow,
    MathFlow,
    NiftisToPamFlow,
    PamToNiftisFlow,
    SplitFlow,
    TensorToPamFlow,
)

ne, have_ne, _ = optional_package("numexpr")

fname_log = mkstemp()[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    filename=fname_log,
    filemode="w",
)

is_big_endian = "big" in sys.byteorder.lower()


def test_io_info():
    fimg, fbvals, fbvecs = get_fnames(name="small_101D")
    io_info_flow = IoInfoFlow()
    io_info_flow.run([fimg, fbvals, fbvecs])

    fimg, fbvals, fvecs = get_fnames(name="small_25")
    io_info_flow = IoInfoFlow()
    io_info_flow.run([fimg, fbvals, fvecs])

    io_info_flow = IoInfoFlow()
    io_info_flow.run([fimg, fbvals, fvecs], b0_threshold=20, bvecs_tol=0.001)

    filepath_dix, _, _ = get_fnames(name="gold_standard_tracks")
    if not is_big_endian:
        io_info_flow = IoInfoFlow()
        io_info_flow.run(filepath_dix["gs.trx"])

    io_info_flow = IoInfoFlow()
    io_info_flow.run(filepath_dix["gs.trk"])

    io_info_flow = IoInfoFlow()
    npt.assert_raises(SystemExit, io_info_flow.run, filepath_dix["gs.tck"])
    io_info_flow = IoInfoFlow()
    npt.assert_raises(OSError, io_info_flow.run, "fake.vtk")

    io_info_flow = IoInfoFlow()
    io_info_flow.run(filepath_dix["gs.tck"], reference=filepath_dix["gs.nii"])

    with open(fname_log, "r") as file:
        lines = file.readlines()
        try:
            npt.assert_equal(lines[-3], "INFO Total number of unit bvectors 25\n")
        except IndexError:  # logging maybe disabled in IDE setting
            pass


def test_io_fetch():
    fetch_flow = FetchFlow()
    with TemporaryDirectory() as out_dir:
        fetch_flow.run(["bundle_fa_hcp"])
        npt.assert_equal(os.path.isdir(os.path.join(dipy_home, "bundle_fa_hcp")), True)

        fetch_flow.run(["bundle_fa_hcp"], out_dir=out_dir)
        npt.assert_equal(os.path.isdir(os.path.join(out_dir, "bundle_fa_hcp")), True)


def test_io_fetch_fetcher_datanames():
    available_data = FetchFlow.get_fetcher_datanames()

    module_path = "dipy.data.fetcher"
    if module_path in sys.modules:
        fetcher_module = importlib.reload(sys.modules[module_path])
    else:
        fetcher_module = importlib.import_module(module_path)

    ignored_fetchers = ["fetch_data"]
    fetcher_list = {
        name.replace("fetch_", ""): func
        for name, func in getmembers(fetcher_module, isfunction)
        if name.lower().startswith("fetch_") and name.lower() not in ignored_fetchers
    }

    num_expected_fetch_methods = len(fetcher_list)
    npt.assert_equal(len(available_data), num_expected_fetch_methods)
    npt.assert_equal(
        all(dataset_name in available_data.keys() for dataset_name in fetcher_list),
        True,
    )


def test_split_flow():
    with TemporaryDirectory() as out_dir:
        split_flow = SplitFlow()
        data_path, _, _ = get_fnames()
        volume, affine = load_nifti(data_path)
        split_flow.run(data_path, out_dir=out_dir)
        assert_true(os.path.isfile(split_flow.last_generated_outputs["out_split"]))
        split_flow._force_overwrite = True
        split_flow.run(data_path, vol_idx=0, out_dir=out_dir)
        split_path = split_flow.last_generated_outputs["out_split"]
        assert_true(os.path.isfile(split_path))
        split_data, split_affine = load_nifti(split_path)
        npt.assert_equal(split_data.shape, volume[..., 0].shape)
        npt.assert_array_almost_equal(split_affine, affine)


def test_concatenate_flow():
    with TemporaryDirectory() as out_dir:
        concatenate_flow = ConcatenateTractogramFlow()
        data_path, _, _ = get_fnames(name="gold_standard_tracks")
        input_files = [
            v
            for k, v in data_path.items()
            if k in ["gs.trk", "gs.tck", "gs.trx", "gs.fib"]
        ]
        concatenate_flow.run(*input_files, out_dir=out_dir)
        assert_true(
            concatenate_flow.last_generated_outputs["out_extension"].endswith("trx")
        )
        assert_true(
            os.path.isfile(
                concatenate_flow.last_generated_outputs["out_tractogram"] + ".trx"
            )
        )

        trk = load_tractogram(
            concatenate_flow.last_generated_outputs["out_tractogram"] + ".trx", "same"
        )
        npt.assert_equal(len(trk), 13)


def test_convert_sh_flow():
    with TemporaryDirectory() as out_dir:
        filepath_in = os.path.join(out_dir, "sh_coeff_img.nii.gz")
        filename_out = "sh_coeff_img_converted.nii.gz"
        filepath_out = os.path.join(out_dir, filename_out)

        # Create an input image
        dim0, dim1, dim2 = 2, 3, 3  # spatial dimensions of array
        num_sh_coeffs = 15  # 15 sh coeffs means l_max is 4
        img_in = np.arange(dim0 * dim1 * dim2 * num_sh_coeffs, dtype=float).reshape(
            dim0, dim1, dim2, num_sh_coeffs
        )
        save_nifti(filepath_in, img_in, np.eye(4))

        # Compute expected result to compare against later
        expected_img_out = convert_sh_descoteaux_tournier(img_in)

        # Run the workflow and load the output
        workflow = ConvertSHFlow()
        workflow.run(
            filepath_in,
            out_dir=out_dir,
            out_file=filename_out,
        )
        img_out, _ = load_nifti(filepath_out)

        # Compare
        npt.assert_array_almost_equal(img_out, expected_img_out)


def test_convert_tractogram_flow():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames(name="gold_standard_tracks")
        input_files = [
            v
            for k, v in data_path.items()
            if k
            in [
                "gs.tck",
            ]
        ]

        convert_tractogram_flow = ConvertTractogramFlow(mix_names=True)
        convert_tractogram_flow.run(
            input_files, reference=data_path["gs.nii"], out_dir=out_dir
        )

        convert_tractogram_flow._force_overwrite = True
        npt.assert_raises(
            ValueError, convert_tractogram_flow.run, input_files, out_dir=out_dir
        )

        if not is_big_endian:
            npt.assert_warns(
                UserWarning,
                convert_tractogram_flow.run,
                data_path["gs.trx"],
                out_dir=out_dir,
                out_tractogram="gs_converted.trx",
            )


def test_convert_tensors_flow():
    with TemporaryDirectory() as out_dir:
        filepath_in = os.path.join(out_dir, "tensors_img.nii.gz")
        filename_out = "tensors_converted.nii.gz"
        filepath_out = os.path.join(out_dir, filename_out)

        # Create an input image
        fdata, fbval, fbvec = get_fnames(name="small_25")
        data, affine = load_nifti(fdata)
        gtab = grad.gradient_table(fbval, bvecs=fbvec)
        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(data)

        tensor_vals = dti.lower_triangular(tenfit.quadratic_form)
        ten_img = nifti1_symmat(tensor_vals, affine=affine)

        save_nifti(filepath_in, ten_img.get_fdata().squeeze(), affine)

        # Compute expected result to compare against later
        expected_img_out = reconst_utils.convert_tensors(
            ten_img.get_fdata(), "dipy", "mrtrix"
        )

        # Run the workflow and load the output
        workflow = ConvertTensorsFlow()
        workflow.run(
            filepath_in,
            from_format="dipy",
            to_format="mrtrix",
            out_dir=out_dir,
            out_tensor=filename_out,
        )

        img_out, _ = load_nifti(filepath_out)
        npt.assert_array_almost_equal(img_out, expected_img_out)


def generate_random_pam():
    pam = PeaksAndMetrics()
    pam.affine = np.eye(4)
    pam.peak_dirs = np.random.rand(15, 15, 15, 5, 3)
    pam.peak_values = np.zeros((15, 15, 15, 5))
    pam.peak_indices = np.zeros((15, 15, 15, 5))
    pam.shm_coeff = np.zeros((15, 15, 15, 45))
    pam.sphere = default_sphere
    pam.B = np.zeros((45, default_sphere.vertices.shape[0]))
    pam.total_weight = 0.5
    pam.ang_thr = 60
    pam.gfa = np.zeros((10, 10, 10))
    pam.qa = np.zeros((10, 10, 10, 5))
    pam.odf = np.zeros((10, 10, 10, default_sphere.vertices.shape[0]))
    return pam


def test_niftis_to_pam_flow():
    pam = generate_random_pam()
    with TemporaryDirectory() as out_dir:
        fname = pjoin(out_dir, "test.pam5")
        save_pam(fname, pam)

        args = [fname, out_dir]
        flow = PamToNiftisFlow()
        flow.run(*args)

        args = [
            flow.last_generated_outputs["out_peaks_dir"],
            flow.last_generated_outputs["out_peaks_values"],
            flow.last_generated_outputs["out_peaks_indices"],
        ]

        flow2 = NiftisToPamFlow()
        flow2.run(*args, out_dir=out_dir)
        pam_file = flow2.last_generated_outputs["out_pam"]
        assert_true(os.path.isfile(pam_file))

        res_pam = load_pam(pam_file)
        npt.assert_array_equal(pam.affine, res_pam.affine)
        npt.assert_array_almost_equal(pam.peak_dirs, res_pam.peak_dirs)
        npt.assert_array_almost_equal(pam.peak_values, res_pam.peak_values)
        npt.assert_array_almost_equal(pam.peak_indices, res_pam.peak_indices)


def test_tensor_to_pam_flow():
    fdata, fbval, fbvec = get_fnames(name="small_25")
    gtab = grad.gradient_table(fbval, bvecs=fbvec)
    data, affine = load_nifti(fdata)
    dm = dti.TensorModel(gtab)
    df = dm.fit(data)
    df.evals[0, 0, 0] = np.array([0, 0, 0])

    with TemporaryDirectory() as out_dir:
        f_mevals, f_mevecs = (
            pjoin(out_dir, "evals.nii.gz"),
            pjoin(out_dir, "evecs.nii.gz"),
        )
        save_nifti(f_mevals, df.evals, affine)
        save_nifti(f_mevecs, df.evecs, affine)

        args = [f_mevals, f_mevecs]
        flow = TensorToPamFlow()
        flow.run(*args, out_dir=out_dir)
        pam_file = flow.last_generated_outputs["out_pam"]
        assert_true(os.path.isfile(pam_file))

        pam = load_pam(pam_file)
        npt.assert_array_equal(pam.affine, affine)
        npt.assert_array_almost_equal(pam.peak_dirs[..., :3, :], df.evecs)
        npt.assert_array_almost_equal(pam.peak_values[..., :3], df.evals)


def test_pam_to_niftis_flow():
    pam = generate_random_pam()

    with TemporaryDirectory() as out_dir:
        fname = pjoin(out_dir, "test.pam5")
        save_pam(fname, pam)

        args = [fname, out_dir]
        flow = PamToNiftisFlow()
        flow.run(*args)
        assert_true(os.path.isfile(flow.last_generated_outputs["out_peaks_dir"]))
        assert_true(os.path.isfile(flow.last_generated_outputs["out_peaks_values"]))
        assert_true(os.path.isfile(flow.last_generated_outputs["out_peaks_indices"]))
        assert_true(os.path.isfile(flow.last_generated_outputs["out_shm"]))
        assert_true(os.path.isfile(flow.last_generated_outputs["out_gfa"]))
        assert_true(os.path.isfile(flow.last_generated_outputs["out_sphere"]))


def test_math():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames(name="small_101D")
        data_path_a = pjoin(out_dir, "data_a.nii.gz")
        data_path_b = pjoin(out_dir, "data_b.nii.gz")
        shutil.copy(data_path, data_path_a)
        shutil.copy(data_path, data_path_b)

        data, _ = load_nifti(data_path)
        operations = ["vol1*3", "vol1+vol2+vol3", "5*vol1-vol2-vol3", "vol3*2 + vol2"]
        kwargs = [{"dtype": "i"}, {"dtype": "float32"}, {}, {}]

        if have_ne:
            for op, kwarg in zip(operations, kwargs):
                math_flow = MathFlow()
                math_flow.run(
                    op, [data_path_a, data_path_b, data_path], out_dir=out_dir, **kwarg
                )
                out_path = pjoin(out_dir, "math_out.nii.gz")
                out_data, _ = load_nifti(out_path)
                npt.assert_array_equal(out_data, data * 3)
                if kwarg:
                    npt.assert_equal(out_data.dtype, np.dtype(kwarg["dtype"]))

            # Test broadcasting 3D/4D
            data_3d = np.ones(data.shape[:-1]) * 15
            data_3d_path = pjoin(out_dir, "data_3d.nii.gz")
            save_nifti(data_3d_path, data_3d, np.eye(4))
            math_flow = MathFlow()
            math_flow.run(
                "vol1*vol2",
                [data_path_a, data_3d_path],
                disable_check=True,
                out_dir=out_dir,
            )

            # Test boolean data type
            data_bool = np.ones(data.shape, dtype=np.uint8)
            data_bool_2 = np.zeros(data.shape, dtype=np.uint8)
            data_bool_path = pjoin(out_dir, "data_bool.nii.gz")
            data_bool_2_path = pjoin(out_dir, "data_bool_2.nii.gz")
            save_nifti(data_bool_path, data_bool, np.eye(4))
            save_nifti(data_bool_2_path, data_bool_2, np.eye(4))
            math_flow = MathFlow()
            math_flow.run(
                "vol1*vol2",
                [data_bool_path, data_bool_2_path],
                disable_check=True,
                dtype="bool",
                out_dir=out_dir,
            )
            out_path = pjoin(out_dir, "math_out.nii.gz")
            out_data, _ = load_nifti(out_path)
            npt.assert_array_equal(out_data, data_bool * data_bool_2)
            npt.assert_equal(out_data.dtype, np.uint8)

        else:
            math_flow = MathFlow()
            npt.assert_raises(TripWireError, math_flow.run, "vol1*3", [data_path_a])


@pytest.mark.skipif(not have_ne, reason="numexpr not installed")
def test_math_error():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames(name="small_101D")
        data_path_2, _, _ = get_fnames(name="small_64D")
        data_path_a = pjoin(out_dir, "data_a.nii.gz")
        data_path_b = pjoin(out_dir, "data_b.gz")
        data_path_c = pjoin(out_dir, "data_c.nii")
        shutil.copy(data_path, data_path_a)
        shutil.copy(data_path, data_path_b)

        math_flow = MathFlow()
        npt.assert_raises(
            SyntaxError, math_flow.run, "vol1*", [data_path_a], out_dir=out_dir
        )
        npt.assert_raises(
            SystemExit,
            math_flow.run,
            "vol1*2",
            [data_path_a],
            dtype="k",
            out_dir=out_dir,
        )
        npt.assert_raises(
            SystemExit, math_flow.run, "vol1*2", [data_path_b], out_dir=out_dir
        )
        npt.assert_raises(
            SystemExit, math_flow.run, "vol1*2", [data_path_c], out_dir=out_dir
        )
        npt.assert_raises(
            SystemExit, math_flow.run, "vol1*vol3", [data_path_a], out_dir=out_dir
        )
        npt.assert_raises(
            SystemExit,
            math_flow.run,
            "vol1*vol2",
            [data_path, data_path_2],
            out_dir=out_dir,
        )


def test_extract_b0_flow():
    with TemporaryDirectory() as out_dir:
        fdata, fbval, fbvec = get_fnames(name="small_25")
        data, affine = load_nifti(fdata)
        b0_data = data[..., 0]
        b0_path = pjoin(out_dir, "b0_expected.nii.gz")
        save_nifti(b0_path, b0_data, affine)

        extract_b0_flow = ExtractB0Flow()
        extract_b0_flow.run(fdata, fbval, out_dir=out_dir, strategy="first")
        npt.assert_equal(
            extract_b0_flow.last_generated_outputs["out_b0"],
            pjoin(out_dir, "b0.nii.gz"),
        )
        res, _ = load_nifti(extract_b0_flow.last_generated_outputs["out_b0"])
        npt.assert_array_equal(res, b0_data)


def test_extract_shell_flow():
    with TemporaryDirectory() as out_dir:
        fdata, fbval, fbvec = get_fnames(name="small_25")
        data, affine = load_nifti(fdata)

        extract_shell_flow = ExtractShellFlow()
        extract_shell_flow.run(
            fdata, fbval, fbvec, bvals_to_extract="2000", out_dir=out_dir
        )
        res, _ = load_nifti(pjoin(out_dir, "shell_2000.nii.gz"))
        npt.assert_array_equal(res, data[..., 1:])

        extract_shell_flow._force_overwrite = True
        extract_shell_flow.run(
            fdata,
            fbval,
            fbvec,
            bvals_to_extract="0, 2000",
            group_shells=False,
            out_dir=out_dir,
        )
        npt.assert_equal(os.path.isfile(pjoin(out_dir, "shell_0.nii.gz")), True)
        npt.assert_equal(os.path.isfile(pjoin(out_dir, "shell_2000.nii.gz")), True)
        res0, _ = load_nifti(pjoin(out_dir, "shell_0.nii.gz"))
        res2000, _ = load_nifti(pjoin(out_dir, "shell_2000.nii.gz"))
        npt.assert_array_equal(np.squeeze(res0), data[..., 0])
        npt.assert_array_equal(res2000[..., 9], data[..., 10])


def test_extract_volume_flow():
    with TemporaryDirectory() as out_dir:
        fdata, _, _ = get_fnames(name="small_25")
        data, affine = load_nifti(fdata)

        extract_volume_flow = ExtractVolumeFlow()
        extract_volume_flow.run(fdata, vol_idx="0-3,5", out_dir=out_dir)
        res, _ = load_nifti(extract_volume_flow.last_generated_outputs["out_vol"])
        npt.assert_equal(res.shape[-1], 5)

        extract_volume_flow._force_overwrite = True
        extract_volume_flow.run(fdata, vol_idx="0-3,5", grouped=False, out_dir=out_dir)
        npt.assert_equal(os.path.isfile(pjoin(out_dir, "volume_2.nii.gz")), True)
        npt.assert_equal(os.path.isfile(pjoin(out_dir, "volume_5.nii.gz")), True)
        res2, _ = load_nifti(pjoin(out_dir, "volume_2.nii.gz"))
        res5, _ = load_nifti(pjoin(out_dir, "volume_5.nii.gz"))
        npt.assert_array_equal(res2, data[..., 2])
        npt.assert_array_equal(res5, data[..., 5])
