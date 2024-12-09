import importlib
from inspect import getmembers, isfunction
import logging
import os
from os.path import join as pjoin
import sys
from tempfile import TemporaryDirectory, mkstemp

import numpy as np
import numpy.testing as npt

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
from dipy.workflows.io import (
    ConcatenateTractogramFlow,
    ConvertSHFlow,
    ConvertTensorsFlow,
    ConvertTractogramFlow,
    FetchFlow,
    IoInfoFlow,
    NiftisToPamFlow,
    PamToNiftisFlow,
    SplitFlow,
    TensorToPamFlow,
)

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
    npt.assert_raises(TypeError, io_info_flow.run, filepath_dix["gs.tck"])

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

    ignored_fetchers = ["fetch_hbn", "fetch_hcp", "fetch_data"]
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
