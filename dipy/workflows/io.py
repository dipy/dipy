import importlib
from inspect import getmembers, isfunction
import logging
import os
import sys
import warnings

import numpy as np
import trx.trx_file_memmap as tmm

from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.io.image import load_nifti, save_nifti
from dipy.io.peaks import (
    load_pam,
    niftis_to_pam,
    pam_to_niftis,
    tensor_to_pam,
)
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.reconst.shm import convert_sh_descoteaux_tournier
from dipy.reconst.utils import convert_tensors
from dipy.tracking.streamlinespeed import length
from dipy.utils.tractogram import concatenate_tractogram
from dipy.workflows.workflow import Workflow


class IoInfoFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "io_info"

    def run(
        self,
        input_files,
        b0_threshold=50,
        bvecs_tol=0.01,
        bshell_thr=100,
        reference=None,
    ):
        """Provides useful information about different files used in
        medical imaging. Any number of input files can be provided. The
        program identifies the type of file by its extension.

        Parameters
        ----------
        input_files : variable string
            Any number of Nifti1, bvals or bvecs files.
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Threshold used to check that norm(bvec) = 1 +/- bvecs_tol
            b-vectors are unit vectors.
        bshell_thr : float, optional
            Threshold for distinguishing b-values in different shells.
        reference : string, optional
            Reference anatomy for tck/vtk/fib/dpy file.
            support (.nii or .nii.gz).

        """
        np.set_printoptions(3, suppress=True)

        io_it = self.get_io_iterator()

        for input_path in io_it:
            mult_ = len(input_path)
            logging.info(f"-----------{mult_ * '-'}")
            logging.info(f"Looking at {input_path}")
            logging.info(f"-----------{mult_ * '-'}")

            ipath_lower = input_path.lower()
            extension = os.path.splitext(ipath_lower)[1]

            if ipath_lower.endswith(".nii") or ipath_lower.endswith(".nii.gz"):
                data, affine, img, vox_sz, affcodes = load_nifti(
                    input_path, return_img=True, return_voxsize=True, return_coords=True
                )
                logging.info(f"Data size {data.shape}")
                logging.info(f"Data type {data.dtype}")

                if data.ndim == 3:
                    logging.info(
                        f"Data min {data.min()} max {data.max()} avg {data.mean()}"
                    )
                    logging.info(
                        f"2nd percentile {np.percentile(data, 2)} "
                        f"98th percentile {np.percentile(data, 98)}"
                    )
                if data.ndim == 4:
                    logging.info(
                        f"Data min {data[..., 0].min()} "
                        f"max {data[..., 0].max()} "
                        f"avg {data[..., 0].mean()} of vol 0"
                    )
                    msg = (
                        f"2nd percentile {np.percentile(data[..., 0], 2)} "
                        f"98th percentile {np.percentile(data[..., 0], 98)} "
                        f"of vol 0"
                    )
                    logging.info(msg)
                logging.info(f"Native coordinate system {''.join(affcodes)}")
                logging.info(f"Affine Native to RAS matrix \n{affine}")
                logging.info(f"Voxel size {np.array(vox_sz)}")
                if np.sum(np.abs(np.diff(vox_sz))) > 0.1:
                    msg = "Voxel size is not isotropic. Please reslice.\n"
                    logging.warning(msg, stacklevel=2)

            if os.path.basename(input_path).lower().find("bval") > -1:
                bvals = np.loadtxt(input_path)
                logging.info(f"b-values \n{bvals}")
                logging.info(f"Total number of b-values {len(bvals)}")
                shells = np.sum(np.diff(np.sort(bvals)) > bshell_thr)
                logging.info(f"Number of gradient shells {shells}")
                logging.info(
                    f"Number of b0s {np.sum(bvals <= b0_threshold)} "
                    f"(b0_thr {b0_threshold})\n"
                )

            if os.path.basename(input_path).lower().find("bvec") > -1:
                bvecs = np.loadtxt(input_path)
                logging.info(f"Bvectors shape on disk is {bvecs.shape}")
                rows, cols = bvecs.shape
                if rows < cols:
                    bvecs = bvecs.T
                logging.info(f"Bvectors are \n{bvecs}")
                norms = np.array([np.linalg.norm(bvec) for bvec in bvecs])
                res = np.where((norms <= 1 + bvecs_tol) & (norms >= 1 - bvecs_tol))
                ncl1 = np.sum(norms < 1 - bvecs_tol)
                logging.info(f"Total number of unit bvectors {len(res[0])}")
                logging.info(f"Total number of non-unit bvectors {ncl1}\n")

            if extension in [".trk", ".tck", ".trx", ".vtk", ".vtp", ".fib", ".dpy"]:
                sft = None
                if extension in [".trk", ".trx"]:
                    sft = load_tractogram(input_path, "same", bbox_valid_check=False)
                else:
                    sft = load_tractogram(input_path, reference, bbox_valid_check=False)

                lengths_mm = list(length(sft.streamlines))

                sft.to_voxmm()

                lengths, steps = [], []
                for streamline in sft.streamlines:
                    lengths += [len(streamline)]
                    steps += [np.sqrt(np.sum(np.diff(streamline, axis=0) ** 2, axis=1))]
                steps = np.hstack(steps)

                logging.info(f"Number of streamlines: {len(sft)}")
                logging.info(f"min_length_mm: {float(np.min(lengths_mm))}")
                logging.info(f"mean_length_mm: {float(np.mean(lengths_mm))}")
                logging.info(f"max_length_mm: {float(np.max(lengths_mm))}")
                logging.info(f"std_length_mm: {float(np.std(lengths_mm))}")
                logging.info(f"min_length_nb_points: {float(np.min(lengths))}")
                logging.info("mean_length_nb_points: " f"{float(np.mean(lengths))}")
                logging.info(f"max_length_nb_points: {float(np.max(lengths))}")
                logging.info(f"std_length_nb_points: {float(np.std(lengths))}")
                logging.info(f"min_step_size: {float(np.min(steps))}")
                logging.info(f"mean_step_size: {float(np.mean(steps))}")
                logging.info(f"max_step_size: {float(np.max(steps))}")
                logging.info(f"std_step_size: {float(np.std(steps))}")
                logging.info(
                    "data_per_point_keys: " f"{list(sft.data_per_point.keys())}"
                )
                logging.info(
                    "data_per_streamline_keys: "
                    f"{list(sft.data_per_streamline.keys())}"
                )

        np.set_printoptions()


class FetchFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "fetch"

    @staticmethod
    def get_fetcher_datanames():
        """Gets available dataset and function names.

        Returns
        -------
        available_data: dict
            Available dataset and function names.

        """

        fetcher_module = FetchFlow.load_module("dipy.data.fetcher")

        available_data = dict(
            {
                (name.replace("fetch_", ""), func)
                for name, func in getmembers(fetcher_module, isfunction)
                if name.lower().startswith("fetch_")
                and func is not fetcher_module.fetch_data
                if name.lower() not in ["fetch_hbn", "fetch_hcp"]
            }
        )

        return available_data

    @staticmethod
    def load_module(module_path):
        """Load / reload an external module.

        Parameters
        ----------
        module_path: string
            the path to the module relative to the main script

        Returns
        -------
        module: module object

        """
        if module_path in sys.modules:
            return importlib.reload(sys.modules[module_path])
        else:
            return importlib.import_module(module_path)

    def run(self, data_names, out_dir=""):
        """Download files to folder and check their md5 checksums.

        To see all available datasets, please type "list" in data_names.

        Parameters
        ----------
        data_names : variable string
            Any number of Nifti1, bvals or bvecs files.
        out_dir : string, optional
            Output directory. (default current directory)

        """
        if out_dir:
            dipy_home = os.environ.get("DIPY_HOME", None)
            os.environ["DIPY_HOME"] = out_dir

        available_data = FetchFlow.get_fetcher_datanames()

        data_names = [name.lower() for name in data_names]

        if "all" in data_names:
            for name, fetcher_function in available_data.items():
                logging.info("------------------------------------------")
                logging.info(f"Fetching at {name}")
                logging.info("------------------------------------------")
                fetcher_function()

        elif "list" in data_names:
            logging.info(
                "Please, select between the following data names: "
                f"{', '.join(available_data.keys())}"
            )

        else:
            skipped_names = []
            for data_name in data_names:
                if data_name not in available_data.keys():
                    skipped_names.append(data_name)
                    continue

                logging.info("------------------------------------------")
                logging.info(f"Fetching at {data_name}")
                logging.info("------------------------------------------")
                available_data[data_name]()

            nb_success = len(data_names) - len(skipped_names)
            print("\n")
            logging.info(f"Fetched {nb_success} / {len(data_names)} Files ")
            if skipped_names:
                logging.warn(f"Skipped data name(s): {' '.join(skipped_names)}")
                logging.warn(
                    "Please, select between the following data names: "
                    f"{', '.join(available_data.keys())}"
                )

        if out_dir:
            if dipy_home:
                os.environ["DIPY_HOME"] = dipy_home
            else:
                os.environ.pop("DIPY_HOME", None)

            # We load the module again so that if we run another one of these
            # in the same process, we don't have the env variable pointing
            # to the wrong place
            self.load_module("dipy.data.fetcher")


class SplitFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "split"

    def run(self, input_files, vol_idx=0, out_dir="", out_split="split.nii.gz"):
        """Splits the input 4D file and extracts the required 3D volume.

        Parameters
        ----------
        input_files : variable string
            Any number of Nifti1 files
        vol_idx : int, optional
            Index of the 3D volume to extract.
        out_dir : string, optional
            Output directory. (default current directory)
        out_split : string, optional
            Name of the resulting split volume

        """
        io_it = self.get_io_iterator()
        for fpath, osplit in io_it:
            logging.info(f"Splitting {fpath}")
            data, affine, image = load_nifti(fpath, return_img=True)

            if vol_idx == 0:
                logging.info("Splitting and extracting 1st b0")

            split_vol = data[..., vol_idx]
            save_nifti(osplit, split_vol, affine, hdr=image.header)

            logging.info(f"Split volume saved as {osplit}")


class ConcatenateTractogramFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "concatracks"

    def run(
        self,
        tractogram_files,
        reference=None,
        delete_dpv=False,
        delete_dps=False,
        delete_groups=False,
        check_space_attributes=True,
        preallocation=False,
        out_dir="",
        out_extension="trx",
        out_tractogram="concatenated_tractogram",
    ):
        """Concatenate multiple tractograms into one.

        Parameters
        ----------
        tractogram_list : variable string
            The stateful tractogram filenames to concatenate
        reference : string, optional
            Reference anatomy for tck/vtk/fib/dpy file.
            support (.nii or .nii.gz).
        delete_dpv : bool, optional
            Delete dpv keys that do not exist in all the provided TrxFiles
        delete_dps : bool, optional
            Delete dps keys that do not exist in all the provided TrxFile
        delete_groups : bool, optional
            Delete all the groups that currently exist in the TrxFiles
        check_space_attributes : bool, optional
            Verify that dimensions and size of data are similar between all the
            TrxFiles
        preallocation : bool, optional
            Preallocated TrxFile has already been generated and is the first
            element in trx_list (Note: delete_groups must be set to True as
            well)
        out_dir : string, optional
            Output directory. (default current directory)
        out_extension : string, optional
            Extension of the resulting tractogram
        out_tractogram : string, optional
            Name of the resulting tractogram

        """
        io_it = self.get_io_iterator()

        trx_list = []
        has_group = False
        for fpath, _, _ in io_it:
            if fpath.lower().endswith(".trx") or fpath.lower().endswith(".trk"):
                reference = "same"

            if not reference:
                raise ValueError(
                    "No reference provided. It is needed for tck,"
                    "fib, dpy or vtk files"
                )

            tractogram_obj = load_tractogram(fpath, reference, bbox_valid_check=False)

            if not isinstance(tractogram_obj, tmm.TrxFile):
                tractogram_obj = tmm.TrxFile.from_sft(tractogram_obj)
            elif len(tractogram_obj.groups):
                has_group = True
            trx_list.append(tractogram_obj)

        trx = concatenate_tractogram(
            trx_list,
            delete_dpv=delete_dpv,
            delete_dps=delete_dps,
            delete_groups=delete_groups or not has_group,
            check_space_attributes=check_space_attributes,
            preallocation=preallocation,
        )

        valid_extensions = ["trk", "trx", "tck", "fib", "dpy", "vtk"]
        if out_extension.lower() not in valid_extensions:
            raise ValueError(
                f"Invalid extension. Valid extensions are: {valid_extensions}"
            )

        out_fpath = os.path.join(out_dir, f"{out_tractogram}.{out_extension}")
        save_tractogram(trx.to_sft(), out_fpath, bbox_valid_check=False)


class ConvertSHFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "convert_dipy_mrtrix"

    def run(
        self,
        input_files,
        out_dir="",
        out_file="sh_convert_dipy_mrtrix_out.nii.gz",
    ):
        """Converts SH basis representation between DIPY and MRtrix3 formats.
        Because this conversion is equal to its own inverse, it can be used to
        convert in either direction: DIPY to MRtrix3 or vice versa.

        Parameters
        ----------
        input_files : string
            Path to the input files. This path may contain wildcards to
            process multiple inputs at once.

        out_dir : string, optional
            Where the resulting file will be saved. (default '')

        out_file : string, optional
            Name of the result file to be saved.
            (default 'sh_convert_dipy_mrtrix_out.nii.gz')
        """

        io_it = self.get_io_iterator()

        for in_file, out_file in io_it:
            data, affine, image = load_nifti(in_file, return_img=True)
            data = convert_sh_descoteaux_tournier(data)
            save_nifti(out_file, data, affine, hdr=image.header)


class ConvertTensorsFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "convert_tensors"

    def run(
        self,
        tensor_files,
        from_format="mrtrix",
        to_format="dipy",
        out_dir=".",
        out_tensor="converted_tensor",
    ):
        """Converts tensor representation between different formats.

        Parameters
        ----------
        tensor_files : variable string
            Any number of tensor files
        from_format : string, optional
            Format of the input tensor files. Valid options are 'dipy',
            'mrtrix', 'ants', 'fsl'.
        to_format : string, optional
            Format of the output tensor files. Valid options are 'dipy',
            'mrtrix', 'ants', 'fsl'.
        out_dir : string, optional
            Output directory. (default current directory)
        out_tensor : string, optional
            Name of the resulting tensor file

        """
        io_it = self.get_io_iterator()
        for fpath, otensor in io_it:
            logging.info(f"Converting {fpath}")
            data, affine, image = load_nifti(fpath, return_img=True)
            data = convert_tensors(data, from_format, to_format)
            save_nifti(otensor, data, affine, hdr=image.header)


class ConvertTractogramFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "convert_tractogram"

    def run(
        self,
        input_files,
        reference=None,
        pos_dtype="float32",
        offsets_dtype="uint32",
        out_dir="",
        out_tractogram="converted_tractogram.trk",
    ):
        """Converts tractogram between different formats.

        Parameters
        ----------
        input_files : variable string
            Any number of tractogram files
        reference : string, optional
            Reference anatomy for tck/vtk/fib/dpy file.
            support (.nii or .nii.gz).
        pos_dtype : string, optional
            Data type of the tractogram points, used for vtk files.
        offsets_dtype : string, optional
            Data type of the tractogram offsets, used for vtk files.
        out_dir : string, optional
            Output directory. (default current directory)
        out_tractogram : string, optional
            Name of the resulting tractogram

        """
        io_it = self.get_io_iterator()

        for fpath, otracks in io_it:
            in_extension = fpath.lower().split(".")[-1]
            out_extension = otracks.lower().split(".")[-1]

            if in_extension == out_extension:
                warnings.warn(
                    "Input and output are the same file format. Skipping...",
                    stacklevel=2,
                )
                continue

            if not reference and in_extension in ["trx", "trk"]:
                reference = "same"

            if not reference and in_extension not in ["trx", "trk"]:
                raise ValueError(
                    "No reference provided. It is needed for tck,"
                    "fib, dpy or vtk files"
                )

            sft = load_tractogram(fpath, reference, bbox_valid_check=False)

            if out_extension != "trx":
                if out_extension == "vtk":
                    if sft.streamlines._data.dtype.name != pos_dtype:
                        sft.streamlines._data = sft.streamlines._data.astype(pos_dtype)
                    if offsets_dtype == "uint64" or offsets_dtype == "uint32":
                        offsets_dtype = offsets_dtype[1:]
                    if sft.streamlines._offsets.dtype.name != offsets_dtype:
                        sft.streamlines._offsets = sft.streamlines._offsets.astype(
                            offsets_dtype
                        )
                save_tractogram(sft, otracks, bbox_valid_check=False)
            else:
                trx = tmm.TrxFile.from_sft(sft)
                if trx.streamlines._data.dtype.name != pos_dtype:
                    trx.streamlines._data = trx.streamlines._data.astype(pos_dtype)
                if trx.streamlines._offsets.dtype.name != offsets_dtype:
                    trx.streamlines._offsets = trx.streamlines._offsets.astype(
                        offsets_dtype
                    )
                tmm.save(trx, otracks)
                trx.close()


class NiftisToPamFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "niftis_to_pam"

    def run(
        self,
        peaks_dir_files,
        peaks_values_files,
        peaks_indices_files,
        shm_files=None,
        gfa_files=None,
        sphere_files=None,
        default_sphere_name="repulsion724",
        out_dir="",
        out_pam="peaks.pam5",
    ):
        """Convert multiple nifti files to a single pam5 file.

        Parameters
        ----------
        peaks_dir_files : string
            Path to the input peaks directions volume. This path may contain
            wildcards to process multiple inputs at once.
        peaks_values_files : string
            Path to the input peaks values volume. This path may contain
            wildcards to process multiple inputs at once.
        peaks_indices_files : string
            Path to the input peaks indices volume. This path may contain
            wildcards to process multiple inputs at once.
        shm_files : string, optional
            Path to the input spherical harmonics volume. This path may
            contain wildcards to process multiple inputs at once.
        gfa_files : string, optional
            Path to the input generalized FA volume. This path may contain
            wildcards to process multiple inputs at once.
        sphere_files : string, optional
            Path to the input sphere vertices. This path may contain
            wildcards to process multiple inputs at once. If it is not define,
            default_sphere option will be used.
        default_sphere_name : string, optional
            Specify default sphere to use for spherical harmonics
            representation. This option can be superseded by
            sphere_files option. Possible options: ['symmetric362', 'symmetric642',
            'symmetric724', 'repulsion724', 'repulsion100', 'repulsion200'].
        out_dir : string, optional
            Output directory (default input file directory).
        out_pam : string, optional
            Name of the peaks volume to be saved.

        """
        io_it = self.get_io_iterator()

        msg = f"pam5 files saved in {out_dir or 'current directory'}"

        for fpeak_dirs, fpeak_values, fpeak_indices, opam in io_it:
            logging.info("Converting nifti files to pam5")
            peak_dirs, affine = load_nifti(fpeak_dirs)
            peak_values, _ = load_nifti(fpeak_values)
            peak_indices, _ = load_nifti(fpeak_indices)

            if sphere_files:
                xyz = np.loadtxt(sphere_files)
                sphere = Sphere(xyz=xyz)
            else:
                sphere = get_sphere(name=default_sphere_name)

            niftis_to_pam(
                affine=affine,
                peak_dirs=peak_dirs,
                sphere=sphere,
                peak_values=peak_values,
                peak_indices=peak_indices,
                pam_file=opam,
            )
            logging.info(msg.replace("pam5", opam))


class TensorToPamFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "tensor_to_niftis"

    def run(
        self,
        evals_files,
        evecs_files,
        sphere_files=None,
        default_sphere_name="repulsion724",
        out_dir="",
        out_pam="peaks.pam5",
    ):
        """Convert multiple tensor files(evals, evecs) to pam5 files.

        Parameters
        ----------
        evals_files : string
            Path to the input eigen values volumes. This path may contain
            wildcards to process multiple inputs at once.
        evecs_files : string
            Path to the input eigen vectors volumes. This path may contain
            wildcards to process multiple inputs at once.
        sphere_files : string, optional
            Path to the input sphere vertices. This path may contain
            wildcards to process multiple inputs at once. If it is not define,
            default_sphere option will be used.
        default_sphere_name : string, optional
            Specify default sphere to use for spherical harmonics
            representation. This option can be superseded by sphere_files
            option. Possible options: ['symmetric362', 'symmetric642',
            'symmetric724', 'repulsion724', 'repulsion100', 'repulsion200'].
        out_dir : string, optional
            Output directory (default input file directory).
        out_pam : string, optional
            Name of the peaks volume to be saved.

        """
        io_it = self.get_io_iterator()

        msg = f"pam5 files saved in {out_dir or 'current directory'}"

        for fevals, fevecs, opam in io_it:
            logging.info("Converting tensor files to pam5...")
            evals, affine = load_nifti(fevals)
            evecs, _ = load_nifti(fevecs)

            if sphere_files:
                xyz = np.loadtxt(sphere_files)
                sphere = Sphere(xyz=xyz)
            else:
                sphere = get_sphere(name=default_sphere_name)

            tensor_to_pam(evals, evecs, affine, sphere=sphere, pam_file=opam)
            logging.info(msg.replace("pam5", opam))


class PamToNiftisFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "pam_to_niftis"

    def run(
        self,
        pam_files,
        out_dir="",
        out_peaks_dir="peaks_dirs.nii.gz",
        out_peaks_values="peaks_values.nii.gz",
        out_peaks_indices="peaks_indices.nii.gz",
        out_shm="shm.nii.gz",
        out_gfa="gfa.nii.gz",
        out_sphere="sphere.txt",
        out_b="B.nii.gz",
        out_qa="qa.nii.gz",
    ):
        """Convert pam5 files to multiple nifti files.

        Parameters
        ----------
        pam_files : string
            Path to the input peaks volumes. This path may contain wildcards to
            process multiple inputs at once.
        out_dir : string, optional
            Output directory (default input file directory).
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved.
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved.
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved.
        out_shm : string, optional
            Name of the spherical harmonics volume to be saved.
        out_gfa : string, optional
            Generalized FA volume name to be saved.
        out_sphere : string, optional
            Sphere vertices name to be saved.
        out_b : string, optional
            Name of the B Matrix to be saved.
        out_qa : string, optional
            Name of the Quantitative Anisotropy file to be saved.

        """
        io_it = self.get_io_iterator()

        msg = f"Nifti files saved in {out_dir or 'current directory'}"
        for (
            ipam,
            opeaks_dir,
            opeaks_values,
            opeaks_indices,
            oshm,
            ogfa,
            osphere,
            ob,
            oqa,
        ) in io_it:
            logging.info("Converting %s file to niftis...", ipam)
            pam = load_pam(ipam)
            pam_to_niftis(
                pam,
                fname_peaks_dir=opeaks_dir,
                fname_shm=oshm,
                fname_peaks_values=opeaks_values,
                fname_peaks_indices=opeaks_indices,
                fname_sphere=osphere,
                fname_gfa=ogfa,
                fname_b=ob,
                fname_qa=oqa,
            )
            logging.info(msg)
