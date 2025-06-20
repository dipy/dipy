import enum
import importlib
from inspect import getmembers, isfunction
import logging
import os
import re
import sys
import warnings

import numpy as np
import trx.trx_file_memmap as tmm

from dipy.core.gradients import (
    extract_b0,
    extract_dwi_shell,
    gradient_table,
    mask_non_weighted_bvals,
)
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
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
from dipy.utils.optpkg import optional_package
from dipy.utils.tractogram import concatenate_tractogram
from dipy.workflows.utils import handle_vol_idx
from dipy.workflows.workflow import Workflow

ne, have_ne, _ = optional_package("numexpr")


class StatsPropertyName(enum.Enum):
    """Statistical data property names."""

    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    MEAN = "mean"
    STD_DEV = "std dev"


class PercentilePropertyName(enum.Enum):
    """Percentile data property names."""

    PERCENTILE_2 = "2nd percentile"
    PERCENTILE_98 = "98th percentile"


class VolumetricPropertyName(enum.Enum):
    """Volumetric data property names."""

    AFFINE = "Affine matrix"
    VOXEL_ORDER = "Voxel order"
    DATA_TYPE = "Data type"
    DIMENSIONS = "Dimensions"
    VOXEL_SIZE = "Voxel size"


class BvalPropertyName(enum.Enum):
    """b-value data property names."""

    B0_THRESHOLD = "b0 threshold"
    B_VALUES = "b-values"
    NUMBER_B0s = "Number of b0s"
    NUMBER_B_VALUES = "Total number of b-values"
    NUMBER_SHELLS = "Number of gradient shells"


class BvecPropertyName(enum.Enum):
    """b-vector data property names."""

    B_VECTORS = "b-vectors"
    B_VECTORS_SHAPE = "Shape of b-vectors on disk"
    NUMBER_NONUNIT_B_VECTORS = "Total number of non-unit b-vectors"
    NUMBER_UNIT_B_VECTORS = "Total number of unit b-vectors"


class TractographyPropertyName(enum.Enum):
    """Tractography data property names."""

    DPP_KEYS = "Data per point keys"
    DPS_KEYS = "Data per streamline keys"
    LENGTH_MM = "Length (mm)"
    LENGTH_NUM_PTS = "Length (nb points)"
    NUM_STRML = "Number of streamlines"
    ORIGIN = "Origin"
    SPACE = "Space"
    STEP_SIZE = "Step size (mm)"


def _print_property_information(prop, val, alignment_space):
    """Print a property, value pair left-aligned at the given width.

    Parameters
    ----------
    prop : str
        Name of the property.
    val : scalar, str, ndarray, list
        Value to be property.
    alignment_space : int
        Character width for the property, value pair alignment.
    """

    logging.info(f"{prop + ':':<{alignment_space}}{val}")


def _print_stats_information(data, alignment_space, tab):
    """Print statistical information left-aligned at the given width.

    Prints minimum, maximum, median, mean and standard deviation values of
    the given data indented by ``tab`` number of whitespaces.

    Parameters
    ----------
    data : ndarray
        Name of the piece of information.
    alignment_space : int
        Character width for the property, value pair alignment.
    tab : str
        Whitespace characters corresponding to a tab character for the
        indentation.
    """

    logging.info(f"{tab + 'min:':<{alignment_space}}{np.min(data)}")
    logging.info(f"{tab + 'max:':<{alignment_space}}{np.max(data)}")
    logging.info(f"{tab + 'median:':<{alignment_space}}{np.median(data)}")
    logging.info(f"{tab + 'mean:':<{alignment_space}}{np.mean(data)}")
    logging.info(f"{tab + 'std dev:':<{alignment_space}}{np.std(data)}")


def _print_volumetric_information(data, affine, vox_sz, affcodes, alignment_space, tab):
    """Print volumetric information.

    Parameters
    ----------
    data : ndarray
        Data whose properties are to be printed.
    affine : ndarray
        Affine matrix.
    vox_sz : tuple
        Voxel size.
    affcodes : tuple
        Voxel order (anatomical coordinate system).
    alignment_space : int
        Character width for the property, value pair alignment.
    tab : str
        Whitespace characters corresponding to a tab character for the
        indentation.
    """

    def _print_voxel_information(_name, _data, _alignment_space, _tab):
        """Print voxel information.

        Parameters
        ----------
        _name : str
           Name of the information piece.
        _data : ndarray
            Data whose properties are to be printed.
        _alignment_space : int
            Character width for the property, value pair alignment.
        _tab : str
            Whitespace characters corresponding to a tab character
            for the indentation.
        """

        logging.info(f"{_name}:")
        _print_stats_information(_data, _alignment_space, _tab)
        _print_property_information(
            _tab + PercentilePropertyName.PERCENTILE_2.value,
            np.percentile(_data, 2),
            _alignment_space,
        )
        _print_property_information(
            _tab + PercentilePropertyName.PERCENTILE_98.value,
            np.percentile(_data, 98),
            _alignment_space,
        )

    _print_property_information(
        VolumetricPropertyName.DIMENSIONS.value, data.shape, alignment_space
    )
    _print_property_information(
        VolumetricPropertyName.DATA_TYPE.value, data.dtype, alignment_space
    )

    if data.ndim == 3:
        _print_voxel_information("Data", data, alignment_space, tab)
    if data.ndim == 4:
        _print_voxel_information("Data (0th vol)", data[..., 0], alignment_space, tab)

    _print_property_information(
        VolumetricPropertyName.VOXEL_ORDER.value,
        "".join(affcodes),
        alignment_space,
    )
    logging.info(f"Affine matrix:\n{affine}")
    _print_property_information(
        VolumetricPropertyName.VOXEL_SIZE.value,
        tuple(map(float, vox_sz)),
        alignment_space,
    )

    if np.sum(np.abs(np.diff(vox_sz))) > 0.1:
        msg = "Voxel size is not isotropic. Please reslice.\n"
        logging.warning(msg, stacklevel=2)


def _print_bval_data_information(bvals, b0_threshold, bshell_thr, alignment_space):
    """Print b-value information.

    Parameters
    ----------
    bvals : ndarray
        b-values.
    b0_threshold : float
        b0 threshold.
    bshell_thr : float
        Threshold value to determine shells.
    alignment_space : int
        Character width for the property, value pair alignment.
    """

    logging.info(f"{BvalPropertyName.B_VALUES.value}:\n{bvals}")
    _print_property_information(
        BvalPropertyName.NUMBER_B_VALUES.value, len(bvals), alignment_space
    )
    shells = np.sum(np.diff(np.sort(bvals)) > bshell_thr)
    _print_property_information(
        BvalPropertyName.NUMBER_SHELLS.value, shells, alignment_space
    )
    num_b0s = np.sum(bvals <= b0_threshold)
    logging.info(
        f"{BvalPropertyName.NUMBER_B0s.value + ':':<{alignment_space}}{num_b0s}"
        f" ({BvalPropertyName.B0_THRESHOLD.value}: {b0_threshold})"
    )


def _print_bvec_data_information(bvecs, bvecs_tol, alignment_space):
    """Print b-vector information.

    Parameters
    ----------
    bvecs : ndarray
        b-vectors.
    bvecs_tol : float
        Threshold used to check that
        :math:`norm(\text{bvec}) = 1 \\pm \text{bvecs_tol}` b-vectors are
        unit vectors.
    alignment_space : int
        Character width for the property, value pair alignment.
    """

    _print_property_information(
        BvecPropertyName.B_VECTORS_SHAPE.value, bvecs.shape, alignment_space
    )
    rows, cols = bvecs.shape
    if rows < cols:
        bvecs = bvecs.T
    logging.info(f"{BvecPropertyName.B_VECTORS.value}\n{bvecs}")
    norms = np.array([np.linalg.norm(bvec) for bvec in bvecs])
    res = np.where((norms <= 1 + bvecs_tol) & (norms >= 1 - bvecs_tol))
    ncl1 = np.sum(norms < 1 - bvecs_tol)
    _print_property_information(
        BvecPropertyName.NUMBER_UNIT_B_VECTORS.value, len(res[0]), alignment_space
    )
    _print_property_information(
        BvecPropertyName.NUMBER_NONUNIT_B_VECTORS.value, ncl1, alignment_space
    )


def _print_tractography_information(
    sft, lengths_mm, lengths, step_size, alignment_space, tab
):
    """Print tractogram information.

    Parameters
    ----------
    sft : StatefulTractogram
       Tractogram.
    lengths_mm : list
        Length of the streamlines in millimeters.
    lengths : list
        Length of the streamlines in number of points.
    step_size : ndarray
        Step sizes.
    alignment_space : int
        Character width for the property, value pair alignment.
    tab : str
        Whitespace characters corresponding to a tab character for the
        indentation.
    """

    _print_property_information(
        TractographyPropertyName.NUM_STRML.value, len(sft), alignment_space
    )
    logging.info(f"{TractographyPropertyName.LENGTH_MM.value}:")
    _print_stats_information(lengths_mm, alignment_space, tab)
    logging.info(f"{TractographyPropertyName.LENGTH_NUM_PTS.value}:")
    _print_stats_information(lengths, alignment_space, tab)
    logging.info(f"{TractographyPropertyName.STEP_SIZE.value}:")
    _print_stats_information(step_size, alignment_space, tab)
    _print_property_information(
        TractographyPropertyName.DPP_KEYS.value,
        list(sft.data_per_point.keys()),
        alignment_space,
    )
    _print_property_information(
        TractographyPropertyName.DPS_KEYS.value,
        list(sft.data_per_streamline.keys()),
        alignment_space,
    )
    logging.info(f"Affine matrix:\n{sft.affine}")
    _print_property_information(
        VolumetricPropertyName.DIMENSIONS.value,
        tuple(map(int, sft.dimensions)),
        alignment_space,
    )
    _print_property_information(
        VolumetricPropertyName.DATA_TYPE.value,
        sft.streamlines.get_data().dtype,
        alignment_space,
    )
    _print_property_information(
        TractographyPropertyName.ORIGIN.value, sft.origin, alignment_space
    )
    _print_property_information(
        TractographyPropertyName.SPACE.value, sft.space, alignment_space
    )
    _print_property_information(
        VolumetricPropertyName.VOXEL_ORDER.value,
        sft.voxel_order,
        alignment_space,
    )
    _print_property_information(
        VolumetricPropertyName.VOXEL_SIZE.value,
        tuple(map(float, sft.voxel_sizes)),
        alignment_space,
    )


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
            Any number of NIfTI, bvals, bvecs or tractography data files.
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Threshold used to check that
            :math:`norm(\text{bvec}) = 1 \\pm \text{bvecs_tol}` b-vectors are
            unit vectors.
        bshell_thr : float, optional
            Threshold for distinguishing b-values in different shells.
        reference : string, optional
            Reference anatomy for ``*.tck``, ``*.vtk``/``*.vtp``, ``*.fib``, and
            ``*.dpy`` tractography files.

        """
        np.set_printoptions(3, suppress=True)

        io_it = self.get_io_iterator()

        vol_property_length = (
            len(max([item.value for item in VolumetricPropertyName], key=len)) + 2
        )
        stats_property_length = (
            len(max([item.value for item in StatsPropertyName], key=len)) + 2
        )
        pctl_property_length = (
            len(max([item.value for item in PercentilePropertyName], key=len)) + 2
        )
        tab = "\t".expandtabs(4)

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
                apply_tab_offset = bool(
                    max(
                        range(3),
                        key=lambda i: [
                            vol_property_length,
                            stats_property_length,
                            pctl_property_length,
                        ][i],
                    )
                )
                _print_volumetric_information(
                    data,
                    affine,
                    vox_sz,
                    affcodes,
                    max(
                        vol_property_length, stats_property_length, pctl_property_length
                    )
                    + apply_tab_offset * len(tab),
                    tab,
                )

            if os.path.basename(input_path).lower().find("bval") > -1:
                bval_property_length = (
                    len(max([item.value for item in BvalPropertyName], key=len)) + 2
                )

                bvals = np.loadtxt(input_path)
                _print_bval_data_information(
                    bvals, b0_threshold, bshell_thr, bval_property_length
                )

            if os.path.basename(input_path).lower().find("bvec") > -1:
                bvec_property_length = (
                    len(max([item.value for item in BvecPropertyName], key=len)) + 2
                )

                bvecs = np.loadtxt(input_path)
                _print_bvec_data_information(bvecs, bvecs_tol, bvec_property_length)

            if extension in [".trk", ".tck", ".trx", ".vtk", ".vtp", ".fib", ".dpy"]:
                tractogr_property_length = (
                    len(max([item.value for item in TractographyPropertyName], key=len))
                    + 2
                )
                apply_tab_offset = not bool(
                    max(
                        range(3),
                        key=lambda i: [
                            stats_property_length,
                            vol_property_length,
                            tractogr_property_length,
                        ][i],
                    )
                )
                if extension in [".trk", ".trx"]:
                    sft = load_tractogram(input_path, "same", bbox_valid_check=False)
                else:
                    if not reference or not os.path.exists(reference):
                        msg = (
                            "No reference provided. It is needed for tck, fib, dpy or "
                            "vtk files to load properly. Please provide a reference, "
                            "Nifti or Trk file using the option "
                            "--reference my_files.nii.gz ."
                        )
                        logging.error(msg, stacklevel=2)
                        sys.exit(1)

                    sft = load_tractogram(input_path, reference, bbox_valid_check=False)

                lengths_mm = list(length(sft.streamlines))

                sft.to_voxmm()

                lengths, steps = [], []
                for streamline in sft.streamlines:
                    lengths += [len(streamline)]
                    steps += [np.sqrt(np.sum(np.diff(streamline, axis=0) ** 2, axis=1))]
                steps = np.hstack(steps)

                _print_tractography_information(
                    sft,
                    lengths_mm,
                    lengths,
                    steps,
                    max(
                        vol_property_length,
                        stats_property_length,
                        tractogr_property_length,
                    )
                    + apply_tab_offset * len(tab),
                    tab,
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

    def run(
        self,
        data_names,
        subjects=None,
        include_optional=False,
        include_afq=False,
        hcp_bucket="hcp-openaccess",
        hcp_profile_name="hcp",
        hcp_study="HCP_1200",
        hcp_aws_access_key_id=None,
        hcp_aws_secret_access_key=None,
        out_dir="",
    ):
        """Download files to folder and check their md5 checksums.

        To see all available datasets, please type "list" in data_names.

        Parameters
        ----------
        data_names : variable string
            Any number of Nifti1, bvals or bvecs files.
        subjects : variable string, optional
            Identifiers of the subjects to download. Used only by the HBN & HCP dataset.
            For example with HBN dataset: --subject NDARAA948VFH NDAREK918EC2
        include_optional : bool, optional
            Include optional datasets.
        include_afq : bool, optional
            Whether to include pyAFQ derivatives. Used only by the HBN dataset.
        hcp_bucket : string, optional
            The name of the HCP S3 bucket.
        hcp_profile_name : string, optional
            The name of the AWS profile used for access.
        hcp_study : string, optional
            Which HCP study to grab.
        hcp_aws_access_key_id : string, optional
            AWS credentials to HCP AWS S3. Will only be used if `profile_name` is
            set to False.
        hcp_aws_secret_access_key : string, optional
            AWS credentials to HCP AWS S3. Will only be used if `profile_name` is
            set to False.
        out_dir : string, optional
            Output directory.

        """
        if out_dir:
            dipy_home = os.environ.get("DIPY_HOME", None)
            os.environ["DIPY_HOME"] = out_dir

        available_data = FetchFlow.get_fetcher_datanames()

        data_names = [name.lower() for name in data_names]

        if "all" in data_names:
            logging.warning("Skipping HCP and HBN datasets.")
            available_data.pop("hcp", None)
            available_data.pop("hbn", None)
            for name, fetcher_function in available_data.items():
                if name in ["hcp", "hbn"]:
                    continue
                logging.info("------------------------------------------")
                logging.info(f"Fetching at {name}")
                logging.info("------------------------------------------")
                fetcher_function(include_optional=include_optional)

        elif "list" in data_names:
            logging.info(
                "Please, select between the following data names: \n"
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
                if data_name == "hcp":
                    if not subjects:
                        logging.error(
                            "Please provide the subjects to download the HCP dataset."
                        )
                        continue
                    try:
                        available_data[data_name](
                            subjects=subjects,
                            hcp_bucket=hcp_bucket,
                            profile_name=hcp_profile_name,
                            study=hcp_study,
                            aws_access_key_id=hcp_aws_access_key_id,
                            aws_secret_access_key=hcp_aws_secret_access_key,
                        )
                    except Exception as e:
                        logging.error(
                            f"Error while fetching HCP dataset: {str(e)}", exc_info=True
                        )
                elif data_name == "hbn":
                    if not subjects:
                        logging.error(
                            "Please provide the subjects to download the HBN dataset."
                        )
                        continue
                    try:
                        available_data[data_name](
                            subjects=subjects, include_afq=include_afq
                        )
                    except Exception as e:
                        logging.error(
                            f"Error while fetching HBN dataset: {str(e)}", exc_info=True
                        )
                else:
                    available_data[data_name](include_optional=include_optional)

            nb_success = len(data_names) - len(skipped_names)
            print("\n")
            logging.info(f"Fetched {nb_success} / {len(data_names)} Files ")
            if skipped_names:
                logging.warning(f"Skipped data name(s): {' '.join(skipped_names)}")
                logging.warning(
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
            Output directory.
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


class ExtractB0Flow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "extract_b0"

    def run(
        self,
        input_files,
        bvalues_files,
        b0_threshold=50,
        group_contiguous_b0=False,
        strategy="mean",
        out_dir="",
        out_b0="b0.nii.gz",
    ):
        """Extract on or multiple b0 volume from the input 4D file.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bvalues_files : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        group_contiguous_b0 : bool, optional
            If True, each contiguous b0 volumes are grouped together.
        strategy : str, optional
            The extraction strategy, of either:

                - first: select the first b0 found.
                - all: select them all.
                - mean: average them.

            When used in conjunction with the batch parameter set to True, the
            strategy is applied individually on each continuous set found.
        out_dir : string, optional
            Output directory.
        out_b0 : string, optional
            Name of the resulting b0 volume.

        """
        io_it = self.get_io_iterator()
        for dwi, bval, ob0 in io_it:
            logging.info("Extracting b0 from {0}".format(dwi))
            data, affine, image = load_nifti(dwi, return_img=True)

            bvals, bvecs = read_bvals_bvecs(bval, None)
            # If all b-values are smaller or equal to the b0 threshold, it is
            # assumed that no thresholding is requested
            if any(mask_non_weighted_bvals(bvals, b0_threshold)):
                if b0_threshold < bvals.min():
                    warnings.warn(
                        f"b0_threshold (value: {b0_threshold}) is too low, "
                        "increase your b0_threshold. It should be higher than the "
                        f"first b0 value ({bvals.min()}).",
                        stacklevel=2,
                    )

            bvecs = np.random.randn(bvals.shape[0], 3)
            norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
            bvecs = bvecs / norms
            gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=b0_threshold)
            b0s_result = extract_b0(
                data,
                gtab.b0s_mask,
                group_contiguous_b0=group_contiguous_b0,
                strategy=strategy,
            )

            if b0s_result.ndim == 3:
                save_nifti(ob0, b0s_result, affine, hdr=image.header)
                logging.info("b0 saved as {0}".format(ob0))
            elif b0s_result.ndim == 4:
                for i in range(b0s_result.shape[-1]):
                    save_nifti(
                        ob0.replace(".nii", f"_{i}.nii"),
                        b0s_result[..., i],
                        affine,
                        hdr=image.header,
                    )
                    logging.info(
                        "b0 saved as {0}".format(ob0.replace(".nii", f"_{i}.nii"))
                    )
            else:
                logging.error("No b0 volumes found")


class ExtractShellFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "extract_shell"

    def run(
        self,
        input_files,
        bvalues_files,
        bvectors_files,
        bvals_to_extract=None,
        b0_threshold=50,
        bvecs_tol=0.01,
        tol=20,
        group_shells=True,
        out_dir="",
        out_shell="shell.nii.gz",
    ):
        """Extract shells from the input 4D file.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bvalues_files : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors_files : string
            Path to the bvectors files. This path may contain wildcards to use
            multiple bvectors files at once.
        bvals_to_extract : string, optional
            List of b-values to extract. You can provide a single b-values or a range
            of b-values separated by a dash. For example, to extract b-values 0, 1,
            and 2, you can use '0-2'. You can also provide a list of b-values separated
            by a comma. For example, to extract b-values 0, 1, 2, 8, 10, 11 and 12,
            you can use '0-2,8,10-12'.
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Threshold used to check that norm(bvec) = 1 +/- bvecs_tol
        tol : int, optional
            Tolerance range for b-value selection. A value of 20 means volumes with
            b-values within ±20 units of the specified b-values will be extracted.
        group_shells : bool, optional
            If True, extracted volumes are grouped into a single array. If False,
            returns a list of separate volumes.
        out_dir : string, optional
            Output directory.
        out_shell : string, optional
            Name of the resulting shell volume.

        """
        io_it = self.get_io_iterator()
        if bvals_to_extract is None:
            logging.error(
                "Please provide a list of b-values to extract."
                " e.g: --bvals_to_extract 1000 2000 3000"
            )
            sys.exit(1)

        bvals_to_extract = handle_vol_idx(bvals_to_extract)

        for dwi, bval, bvec, oshell in io_it:
            logging.info("Extracting shell from {0}".format(dwi))
            data, affine, image = load_nifti(dwi, return_img=True)

            bvals, bvecs = read_bvals_bvecs(bval, bvec)
            # If all b-values are smaller or equal to the b0 threshold, it is
            # assumed that no thresholding is requested
            if any(mask_non_weighted_bvals(bvals, b0_threshold)):
                if b0_threshold < bvals.min():
                    warnings.warn(
                        f"b0_threshold (value: {b0_threshold}) is too low, "
                        "increase your b0_threshold. It should be higher than the "
                        f"first b0 value ({bvals.min()}).",
                        stacklevel=2,
                    )
            gtab = gradient_table(
                bvals, bvecs=bvecs, b0_threshold=b0_threshold, atol=bvecs_tol
            )
            indices, shell_data, output_bvals, output_bvecs = extract_dwi_shell(
                data,
                gtab,
                bvals_to_extract,
                tol=tol,
                group_shells=group_shells,
            )

            for i, shell in enumerate(shell_data):
                shell_value = np.unique(output_bvals[i]).astype(int).astype(str)
                shell_value = "_".join(shell_value.tolist())
                save_nifti(
                    oshell.replace(".nii", f"_{shell_value}.nii"),
                    shell,
                    affine,
                    hdr=image.header,
                )
                logging.info(
                    "b0 saved as {0}".format(
                        oshell.replace(".nii", f"_{shell_value}.nii")
                    )
                )


class ExtractVolumeFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "extract_volume"

    def run(
        self, input_files, vol_idx=0, grouped=True, out_dir="", out_vol="volume.nii.gz"
    ):
        """Extracts the required volume from the input 4D file.

        Parameters
        ----------
        input_files : string
            Any number of Nifti1 files
        vol_idx : string, optional
            Indexes of the 3D volume to extract. Index start from 0. You can provide
            a single index or a range of indexes separated by a dash. For example,
            to extract volumes 0, 1, and 2, you can use '0-2'. You can also provide
            a list of indexes separated by a comma. For example, to extract volumes
            0, 1, 2, 8, 10, 11 and 12 , you can use '0-2,8,10-12'.
        grouped : bool, optional
            If True, extracted volumes are grouped into a single array. If False,
            save a list of separate volumes.
        out_dir : string, optional
            Output directory.
        out_vol : string, optional
            Name of the resulting volume.

        """
        io_it = self.get_io_iterator()
        vol_idx = handle_vol_idx(vol_idx)

        for fpath, ovol in io_it:
            logging.info("Extracting volume from {0}".format(fpath))
            data, affine, image = load_nifti(fpath, return_img=True)

            if grouped:
                split_vol = data[..., vol_idx]
                save_nifti(ovol, split_vol, affine, hdr=image.header)
                logging.info("Volume saved as {0}".format(ovol))
            else:
                for i in vol_idx:
                    fname = ovol.replace(".nii", f"_{i}.nii")
                    split_vol = data[..., i]
                    save_nifti(fname, split_vol, affine, hdr=image.header)
                    logging.info("Volume saved as {0}".format(fname))


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
            Output directory.
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
            Output directory.
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
            Output directory.
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


class MathFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "math_flow"

    def broadcast_arrays(self, vol_dict, operation):
        variables = re.findall(r"\b\w+\b", operation)
        variables = [var for var in variables if var in vol_dict]

        shapes = [vol_dict[var].shape for var in variables]
        max_dims = max(len(shape) for shape in shapes)

        for var in variables:
            shape = vol_dict[var].shape
            if len(shape) < max_dims:
                new_shape = shape + (1,) * (max_dims - len(shape))
                vol_dict[var] = vol_dict[var].reshape(new_shape)

        updated_shapes = [vol_dict[var].shape for var in variables]
        try:
            broadcast_shape = np.broadcast_shapes(*updated_shapes)
        except ValueError as e:
            raise ValueError(f"Shape mismatch after adding dimensions: {e}") from e

        for var in variables:
            if vol_dict[var].shape != broadcast_shape:
                vol_dict[var] = np.broadcast_to(vol_dict[var], broadcast_shape)

        return vol_dict

    def run(
        self,
        operation,
        input_files,
        dtype=None,
        disable_check=False,
        out_dir="",
        out_file="math_out.nii.gz",
    ):
        """Perform mathematical operations on volume input files.

        This workflow allows the user to perform mathematical operations on
        multiple input files. e.g. to add two volumes together, subtract one:
        ``dipy_math "vol1 + vol2 - vol3" t1.nii.gz t1_a.nii.gz t1_b.nii.gz``
        The input files must be in Nifti format and have the same shape.

        Parameters
        ----------
        operation : string
            Mathematical operation to perform. supported operators are:
                - Bitwise operators (and, or, not, xor): ``&, |, ~, ^``
                - Comparison operators: ``<, <=, ==, !=, >=, >``
                - Unary arithmetic operators: ``-``
                - Binary arithmetic operators: ``+, -, *, /, **, <<, >>``
            Supported functions are:
                - ``where(bool, number1, number2) -> number``: number1 if the bool
                  condition is true, number2 otherwise.
                - ``{sin,cos,tan}(float|complex) -> float|complex``: trigonometric sine,
                  cosine or tangent.
                - ``{arcsin,arccos,arctan}(float|complex) -> float|complex``:
                  trigonometric inverse sine, cosine or tangent.
                - ``arctan2(float1, float2) -> float``: trigonometric inverse tangent of
                  float1/float2.
                - ``{sinh,cosh,tanh}(float|complex) -> float|complex``: hyperbolic
                  sine, cosine or tangent.
                - ``{arcsinh,arccosh,arctanh}(float|complex) -> float|complex``:
                  hyperbolic inverse sine, cosine or tangent.
                - ``{log,log10,log1p}(float|complex) -> float|complex``: natural,
                  base-10 and log(1+x) logarithms.
                - ``{exp,expm1}(float|complex) -> float|complex``: exponential and
                  exponential minus one.
                - ``sqrt(float|complex) -> float|complex``: square root.
                - ``abs(float|complex) -> float|complex``: absolute value.
                - ``conj(complex) -> complex``: conjugate value.
                - ``{real,imag}(complex) -> float``: real or imaginary part of complex.
                - ``complex(float, float) -> complex``: complex from real and imaginary
                  parts.
                - ``contains(np.str, np.str) -> bool``: returns True for every string
                  in op1 that contains op2.
        input_files : variable string
            Any number of Nifti1 files
        dtype : string, optional
            Data type of the resulting file.
        disable_check : bool, optional
            If True, the workflow will not check if all input files have the same
            shape and affine matrix.
        out_dir : string, optional
            Output directory
        out_file : string, optional
            Name of the resulting file to be saved.
        """
        vol_dict = {}
        ref_affine = None
        ref_shape = None
        info_msg = ""
        have_errors = False
        for i, fname in enumerate(input_files, start=1):
            if not os.path.isfile(fname):
                logging.error(f"Input file {fname} does not exist.")
                raise SystemExit()

            if not (fname.endswith(".nii.gz") or fname.endswith(".nii")):
                msg = (
                    f"Wrong volume type: {fname}. Only Nifti files are supported"
                    " (*.nii or *.nii.gz)."
                )
                logging.error(msg)
                raise SystemExit()

            data, affine = load_nifti(fname)
            vol_dict[f"vol{i}"] = data
            info_msg += f"{fname}:\n- vol index: {i}\n- shape: {data.shape}"
            info_msg += f"\n- affine:\n{affine}\n"
            if ref_affine is None:
                ref_affine = affine
                ref_shape = data.shape
                continue

            have_errors = (
                have_errors
                or not np.all(np.isclose(ref_affine, affine, rtol=1e-05, atol=1e-08))
                or not np.array_equal(ref_shape, data.shape)
            )

        if have_errors:
            logging.warning(info_msg)
            if not disable_check:
                msg = "All input files must have the same shape and affine matrix."
                logging.error(msg)
                raise SystemExit()

        try:
            if disable_check:
                vol_dict = self.broadcast_arrays(vol_dict, operation)
            res = ne.evaluate(operation, local_dict=vol_dict)
        except KeyError as e:
            msg = (
                f"Impossible key {e} in the operation. You have {len(input_files)}"
                f" volumes available with the following keys: {list(vol_dict.keys())}"
            )
            logging.error(msg)
            raise SystemExit() from e

        if dtype:
            try:
                res = res.astype(dtype)
            except TypeError as e:
                msg = (
                    f"Impossible to cast to {dtype}. Check possible numpy type here:"
                    "https://numpy.org/doc/stable/reference/arrays.interface.html"
                )
                logging.error(msg)
                raise SystemExit() from e

        if res.dtype == bool:
            res = res.astype(np.uint8)
        out_fname = os.path.join(out_dir, out_file)
        logging.info(f"Saving result to {out_fname}")
        save_nifti(out_fname, res, affine)
