import contextlib
from hashlib import md5
import json
import os
from pathlib import Path
import random
from shutil import copyfileobj
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

import nibabel as nib
import numpy as np
from tqdm.auto import tqdm

from dipy.core.gradients import (
    gradient_table,
    gradient_table_from_gradient_strength_bvecs,
)
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.io.streamline import load_tractogram
from dipy.testing.decorators import warning_for_keywords
from dipy.utils.logging import logger
from dipy.utils.optpkg import TripWire, optional_package

# Set a user-writeable file-system location to put files:
if "DIPY_HOME" in os.environ:
    dipy_home = Path(os.environ["DIPY_HOME"])
else:
    dipy_home = Path("~").expanduser() / ".dipy"

# The URL to the University of Washington Researchworks repository:
UW_RW_URL = "https://digital.lib.washington.edu/researchworks/bitstream/handle/"

# Mirror server for DIPY data (fallback when primary sources fail)
DIPY_MIRROR_URL = "https://workshop.dipy.org/services/data/"

# Base URLs that can use the mirror fallback
MIRRORABLE_HOSTS = [
    "ndownloader.figshare.com",
    "zenodo.org",
    "digital.lib.washington.edu",
    "stacks.stanford.edu",
]

boto3, has_boto3, _ = optional_package("boto3")

HEADER_LIST = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"},
    # Firefox 77 Mac
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0",  # noqa
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",  # noqa
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    },
    # Firefox 77 Windows
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0",  # noqa
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",  # noqa
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    },
    # Chrome 83 Mac
    {
        "Connection": "keep-alive",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",  # noqa
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",  # noqa
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
        "Referer": "https://www.google.com/",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    },
    # Chrome 83 Windows
    {
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",  # noqa
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",  # noqa
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Sec-Fetch-Dest": "document",
        "Referer": "https://www.google.com/",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
    },
]


class FetcherError(Exception):
    pass


@warning_for_keywords()
def copyfileobj_withprogress(fsrc, fdst, total_length, *, length=16 * 1024):
    for _ in tqdm(
        range(0, int(total_length), length),
        unit=" MB",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{unit} [{elapsed}]",
    ):
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)


def _already_there_msg(folder):
    """
    Prints a message indicating that a certain data-set is already in place
    """
    msg = "Dataset is already in place. If you want to fetch it again "
    msg += f"please first remove the folder {folder} "
    logger.info(msg)


def _get_file_md5(filename):
    """Compute the md5 checksum of a file"""
    md5_data = md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5_data.block_size), b""):
            md5_data.update(chunk)
    return md5_data.hexdigest()


@warning_for_keywords()
def check_md5(filename, *, stored_md5=None):
    """
    Computes the md5 of filename and check if it matches with the supplied
    string md5

    Parameters
    ----------
    filename : string
        Path to a file.
    md5 : string
        Known md5 of filename to check against. If None (default), checking is
        skipped
    """
    if stored_md5 is not None:
        computed_md5 = _get_file_md5(filename)
        if stored_md5 != computed_md5:
            msg = f"""The downloaded file, {filename}, does not have the expected md5
   checksum of "{stored_md5}". Instead, the md5 checksum was: "{computed_md5}". This
   could mean that something is wrong with the file or that the upstream file has been
   updated. You can try downloading the file again or updating to the newest version of
   dipy."""
            raise FetcherError(msg)


def _get_mirror_url(original_url):
    """Convert original URL to mirror URL preserving path structure.

    Parameters
    ----------
    original_url : str
        The original download URL.

    Returns
    -------
    str or None
        Mirror URL if original can be mirrored, None otherwise.

    Examples
    --------
    >>> url = "https://stacks.stanford.edu/file/druid:yx282xq2090/dwi.nii.gz"
    >>> _get_mirror_url(url)
    'https://workshop.dipy.org/services/data/file/druid:yx282xq2090/dwi.nii.gz'

    """
    for host in MIRRORABLE_HOSTS:
        if host in original_url:
            parts = original_url.split(host, 1)
            if len(parts) == 2:
                path_after_host = parts[1].lstrip("/")
                mirror_url = DIPY_MIRROR_URL + path_after_host
                logger.debug(f"Mirror URL: {original_url} -> {mirror_url}")
                return mirror_url
    return None


def _get_file_data(
    fname, url, *, use_headers=False, timeout=60, max_retries=5, stored_md5=None
):
    """Get data from url and write it to file with retry logic and mirror fallback.

    Parameters
    ----------
    fname : str
        The filename to write the data to.
    url : str
        The URL to get the data from.
    use_headers : bool, optional
        Whether to use headers when downloading files.
    timeout : int, optional
        Timeout in seconds for the HTTP request.
    max_retries : int, optional
        Maximum number of retry attempts.
    stored_md5 : str, optional
        Expected MD5 checksum. If provided, validates after download.

    Raises
    ------
    FetcherError
        If download fails after all retry attempts.

    """
    # Check if mirror is available for this URL
    mirror_url = _get_mirror_url(url)

    is_figshare = "figshare.com" in url
    is_zenodo = "zenodo.org" in url

    if is_figshare or is_zenodo:
        use_headers = True

    if is_figshare or is_zenodo:
        time.sleep(random.uniform(0.5, 2.0))

    last_exception = None
    for attempt in range(max_retries):
        # Alternate between original URL and mirror on retries
        if mirror_url and attempt > 0:
            # Use mirror on odd attempts, original on even attempts
            current_url = mirror_url if attempt % 2 == 1 else url
            if current_url == mirror_url:
                logger.info(f"Trying mirror server: {mirror_url}")
            else:
                logger.info(f"Retrying original URL: {url}")
        else:
            current_url = url

        req = current_url
        if use_headers:
            hdr = random.choice(HEADER_LIST)
            req = Request(current_url, headers=hdr)

        try:
            with contextlib.closing(urlopen(req, timeout=timeout)) as opener:
                try:
                    response_size = opener.headers["content-length"]
                except KeyError:
                    response_size = None

                with open(fname, "wb") as data:
                    if response_size is None:
                        copyfileobj(opener, data)
                    else:
                        copyfileobj_withprogress(opener, data, response_size)

            # Validate MD5 checksum if provided
            if stored_md5 is not None:
                try:
                    check_md5(fname, stored_md5=stored_md5)
                except FetcherError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"MD5 checksum mismatch for {current_url}. "
                            f"Retrying with alternate source..."
                        )
                        # Remove bad file
                        if Path(fname).exists():
                            os.remove(fname)
                        # Continue to next attempt (will try mirror)
                        continue
                    else:
                        # Last attempt failed, raise the error
                        raise

            # Success - exit the retry loop
            return

        except HTTPError as e:
            last_exception = e
            # Special handling for 403 Forbidden (rate limiting)
            if e.code == 403:
                if attempt < max_retries - 1:
                    # Longer wait for rate limiting (exponential + jitter)
                    base_wait = 2 ** (attempt + 2)  # 4, 8, 16, 32 seconds
                    jitter = random.uniform(1, 5)
                    wait_time = base_wait + jitter
                    logger.warning(
                        f"Download blocked (403 Forbidden) for {current_url}. "
                        f"Likely rate limiting. Waiting {wait_time:.1f}s "
                        f"before retry {attempt + 2}/{max_retries}..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Download blocked after {max_retries} attempts. "
                        f"URL: {current_url}. This may indicate rate limiting."
                    )
            else:
                # Other HTTP errors (404, 500, 504, etc.)
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(
                        f"Download attempt {attempt + 1}/{max_retries} "
                        f"failed for {current_url}: HTTP {e.code}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)

        except (URLError, TimeoutError, OSError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Exponential backoff: 2, 4, 8, 16 seconds
                wait_time = 2 ** (attempt + 1)
                logger.warning(
                    f"Download attempt {attempt + 1}/{max_retries} failed "
                    f"for {current_url}: {e}. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Download failed after {max_retries} attempts for {current_url}"
                )

    # If we get here, all retries failed
    error_msg = f"Failed to download {url} after {max_retries} attempts."
    if isinstance(last_exception, HTTPError) and last_exception.code == 403:
        error_msg += (
            " Got 403 Forbidden - likely rate limiting. "
            "Try reducing parallel downloads or wait before retrying."
        )
    error_msg += f" Last error: {last_exception}"
    raise FetcherError(error_msg)


@warning_for_keywords()
def fetch_data(
    files, folder, *, data_size=None, use_headers=False, raise_on_error=True
):
    """Downloads files to folder and checks their md5 checksums

    Parameters
    ----------
    files : dictionary
        For each file in `files` the value should be (url, md5). The file will
        be downloaded from url if the file does not already exist or if the
        file exists but the md5 checksum does not match.
    folder : str or Path
        The directory where to save the file, the directory will be created if
        it does not already exist.
    data_size : str, optional
        A string describing the size of the data (e.g. "91 MB") to be logged to
        the screen. Default does not produce any information about data size.
    use_headers : bool, optional
        Whether to use headers when downloading files.
    raise_on_error : bool, optional
        If True (default), raises FetcherError on download/checksum failures.
        If False, logs warnings and continues with remaining files.

    Raises
    ------
    FetcherError
        Raises if raise_on_error is True and the md5 checksum of the file
        does not match the expected value or download fails.

    """
    if not Path(folder).exists():
        logger.info(f"Creating new folder {folder}")
        os.makedirs(folder)

    if data_size is not None:
        logger.info(f"Data size is approximately {data_size}")

    all_skip = True
    failed_files = []
    successful_downloads = 0

    for f in files:
        url, md5 = files[f]
        fullpath = Path(folder) / f
        if fullpath.exists() and (_get_file_md5(fullpath) == md5):
            continue
        all_skip = False
        logger.info(f'Downloading "{f}" to {folder}')
        logger.info(f"From: {url}")
        try:
            _get_file_data(fullpath, url, use_headers=use_headers, stored_md5=md5)
            successful_downloads += 1
        except (FetcherError, Exception) as e:
            failed_files.append((f, url, str(e)))
            if raise_on_error:
                raise
            else:
                logger.warning(f"Failed to download {f}: {e}")
                # Clean up partial download
                if fullpath.exists():
                    try:
                        os.remove(fullpath)
                    except OSError:
                        pass

    if all_skip:
        _already_there_msg(folder)
    else:
        if failed_files:
            logger.warning(
                f"Downloaded {successful_downloads} files successfully, "
                f"but {len(failed_files)} files failed to download."
            )
            for fname, furl, error in failed_files:
                logger.warning(f"  - {fname} from {furl}: {error}")
        else:
            logger.info(f"Files successfully downloaded to {folder}")


@warning_for_keywords()
def _make_fetcher(
    name,
    folder,
    baseurl,
    remote_fnames,
    local_fnames,
    *,
    md5_list=None,
    optional_fnames=None,
    doc="",
    data_size=None,
    msg=None,
    unzip=False,
    use_headers=False,
):
    """Create a new fetcher

    Parameters
    ----------
    name : str
        The name of the fetcher function.
    folder : str or Path
        The full path to the folder in which the files would be placed locally.
        Typically, this is something like 'Path(dipy_home) / 'foo''
    baseurl : str
        The URL from which this fetcher reads files
    remote_fnames : list of strings
        The names of the files in the baseurl location
    local_fnames : list of strings or Paths
        The names of the files to be saved on the local filesystem
    md5_list : list of strings, optional
        The md5 checksums of the files. Used to verify the content of the
        files. Default: None, skipping checking md5.
    optional_fnames : str, optional
        The name of the optional file to be saved on the local filesystem.
    doc : str, optional.
        Documentation of the fetcher.
    data_size : str, optional.
        If provided, is sent as a message to the user before downloading
        starts.
    msg : str, optional.
        A message to print to screen when fetching takes place. Default (None)
        is to print nothing
    unzip : bool, optional
        Whether to unzip the file(s) after downloading them. Supports zip, gz,
        and tar.gz files.
    use_headers : bool, optional
        Whether to use headers when downloading files.

    returns
    -------
    fetcher : function
        A function that, when called, fetches data according to the designated
        inputs
    """
    optional_fnames = optional_fnames or []

    def fetcher(*, include_optional=False):
        files = {}
        for (
            i,
            (f, n),
        ) in enumerate(zip(remote_fnames, local_fnames)):
            if n in optional_fnames and not include_optional:
                continue
            files[str(n)] = (baseurl + f, md5_list[i] if md5_list is not None else None)

        fetch_data(files, folder, data_size=data_size, use_headers=use_headers)

        if msg is not None:
            logger.info(msg)
        if unzip:
            for f in local_fnames:
                p = Path(f)
                if p.suffix in (".gz", ".bz2"):
                    if p.with_suffix("").suffix == ".tar":
                        ar = tarfile.open(Path(folder) / f)
                        ar.extractall(path=folder)
                        ar.close()
                    else:
                        raise ValueError("File extension is not recognized")
                elif p.suffix == ".zip":
                    z = zipfile.ZipFile(Path(folder) / f, "r")
                    files[str(f)] += (tuple(z.namelist()),)
                    z.extractall(folder)
                    z.close()
                else:
                    raise ValueError("File extension is not recognized")

        return files, folder

    fetcher.__name__ = name
    fetcher.__doc__ = doc
    return fetcher


fetch_isbi2013_2shell = _make_fetcher(
    "fetch_isbi2013_2shell",
    dipy_home / "isbi2013",
    UW_RW_URL + "1773/38465/",
    ["phantom64.nii.gz", "phantom64.bval", "phantom64.bvec"],
    [Path("phantom64.nii.gz"), Path("phantom64.bval"), Path("phantom64.bvec")],
    md5_list=[
        "42911a70f232321cf246315192d69c42",
        "90e8cf66e0f4d9737a3b3c0da24df5ea",
        "4b7aa2757a1ccab140667b76e8075cb1",
    ],
    doc="Download a 2-shell software phantom dataset",
    data_size="",
)

fetch_stanford_labels = _make_fetcher(
    "fetch_stanford_labels",
    dipy_home / "stanford_hardi",
    "https://stacks.stanford.edu/file/druid:yx282xq2090/",
    ["aparc-reduced.nii.gz", "label_info.txt"],
    [Path("aparc-reduced.nii.gz"), Path("label_info.txt")],
    md5_list=["742de90090d06e687ce486f680f6d71a", "39db9f0f5e173d7a2c2e51b07d5d711b"],
    doc="Download reduced freesurfer aparc image from stanford web site",
)

fetch_sherbrooke_3shell = _make_fetcher(
    "fetch_sherbrooke_3shell",
    dipy_home / "sherbrooke_3shell",
    UW_RW_URL + "1773/38475/",
    ["HARDI193.nii.gz", "HARDI193.bval", "HARDI193.bvec"],
    [Path("HARDI193.nii.gz"), Path("HARDI193.bval"), Path("HARDI193.bvec")],
    md5_list=[
        "0b735e8f16695a37bfbd66aab136eb66",
        "e9b9bb56252503ea49d31fb30a0ac637",
        "0c83f7e8b917cd677ad58a078658ebb7",
    ],
    doc="Download a 3shell HARDI dataset with 192 gradient direction",
)


fetch_stanford_hardi = _make_fetcher(
    "fetch_stanford_hardi",
    dipy_home / "stanford_hardi",
    "https://stacks.stanford.edu/file/druid:yx282xq2090/",
    ["dwi.nii.gz", "dwi.bvals", "dwi.bvecs"],
    [Path("HARDI150.nii.gz"), Path("HARDI150.bval"), Path("HARDI150.bvec")],
    md5_list=[
        "0b18513b46132b4d1051ed3364f2acbc",
        "4e08ee9e2b1d2ec3fddb68c70ae23c36",
        "4c63a586f29afc6a48a5809524a76cb4",
    ],
    doc="Download a HARDI dataset with 160 gradient directions",
)

fetch_resdnn_tf_weights = _make_fetcher(
    "fetch_resdnn_tf_weights",
    dipy_home / "histo_resdnn_weights",
    "https://ndownloader.figshare.com/files/",
    ["22736240"],
    [Path("resdnn_weights_mri_2018.h5")],
    md5_list=["f0e118d72ab804a464494bd9015227f4"],
    doc="Download ResDNN Tensorflow model weights for Nath et. al 2018",
)

fetch_resdnn_torch_weights = _make_fetcher(
    "fetch_resdnn_torch_weights",
    dipy_home / "histo_resdnn_weights",
    "https://ndownloader.figshare.com/files/",
    ["50019429"],
    [Path("histo_weights.pth")],
    md5_list=["ca13692bbbaea725ff8b5df2d3a2779a"],
    doc="Download ResDNN Pytorch model weights for Nath et. al 2018",
)

fetch_synb0_weights = _make_fetcher(
    "fetch_synb0_weights",
    dipy_home / "synb0",
    "https://ndownloader.figshare.com/files/",
    ["36379914", "36379917", "36379920", "36379923", "36379926"],
    [
        Path("synb0_default_weights1.h5"),
        Path("synb0_default_weights2.h5"),
        Path("synb0_default_weights3.h5"),
        Path("synb0_default_weights4.h5"),
        Path("synb0_default_weights5.h5"),
    ],
    md5_list=[
        "a9362c75bc28616167a11a42fe5d004e",
        "9dc9353d6ff741d8e22b8569f157c56e",
        "e548f341e4f12d63dfbed306233fddce",
        "8cb7a3492d08e4c9b8938277d6fd9b75",
        "5e796f892605b3bdb9cb9678f1c6ac11",
    ],
    doc="Download Synb0 model weights for Schilling et. al 2019",
)

fetch_synb0_test = _make_fetcher(
    "fetch_synb0_test",
    dipy_home / "synb0",
    "https://ndownloader.figshare.com/files/",
    ["36379911", "36671850"],
    [Path("test_input_synb0.npz"), Path("test_output_synb0.npz")],
    md5_list=["987203aa73de2dac8770f39ed506dc0c", "515544fbcafd9769785502821b47b661"],
    doc="Download Synb0 test data for Schilling et. al 2019",
)

fetch_deepn4_tf_weights = _make_fetcher(
    "fetch_deepn4_tf_weights",
    dipy_home / "deepn4",
    "https://ndownloader.figshare.com/files/",
    ["44673313"],
    [Path("model_weights.h5")],
    md5_list=["ef264edd554177a180cf99162dbd2745"],
    doc="Download DeepN4 model weights for Kanakaraj et. al 2024",
)

fetch_deepn4_torch_weights = _make_fetcher(
    "fetch_deepn4_torch_weights",
    dipy_home / "deepn4",
    "https://ndownloader.figshare.com/files/",
    ["52285805"],
    [Path("deepn4_torch_weights")],
    md5_list=["97c5a5f8356a3d0eeca1c6bb7949c8b8"],
    doc="Download DeepN4 model weights for Kanakaraj et. al 2024",
)

fetch_deepn4_test = _make_fetcher(
    "fetch_deepn4_test",
    dipy_home / "deepn4",
    "https://ndownloader.figshare.com/files/",
    ["60829657"],
    ["d4_data.npz"],
    md5_list=["d3cc2020107599139fe02bbadacee8ef"],
    doc="Download DeepN4 test data for Kanakaraj et. al 2024",
)

fetch_evac_tf_weights = _make_fetcher(
    "fetch_evac_tf_weights",
    dipy_home / "evac",
    "https://ndownloader.figshare.com/files/",
    ["43037191"],
    [Path("evac_default_weights.h5")],
    md5_list=["491cfa4f9a2860fad6c19f2b71b918e1"],
    doc="Download EVAC+ model weights for Park et. al 2022",
)

fetch_evac_torch_weights = _make_fetcher(
    "fetch_evac_torch_weights",
    dipy_home / "evac",
    "https://ndownloader.figshare.com/files/",
    ["50019432"],
    [Path("evac_weights.pth")],
    md5_list=["b2bab512a0f899f089d9dd9ffc44f65b"],
    doc="Download EVAC+ model weights for Park et. al 2022",
)

fetch_evac_test = _make_fetcher(
    "fetch_evac_test",
    dipy_home / "evac",
    "https://ndownloader.figshare.com/files/",
    ["60829660"],
    ["evac_test_data.npz"],
    md5_list=["d1f0dd6d38b4d358b2ce53a495fae31e"],
    doc="Download EVAC+ test data for Park et. al 2022",
)

fetch_synthseg_torch_weights = _make_fetcher(
    "fetch_synthseg_torch_weights",
    Path(dipy_home) / "synthseg",
    "https://ndownloader.figshare.com/files/",
    ["60274412"],
    ["synthseg_model_weights.pth"],
    md5_list=["83d54a150410bdc1f04d70a84ff8f1ee"],
    doc="Download SynthSeg model weights for Billot et. al 2023",
)

fetch_synthseg_test = _make_fetcher(
    "fetch_synthseg_test",
    Path(dipy_home) / "synthseg",
    "https://ndownloader.figshare.com/files/",
    ["60572540"],
    ["synthseg_test_data.npz"],
    md5_list=["70377d1858dc870eca389f30cfc4e193"],
    doc="Download SynthSeg test data for Billot et. al 2023",
)

fetch_stanford_t1 = _make_fetcher(
    "fetch_stanford_t1",
    dipy_home / "stanford_hardi",
    "https://stacks.stanford.edu/file/druid:yx282xq2090/",
    ["t1.nii.gz"],
    ["t1.nii.gz"],
    md5_list=["a6a140da6a947d4131b2368752951b0a"],
)

fetch_stanford_pve_maps = _make_fetcher(
    "fetch_stanford_pve_maps",
    dipy_home / "stanford_hardi",
    "https://stacks.stanford.edu/file/druid:yx282xq2090/",
    ["pve_csf.nii.gz", "pve_gm.nii.gz", "pve_wm.nii.gz"],
    [Path("pve_csf.nii.gz"), Path("pve_gm.nii.gz"), Path("pve_wm.nii.gz")],
    md5_list=[
        "2c498e4fed32bca7f726e28aa86e9c18",
        "1654b20aeb35fc2734a0d7928b713874",
        "2e244983cf92aaf9f9d37bc7716b37d5",
    ],
)

fetch_stanford_tracks = _make_fetcher(
    "fetch_stanford_tracks",
    dipy_home / "stanford_hardi",
    "https://raw.githubusercontent.com/dipy/dipy_datatest/main/",
    [
        "hardi-lr-superiorfrontal.trk",
    ],
    [
        Path("hardi-lr-superiorfrontal.trk"),
    ],
    md5_list=[
        "2d49aaf6ad6c10d8d069bfb319bf3541",
    ],
    doc="Download stanford track for examples",
    data_size="1.4MB",
)

fetch_taiwan_ntu_dsi = _make_fetcher(
    "fetch_taiwan_ntu_dsi",
    dipy_home / "taiwan_ntu_dsi",
    UW_RW_URL + "1773/38480/",
    ["DSI203.nii.gz", "DSI203.bval", "DSI203.bvec", "DSI203_license.txt"],
    [
        Path("DSI203.nii.gz"),
        Path("DSI203.bval"),
        Path("DSI203.bvec"),
        Path("DSI203_license.txt"),
    ],
    md5_list=[
        "950408c0980a7154cb188666a885a91f",
        "602e5cb5fad2e7163e8025011d8a6755",
        "a95eb1be44748c20214dc7aa654f9e6b",
        "7fa1d5e272533e832cc7453eeba23f44",
    ],
    doc="Download a DSI dataset with 203 gradient directions",
    msg="See DSI203_license.txt for LICENSE. For the complete datasets"
    + " please visit https://dsi-studio.labsolver.org",
    data_size="91MB",
)

fetch_syn_data = _make_fetcher(
    "fetch_syn_data",
    dipy_home / "syn_test",
    UW_RW_URL + "1773/38476/",
    ["t1.nii.gz", "b0.nii.gz"],
    [Path("t1.nii.gz"), Path("b0.nii.gz")],
    md5_list=["701bda02bb769655c7d4a9b1df2b73a6", "e4b741f0c77b6039e67abb2885c97a78"],
    data_size="12MB",
    doc="Download t1 and b0 volumes from the same session",
)

fetch_mni_template = _make_fetcher(
    "fetch_mni_template",
    dipy_home / "mni_template",
    "https://ndownloader.figshare.com/files/",
    [
        "5572676?private_link=4b8666116a0128560fb5",
        "5572673?private_link=93216e750d5a7e568bda",
        "5572670?private_link=33c92d54d1afb9aa7ed2",
        "5572661?private_link=584319b23e7343fed707",
    ],
    [
        Path("mni_icbm152_t2_tal_nlin_asym_09a.nii"),
        Path("mni_icbm152_t1_tal_nlin_asym_09a.nii"),
        Path("mni_icbm152_t1_tal_nlin_asym_09c_mask.nii"),
        Path("mni_icbm152_t1_tal_nlin_asym_09c.nii"),
    ],
    md5_list=[
        "f41f2e1516d880547fbf7d6a83884f0d",
        "1ea8f4f1e41bc17a94602e48141fdbc8",
        "a243e249cd01a23dc30f033b9656a786",
        "3d5dd9b0cd727a17ceec610b782f66c1",
    ],
    doc="fetch the MNI 2009a T1 and T2, and 2009c T1 and T1 mask files",
    data_size="70MB",
)

fetch_scil_b0 = _make_fetcher(
    "fetch_scil_b0",
    dipy_home,
    UW_RW_URL + "1773/38479/",
    ["datasets_multi-site_all_companies.zip"],
    [Path("datasets_multi-site_all_companies.zip")],
    md5_list=["e9810fa5bf21b99da786647994d7d5b7"],
    doc="Download b=0 datasets from multiple MR systems (GE, Philips, "
    + "Siemens) and different magnetic fields (1.5T and 3T)",
    data_size="9.2MB",
    unzip=True,
)

fetch_bundles_2_subjects = _make_fetcher(
    "fetch_bundles_2_subjects",
    dipy_home / "exp_bundles_and_maps",
    UW_RW_URL + "1773/38477/",
    ["bundles_2_subjects.tar.gz"],
    [Path("bundles_2_subjects.tar.gz")],
    md5_list=["97756fbef11ce2df31f1bedf1fc7aac7"],
    data_size="234MB",
    doc="Download 2 subjects from the SNAIL dataset with their bundles",
    unzip=True,
)

fetch_ivim = _make_fetcher(
    "fetch_ivim",
    dipy_home / "ivim",
    "https://ndownloader.figshare.com/files/",
    ["5305243", "5305246", "5305249"],
    [Path("ivim.nii.gz"), Path("ivim.bval"), Path("ivim.bvec")],
    md5_list=[
        "cda596f89dc2676af7d9bf1cabccf600",
        "f03d89f84aa9a9397103a400e43af43a",
        "fb633a06b02807355e49ccd85cb92565",
    ],
    doc="Download IVIM dataset",
)

fetch_cfin_multib = _make_fetcher(
    "fetch_cfin_multib",
    dipy_home / "cfin_multib",
    UW_RW_URL + "/1773/38488/",
    [
        "T1.nii",
        "__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii",
        "__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bval",
        "__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bvec",
    ],
    [
        Path("T1.nii"),
        Path("__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii"),
        Path("__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bval"),
        Path("__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bvec"),
    ],
    md5_list=[
        "889883b5e7d93a6e372bc760ea887e7c",
        "9daea1d01d68fd0055a3b34f5ffd5f6e",
        "3ee44135fde7ea5c9b8c801414bdde2c",
        "948373391de950e7cc1201ba9f696bf0",
    ],
    doc="Download CFIN multi b-value diffusion data",
    msg=(
        "This data was provided by Brian Hansen and Sune Jespersen"
        + " More details about the data are available in their paper: "
        + " https://www.nature.com/articles/sdata201672"
    ),
)

fetch_file_formats = _make_fetcher(
    "bundle_file_formats_example",
    dipy_home / "bundle_file_formats_example",
    "https://zenodo.org/record/3352379/files/",
    [
        "cc_m_sub.trk",
        "laf_m_sub.tck",
        "lpt_m_sub.fib",
        "raf_m_sub.vtk",
        "rpt_m_sub.dpy",
        "template0.nii.gz",
    ],
    [
        Path("cc_m_sub.trk"),
        Path("laf_m_sub.tck"),
        Path("lpt_m_sub.fib"),
        Path("raf_m_sub.vtk"),
        Path("rpt_m_sub.dpy"),
        Path("template0.nii.gz"),
    ],
    md5_list=[
        "78ed7bead3e129fb4b4edd6da1d7e2d2",
        "20009796ccd43dc8d2d5403b25dff717",
        "8afa8419e2efe04ede75cce1f53c77d8",
        "9edcbea30c7a83b467c3cdae6ce963c8",
        "42bff2538a650a7ff1e57bfd9ed90ad6",
        "99c37a2134026d2c4bbb7add5088ddc6",
    ],
    doc="Download 5 bundles in various file formats and their reference",
    data_size="25MB",
)

fetch_bundle_atlas_hcp842 = _make_fetcher(
    "fetch_bundle_atlas_hcp842",
    dipy_home / "bundle_atlas_hcp842",
    "https://ndownloader.figshare.com/files/",
    ["13638644"],
    [Path("Atlas_80_Bundles.zip")],
    md5_list=["78331d527a10ec000d4f33bac472e099"],
    doc="Download atlas tractogram from the hcp842 dataset with 80 bundles",
    data_size="300MB",
    unzip=True,
)

fetch_30_bundle_atlas_hcp842 = _make_fetcher(
    "fetch_30_bundle_atlas_hcp842",
    dipy_home / "bundle_atlas_hcp842",
    "https://ndownloader.figshare.com/files/",
    ["26842853"],
    [Path("Atlas_30_Bundles.zip")],
    md5_list=["f3922cdbea4216823798fade128d6782"],
    doc="Download atlas tractogram from the hcp842 dataset with 30 bundles",
    data_size="207.09MB",
    unzip=True,
)

fetch_target_tractogram_hcp = _make_fetcher(
    "fetch_target_tractogram_hcp",
    dipy_home / "target_tractogram_hcp",
    "https://ndownloader.figshare.com/files/",
    ["12871127"],
    [Path("hcp_tractogram.zip")],
    md5_list=["fa25ef19c9d3748929b6423397963b6a"],
    doc="Download tractogram of one of the hcp dataset subjects",
    data_size="541MB",
    unzip=True,
)


fetch_bundle_fa_hcp = _make_fetcher(
    "fetch_bundle_fa_hcp",
    dipy_home / "bundle_fa_hcp",
    "https://ndownloader.figshare.com/files/",
    ["14035265"],
    [Path("hcp_bundle_fa.nii.gz")],
    md5_list=["2d5c0036b0575597378ddf39191028ea"],
    doc=("Download map of FA within two bundles in one of the hcp dataset subjects"),
    data_size="230kb",
)


fetch_qtdMRI_test_retest_2subjects = _make_fetcher(
    "fetch_qtdMRI_test_retest_2subjects",
    dipy_home / "qtdMRI_test_retest_2subjects",
    "https://zenodo.org/record/996889/files/",
    [
        "subject1_dwis_test.nii.gz",
        "subject2_dwis_test.nii.gz",
        "subject1_dwis_retest.nii.gz",
        "subject2_dwis_retest.nii.gz",
        "subject1_ccmask_test.nii.gz",
        "subject2_ccmask_test.nii.gz",
        "subject1_ccmask_retest.nii.gz",
        "subject2_ccmask_retest.nii.gz",
        "subject1_scheme_test.txt",
        "subject2_scheme_test.txt",
        "subject1_scheme_retest.txt",
        "subject2_scheme_retest.txt",
    ],
    [
        Path("subject1_dwis_test.nii.gz"),
        Path("subject2_dwis_test.nii.gz"),
        Path("subject1_dwis_retest.nii.gz"),
        Path("subject2_dwis_retest.nii.gz"),
        Path("subject1_ccmask_test.nii.gz"),
        Path("subject2_ccmask_test.nii.gz"),
        Path("subject1_ccmask_retest.nii.gz"),
        Path("subject2_ccmask_retest.nii.gz"),
        Path("subject1_scheme_test.txt"),
        Path("subject2_scheme_test.txt"),
        Path("subject1_scheme_retest.txt"),
        Path("subject2_scheme_retest.txt"),
    ],
    md5_list=[
        "ebd7441f32c40e25c28b9e069bd81981",
        "dd6a64dd68c8b321c75b9d5fb42c275a",
        "830a7a028a66d1b9812f93309a3f9eae",
        "d7f1951e726c35842f7ea0a15d990814",
        "ddb8dfae908165d5e82c846bcc317cab",
        "5630c06c267a0f9f388b07b3e563403c",
        "02e9f92b31e8980f658da99e532e14b5",
        "6e7ce416e7cfda21cecce3731f81712b",
        "957cb969f97d89e06edd7a04ffd61db0",
        "5540c0c9bd635c29fc88dd599cbbf5e6",
        "5540c0c9bd635c29fc88dd599cbbf5e6",
        "5540c0c9bd635c29fc88dd599cbbf5e6",
    ],
    doc="Downloads test-retest qt-dMRI acquisitions of two C57Bl6 mice.",
    data_size="298.2MB",
)


fetch_gold_standard_io = _make_fetcher(
    "fetch_gold_standard_io",
    dipy_home / "gold_standard_io",
    "https://zenodo.org/record/14538513/files/",
    [
        "gs_streamlines.trk",
        "gs_streamlines.tck",
        "gs_streamlines.trx",
        "gs_streamlines.fib",
        "gs_streamlines.vtk",
        "gs_streamlines.dpy",
        "gs_mesh_faces.txt",
        "gs_mesh.gii",
        "gs_mesh_lpsmm_corner.ply",
        "gs_mesh_lpsmm_center.ply",
        "gs_mesh_rasmm_center.ply",
        "gs_mesh_rasmm_corner.ply",
        "gs_mesh_vox_center.ply",
        "gs_mesh_vox_corner.ply",
        "gs_mesh_voxmm_center.ply",
        "gs_mesh_voxmm_corner.ply",
        "gs_volume.nii",
        "gs_volume_3mm.nii",
        "gs_streamlines_rasmm_space.txt",
        "gs_streamlines_voxmm_space.txt",
        "gs_streamlines_vox_space.txt",
        "gs_mesh_rasmm_corner.txt",
        "gs_mesh_rasmm_center.txt",
        "gs_mesh_lpsmm_corner.txt",
        "gs_mesh_lpsmm_center.txt",
        "gs_mesh_voxmm_corner.txt",
        "gs_mesh_voxmm_center.txt",
        "gs_mesh_vox_corner.txt",
        "gs_mesh_vox_center.txt",
        "points_data.json",
        "streamlines_data.json",
    ],
    [
        Path("gs_streamlines.trk"),
        Path("gs_streamlines.tck"),
        Path("gs_streamlines.trx"),
        Path("gs_streamlines.fib"),
        Path("gs_streamlines.vtk"),
        Path("gs_streamlines.dpy"),
        Path("gs_mesh_faces.txt"),
        Path("gs_mesh.gii"),
        Path("gs_mesh_lpsmm_corner.ply"),
        Path("gs_mesh_lpsmm_center.ply"),
        Path("gs_mesh_rasmm_center.ply"),
        Path("gs_mesh_rasmm_corner.ply"),
        Path("gs_mesh_vox_center.ply"),
        Path("gs_mesh_vox_corner.ply"),
        Path("gs_mesh_voxmm_center.ply"),
        Path("gs_mesh_voxmm_corner.ply"),
        Path("gs_volume.nii"),
        Path("gs_volume_3mm.nii"),
        Path("gs_streamlines_rasmm_space.txt"),
        Path("gs_streamlines_voxmm_space.txt"),
        Path("gs_streamlines_vox_space.txt"),
        Path("gs_mesh_rasmm_corner.txt"),
        Path("gs_mesh_rasmm_center.txt"),
        Path("gs_mesh_lpsmm_corner.txt"),
        Path("gs_mesh_lpsmm_center.txt"),
        Path("gs_mesh_voxmm_corner.txt"),
        Path("gs_mesh_voxmm_center.txt"),
        Path("gs_mesh_vox_corner.txt"),
        Path("gs_mesh_vox_center.txt"),
        Path("points_data.json"),
        Path("streamlines_data.json"),
    ],
    md5_list=[
        "3acf565779f4d5107f96b2ef90578d64",
        "151a30cf356c002060d720bf9d577245",
        "a6587f1a3adc4df076910c4d72eb4161",
        "e9818e07bef5bd605dea0877df14a2b0",
        "e9818e07bef5bd605dea0877df14a2b0",
        "248606297e400d1a9b1786845aad8de3",
        "775852c88be8f8c2c69f0b7fdcb8d310",
        "04889f220a57c43dd955987e516b51b2",
        "cc05be6621e7986061844e2956339af3",
        "d2fc1c7f64c472b95a769b620a33e51f",
        "18a37f374fb13021d115bb2db6b2ab19",
        "f2c95797b5e12a3117619e196138e19d",
        "d2d36eef82aef9897de95d9d38e95eaf",
        "0ed5b7aa807f4c20ba681db76150e3cd",
        "6a1d90454284b31c4e0ec3ce5c3d4ce7",
        "79fcf19eb9c8b9a94f0a47ed833ae901",
        "a2d4d8f62d1de0ab9927782c7d51cb27",
        "217b3ae0712a02b2463b8eedfe9a0a68",
        "ca193a5508d3313d542231aaf262960f",
        "3284de59dfd9ca3130e6e01258ed9022",
        "a2a89c387f45adab733652a92f6602d5",
        "30e36aa7103b7900fc4bb92807975b6b",
        "0a33b20061af68bb22bc4c755702fd4a",
        "617dab59e802f5e30eae6f327df025de",
        "b299f01365b81e588375fc824beb8441",
        "3310302ec55808d52b22544d0b83aabb",
        "c653b6927356c489b81b7124fa9fb375",
        "7bd00557ebddbf261684cefee1e53685",
        "c4969777454d557850fd2cf5d1229acc",
        "4bcca0c6195871fc05e93cdfabec22b4",
        "578f29052ac03a6d8a98580eb7c70d97",
    ],
    doc="Downloads the gold standard for streamlines and meshes io testing.",
    data_size="51.KB",
)


fetch_real_data_io = _make_fetcher(
    "fetch_real_data_io",
    dipy_home / "real_data_io",
    "https://zenodo.org/record/14537772/files/",
    [
        "anat.nii.gz",
        "baf_lh.orig",
        "baf_lh.pial",
        "baf_lh.smoothwm",
        "baf_rh.orig",
        "baf_rh.pial",
        "baf_rh.smoothwm",
        "baf_t1.nii.gz",
        "naf_lh.pial",
        "naf_mni_masked.nii.gz",
        "pial.L.surf.gii",
        "pial.L.surf.gii.gz",
        "saf_lh.orig",
        "saf_lh.pial",
        "saf_lh.smoothwm",
        "saf_rh.orig",
        "saf_rh.pial",
        "saf_rh.smoothwm",
        "saf_t1.nii.gz",
        "smoothwm.L.surf.gii",
        "smoothwm.L.surf.gii.gz",
    ],
    [
        Path("anat.nii.gz"),
        Path("baf_lh.orig"),
        Path("baf_lh.pial"),
        Path("baf_lh.smoothwm"),
        Path("baf_rh.orig"),
        Path("baf_rh.pial"),
        Path("baf_rh.smoothwm"),
        Path("baf_t1.nii.gz"),
        Path("naf_lh.pial"),
        Path("naf_mni_masked.nii.gz"),
        Path("pial.L.surf.gii"),
        Path("pial.L.surf.gii.gz"),
        Path("saf_lh.orig"),
        Path("saf_lh.pial"),
        Path("saf_lh.smoothwm"),
        Path("saf_rh.orig"),
        Path("saf_rh.pial"),
        Path("saf_rh.smoothwm"),
        Path("saf_t1.nii.gz"),
        Path("smoothwm.L.surf.gii"),
        Path("smoothwm.L.surf.gii.gz"),
    ],
    md5_list=[
        "deb754c933076edf360be8a1842d651b",
        "162506eaf2ab8f6c8bb7498b6d3dbe75",
        "7ae3f808e5d497759e16e9c992a99912",
        "211a7c015c1050b338006d2a1b79a1c9",
        "6c2c1e34aecc977e4a02eca42a3da6e2",
        "243f193388473858dbd80acd4f5c1be4",
        "96f58e8fc72f51bc0ea360aefeaa9914",
        "30a270c56730cc60ec7539d493a76e98",
        "8e6087f510a0dfb513ad46c14d82a4bf",
        "ea6c119442d23a25033de19b55c607d3",
        "252928164e241871003190ac6d013cd3",
        "f1b7b0785036903c5ebf8d4ed61e431c",
        "486de9dbbb8250903ad930372f9fb0be",
        "f853d34e01abb68bf1ca2cd3ec18d3c2",
        "a5b25fbe6aad62d36fba77c0a2fb21b0",
        "187ee9b18312d3e4b5c8f13e4acf3d9e",
        "15387c2de6ef710d94374d734593a5bb",
        "f63648f5b8fe2a1517d6f9c034f284bb",
        "3734ce8cbd95e6f1587603dd79b631ef",
        "54aa36961ca4a19f7843d7a5a1e721d0",
        "0b6dbdda2416cf2db2bf56603a154845",
    ],
    doc="Downloads the gold standard for streamlines and meshes io testing.",
    data_size="51.KB",
)


fetch_qte_lte_pte = _make_fetcher(
    "fetch_qte_lte_pte",
    dipy_home / "qte_lte_pte",
    "https://zenodo.org/record/4624866/files/",
    ["lte-pte.nii.gz", "lte-pte.bval", "lte-pte.bvec", "mask.nii.gz"],
    [
        Path("lte-pte.nii.gz"),
        Path("lte-pte.bval"),
        Path("lte-pte.bvec"),
        Path("mask.nii.gz"),
    ],
    md5_list=[
        "f378b2cd9f57625512002b9e4c0f1660",
        "5c25d24dd3df8590582ed690507a8769",
        "31abe55dfda7ef5fdf5015d0713be9b0",
        "1b7b83b8a60295f52d80c3855a12b275",
    ],
    doc="Download QTE data with linear and planar tensor encoding.",
    data_size="41.5 MB",
)


fetch_cti_rat1 = _make_fetcher(
    "fetch_cti_rat1",
    dipy_home / "cti_rat1",
    "https://zenodo.org/record/8276773/files/",
    [
        "Rat1_invivo_cti_data.nii",
        "bvals1.bval",
        "bvec1.bvec",
        "bvals2.bval",
        "bvec2.bvec",
        "Rat1_mask.nii",
    ],
    [
        Path("Rat1_invivo_cti_data.nii"),
        Path("bvals1.bval"),
        Path("bvec1.bvec"),
        Path("bvals2.bval"),
        Path("bvec2.bvec"),
        Path("Rat1_mask.nii"),
    ],
    md5_list=[
        "2f855e7826f359d80cfd6f094d3a7008",
        "1deed2a91e20104ca42d7482cc096a9a",
        "40a4f5131b8a64608d16b0c6c5ad0837",
        "1979c7dc074e00f01103cbdf83ed78db",
        "653d9344060803d5576f43c65ce45ccb",
        "34bc3d5acea9442d05ef185717780440",
    ],
    doc="Download Rat Brain DDE data for CTI reconstruction"
    + " (Rat #1 data from Henriques et al. MRM 2021).",
    data_size="152.92 MB",
    msg=(
        "More details about the data are available in the paper: "
        + "https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28938"
    ),
)


fetch_fury_surface = _make_fetcher(
    "fetch_fury_surface",
    dipy_home / "fury_surface",
    "https://raw.githubusercontent.com/fury-gl/fury-data/master/surfaces/",
    ["100307_white_lh.vtk"],
    [Path("100307_white_lh.vtk")],
    md5_list=["dbec91e29af15541a5cb36d80977b26b"],
    doc="Surface for testing and examples",
    data_size="11MB",
)

fetch_DiB_70_lte_pte_ste = _make_fetcher(
    "fetch_DiB_70_lte_pte_ste",
    dipy_home / "DiB_70_lte_pte_ste",
    "https://github.com/filip-szczepankiewicz/Szczepankiewicz_DIB_2019/"
    "raw/master/DATA/brain/NII_Boito_SubSamples/",
    [
        "DiB_70_lte_pte_ste/DiB_70_lte_pte_ste.nii.gz",
        "DiB_70_lte_pte_ste/bval_DiB_70_lte_pte_ste.bval",
        "DiB_70_lte_pte_ste/bvec_DiB_70_lte_pte_ste.bvec",
        "DiB_mask.nii.gz",
    ],
    [
        Path("DiB_70_lte_pte_ste.nii.gz"),
        Path("bval_DiB_70_lte_pte_ste.bval"),
        Path("bvec_DiB_70_lte_pte_ste.bvec"),
        Path("DiB_mask.nii.gz"),
    ],
    doc="Download QTE data with linear, planar, "
    + "and spherical tensor encoding. If using this data please cite "
    + "F Szczepankiewicz, S Hoge, C-F Westin. Linear, planar and "
    + "spherical tensor-valued diffusion MRI data by free waveform "
    + "encoding in healthy brain, water, oil and liquid crystals. "
    + "Data in Brief (2019),"
    + "DOI: https://doi.org/10.1016/j.dib.2019.104208",
    md5_list=[
        "11f2e0d53e19061654eb3cdfc8fe9827",
        "15021885b4967437c8cf441c09045c25",
        "1e6b867182da249f81aa9abd50e8b9f7",
        "2ea48d80b6ae1c3da50cb44e615b09e5",
    ],
    data_size="51.1 MB",
)

fetch_DiB_217_lte_pte_ste = _make_fetcher(
    "fetch_DiB_217_lte_pte_ste",
    dipy_home / "DiB_217_lte_pte_ste",
    "https://github.com/filip-szczepankiewicz/Szczepankiewicz_DIB_2019/"
    "raw/master/DATA/brain/NII_Boito_SubSamples/",
    [
        "DiB_217_lte_pte_ste/DiB_217_lte_pte_ste_1.nii.gz",
        "DiB_217_lte_pte_ste/DiB_217_lte_pte_ste_2.nii.gz",
        "DiB_217_lte_pte_ste/bval_DiB_217_lte_pte_ste.bval",
        "DiB_217_lte_pte_ste/bvec_DiB_217_lte_pte_ste.bvec",
        "DiB_mask.nii.gz",
    ],
    [
        Path("DiB_217_lte_pte_ste_1.nii.gz"),
        Path("DiB_217_lte_pte_ste_2.nii.gz"),
        Path("bval_DiB_217_lte_pte_ste.bval"),
        Path("bvec_DiB_217_lte_pte_ste.bvec"),
        Path("DiB_mask.nii.gz"),
    ],
    doc="Download QTE data with linear, planar, "
    + "and spherical tensor encoding. If using this data please cite "
    + "F Szczepankiewicz, S Hoge, C-F Westin. Linear, planar and "
    + "spherical tensor-valued diffusion MRI data by free waveform "
    + "encoding in healthy brain, water, oil and liquid crystals. "
    + "Data in Brief (2019),"
    + "DOI: https://doi.org/10.1016/j.dib.2019.104208",
    md5_list=[
        "424e9cf75b20bc1f7ae1acde26b26da0",
        "8e70d14fb8f08065a7a0c4d3033179c6",
        "1f657215a475676ce333299038df3a39",
        "4220ca92f2906c97ed4c7287eb62c6f0",
        "2ea48d80b6ae1c3da50cb44e615b09e5",
    ],
    data_size="166.3 MB",
)


fetch_ptt_minimal_dataset = _make_fetcher(
    "fetch_ptt_minimal_dataset",
    dipy_home / "ptt_dataset",
    "https://raw.githubusercontent.com/dipy/dipy_datatest/main/",
    ["ptt_fod.nii", "ptt_seed_coords.txt", "ptt_seed_image.nii"],
    [Path("ptt_fod.nii"), Path("ptt_seed_coords.txt"), Path("ptt_seed_image.nii")],
    md5_list=[
        "6e454f8088b64e7b85218c71010d8dbe",
        "8c2d71fb95020e2bb1743623eb11c2a6",
        "9cb88f88d664019ba80c0b372c8bafec",
    ],
    doc="Download FOD and seeds for PTT testing and examples",
    data_size="203KB",
)


fetch_bundle_warp_dataset = _make_fetcher(
    "fetch_bundle_warp_dataset",
    dipy_home / "bundle_warp",
    "https://ndownloader.figshare.com/files/",
    ["40026343", "40026346"],
    [
        Path("m_UF_L.trk"),
        Path("s_UF_L.trk"),
    ],
    md5_list=["4db38ca1e80c16d6e3a97f88f0611187", "c1499005baccfab865ce38368d7a4c7f"],
    doc="Download Bundle Warp dataset",
)


fetch_disco1_dataset = _make_fetcher(
    "fetch_disco1_dataset",
    dipy_home / "disco" / "disco_1",
    "https://data.mendeley.com/public-files/datasets/fgf86jdfg6/files/",
    [
        "028147aa-f17f-4514-80e6-24c7419da75e/file_downloaded",
        "e714ff30-be78-4284-b282-84ed011885de/file_downloaded",
        "a45ac785-4977-434b-b05e-c97ed4a1e6c7/file_downloaded",
        "c3d1de3e-6dcf-4142-99bf-4dd42d445a0a/file_downloaded",
        "35f025f6-d4ab-45e6-9f58-e7685862117f/file_downloaded",
        "02ad744f-4910-40a2-870b-5507d6195926/file_downloaded",
        "c030a2dc-6b80-43d7-a63c-87a28c77c73b/file_downloaded",
        "c6a54096-d7ea-4d05-a921-6aeb0c1bc625/file_downloaded",
        "d95bfd9a-8886-48ec-acc5-5526f3bca32c/file_downloaded",
        "8ef464f7-e648-4583-a208-3ee87a08a2a1/file_downloaded",
        "cbaa3708-7f39-46ec-8cea-51285956bf5a/file_downloaded",
        "964040aa-e1f0-48f7-8eb7-87553f4190e5/file_downloaded",
        "e8bea304-35ce-45de-95be-c51e2ed917f6/file_downloaded",
        "ec5da4f1-53bd-4103-8722-ea3211223150/file_downloaded",
        "c1ddba00-9ea1-435a-946a-bca43079eadf/file_downloaded",
        "0aa2d1a0-ae07-4eff-af6a-22183249e9fa/file_downloaded",
        "744d468f-bd0e-44a8-8395-893889c16462/file_downloaded",
        "311afc50-9bf5-469a-a298-091d252940c9/file_downloaded",
        "68f37ddf-8f06-4d3e-80c5-842b495516ba/file_downloaded",
        "d272c9aa-85d9-40d1-9a87-1044d361da0c/file_downloaded",
        "0ba92dbc-4489-4626-8bfd-5d56b7557699/file_downloaded",
        "099c3f98-4574-4315-be5e-a2b183695e71/file_downloaded",
        "231bbeac-bdec-4815-9238-9a2370bdca6b/file_downloaded",
        "cd073c2f-c3b0-4a46-9c3d-63a853a8af49/file_downloaded",
        "470f235c-04f5-4544-bf4b-f11ac0d8fbd2/file_downloaded",
        "276b637b-6244-4927-b1af-8c1ca12b8a0c/file_downloaded",
        "1c524b3f-ca6b-4f6f-9e6f-25431496000a/file_downloaded",
        "24f8fe7c-8ab2-40c8-baec-4d1f29eba4f2/file_downloaded",
        "5cc5fb58-6058-40dd-9969-019f7814a4a1/file_downloaded",
        "aabcc286-b607-46e1-9229-80638630d518/file_downloaded",
        "87b6cee7-0645-4405-b5b3-af89c04a2ac9/file_downloaded",
        "ad47bf18-8b69-4b0c-8706-ada53365fd99/file_downloaded",
        "417505fb-3a7a-4cbc-ac8a-92acc26bb958/file_downloaded",
        "7708e776-f07c-47a9-9982-b5fbd9dd4390/file_downloaded",
        "654863c6-6154-4da6-9f4c-3b0c76535dd7/file_downloaded",
        "5f8d028c-82a1-4837-b34f-ad8251c0685b/file_downloaded",
        "bd2fa6f3-f050-44a6-ad18-6c31bb037eb7/file_downloaded",
        "b432035b-5429-4255-8e21-88184802d005/file_downloaded",
        "1f1a4391-9740-4e17-89c3-79430aab0f91/file_downloaded",
        "6c90c685-ae16-4f21-ae13-71e89ca9eb66/file_downloaded",
        "f8878550-061b-40e0-a0c7-5aac5a33e542/file_downloaded",
    ],
    [
        Path("DiSCo_gradients.bvals"),
        Path("DiSCo_gradients_dipy.bvecs"),
        Path("DiSCo_gradients_fsl.bvecs"),
        Path("DiSCo_gradients_mrtrix.b"),
        Path("DiSCo_gradients.scheme"),
        Path("lowRes_DiSCo1_DWI_RicianNoise-snr40.nii.gz"),
        Path("lowRes_DiSCo1_ROIs.nii.gz"),
        Path("lowRes_DiSCo1_mask.nii.gz"),
        Path("lowRes_DiSCo1_DWI_RicianNoise-snr50.nii.gz"),
        Path("lowRes_DiSCo1_ROIs-mask.nii.gz"),
        Path("lowRes_DiSCo1_DWI_Intra.nii.gz"),
        Path("lowRes_DiSCo1_Strand_Bundle_Count.nii.gz"),
        Path("lowRes_DiSCo1_Strand_Count.nii.gz"),
        Path("lowRes_DiSCo1_Strand_Intra_Volume_Fraction.nii.gz"),
        Path("lowRes_DiSCo1_DWI_RicianNoise-snr30.nii.gz"),
        Path("lowRes_DiSCo1_DWI_RicianNoise-snr20.nii.gz"),
        Path("lowRes_DiSCo1_DWI.nii.gz"),
        Path("lowRes_DiSCo1_Strand_ODFs.nii.gz"),
        Path("lowRes_DiSCo1_Strand_Average_Diameter.nii.gz"),
        Path("lowRes_DiSCo1_DWI_RicianNoise-snr10.nii.gz"),
        Path("highRes_DiSCo1_Strand_ODFs.nii.gz"),
        Path("highRes_DiSCo1_DWI_RicianNoise-snr50.nii.gz"),
        Path("highRes_DiSCo1_DWI.nii.gz"),
        Path("highRes_DiSCo1_ROIs.nii.gz"),
        Path("highRes_DiSCo1_DWI_RicianNoise-snr40.nii.gz"),
        Path("highRes_DiSCo1_mask.nii.gz"),
        Path("highRes_DiSCo1_Strand_Bundle_Count.nii.gz"),
        Path("highRes_DiSCo1_DWI_RicianNoise-snr20.nii.gz"),
        Path("highRes_DiSCo1_DWI_RicianNoise-snr30.nii.gz"),
        Path("highRes_DiSCo1_Strand_Average_Diameter.nii.gz"),
        Path("highRes_DiSCo1_Strand_Streamline_Count.nii.gz"),
        Path("highRes_DiSCo1_DWI_Intra.nii.gz"),
        Path("highRes_DiSCo1_DWI_RicianNoise-snr10.nii.gz"),
        Path("highRes_DiSCo1_Strand_Intra_Volume_Fraction.nii.gz"),
        Path("highRes_DiSCo1_ROIs-mask.nii.gz"),
        Path("DiSCo1_Connectivity_Matrix_Strands_Count.txt"),
        Path("DiSCo1_Connectivity_Matrix_Cross-Sectional_Area.txt"),
        Path("DiSCo1_Strands_ROIs_Pairs.txt"),
        Path("DiSCo1_Strands_Diameters.txt"),
        Path("DiSCo1_Strands_Trajectories.tck"),
        Path("DiSCo1_Strands_Trajectories.trk"),
    ],
    md5_list=[
        "c03cfec8ee605a54866ef09c3c8ba31d",
        "b8443aee0a2d2b60ee658a2342cccf0f",
        "4ac749bb584e6963851ee170e137e3ec",
        "207d57a6c16cdf230b45d4df5b4e1dee",
        "2346de6ad563a10e40b8ebec3ac4e0ff",
        "1981ac168ea76ce4d70e3dcd1d749f13",
        "9a06a057df5b2ce5cb05e8c9f59be14f",
        "ac9900538f4563c39032dcc289c36a51",
        "beb0f999bc26cd32ba2079db77e7840a",
        "8d5659e78d7d028e3da8f5af89402669",
        "db335f7d52b85c1d0de7ca61a2fca64b",
        "f1c49868436e0beca3b7cc50416c664e",
        "b967371fe44c4d736edcd65b8da12b93",
        "d4da2df8cd7cbf24cdd0af09247ff0a2",
        "80316e4e687ee3c2e2a9221929d1e826",
        "4c3580f512b325552302792a64da87f0",
        "56ee11c59348f3a61079483bd249032d",
        "580d72144a5e19a119e6839d8f1c623d",
        "f7bdfea9c3163cac7d4f67f49928b682",
        "7cf61f1088971a8799df80c29dc7a9f1",
        "da762d46e6b7d513be0411ac66769c5c",
        "5d7bac8737a4543eee60f7aa26073df9",
        "370bdf6571aa061daaa654a107eae6fd",
        "50fd6e201536b4a4815dd102d9c80974",
        "8269aed23fc643bb98c36945462e866a",
        "86becb33b54a3c4291f0cd77ec87bb3e",
        "d240cccd9d84e69bf146a5d06bb898bb",
        "e75a1838ff8e5860bdbde1162e5aa80f",
        "f7d9507884960f86918f6328962d98d2",
        "be4f8221ccdd5a139f16ac18353d12c6",
        "d8d33dbdc820289f5575b5c064c6dfe9",
        "53740e3d2851d5f301d93730268928f2",
        "ed98e0f26abc85af5735fe228cb86b4c",
        "50a9385f567138c23df4ddf333c785c0",
        "2d845131db87b6a2230a325e076bdd7f",
        "c53f6e42cfd796a561b1af434a61c91d",
        "f97730469c53deaa5b827effebef3e78",
        "d34fd49147cd061b9417328118567d1c",
        "f0d28f1d7eec1fad866607ee59686476",
        "e475641a08ebafeecb79a21e618d7081",
        "8c2a338606c0cb7de6e34b8317f59cd6",
    ],
    optional_fnames=[
        "DiSCo_gradients_fsl.bvecs",
        "DiSCo_gradients_mrtrix.b",
        "DiSCo_gradients.scheme",
        "highRes_DiSCo1_DWI.nii.gz",
        "lowRes_DiSCo1_DWI_RicianNoise-snr40.nii.gz",
        "lowRes_DiSCo1_ROIs.nii.gz",
        "lowRes_DiSCo1_mask.nii.gz",
        "lowRes_DiSCo1_DWI_RicianNoise-snr50.nii.gz",
        "lowRes_DiSCo1_ROIs-mask.nii.gz",
        "lowRes_DiSCo1_DWI_Intra.nii.gz",
        "lowRes_DiSCo1_Strand_Bundle_Count.nii.gz",
        "lowRes_DiSCo1_Strand_Count.nii.gz",
        "lowRes_DiSCo1_Strand_Intra_Volume_Fraction.nii.gz",
        "lowRes_DiSCo1_DWI_RicianNoise-snr30.nii.gz",
        "lowRes_DiSCo1_DWI_RicianNoise-snr20.nii.gz",
        "lowRes_DiSCo1_DWI.nii.gz",
        "lowRes_DiSCo1_Strand_ODFs.nii.gz",
        "lowRes_DiSCo1_Strand_Average_Diameter.nii.gz",
        "lowRes_DiSCo1_DWI_RicianNoise-snr10.nii.gz",
        "highRes_DiSCo1_Strand_ODFs.nii.gz",
        "highRes_DiSCo1_DWI_RicianNoise-snr50.nii.gz",
        "highRes_DiSCo1_DWI_RicianNoise-snr40.nii.gz",
        "highRes_DiSCo1_Strand_Bundle_Count.nii.gz",
        "highRes_DiSCo1_DWI_RicianNoise-snr20.nii.gz",
        "highRes_DiSCo1_DWI_RicianNoise-snr30.nii.gz",
        "highRes_DiSCo1_Strand_Average_Diameter.nii.gz",
        "highRes_DiSCo1_Strand_Streamline_Count.nii.gz",
        "highRes_DiSCo1_DWI_Intra.nii.gz",
        "highRes_DiSCo1_Strand_Intra_Volume_Fraction.nii.gz",
        "DiSCo1_Connectivity_Matrix_Strands_Count.txt",
        "DiSCo1_Strands_ROIs_Pairs.txt",
        "DiSCo1_Strands_Diameters.txt",
        "DiSCo1_Strands_Trajectories.trk",
    ],
    doc=(
        "Download DISCO 1 dataset: The Diffusion-Simulated Connectivity "
        "Dataset. DOI: 10.17632/fgf86jdfg6.3"
    ),
    data_size="1.05 GB",
    use_headers=True,
)


fetch_disco2_dataset = _make_fetcher(
    "fetch_disco2_dataset",
    dipy_home / "disco" / "disco_2",
    "https://data.mendeley.com/public-files/datasets/fgf86jdfg6/files/",
    [
        "028147aa-f17f-4514-80e6-24c7419da75e/file_downloaded",
        "e714ff30-be78-4284-b282-84ed011885de/file_downloaded",
        "a45ac785-4977-434b-b05e-c97ed4a1e6c7/file_downloaded",
        "c3d1de3e-6dcf-4142-99bf-4dd42d445a0a/file_downloaded",
        "35f025f6-d4ab-45e6-9f58-e7685862117f/file_downloaded",
        "e7a50803-a58d-47dc-b026-6a6e20602546/file_downloaded",
        "12158d68-0050-4528-88db-5fb606b2151d/file_downloaded",
        "361df9ec-0542-4e05-8f2d-1c605a5a915a/file_downloaded",
        "2bcf46ae-0b07-4dd0-baf8-654a05a4aca7/file_downloaded",
        "6a514a78-8a8a-4cdf-947a-d1b01ced4b0d/file_downloaded",
        "a387810f-54fe-4493-8842-f3fbeea687fe/file_downloaded",
        "2401238c-ae37-4a99-81a8-50ba90110650/file_downloaded",
        "980a6702-c192-4a4e-b706-f3ec2fb5e800/file_downloaded",
        "c5691dae-5b10-4c67-8a58-2df8314c85ec/file_downloaded",
        "d4eabb32-8b28-461c-8248-a0bacf4e44f2/file_downloaded",
        "dd98fb48-c48f-4d72-85a9-38a2d8b2bb3f/file_downloaded",
        "fed31d0d-7c75-4073-82e5-2c9ba30daa5a/file_downloaded",
        "012853cb-0d7f-4135-8407-a99c4e022024/file_downloaded",
        "46d0466f-75bd-4646-af10-5215477339d8/file_downloaded",
        "d1adcbdd-1d11-43cd-a72a-6830e91b7c33/file_downloaded",
        "f3e9d4a5-dc9d-4112-8a67-b63b40c87e5c/file_downloaded",
        "825d2154-e892-474d-ae31-e490467dfa3c/file_downloaded",
        "bf62b3fd-fff7-4658-b143-9ec948fc256f/file_downloaded",
        "173dacc8-683c-42c0-b270-67f59c77d49d/file_downloaded",
        "f981cbad-84c9-42bb-bf19-978f12e6c972/file_downloaded",
        "d122bf02-98f7-46b2-9286-98473216d64e/file_downloaded",
        "50b68dd1-1037-4cb4-8d1a-025cff38da55/file_downloaded",
        "5f3e04b6-9bd5-4010-bc2f-ea16b7ac8b1c/file_downloaded",
        "f8d7ac40-7d0e-41ba-9788-4f973be5968b/file_downloaded",
        "d4adf4a7-1646-4530-85cb-66c2779e1cb6/file_downloaded",
        "6b1f740e-2b20-440f-9bec-344540d80d1d/file_downloaded",
        "7ea07af5-5cb8-4857-bf30-17a0d4f39f83/file_downloaded",
        "7ad1c5ab-9606-481c-a08e-0770798b224f/file_downloaded",
        "8d8c8fe9-7747-4ba2-b759-f27cf160b88f/file_downloaded",
        "143de683-9bce-43b9-b08e-2e16fea05a53/file_downloaded",
        "f3f5f798-0d01-41de-ad96-6a2e2d04eeb1/file_downloaded",
        "0dc1a52f-7770-4492-8dc7-27da1f9d2ec9/file_downloaded",
        "bba6f22f-1212-49c2-b07f-3b62ae435b4d/file_downloaded",
        "5928d70c-01f3-47ab-8f29-0484f9a014a2/file_downloaded",
        "6be306cb-1dfd-4fe8-912d-ec117ff0d70d/file_downloaded",
        "e49d38af-e262-46ef-ad44-c51c29ee2178/file_downloaded",
    ],
    [
        Path("DiSCo_gradients.bvals"),
        Path("DiSCo_gradients_dipy.bvecs"),
        Path("DiSCo_gradients_fsl.bvecs"),
        Path("DiSCo_gradients_mrtrix.b"),
        Path("DiSCo_gradients.scheme"),
        Path("lowRes_DiSCo2_DWI_RicianNoise-snr50.nii.gz"),
        Path("lowRes_DiSCo2_Strand_Average_Diameter.nii.gz"),
        Path("lowRes_DiSCo2_DWI_RicianNoise-snr40.nii.gz"),
        Path("lowRes_DiSCo2_DWI.nii.gz"),
        Path("lowRes_DiSCo2_ROIs.nii.gz"),
        Path("lowRes_DiSCo2_mask.nii.gz"),
        Path("lowRes_DiSCo2_DWI_Intra.nii.gz"),
        Path("lowRes_DiSCo2_ROIs-mask.nii.gz"),
        Path("lowRes_DiSCo2_Strand_Count.nii.gz"),
        Path("lowRes_DiSCo2_DWI_RicianNoise-snr20.nii.gz"),
        Path("lowRes_DiSCo2_DWI_RicianNoise-snr30.nii.gz"),
        Path("lowRes_DiSCo2_Strand_Intra_Volume_Fraction.nii.gz"),
        Path("lowRes_DiSCo2_DWI_RicianNoise-snr10.nii.gz"),
        Path("lowRes_DiSCo2_Strand_ODFs.nii.gz"),
        Path("lowRes_DiSCo2_Strand_Bundle_Count.nii.gz"),
        Path("highRes_DiSCo2_DWI_RicianNoise-snr40.nii.gz"),
        Path("highRes_DiSCo2_DWI_RicianNoise-snr50.nii.gz"),
        Path("highRes_DiSCo2_ROIs-mask.nii.gz"),
        Path("highRes_DiSCo2_Strand_ODFs.nii.gz"),
        Path("highRes_DiSCo2_Strand_Count.nii.gz"),
        Path("highRes_DiSCo2_ROIs.nii.gz"),
        Path("highRes_DiSCo2_mask.nii.gz"),
        Path("highRes_DiSCo2_Strand_Average_Diameter.nii.gz"),
        Path("highRes_DiSCo2_DWI_RicianNoise-snr30.nii.gz"),
        Path("highRes_DiSCo2_DWI_Intra.nii.gz"),
        Path("highRes_DiSCo2_DWI_RicianNoise-snr20.nii.gz"),
        Path("highRes_DiSCo2_DWI.nii.gz"),
        Path("highRes_DiSCo2_Strand_Bundle_Count.nii.gz"),
        Path("highRes_DiSCo2_DWI_RicianNoise-snr10.nii.gz"),
        Path("highRes_DiSCo2_Strand_Intra_Volume_Fraction.nii.gz"),
        Path("DiSCo2_Strands_Diameters.txt"),
        Path("DiSCo2_Connectivity_Matrix_Strands_Count.txt"),
        Path("DiSCo2_Strands_Trajectories.trk"),
        Path("DiSCo2_Strands_ROIs_Pairs.txt"),
        Path("DiSCo2_Strands_Trajectories.tck"),
        Path("DiSCo2_Connectivity_Matrix_Cross-Sectional_Area.txt"),
    ],
    md5_list=[
        "c03cfec8ee605a54866ef09c3c8ba31d",
        "b8443aee0a2d2b60ee658a2342cccf0f",
        "4ac749bb584e6963851ee170e137e3ec",
        "207d57a6c16cdf230b45d4df5b4e1dee",
        "2346de6ad563a10e40b8ebec3ac4e0ff",
        "425a97bce000265d9e4868fcb54a6bd7",
        "7cb8fd4ae58973235903adfcf338cb87",
        "bb5ba3ba1163f4335dae76f905e181d0",
        "350fb02f7ceeaf95c5cd13a4721cef5f",
        "a8476809cca854668ae4dd3bfd1366ec",
        "e55bca1aef80039eafdadec4b8c3ff21",
        "5d9b46a5130a7734a2a0544ed1f5cc72",
        "90869179977e2b78bd0a8c4f016ee7e3",
        "c6207b65665f5b82a5413e65627156b3",
        "1f7e32d2780bbc1a2b5f7b592c9fd53b",
        "36de56a359a67ec68c15cc484d79e1da",
        "6ddfa6a7e5083cb71f3ae4d23711a6a8",
        "21f0cf235c8c1705e7bd9ebd9a39479a",
        "660f549e8180f17ea69de82661b427e9",
        "19c037c7813c7574b809baedd9bb64fd",
        "77a1f4111eefc01b26d7ec8b6473894d",
        "cde18c2581ced8755958a153332c375a",
        "88b12d4c24bd833430b3125340c96801",
        "4d70e9b246101f5f2ba8a44d92beb2e8",
        "c01bb0a1d2bdb95565b92eb022ff2124",
        "31f664676e94280c5ebdf04b544d200f",
        "41a3155f6fb44da50c515bb867006ab1",
        "9494f6a6f5c4e44b88051f0284757517",
        "5d9f8bd4a540891082f2dda976f70478",
        "8e2e5900008ae6923fe56447d7b493c7",
        "e961d27613acc250b6407d29d5020154",
        "ef330fa395d4e48d0286652927116275",
        "00c620b144253ef114fee1fe8e95a87f",
        "7e4cf83496996aefd9f27c6077c71d74",
        "e6be6c73425337e331fda4e4bdcbe749",
        "0b6c1770d48f39f704986ba69970541e",
        "69da5bb232b41dbeeea9d409ca94955f",
        "0b43ffcc46217da86a472a7b1e02ac06",
        "4e774e280cc57b1336e4fa9e479ee4ee",
        "cc5442ea2ff2ddfcd1330016e2a68399",
        "7f276c0276561df13c86008776faf6b2",
    ],
    optional_fnames=[
        "DiSCo_gradients_fsl.bvecs",
        "DiSCo_gradients_mrtrix.b",
        "DiSCo_gradients.scheme",
        "lowRes_DiSCo2_DWI_RicianNoise-snr50.nii.gz",
        "lowRes_DiSCo2_Strand_Average_Diameter.nii.gz",
        "lowRes_DiSCo2_DWI_RicianNoise-snr40.nii.gz",
        "lowRes_DiSCo2_DWI.nii.gz",
        "lowRes_DiSCo2_ROIs.nii.gz",
        "lowRes_DiSCo2_mask.nii.gz",
        "lowRes_DiSCo2_DWI_Intra.nii.gz",
        "lowRes_DiSCo2_ROIs-mask.nii.gz",
        "lowRes_DiSCo2_Strand_Count.nii.gz",
        "lowRes_DiSCo2_DWI_RicianNoise-snr20.nii.gz",
        "lowRes_DiSCo2_DWI_RicianNoise-snr30.nii.gz",
        "lowRes_DiSCo2_Strand_Intra_Volume_Fraction.nii.gz",
        "lowRes_DiSCo2_DWI_RicianNoise-snr10.nii.gz",
        "lowRes_DiSCo2_Strand_ODFs.nii.gz",
        "lowRes_DiSCo2_Strand_Bundle_Count.nii.gz",
        "highRes_DiSCo2_DWI_RicianNoise-snr40.nii.gz",
        "highRes_DiSCo2_DWI_RicianNoise-snr50.nii.gz",
        "highRes_DiSCo2_Strand_ODFs.nii.gz",
        "highRes_DiSCo2_Strand_Count.nii.gz",
        "highRes_DiSCo2_Strand_Average_Diameter.nii.gz",
        "highRes_DiSCo2_DWI_RicianNoise-snr30.nii.gz",
        "highRes_DiSCo2_DWI_Intra.nii.gz",
        "highRes_DiSCo2_DWI_RicianNoise-snr20.nii.gz",
        "highRes_DiSCo2_Strand_Bundle_Count.nii.gz",
        "highRes_DiSCo2_Strand_Intra_Volume_Fraction.nii.gz",
        "highRes_DiSCo2_DWI.nii.gz",
        "DiSCo2_Strands_Diameters.txt",
        "DiSCo2_Connectivity_Matrix_Strands_Count.txt",
        "DiSCo2_Strands_ROIs_Pairs.txt",
        "DiSCo2_Strands_Trajectories.trk",
    ],
    doc=(
        "Download DISCO 2 dataset: The Diffusion-Simulated Connectivity "
        "Dataset. DOI: 10.17632/fgf86jdfg6.3"
    ),
    data_size="1.05 GB",
    use_headers=True,
)


fetch_disco3_dataset = _make_fetcher(
    "fetch_disco3_dataset",
    dipy_home / "disco" / "disco_3",
    "https://data.mendeley.com/public-files/datasets/fgf86jdfg6/files/",
    [
        "028147aa-f17f-4514-80e6-24c7419da75e/file_downloaded",
        "e714ff30-be78-4284-b282-84ed011885de/file_downloaded",
        "a45ac785-4977-434b-b05e-c97ed4a1e6c7/file_downloaded",
        "c3d1de3e-6dcf-4142-99bf-4dd42d445a0a/file_downloaded",
        "35f025f6-d4ab-45e6-9f58-e7685862117f/file_downloaded",
        "3cc65986-2e64-4397-a34d-386dfe286a89/file_downloaded",
        "ac384df6-8e9a-4c44-89db-624f89c12015/file_downloaded",
        "7feb06e6-0bdc-4fba-b7df-c67c3cb43301/file_downloaded",
        "c8bddc43-f1f2-404c-98d4-9ab34380fc8f/file_downloaded",
        "00bdcd82-dde0-499d-9bef-581755ad7e57/file_downloaded",
        "8abbc092-9368-4154-9e20-b28e3b6d03b4/file_downloaded",
        "e0773f2f-7fbe-4139-8e71-2d3267360ab5/file_downloaded",
        "8670b9c4-9275-4da6-adbf-398889a606d7/file_downloaded",
        "ec2306f2-7d60-452a-82d8-66962ffb903d/file_downloaded",
        "00cf30d9-14fa-484b-b6e1-3d484e43a840/file_downloaded",
        "b61b83aa-522b-4239-9a7b-74c80aea3c07/file_downloaded",
        "75176143-5ddf-4dfa-b2cd-b87da0abfc7b/file_downloaded",
        "d5fe5abf-b72e-474a-9316-714e0739feec/file_downloaded",
        "45a139fc-2d99-4832-8d95-7a7df2acf711/file_downloaded",
        "68f30089-1bf9-45d9-82d8-d77855823229/file_downloaded",
        "4162d753-0d7c-4492-aa4f-88554ee64fed/file_downloaded",
        "5ca6e137-2e65-41cd-bb4b-e1d21dcb5a28/file_downloaded",
        "8e88dccf-d7b3-4692-ab0f-26810ba168e0/file_downloaded",
        "fd97249f-ab0b-4ffa-a3ba-c3d0b3b1c947/file_downloaded",
        "73636f0f-7fc6-48c9-a140-7a11df9847f2/file_downloaded",
        "186a2116-21a8-42fc-b6d1-f303b0ba0b69/file_downloaded",
        "e087b436-d1be-4c8d-8c56-3de9ef7fb054/file_downloaded",
        "305ae78b-725c-4bb1-b435-1bca70c18b6b/file_downloaded",
        "2e62eb11-660b-4f09-920d-f5ee1e25b02a/file_downloaded",
        "a35e33f5-f23a-48de-9c03-1560ce5fecbe/file_downloaded",
        "f1737350-0b46-468e-b78a-af9731eac720/file_downloaded",
        "86ba2b2c-2094-43ef-be3e-e286dca5c5bd/file_downloaded",
        "c5a4a685-c3d6-4ef2-9ec9-956fce821c9b/file_downloaded",
        "d84ec650-8c05-41a6-a4ef-a4e3a2f66dd3/file_downloaded",
        "7648b6ce-2bc8-4753-8e8c-399978d4e187/file_downloaded",
        "1b702c78-cbae-4ceb-a5f6-89c5cb0f4f88/file_downloaded",
        "835af619-7447-4247-8e5b-2e2895480add/file_downloaded",
        "99752985-840e-4ed2-b068-68f6be7af48f/file_downloaded",
        "c07c7b51-1eab-4852-8b36-bd9901f15d7a/file_downloaded",
        "af6f3bc0-a09a-461a-8e6b-6095d83110ea/file_downloaded",
        "482bcf8f-52eb-4bf8-83b4-765ed935ad81/file_downloaded",
    ],
    [
        Path("DiSCo_gradients.bvals"),
        Path("DiSCo_gradients_dipy.bvecs"),
        Path("DiSCo_gradients_fsl.bvecs"),
        Path("DiSCo_gradients_mrtrix.b"),
        Path("DiSCo_gradients.scheme"),
        Path("lowRes_DiSCo3_Strand_ODFs.nii.gz"),
        Path("lowRes_DiSCo3_DWI_RicianNoise-snr10.nii.gz"),
        Path("lowRes_DiSCo3_Strand_Intra_Volume_Fraction.nii.gz"),
        Path("lowRes_DiSCo3_DWI_RicianNoise-snr30.nii.gz"),
        Path("lowRes_DiSCo3_DWI_RicianNoise-snr20.nii.gz"),
        Path("lowRes_DiSCo3_Strand_Average_Diameter.nii.gz"),
        Path("lowRes_DiSCo3_DWI_Intra.nii.gz"),
        Path("lowRes_DiSCo3_Strand_Count.nii.gz"),
        Path("lowRes_DiSCo3_DWI.nii.gz"),
        Path("lowRes_DiSCo3_Strand_Bundle_Count.nii.gz"),
        Path("lowRes_DiSCo3_ROIs-mask.nii.gz"),
        Path("lowRes_DiSCo3_DWI_RicianNoise-snr40.nii.gz"),
        Path("lowRes_DiSCo3_mask.nii.gz"),
        Path("lowRes_DiSCo3_DWI_RicianNoise-snr50.nii.gz"),
        Path("lowRes_DiSCo3_ROIs.nii.gz"),
        Path("highRes_DiSCo3_DWI_RicianNoise-snr10.nii.gz"),
        Path("highRes_DiSCo3_Strand_Average_Diameter.nii.gz"),
        Path("highRes_DiSCo3_Strand_Count.nii.gz"),
        Path("highRes_DiSCo3_DWI_RicianNoise-snr20.nii.gz"),
        Path("highRes_DiSCo3_DWI.nii.gz"),
        Path("highRes_DiSCo3_DWI_RicianNoise-snr30.nii.gz"),
        Path("highRes_DiSCo3_ROIs-mask.nii.gz"),
        Path("highRes_DiSCo3_Strand_Intra_Volume_Fraction.nii.gz"),
        Path("highRes_DiSCo3_Strand_Bundle_Count.nii.gz"),
        Path("highRes_DiSCo3_DWI_Intra.nii.gz"),
        Path("highRes_DiSCo3_DWI_RicianNoise-snr50.nii.gz"),
        Path("highRes_DiSCo3_Strand_ODFs.nii.gz"),
        Path("highRes_DiSCo3_mask.nii.gz"),
        Path("highRes_DiSCo3_ROIs.nii.gz"),
        Path("highRes_DiSCo3_DWI_RicianNoise-snr40.nii.gz"),
        Path("DiSCo3_Connectivity_Matrix_Cross-Sectional_Area.txt"),
        Path("DiSCo3_Strands_Trajectories.tck"),
        Path("DiSCo3_Strands_Trajectories.trk"),
        Path("DiSCo3_Strands_ROIs_Pairs.txt"),
        Path("DiSCo3_Connectivity_Matrix_Strands_Count.txt"),
        Path("DiSCo3_Strands_Diameters.txt"),
    ],
    md5_list=[
        "c03cfec8ee605a54866ef09c3c8ba31d",
        "b8443aee0a2d2b60ee658a2342cccf0f",
        "4ac749bb584e6963851ee170e137e3ec",
        "207d57a6c16cdf230b45d4df5b4e1dee",
        "2346de6ad563a10e40b8ebec3ac4e0ff",
        "acad92cebd780424280b5e38a5b41445",
        "c5ce15ea50d02621d2a3f27cde7cd208",
        "c827954b40279462b520edfb8cd2ced0",
        "f4e8fca1a434436ffe8fc4392d2e635b",
        "62606109e8a96f41208b45bffbe94aae",
        "a03bc8afe4d0348841c9fed5f03695b4",
        "85f60f649c6f3d4e6ae6bcf88053058d",
        "7fdec9c4f4fb740f595191a4d0202de7",
        "df04eb8f1ce0138f3c7f20488f4dbc00",
        "0bde3cefd74e4c3b4d28ec0d90bcadc9",
        "aa7bfbfc9ff454e7a466a01229b1377b",
        "218561e3ebbee2f08d4f3040ec2f12a9",
        "577ae9e407bb225dae48482a7f88bafb",
        "e85e6edf80bd07030f5a6c1769c84546",
        "49ed07d5fc60ee17a4a4019d7bea7d97",
        "aa86d752bec702172a54c1e8efe51d78",
        "df0dde6428f5c5146b5ce8f353865bca",
        "0f0b68b956415bbe06766e497ee77388",
        "44adc18b39d03b8d83f48cc572aaab78",
        "0460e6420739ae31b7076f5e1be5a1f1",
        "80c07379496d0f2aab66d8041a9c53b1",
        "0e3763e51612b56fa8dc933efc8d1971",
        "676f88a792d9034015a9613001c9bf02",
        "6f399bee3158613c74c1330cee5d9b0d",
        "534e64906cc2bf58ff4cf98ecd301dd5",
        "7d0f7513c85d766c45581f9fa8ab34a5",
        "01b4fe6c6e6459eac293a62428680e3b",
        "2bb5a0b24139f5448cdd51167ce33906",
        "bf6c5a26cb1061be91bec7fa73153cec",
        "e8911b580e7fc7e2d94d238f2c81c43b",
        "647714842c8dea4d192bdd7c2b41fd3b",
        "bba1e4283ed853eb484497db4b6da460",
        "d761384f8fed31c53197ce4916f32296",
        "3b119c1b8487da71412f3275616003dc",
        "47c300fcb48b882c124d69086b224956",
        "9e47185ae517f6f35daa1aa16c51ba45",
    ],
    optional_fnames=[
        "DiSCo_gradients_fsl.bvecs",
        "DiSCo_gradients_mrtrix.b",
        "DiSCo_gradients.scheme",
        "lowRes_DiSCo3_Strand_ODFs.nii.gz",
        "lowRes_DiSCo3_DWI_RicianNoise-snr10.nii.gz",
        "lowRes_DiSCo3_Strand_Intra_Volume_Fraction.nii.gz",
        "lowRes_DiSCo3_DWI_RicianNoise-snr30.nii.gz",
        "lowRes_DiSCo3_DWI_RicianNoise-snr20.nii.gz",
        "lowRes_DiSCo3_Strand_Average_Diameter.nii.gz",
        "lowRes_DiSCo3_DWI_Intra.nii.gz",
        "lowRes_DiSCo3_Strand_Count.nii.gz",
        "lowRes_DiSCo3_DWI.nii.gz",
        "lowRes_DiSCo3_Strand_Bundle_Count.nii.gz",
        "lowRes_DiSCo3_ROIs-mask.nii.gz",
        "lowRes_DiSCo3_DWI_RicianNoise-snr40.nii.gz",
        "lowRes_DiSCo3_mask.nii.gz",
        "lowRes_DiSCo3_DWI_RicianNoise-snr50.nii.gz",
        "lowRes_DiSCo3_ROIs.nii.gz",
        "highRes_DiSCo3_Strand_Average_Diameter.nii.gz",
        "highRes_DiSCo3_Strand_Count.nii.gz",
        "highRes_DiSCo3_DWI_RicianNoise-snr20.nii.gz",
        "highRes_DiSCo3_DWI_RicianNoise-snr30.nii.gz",
        "highRes_DiSCo3_Strand_Intra_Volume_Fraction.nii.gz",
        "highRes_DiSCo3_Strand_Bundle_Count.nii.gz",
        "highRes_DiSCo3_DWI_Intra.nii.gz",
        "highRes_DiSCo3_DWI_RicianNoise-snr50.nii.gz",
        "highRes_DiSCo3_Strand_ODFs.nii.gz",
        "highRes_DiSCo3_DWI_RicianNoise-snr40.nii.gz",
        "highRes_DiSCo3_DWI.nii.gz",
        "DiSCo3_Strands_ROIs_Pairs.txt",
        "DiSCo3_Connectivity_Matrix_Strands_Count.txt",
        "DiSCo3_Strands_Diameters.txt",
        "DiSCo3_Strands_Trajectories.trk",
    ],
    doc=(
        "Download DISCO 3 dataset: The Diffusion-Simulated Connectivity "
        "Dataset. DOI: 10.17632/fgf86jdfg6.3"
    ),
    data_size="1.05 GB",
    use_headers=True,
)


def fetch_disco_dataset(*, include_optional=False):
    """Download All DISCO datasets.

    Notes
    -----
    see DOI: 10.17632/fgf86jdfg6.3

    """
    files_1, folder_1 = fetch_disco1_dataset(include_optional=include_optional)
    files_2, folder_2 = fetch_disco2_dataset(include_optional=include_optional)
    files_3, folder_3 = fetch_disco3_dataset(include_optional=include_optional)

    all_path = (
        [Path(folder_1) / f for f in files_1]
        + [Path(folder_2) / f for f in files_2]
        + [Path(folder_3) / f for f in files_3]
    )

    return all_path


@warning_for_keywords()
def get_fnames(*, name="small_64D", include_optional=False):
    """Provide full paths to example or test datasets.

    Parameters
    ----------
    name : str
        the filename/s of which dataset to return, one of:

        - 'small_64D' small region of interest nifti,bvecs,bvals 64 directions
        - 'small_101D' small region of interest nifti, bvecs, bvals
          101 directions
        - 'aniso_vox' volume with anisotropic voxel size as Nifti
        - 'fornix' 300 tracks in Trackvis format (from Pittsburgh
          Brain Competition)
        - 'gqi_vectors' the scanner wave vectors needed for a GQI acquisitions
          of 101 directions tested on Siemens 3T Trio
        - 'small_25' small ROI (10x8x2) DTI data (b value 2000, 25 directions)
        - 'test_piesno' slice of N=8, K=14 diffusion data
        - 'reg_c' small 2D image used for validating registration
        - 'reg_o' small 2D image used for validation registration
        - 'cb_2' two vectorized cingulum bundles
    include_optional : bool, optional
        If True, include optional datasets.

    Returns
    -------
    fnames : tuple
        filenames for dataset

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.io.image import load_nifti
    >>> from dipy.data import get_fnames
    >>> fimg, fbvals, fbvecs = get_fnames(name='small_101D')
    >>> bvals=np.loadtxt(fbvals)
    >>> bvecs=np.loadtxt(fbvecs).T
    >>> data, affine = load_nifti(fimg)
    >>> data.shape == (6, 10, 10, 102)
    True
    >>> bvals.shape == (102,)
    True
    >>> bvecs.shape == (102, 3)
    True

    """
    DATA_DIR = Path(__file__).parent / "files"
    if name == "small_64D":
        fbvals = Path(DATA_DIR) / "small_64D.bval"
        fbvecs = Path(DATA_DIR) / "small_64D.bvec"
        fimg = Path(DATA_DIR) / "small_64D.nii"
        return fimg, fbvals, fbvecs
    if name == "55dir_grad":
        fbvals = Path(DATA_DIR) / "55dir_grad.bval"
        fbvecs = Path(DATA_DIR) / "55dir_grad.bvec"
        return fbvals, fbvecs
    if name == "small_101D":
        fbvals = Path(DATA_DIR) / "small_101D.bval"
        fbvecs = Path(DATA_DIR) / "small_101D.bvec"
        fimg = Path(DATA_DIR) / "small_101D.nii.gz"
        return fimg, fbvals, fbvecs
    if name == "aniso_vox":
        return Path(DATA_DIR) / "aniso_vox.nii.gz"
    if name == "ascm_test":
        return Path(DATA_DIR) / "ascm_out_test.nii.gz"
    if name == "fornix":
        return Path(DATA_DIR) / "tracks300.trk"
    if name == "gqi_vectors":
        return Path(DATA_DIR) / "ScannerVectors_GQI101.txt"
    if name == "dsi515btable":
        return Path(DATA_DIR) / "dsi515_b_table.txt"
    if name == "dsi4169btable":
        return Path(DATA_DIR) / "dsi4169_b_table.txt"
    if name == "grad514":
        return Path(DATA_DIR) / "grad_514.txt"
    if name == "small_25":
        fbvals = Path(DATA_DIR) / "small_25.bval"
        fbvecs = Path(DATA_DIR) / "small_25.bvec"
        fimg = Path(DATA_DIR) / "small_25.nii.gz"
        return fimg, fbvals, fbvecs
    if name == "small_25_streamlines":
        fstreamlines = Path(DATA_DIR) / "EuDX_small_25.trk"
        return fstreamlines
    if name == "S0_10":
        fimg = Path(DATA_DIR) / "S0_10slices.nii.gz"
        return fimg
    if name == "test_piesno":
        fimg = Path(DATA_DIR) / "test_piesno.nii.gz"
        return fimg
    if name == "reg_c":
        return Path(DATA_DIR) / "C.npy"
    if name == "reg_o":
        return Path(DATA_DIR) / "circle.npy"
    if name == "cb_2":
        return Path(DATA_DIR) / "cb_2.npz"
    if name == "minimal_bundles":
        return Path(DATA_DIR) / "minimal_bundles.zip"
    if name == "t1_coronal_slice":
        return Path(DATA_DIR) / "t1_coronal_slice.npy"
    if name == "t-design":
        N = 45
        return Path(DATA_DIR) / f"tdesign{N}.txt"
    if name == "scil_b0":
        files, folder = fetch_scil_b0()
        files = files["datasets_multi-site_all_companies.zip"][2]
        files = [Path(folder) / f for f in files]
        return [f for f in files if f.is_file()]
    if name == "stanford_hardi":
        files, folder = fetch_stanford_hardi()
        fraw = Path(folder) / "HARDI150.nii.gz"
        fbval = Path(folder) / "HARDI150.bval"
        fbvec = Path(folder) / "HARDI150.bvec"
        return fraw, fbval, fbvec
    if name == "taiwan_ntu_dsi":
        files, folder = fetch_taiwan_ntu_dsi()
        fraw = Path(folder) / "DSI203.nii.gz"
        fbval = Path(folder) / "DSI203.bval"
        fbvec = Path(folder) / "DSI203.bvec"
        return fraw, fbval, fbvec
    if name == "sherbrooke_3shell":
        files, folder = fetch_sherbrooke_3shell()
        fraw = Path(folder) / "HARDI193.nii.gz"
        fbval = Path(folder) / "HARDI193.bval"
        fbvec = Path(folder) / "HARDI193.bvec"
        return fraw, fbval, fbvec
    if name == "isbi2013_2shell":
        files, folder = fetch_isbi2013_2shell()
        fraw = Path(folder) / "phantom64.nii.gz"
        fbval = Path(folder) / "phantom64.bval"
        fbvec = Path(folder) / "phantom64.bvec"
        return fraw, fbval, fbvec
    if name == "stanford_labels":
        files, folder = fetch_stanford_labels()
        return Path(folder) / "aparc-reduced.nii.gz"
    if name == "syn_data":
        files, folder = fetch_syn_data()
        t1_name = Path(folder) / "t1.nii.gz"
        b0_name = Path(folder) / "b0.nii.gz"
        return t1_name, b0_name
    if name == "stanford_t1":
        files, folder = fetch_stanford_t1()
        return Path(folder) / "t1.nii.gz"
    if name == "stanford_pve_maps":
        files, folder = fetch_stanford_pve_maps()
        f_pve_csf = Path(folder) / "pve_csf.nii.gz"
        f_pve_gm = Path(folder) / "pve_gm.nii.gz"
        f_pve_wm = Path(folder) / "pve_wm.nii.gz"
        return f_pve_csf, f_pve_gm, f_pve_wm
    if name == "ivim":
        files, folder = fetch_ivim()
        fraw = Path(folder) / "ivim.nii.gz"
        fbval = Path(folder) / "ivim.bval"
        fbvec = Path(folder) / "ivim.bvec"
        return fraw, fbval, fbvec
    if name == "tissue_data":
        files, folder = fetch_tissue_data()
        t1_name = Path(folder) / "t1_brain.nii.gz"
        t1d_name = Path(folder) / "t1_brain_denoised.nii.gz"
        ap_name = Path(folder) / "power_map.nii.gz"
        return t1_name, t1d_name, ap_name
    if name == "cfin_multib":
        files, folder = fetch_cfin_multib()
        t1_name = Path(folder) / "T1.nii"
        fraw = Path(folder) / "__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii"
        fbval = Path(folder) / "__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bval"
        fbvec = Path(folder) / "__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bvec"
        return fraw, fbval, fbvec, t1_name
    if name == "target_tractrogram_hcp":
        files, folder = fetch_target_tractogram_hcp()
        return (
            Path(folder) / "target_tractogram_hcp",
            "hcp_tractogram",
            "streamlines.trk",
        )
    if name == "bundle_atlas_hcp842":
        files, folder = fetch_bundle_atlas_hcp842()
        return get_bundle_atlas_hcp842()
    if name == "30_bundle_atlas_hcp842":
        files, folder = fetch_30_bundle_atlas_hcp842()
        return get_bundle_atlas_hcp842(size=30)
    if name == "qte_lte_pte":
        _, folder = fetch_qte_lte_pte()
        fdata = Path(folder) / "lte-pte.nii.gz"
        fbval = Path(folder) / "lte-pte.bval"
        fbvec = Path(folder) / "lte-pte.bvec"
        fmask = Path(folder) / "mask.nii.gz"
        return fdata, fbval, fbvec, fmask
    if name == "cti_rat1":
        _, folder = fetch_cti_rat1()
        fdata = Path(folder) / "Rat1_invivo_cti_data.nii"
        fbval1 = Path(folder) / "bvals1.bval"
        fbvec1 = Path(folder) / "bvec1.bvec"
        fbval2 = Path(folder) / "bvals2.bval"
        fbvec2 = Path(folder) / "bvec2.bvec"
        fmask = Path(folder) / "Rat1_mask.nii"
        return fdata, fbval1, fbvec1, fbval2, fbvec2, fmask
    if name == "fury_surface":
        files, folder = fetch_fury_surface()
        surface_name = Path(folder) / "100307_white_lh.vtk"
        return surface_name
    if name == "histo_resdnn_tf_weights":
        files, folder = fetch_resdnn_tf_weights()
        wraw = Path(folder) / "resdnn_weights_mri_2018.h5"
        return wraw
    if name == "histo_resdnn_torch_weights":
        files, folder = fetch_resdnn_torch_weights()
        wraw = Path(folder) / "histo_weights.pth"
        return wraw
    if name == "synb0_default_weights":
        _, folder = fetch_synb0_weights()
        w1 = Path(folder) / "synb0_default_weights1.h5"
        w2 = Path(folder) / "synb0_default_weights2.h5"
        w3 = Path(folder) / "synb0_default_weights3.h5"
        w4 = Path(folder) / "synb0_default_weights4.h5"
        w5 = Path(folder) / "synb0_default_weights5.h5"
        return w1, w2, w3, w4, w5
    if name == "synb0_test_data":
        files, folder = fetch_synb0_test()
        input_array = Path(folder) / "test_input_synb0.npz"
        target_array = Path(folder) / "test_output_synb0.npz"
        return input_array, target_array
    if name == "deepn4_default_tf_weights":
        _, folder = fetch_deepn4_tf_weights()
        w1 = Path(folder) / "model_weights.h5"
        return w1
    if name == "deepn4_default_torch_weights":
        _, folder = fetch_deepn4_torch_weights()
        w1 = Path(folder) / "deepn4_torch_weights"
        return w1
    if name == "deepn4_test_data":
        files, folder = fetch_deepn4_test()
        test_data = Path(folder) / "d4_data.npz"
        return test_data
    if name == "evac_default_tf_weights":
        files, folder = fetch_evac_tf_weights()
        weight = Path(folder) / "evac_default_weights.h5"
        return weight
    if name == "evac_default_torch_weights":
        files, folder = fetch_evac_torch_weights()
        weight = Path(folder) / "evac_weights.pth"
        return weight
    if name == "evac_test_data":
        files, folder = fetch_evac_test()
        test_data = Path(folder) / "evac_test_data.npz"
        return test_data
    if name == "synthseg_torch_weights":
        _, folder = fetch_synthseg_torch_weights()
        w1 = Path(folder) / "synthseg_model_weights.pth"
        return w1
    if name == "synthseg_test_data":
        files, folder = fetch_synthseg_test()
        test_data = Path(folder) / "synthseg_test_data.npz"
        return test_data
    if name == "DiB_70_lte_pte_ste":
        _, folder = fetch_DiB_70_lte_pte_ste()
        fdata = Path(folder) / "DiB_70_lte_pte_ste.nii.gz"
        fbval = Path(folder) / "bval_DiB_70_lte_pte_ste.bval"
        fbvec = Path(folder) / "bvec_DiB_70_lte_pte_ste.bvec"
        fmask = Path(folder) / "DiB_mask.nii.gz"
        return fdata, fbval, fbvec, fmask
    if name == "DiB_217_lte_pte_ste":
        _, folder = fetch_DiB_217_lte_pte_ste()
        fdata_1 = Path(folder) / "DiB_217_lte_pte_ste_1.nii.gz"
        fdata_2 = Path(folder) / "DiB_217_lte_pte_ste_2.nii.gz"
        fbval = Path(folder) / "bval_DiB_217_lte_pte_ste.bval"
        fbvec = Path(folder) / "bvec_DiB_217_lte_pte_ste.bvec"
        fmask = Path(folder) / "DiB_mask.nii.gz"
        return fdata_1, fdata_2, fbval, fbvec, fmask
    if name == "ptt_minimal_dataset":
        files, folder = fetch_ptt_minimal_dataset()
        fod_name = Path(folder) / "ptt_fod.nii"
        seed_coords_name = Path(folder) / "ptt_seed_coords.txt"
        seed_image_name = Path(folder) / "ptt_seed_image.nii"
        return fod_name, seed_coords_name, seed_image_name
    if name == "gold_standard_io":
        filepath_dix = {}
        files, folder = fetch_gold_standard_io()
        for filename in files:
            filepath_dix[filename] = Path(folder) / filename

        with open(filepath_dix["points_data.json"]) as json_file:
            points_data = dict(json.load(json_file))

        with open(filepath_dix["streamlines_data.json"]) as json_file:
            streamlines_data = dict(json.load(json_file))

        return filepath_dix, points_data, streamlines_data
    if name == "real_data_io":
        filepath_dix = {}
        files, folder = fetch_real_data_io()
        for filename in files:
            filepath_dix[filename] = os.path.join(folder, filename)

        return filepath_dix
    if name in ["disco", "disco1", "disco2", "disco3"]:
        local_fetcher = globals().get(f"fetch_{name}_dataset")
        files, folder = local_fetcher(include_optional=include_optional)
        return [Path(folder) / f for f in files]


def read_qtdMRI_test_retest_2subjects():
    r"""Load test-retest qt-dMRI acquisitions of two C57Bl6 mice.

    These datasets were used to study test-retest reproducibility of
    time-dependent q-space indices (q$\tau$-indices) in the corpus callosum of
    two mice :footcite:p:`Fick2018`. The data itself and its details are
    publicly available and can be cited at :footcite:p:`Wassermann2017`.

    The test-retest diffusion MRI spin echo sequences were acquired from two
    C57Bl6 wild-type mice on an 11.7 Tesla Bruker scanner. The test and retest
    acquisition were taken 48 hours from each other. The (processed) data
    consists of 80x160x5 voxels of size 110x110x500μm. Each data set consists
    of 515 Diffusion-Weighted Images (DWIs) spread over 35 acquisition shells.
    The shells are spread over 7 gradient strength shells with a maximum
    gradient strength of 491 mT/m, 5 pulse separation shells between
    [10.8 - 20.0]ms, and a pulse length of 5ms. We manually created a brain
    mask and corrected the data from eddy currents and motion artifacts using
    FSL's eddy. A region of interest was then drawn in the middle slice in the
    corpus callosum, where the tissue is reasonably coherent.

    Returns
    -------
    data : list of length 4
        contains the dwi datasets ordered as
        (subject1_test, subject1_retest, subject2_test, subject2_retest)
    cc_masks : list of length 4
        contains the corpus callosum masks ordered in the same order as data.
    gtabs : list of length 4
        contains the qt-dMRI gradient tables of the data sets.

    References
    ----------
    .. footbibliography::
    """
    data = []
    data_names = [
        "subject1_dwis_test.nii.gz",
        "subject1_dwis_retest.nii.gz",
        "subject2_dwis_test.nii.gz",
        "subject2_dwis_retest.nii.gz",
    ]
    for data_name in data_names:
        data_loc = dipy_home / "qtdMRI_test_retest_2subjects" / data_name
        data.append(load_nifti_data(data_loc))

    cc_masks = []
    mask_names = [
        "subject1_ccmask_test.nii.gz",
        "subject1_ccmask_retest.nii.gz",
        "subject2_ccmask_test.nii.gz",
        "subject2_ccmask_retest.nii.gz",
    ]
    for mask_name in mask_names:
        mask_loc = dipy_home / "qtdMRI_test_retest_2subjects" / mask_name
        cc_masks.append(load_nifti_data(mask_loc))

    gtabs = []
    gtab_txt_names = [
        "subject1_scheme_test.txt",
        "subject1_scheme_retest.txt",
        "subject2_scheme_test.txt",
        "subject2_scheme_retest.txt",
    ]
    for gtab_txt_name in gtab_txt_names:
        txt_loc = dipy_home / "qtdMRI_test_retest_2subjects" / gtab_txt_name
        qtdmri_scheme = np.loadtxt(txt_loc, skiprows=1)
        bvecs = qtdmri_scheme[:, 1:4]
        G = qtdmri_scheme[:, 4] / 1e3  # because dipy takes T/mm not T/m
        small_delta = qtdmri_scheme[:, 5]
        big_delta = qtdmri_scheme[:, 6]
        gtab = gradient_table_from_gradient_strength_bvecs(
            G, bvecs, big_delta, small_delta
        )
        gtabs.append(gtab)

    return data, cc_masks, gtabs


def read_scil_b0():
    """Load GE 3T b0 image form the scil b0 dataset.

    Returns
    -------
    img : obj,
        Nifti1Image

    """
    fnames = get_fnames(name="scil_b0")
    return nib.load(fnames[0])


def read_siemens_scil_b0():
    """Load Siemens 1.5T b0 image from the scil b0 dataset.

    Returns
    -------
    img : obj,
        Nifti1Image

    """
    fnames = get_fnames(name="scil_b0")
    return nib.load(fnames[1])


def read_isbi2013_2shell():
    """Load ISBI 2013 2-shell synthetic dataset.

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable

    """
    fraw, fbval, fbvec = get_fnames(name="isbi2013_2shell")
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs=bvecs)
    img = nib.load(fraw)
    return img, gtab


def read_sherbrooke_3shell():
    """Load Sherbrooke 3-shell HARDI dataset.

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable

    """
    fraw, fbval, fbvec = get_fnames(name="sherbrooke_3shell")
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs=bvecs)
    img = nib.load(fraw)
    return img, gtab


def read_stanford_labels():
    """Read stanford hardi data and label map."""
    # First get the hardi data
    hard_img, gtab = read_stanford_hardi()
    # Fetch and load
    labels_file = get_fnames(name="stanford_labels")
    labels_img = nib.load(labels_file)
    return hard_img, gtab, labels_img


def read_stanford_hardi():
    """Load Stanford HARDI dataset.

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable

    """
    fraw, fbval, fbvec = get_fnames(name="stanford_hardi")
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs=bvecs)
    img = nib.load(fraw)
    return img, gtab


def read_stanford_t1():
    f_t1 = get_fnames(name="stanford_t1")
    img = nib.load(f_t1)
    return img


def read_stanford_pve_maps():
    f_pve_csf, f_pve_gm, f_pve_wm = get_fnames(name="stanford_pve_maps")
    img_pve_csf = nib.load(f_pve_csf)
    img_pve_gm = nib.load(f_pve_gm)
    img_pve_wm = nib.load(f_pve_wm)
    return img_pve_csf, img_pve_gm, img_pve_wm


def read_taiwan_ntu_dsi():
    """Load Taiwan NTU dataset.

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable

    """
    fraw, fbval, fbvec = get_fnames(name="taiwan_ntu_dsi")
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    bvecs[1:] = bvecs[1:] / np.sqrt(np.sum(bvecs[1:] * bvecs[1:], axis=1))[:, None]
    gtab = gradient_table(bvals, bvecs=bvecs)
    img = nib.load(fraw)
    return img, gtab


def read_syn_data():
    """Load t1 and b0 volumes from the same session.

    Returns
    -------
    t1 : obj,
        Nifti1Image
    b0 : obj,
        Nifti1Image

    """
    t1_name, b0_name = get_fnames(name="syn_data")
    t1 = nib.load(t1_name)
    b0 = nib.load(b0_name)
    return t1, b0


def fetch_tissue_data(*, include_optional=False):
    """Download images to be used for tissue classification"""

    t1 = "https://ndownloader.figshare.com/files/6965969"
    t1d = "https://ndownloader.figshare.com/files/6965981"
    ap = "https://ndownloader.figshare.com/files/6965984"

    folder = dipy_home / "tissue_data"

    md5_list = [
        "99c4b77267a6855cbfd96716d5d65b70",  # t1
        "4b87e1b02b19994fbd462490cc784fa3",  # t1d
        "c0ea00ed7f2ff8b28740f18aa74bff6a",
    ]  # ap

    url_list = [t1, t1d, ap]
    fname_list = ["t1_brain.nii.gz", "t1_brain_denoised.nii.gz", "power_map.nii.gz"]

    if not folder.exists():
        logger.info(f"Creating new directory {folder}")
        os.makedirs(folder)
        msg = "Downloading 3 Nifti1 images (9.3MB)..."
        logger.info(msg)

        for i in range(len(md5_list)):
            _get_file_data(Path(folder) / fname_list[i], url_list[i])
            check_md5(Path(folder) / fname_list[i], md5_list[i])

        logger.info("Done.")
        logger.info(f"Files copied in folder {folder}")
    else:
        _already_there_msg(folder)

    return fname_list, folder


@warning_for_keywords()
def read_tissue_data(*, contrast="T1"):
    """Load images to be used for tissue classification

    Parameters
    ----------
    contrast : str
        'T1', 'T1 denoised' or 'Anisotropic Power'

    Returns
    -------
    image : obj,
        Nifti1Image

    """
    folder = dipy_home / "tissue_data"
    t1_name = Path(folder) / "t1_brain.nii.gz"
    t1d_name = Path(folder) / "t1_brain_denoised.nii.gz"
    ap_name = Path(folder) / "power_map.nii.gz"

    md5_dict = {
        "t1": "99c4b77267a6855cbfd96716d5d65b70",
        "t1d": "4b87e1b02b19994fbd462490cc784fa3",
        "ap": "c0ea00ed7f2ff8b28740f18aa74bff6a",
    }

    check_md5(t1_name, md5_dict["t1"])
    check_md5(t1d_name, md5_dict["t1d"])
    check_md5(ap_name, md5_dict["ap"])

    if contrast == "T1 denoised":
        return nib.load(t1d_name)
    elif contrast == "Anisotropic Power":
        return nib.load(ap_name)
    else:
        return nib.load(t1_name)


mni_notes = """
    Notes
    -----
    The templates were downloaded from the MNI (McGill University)
    `website <https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009>`_
    in July 2015.

    The following publications should be referenced when using these templates:

    - :footcite:t:`Fonov2013`
    - :footcite:t:`Fonov2009`

    **License for the MNI templates:**

    Copyright (C) 1993-2004, Louis Collins McConnell Brain Imaging Centre,
    Montreal Neurological Institute, McGill University. Permission to use,
    copy, modify, and distribute this software and its documentation for any
    purpose and without fee is hereby granted, provided that the above
    copyright notice appear in all copies. The authors and McGill University
    make no representations about the suitability of this software for any
    purpose. It is provided "as is" without express or implied warranty. The
    authors are not responsible for any data loss, equipment damage, property
    loss, or injury to subjects or patients resulting from the use or misuse
    of this software package.

    References
    ----------
    .. footbibliography::
"""


@warning_for_keywords()
def read_mni_template(*, version="a", contrast="T2"):
    """Read the MNI template from disk.

    Parameters
    ----------
    version: string
        There are two MNI templates 2009a and 2009c, so options available are:
        "a" and "c".
    contrast : list or string, optional
        Which of the contrast templates to read. For version "a" two contrasts
        are available: "T1" and "T2". Similarly for version "c" there are two
        options, "T1" and "mask". You can input contrast as a string or a list

    Returns
    -------
    list : contains the nibabel.Nifti1Image objects requested, according to the
        order they were requested in the input.

    Examples
    --------
    >>> # Get only the T1 file for version c:
    >>> T1 = read_mni_template("c", contrast = "T1") # doctest: +SKIP
    >>> # Get both files in this order for version a:
    >>> T1, T2 = read_mni_template(contrast = ["T1", "T2"]) # doctest: +SKIP

    """
    files, folder = fetch_mni_template()
    file_dict_a = {
        "T1": Path(folder) / "mni_icbm152_t1_tal_nlin_asym_09a.nii",
        "T2": Path(folder) / "mni_icbm152_t2_tal_nlin_asym_09a.nii",
    }

    file_dict_c = {
        "T1": Path(folder) / "mni_icbm152_t1_tal_nlin_asym_09c.nii",
        "mask": Path(folder) / "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii",
    }

    if contrast == "T2" and version == "c":
        raise ValueError("No T2 image for MNI template 2009c")

    if contrast == "mask" and version == "a":
        raise ValueError("No template mask available for MNI 2009a")

    if not isinstance(contrast, str) and version == "c":
        for k in contrast:
            if k == "T2":
                raise ValueError("No T2 image for MNI template 2009c")

    if version == "a":
        if isinstance(contrast, str):
            return nib.load(file_dict_a[contrast])
        else:
            out_list = []
            for k in contrast:
                out_list.append(nib.load(file_dict_a[k]))
    elif version == "c":
        if isinstance(contrast, str):
            return nib.load(file_dict_c[contrast])
        else:
            out_list = []
            for k in contrast:
                out_list.append(nib.load(file_dict_c[k]))
    else:
        raise ValueError("Only 2009a and 2009c versions are available")
    return out_list


# Add the references to both MNI-related functions:
read_mni_template.__doc__ += mni_notes
fetch_mni_template.__doc__ += mni_notes


@warning_for_keywords()
def fetch_cenir_multib(*, with_raw=False, **kwargs):
    """Fetch 'HCP-like' data, collected at multiple b-values.

    Parameters
    ----------
    with_raw : bool
        Whether to fetch the raw data. Per default, this is False, which means
        that only eddy-current/motion corrected data is fetched
    """
    folder = dipy_home / "cenir_multib"

    fname_list = [
        "4D_dwi_eddycor_B200.nii.gz",
        "dwi_bvals_B200",
        "dwi_bvecs_B200",
        "4D_dwieddycor_B400.nii.gz",
        "bvals_B400",
        "bvecs_B400",
        "4D_dwieddycor_B1000.nii.gz",
        "bvals_B1000",
        "bvecs_B1000",
        "4D_dwieddycor_B2000.nii.gz",
        "bvals_B2000",
        "bvecs_B2000",
        "4D_dwieddycor_B3000.nii.gz",
        "bvals_B3000",
        "bvecs_B3000",
    ]

    md5_list = [
        "fd704aa3deb83c1c7229202cb3db8c48",
        "80ae5df76a575fe5bf9f1164bb0d4cfb",
        "18e90f8a3e6a4db2457e5b1ba1cc98a9",
        "3d0f2b8ef7b6a4a3aa5c4f7a90c9cfec",
        "c38056c40c9cc42372232d6e75c47f54",
        "810d79b4c30cb7dff3b2000017d5f72a",
        "dde8037601a14436b2173f4345b5fd17",
        "97de6a492ae304f39e0b418b6ebac64c",
        "f28a0faa701bdfc66e31bde471a5b992",
        "c5e4b96e3afdee99c0e994eff3b2331a",
        "9c83b8d5caf9c3def240f320f2d2f56c",
        "05446bd261d57193d8dbc097e06db5ff",
        "f0d70456ce424fda2cecd48e64f3a151",
        "336accdb56acbbeff8dac1748d15ceb8",
        "27089f3baaf881d96f6a9da202e3d69b",
    ]
    if with_raw:
        fname_list.extend(
            [
                "4D_dwi_B200.nii.gz",
                "4D_dwi_B400.nii.gz",
                "4D_dwi_B1000.nii.gz",
                "4D_dwi_B2000.nii.gz",
                "4D_dwi_B3000.nii.gz",
            ]
        )
        md5_list.extend(
            [
                "a8c36e76101f2da2ca8119474ded21d5",
                "a0e7939f6d977458afbb2f4659062a79",
                "87fc307bdc2e56e105dffc81b711a808",
                "7c23e8a5198624aa29455f0578025d4f",
                "4e4324c676f5a97b3ded8bbb100bf6e5",
            ]
        )

    files = {}
    baseurl = UW_RW_URL + "1773/33311/"
    for f, m in zip(fname_list, md5_list):
        files[f] = (baseurl + f, m)

    fetch_data(files, folder)
    return files, folder


@warning_for_keywords()
def read_cenir_multib(*, bvals=None):
    """Read CENIR multi b-value data.

    Parameters
    ----------
    bvals : list or int
        The b-values to read from file (200, 400, 1000, 2000, 3000).

    Returns
    -------
    gtab : a GradientTable class instance
    img : nibabel.Nifti1Image

    """
    files, folder = fetch_cenir_multib(with_raw=False)
    if bvals is None:
        bvals = [200, 400, 1000, 2000, 3000]
    if isinstance(bvals, int):
        bvals = [bvals]
    file_dict = {
        200: {
            "DWI": Path(folder) / "4D_dwi_eddycor_B200.nii.gz",
            "bvals": Path(folder) / "dwi_bvals_B200",
            "bvecs": Path(folder) / "dwi_bvecs_B200",
        },
        400: {
            "DWI": Path(folder) / "4D_dwieddycor_B400.nii.gz",
            "bvals": Path(folder) / "bvals_B400",
            "bvecs": Path(folder) / "bvecs_B400",
        },
        1000: {
            "DWI": Path(folder) / "4D_dwieddycor_B1000.nii.gz",
            "bvals": Path(folder) / "bvals_B1000",
            "bvecs": Path(folder) / "bvecs_B1000",
        },
        2000: {
            "DWI": Path(folder) / "4D_dwieddycor_B2000.nii.gz",
            "bvals": Path(folder) / "bvals_B2000",
            "bvecs": Path(folder) / "bvecs_B2000",
        },
        3000: {
            "DWI": Path(folder) / "4D_dwieddycor_B3000.nii.gz",
            "bvals": Path(folder) / "bvals_B3000",
            "bvecs": Path(folder) / "bvecs_B3000",
        },
    }
    data = []
    bval_list = []
    bvec_list = []
    for bval in bvals:
        data.append(load_nifti_data(file_dict[bval]["DWI"]))
        bval_list.extend(np.loadtxt(file_dict[bval]["bvals"]))
        bvec_list.append(np.loadtxt(file_dict[bval]["bvecs"]))

    # All affines are the same, so grab the last one:
    aff = nib.load(file_dict[bval]["DWI"]).affine
    return (
        nib.Nifti1Image(np.concatenate(data, -1), aff),
        gradient_table(bval_list, bvecs=np.concatenate(bvec_list, -1)),
    )


CENIR_notes = """
    Notes
    -----
    Details of the acquisition and processing, and additional meta-data are
    available through UW researchworks:

    https://digital.lib.washington.edu/researchworks/handle/1773/33311
    """

fetch_cenir_multib.__doc__ += CENIR_notes
read_cenir_multib.__doc__ += CENIR_notes


@warning_for_keywords()
def read_bundles_2_subjects(
    *, subj_id="subj_1", metrics=("fa",), bundles=("af.left", "cst.right", "cc_1")
):
    r"""Read images and streamlines from 2 subjects of the SNAIL dataset.

    See :footcite:p:`Renauld2016` and :footcite:p:`Garyfallidis2015` for further
    details about the dataset and processing pipeline.

    Parameters
    ----------
    subj_id : string
        Either ``subj_1`` or ``subj_2``.
    metrics : array-like
        Either ['fa'] or ['t1'] or ['fa', 't1']
    bundles : array-like
        E.g., ['af.left', 'cst.right', 'cc_1']. See all the available bundles
        in the ``exp_bundles_maps/bundles_2_subjects`` directory of your
        ``DIPY_HOME`` of ``$HOME/.dipy`` folder.

    Returns
    -------
    dix : dict
        Dictionary with data of the metrics and the bundles as keys.

    Notes
    -----
    If you are using these datasets please cite the following publications.

    References
    ----------
    .. footbibliography::

    """
    dname = dipy_home / "exp_bundles_and_maps" / "bundles_2_subjects"

    from dipy.tracking.streamline import Streamlines

    res = {}

    if "t1" in metrics:
        data, affine = load_nifti(Path(dname) / subj_id / "t1_warped.nii.gz")
        res["t1"] = data

    if "fa" in metrics:
        fa, affine = load_nifti(Path(dname) / subj_id / "fa_1x1x1.nii.gz")
        res["fa"] = fa

    res["affine"] = affine
    for bun in bundles:
        streams = load_tractogram(
            Path(dname) / subj_id / "bundles" / f"bundles_{bun}.trk",
            "same",
            bbox_valid_check=False,
        ).streamlines

        streamlines = Streamlines(streams)
        res[bun] = streamlines
    return res


def read_ivim():
    """Load IVIM dataset.

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable

    """
    fraw, fbval, fbvec = get_fnames(name="ivim")
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=0)
    img = nib.load(fraw)
    return img, gtab


def read_cfin_dwi():
    """Load CFIN multi b-value DWI data.

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable

    """
    fraw, fbval, fbvec, _ = get_fnames(name="cfin_multib")
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs=bvecs)
    img = nib.load(fraw)
    return img, gtab


def read_cfin_t1():
    """Load CFIN T1-weighted data.

    Returns
    -------
    img : obj,
        Nifti1Image

    """
    _, _, _, fraw = get_fnames(name="cfin_multib")
    img = nib.load(fraw)
    return img  # , gtab


def get_file_formats():
    """

    Returns
    -------
    bundles_list : all bundles (list)
    ref_anat : reference
    """
    ref_anat = dipy_home / "bundle_file_formats_example" / "template0.nii.gz"
    bundles_list = []
    for filename in [
        "cc_m_sub.trk",
        "laf_m_sub.tck",
        "lpt_m_sub.fib",
        "raf_m_sub.vtk",
        "rpt_m_sub.dpy",
    ]:
        bundles_list.append(dipy_home / "bundle_file_formats_example" / filename)

    return bundles_list, ref_anat


@warning_for_keywords()
def get_bundle_atlas_hcp842(*, size=80):
    """
    Returns
    -------
    file1 : string
    file2 : string
    """
    size = 80 if size not in [80, 30] else size

    file1 = (
        dipy_home
        / "bundle_atlas_hcp842"
        / f"Atlas_{size}_Bundles"
        / "whole_brain"
        / "whole_brain_MNI.trk"
    )

    file2 = (
        dipy_home
        / "bundle_atlas_hcp842"
        / f"Atlas_{size}_Bundles"
        / "bundles"
        / "*.trk"
    )

    return file1, file2


def get_two_hcp842_bundles():
    """
    Returns
    -------
    file1 : string
    file2 : string
    """
    file1 = (
        dipy_home / "bundle_atlas_hcp842" / "Atlas_80_Bundles" / "bundles" / "AF_L.trk"
    )

    file2 = (
        dipy_home / "bundle_atlas_hcp842" / "Atlas_80_Bundles" / "bundles" / "CST_L.trk"
    )

    return file1, file2


def get_target_tractogram_hcp():
    """
    Returns
    -------
    file1 : string
    """
    file1 = dipy_home / "target_tractogram_hcp" / "hcp_tractogram" / "streamlines.trk"

    return file1


def read_qte_lte_pte():
    """Read q-space trajectory encoding data with linear and planar tensor
    encoding.

    Returns
    -------
    data_img : nibabel.nifti1.Nifti1Image
        dMRI data image.
    mask_img : nibabel.nifti1.Nifti1Image
        Brain mask image.
    gtab : dipy.core.gradients.GradientTable
        Gradient table.
    """
    fdata, fbval, fbvec, fmask = get_fnames(name="qte_lte_pte")
    data_img = nib.load(fdata)
    mask_img = nib.load(fmask)
    bvals = np.loadtxt(fbval)
    bvecs = np.loadtxt(fbvec)
    btens = np.array(["LTE" for i in range(61)] + ["PTE" for i in range(61)])
    gtab = gradient_table(bvals, bvecs=bvecs, btens=btens)
    return data_img, mask_img, gtab


def read_DiB_70_lte_pte_ste():
    """Read q-space trajectory encoding data with 70 between linear, planar,
    and spherical tensor encoding measurements.

    Returns
    -------
    data_img : nibabel.nifti1.Nifti1Image
        dMRI data image.
    mask_img : nibabel.nifti1.Nifti1Image
        Brain mask image.
    gtab : dipy.core.gradients.GradientTable
        Gradient table.
    """
    fdata, fbval, fbvec, fmask = get_fnames(name="DiB_70_lte_pte_ste")
    data_img = nib.load(fdata)
    mask_img = nib.load(fmask)
    bvals = np.loadtxt(fbval)
    bvecs = np.loadtxt(fbvec)
    btens = np.array(
        ["LTE" for i in range(1)]
        + ["LTE" for i in range(4)]
        + ["PTE" for i in range(3)]
        + ["STE" for i in range(3)]
        + ["STE" for i in range(4)]
        + ["LTE" for i in range(8)]
        + ["PTE" for i in range(5)]
        + ["STE" for i in range(5)]
        + ["LTE" for i in range(21)]
        + ["PTE" for i in range(10)]
        + ["STE" for i in range(6)]
    )
    gtab = gradient_table(bvals, bvecs=bvecs, btens=btens)
    return data_img, mask_img, gtab


def read_DiB_217_lte_pte_ste():
    """Read q-space trajectory encoding data with 217 between linear,
    planar, and spherical tensor encoding.

    Returns
    -------
    data_img : nibabel.nifti1.Nifti1Image
        dMRI data image.
    mask_img : nibabel.nifti1.Nifti1Image
        Brain mask image.
    gtab : dipy.core.gradients.GradientTable
        Gradient table.
    """
    fdata_1, fdata_2, fbval, fbvec, fmask = get_fnames(name="DiB_217_lte_pte_ste")
    _, folder = fetch_DiB_217_lte_pte_ste()
    if Path(Path(folder) / "DiB_217_lte_pte_ste.nii.gz").is_file():
        data_img = nib.load(Path(folder) / "DiB_217_lte_pte_ste.nii.gz")
    else:
        data_1, affine = load_nifti(fdata_1)
        data_2, _ = load_nifti(fdata_2)
        data = np.concatenate((data_1, data_2), axis=3)
        save_nifti(Path(folder) / "DiB_217_lte_pte_ste.nii.gz", data, affine)
        data_img = nib.load(Path(folder) / "DiB_217_lte_pte_ste.nii.gz")
    mask_img = nib.load(fmask)
    bvals = np.loadtxt(fbval)
    bvecs = np.loadtxt(fbvec)
    btens = np.array(
        ["LTE" for i in range(13)]
        + ["LTE" for i in range(10)]
        + ["PTE" for i in range(10)]
        + ["STE" for i in range(10)]
        + ["LTE" for i in range(10)]
        + ["PTE" for i in range(10)]
        + ["STE" for i in range(10)]
        + ["LTE" for i in range(16)]
        + ["PTE" for i in range(16)]
        + ["STE" for i in range(10)]
        + ["LTE" for i in range(46)]
        + ["PTE" for i in range(46)]
        + ["STE" for i in range(10)]
    )
    gtab = gradient_table(bvals, bvecs=bvecs, btens=btens)
    return data_img, mask_img, gtab


def extract_example_tracts(out_dir):
    """Extract 5 'AF_L','CST_R' and 'CC_ForcepsMajor' trk files in out_dir
    folder.

    Parameters
    ----------
    out_dir : str
        Folder in which to extract the files.

    """

    fname = get_fnames(name="minimal_bundles")

    with zipfile.ZipFile(fname, "r") as zip_obj:
        zip_obj.extractall(out_dir)


def read_five_af_bundles():
    """Load 5 small left arcuate fasciculus bundles.

    Returns
    -------
    bundles: list of ArraySequence
        List with loaded bundles.

    """

    subjects = ["sub_1", "sub_2", "sub_3", "sub_4", "sub_5"]

    with tempfile.TemporaryDirectory() as temp_dir:
        extract_example_tracts(temp_dir)

        bundles = []
        for sub in subjects:
            fname = Path(temp_dir) / sub / "AF_L.trk"
            bundle_obj = load_tractogram(fname, "same", bbox_valid_check=False)
            bundles.append(bundle_obj.streamlines)

    return bundles


@warning_for_keywords()
def to_bids_description(
    path, *, fname="dataset_description.json", BIDSVersion="1.4.0", **kwargs
):
    """Dumps a dict into a bids description at the given location"""
    kwargs.update({"BIDSVersion": BIDSVersion})
    desc_file = Path(path) / fname
    with open(desc_file, "w") as outfile:
        json.dump(kwargs, outfile)


@warning_for_keywords()
def fetch_hcp(
    subjects,
    *,
    hcp_bucket="hcp-openaccess",
    profile_name="hcp",
    path=None,
    study="HCP_1200",
    aws_access_key_id=None,
    aws_secret_access_key=None,
):
    """
    Fetch HCP diffusion data and arrange it in a manner that resembles the
    BIDS specification.

    See :footcite:p:`Gorgolewski2016` for details about the BIDS specification.

    Parameters
    ----------
    subjects : list
        Each item is an integer, identifying one of the HCP subjects
    hcp_bucket : string, optional
        The name of the HCP S3 bucket.
    profile_name : string, optional
        The name of the AWS profile used for access.
    path : string, optional
        Path to save files into. Defaults to the value of the ``DIPY_HOME``
        environment variable is set; otherwise, defaults to ``$HOME/.dipy``.
    study : string, optional
        Which HCP study to grab.
    aws_access_key_id : string, optional
        AWS credentials to HCP AWS S3. Will only be used if `profile_name` is
        set to False.
    aws_secret_access_key : string, optional
        AWS credentials to HCP AWS S3. Will only be used if `profile_name` is
        set to False.

    Returns
    -------
    dict with remote and local names of these files,
    path to BIDS derivative dataset

    Notes
    -----
    To use this function with its default setting, you need to have a
    file '~/.aws/credentials', that includes a section:

    [hcp]
    AWS_ACCESS_KEY_ID=XXXXXXXXXXXXXXXX
    AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXXXXXX

    The keys are credentials that you can get from HCP
    (see https://wiki.humanconnectome.org/docs/How%20to%20Get%20Access%20to%20the%20HCP%20OpenAccess%20Amazon%20S3%20Bucket.md)

    Local filenames are changed to match our expected conventions.

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    if not has_boto3:
        raise ValueError(
            "'fetch_hcp' requires boto3 and it is"
            " not currently installed. Please install"
            "it using `pip install boto3`. "
        )

    if profile_name:
        boto3.setup_default_session(profile_name=profile_name)
    elif aws_access_key_id is not None and aws_secret_access_key is not None:
        boto3.setup_default_session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
    else:
        raise ValueError(
            "Must provide either a `profile_name` or ",
            "both `aws_access_key_id` and ",
            "`aws_secret_access_key` as input to 'fetch_hcp'",
        )

    s3 = boto3.resource("s3")
    bucket = s3.Bucket(hcp_bucket)

    if path is None:
        if not dipy_home.exists():
            os.mkdir(dipy_home)
        my_path = dipy_home
    else:
        my_path = path

    base_dir = Path(my_path) / study / "derivatives" / "hcp_pipeline"

    if not Path(base_dir).exists():
        os.makedirs(base_dir, exist_ok=True)

    data_files = {}
    # If user provided incorrect input, these are typical failures that
    # are easy to recover from:
    if isinstance(subjects, (int, str)):
        subjects = [subjects]

    for subject in subjects:
        sub_dir = Path(base_dir) / f"sub-{subject}"
        if not Path(sub_dir).exists():
            os.makedirs(Path(sub_dir) / "dwi", exist_ok=True)
            os.makedirs(Path(sub_dir) / "anat", exist_ok=True)
        data_files[Path(sub_dir) / "dwi" / f"sub-{subject}_dwi.bval"] = (
            f"{study}/{subject}/T1w/Diffusion/bvals"
        )
        data_files[Path(sub_dir) / "dwi" / f"sub-{subject}_dwi.bvec"] = (
            f"{study}/{subject}/T1w/Diffusion/bvecs"
        )
        data_files[Path(sub_dir) / "dwi" / f"sub-{subject}_dwi.nii.gz"] = (
            f"{study}/{subject}/T1w/Diffusion/data.nii.gz"
        )
        data_files[Path(sub_dir) / "anat" / f"sub-{subject}_T1w.nii.gz"] = (
            f"{study}/{subject}/T1w/T1w_acpc_dc.nii.gz"
        )
        data_files[Path(sub_dir) / "anat" / f"sub-{subject}_aparc+aseg_seg.nii.gz"] = (
            f"{study}/{subject}/T1w/aparc+aseg.nii.gz"
        )

    download_files = {}
    for k in data_files.keys():
        if not Path(k).exists():
            download_files[k] = data_files[k]
    if len(download_files.keys()):
        with tqdm(total=len(download_files.keys())) as pbar:
            for k in download_files.keys():
                pbar.set_description_str(f"Downloading {k}")
                bucket.download_file(download_files[k], k)
                pbar.update()

    # Create the BIDS dataset description file text
    hcp_acknowledgements = (
        "Data were provided by the Human Connectome Project, WU-Minn Consortium"
        " (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657)"
        " funded by the 16 NIH Institutes and Centers that support the NIH Blueprint"
        " for Neuroscience Research; and by the McDonnell Center for Systems"
        " Neuroscience at Washington University.",
    )
    to_bids_description(
        Path(my_path) / study,
        **{
            "Name": study,
            "Acknowledgements": hcp_acknowledgements,
            "Subjects": subjects,
        },
    )

    # Create the BIDS derivatives description file text
    to_bids_description(
        base_dir,
        **{
            "Name": study,
            "Acknowledgements": hcp_acknowledgements,
            "GeneratedBy": [{"Name": "hcp_pipeline"}],
        },
    )

    return data_files, Path(my_path) / study


def _hbn_downloader(my_path, derivative, subjects, client):
    base_dir = Path(my_path) / "HBN" / "derivatives" / derivative

    if not base_dir.exists():
        os.makedirs(base_dir, exist_ok=True)

    data_files = {}

    for subject in subjects:
        initial_query = client.list_objects(
            Bucket="fcp-indi", Prefix=f"data/Projects/HBN/BIDS_curated/sub-{subject}/"
        )
        ses = initial_query.get("Contents", None)
        if ses is None:
            raise ValueError(f"Could not find data for subject {subject}")
        else:
            ses = ses[0]["Key"].split("/")[5]

        query = client.list_objects(
            Bucket="fcp-indi",
            Prefix=f"data/Projects/HBN/BIDS_curated/derivatives/{derivative}/sub-{subject}/",
        )  # noqa
        query_content = query.get("Contents", None)
        if query_content is None:
            raise ValueError(f"Could not find derivatives data for subject {subject}")
        file_list = [kk["Key"] for kk in query["Contents"]]
        sub_dir = base_dir / f"sub-{subject}"
        ses_dir = sub_dir / ses
        if derivative == "qsiprep":
            if not sub_dir.exists():
                os.makedirs(sub_dir / "anat", exist_ok=True)
                os.makedirs(sub_dir / "figures", exist_ok=True)
                os.makedirs(ses_dir / "dwi", exist_ok=True)
                os.makedirs(ses_dir / "anat", exist_ok=True)
        if derivative == "afq":
            if not sub_dir.exists():
                os.makedirs(ses_dir / "bundles", exist_ok=True)
                os.makedirs(ses_dir / "clean_bundles", exist_ok=True)
                os.makedirs(ses_dir / "ROIs", exist_ok=True)
                os.makedirs(ses_dir / "tract_profile_plots", exist_ok=True)
                os.makedirs(ses_dir / "viz_bundles", exist_ok=True)

        for remote in file_list:
            full = remote.split("Projects")[-1][1:].replace("/BIDS_curated", "")
            local = Path(my_path) / full
            data_files[local] = remote

    download_files = {}
    for k in data_files.keys():
        if not Path(k).exists():
            download_files[k] = data_files[k]

    if len(download_files.keys()):
        with tqdm(total=len(download_files.keys())) as pbar:
            for k in download_files.keys():
                pbar.set_description_str(f"Downloading {k}")
                client.download_file("fcp-indi", download_files[k], k)
                pbar.update()

    # Create the BIDS dataset description file text
    to_bids_description(
        str(Path(my_path) / "HBN"), **{"Name": "HBN", "Subjects": subjects}
    )

    # Create the BIDS derivatives description file text
    to_bids_description(
        str(base_dir), **{"Name": "HBN", "PipelineDescription": {"Name": "qsiprep"}}
    )

    return data_files


def fetch_hbn(subjects, *, path=None, include_afq=False):
    """
    Fetch preprocessed data from the Healthy Brain Network POD2 study.


    See :footcite:p:`Alexander2017` and :footcite:p:`RichieHalford2022` for
    further details about the study and processing pipeline.

    Parameters
    ----------
    subjects : list
        Identifiers of the subjects to download.
        For example: ["NDARAA948VFH", "NDAREK918EC2"].

    path : string, optional
        Path to save files into. Defaults to the value of the ``DIPY_HOME``
        environment variable is set; otherwise, defaults to ``$HOME/.dipy``.

    include_afq : bool, optional
        Whether to include pyAFQ derivatives

    Returns
    -------
    dict with remote and local names of these files,
    path to BIDS derivative dataset

    References
    ----------
    .. footbibliography::

    """

    if has_boto3:
        from botocore import UNSIGNED
        from botocore.client import Config
    else:
        TripWire(
            "The `fetch_hbn` function requires the boto3"
            + " library, but that is not installed."
        )

    # Anonymous access:
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    if path is None:
        if not dipy_home.exists():
            os.mkdir(dipy_home)
        my_path = dipy_home
    else:
        my_path = path

    # If user provided incorrect input, these are typical failures that
    # are easy to recover from:
    if isinstance(subjects, (int, str)):
        subjects = [subjects]

    data_files = _hbn_downloader(my_path, "qsiprep", subjects, client)

    if include_afq:
        data_files.update(_hbn_downloader(my_path, "afq", subjects, client))

    return data_files, Path(my_path) / "HBN"
