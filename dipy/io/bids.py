"""Read diffusion data organised following the BIDS specification.

This module implements the small subset of BIDS needed to feed DIPY
workflows: entity parsing, dataset enumeration, resolution of the gradient
and JSON files that belong to a diffusion acquisition (including the
inheritance principle), companion anatomical and fieldmap images, and the
``dataset_description.json`` of a derivative dataset.

It deliberately does not index, query or validate a whole dataset. When
``pybids`` happens to be installed, :func:`validate_with_pybids` reports
non-compliant file names as warnings, but discovery never depends on it.
"""

import dataclasses
import json
import os
from pathlib import Path
import re

from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from dipy.version import version as dipy_version

bids_validator, have_bids_validator, _ = optional_package("bids_validator")

BIDS_ENTITY_ORDER = (
    "sub",
    "ses",
    "task",
    "acq",
    "ce",
    "rec",
    "dir",
    "run",
    "mod",
    "echo",
    "flip",
    "inv",
    "mt",
    "part",
    "chunk",
    "space",
    "seg",
    "hemi",
    "desc",
)

NIFTI_EXTENSIONS = (".nii.gz", ".nii")

OPPOSITE_PE = {
    "AP": "PA",
    "PA": "AP",
    "LR": "RL",
    "RL": "LR",
    "i": "i-",
    "i-": "i",
    "j": "j-",
    "j-": "j",
    "k": "k-",
    "k-": "k",
}

METADATA_KEYS = (
    "PhaseEncodingDirection",
    "TotalReadoutTime",
    "EffectiveEchoSpacing",
    "EchoTime",
    "RepetitionTime",
)

_ENTITY_RE = re.compile(r"^([a-zA-Z0-9]+)-(.+)$")

_DATASET_URI_PREFIXES = ("bids::", "bids:raw:")


@dataclasses.dataclass(frozen=True)
class DWIRun:
    """One diffusion acquisition resolved from a BIDS dataset.

    Instances compare by value and hash on ``root`` and ``dwi``, so they can
    be used as dictionary keys or set members despite their dict fields.

    Attributes
    ----------
    root : str
        Absolute path of the BIDS dataset root.
    dwi : str
        Absolute path of the DWI NIfTI file.
    bval : str or None
        Absolute path of the b-values file, or None when absent.
    bvec : str or None
        Absolute path of the b-vectors file, or None when absent.
    sidecar : str or None
        Absolute path of the nearest JSON sidecar, or None when absent.
    metadata : dict
        Merged JSON sidecar contents, nearest file winning.
    t1w : str or None
        Absolute path of the T1w image chosen for this run.
    fmap_epi : str or None
        Absolute path of the opposite phase-encoding EPI fieldmap.
    fmap_metadata : dict
        Merged JSON sidecar contents of ``fmap_epi``.
    entities : dict
        Entity key to label mapping, e.g. ``{"sub": "01", "dir": "AP"}``.
    """

    root: str
    dwi: str
    bval: str | None = None
    bvec: str | None = None
    sidecar: str | None = None
    metadata: dict = dataclasses.field(default_factory=dict)
    t1w: str | None = None
    fmap_epi: str | None = None
    fmap_metadata: dict = dataclasses.field(default_factory=dict)
    entities: dict = dataclasses.field(default_factory=dict)

    def __hash__(self):
        """Hash on the dataset root and the DWI path, which identify a run."""
        return hash((self.root, self.dwi))

    @property
    def subject(self):
        """str : Subject label, without the ``sub-`` prefix."""
        return self.entities.get("sub", "")

    @property
    def session(self):
        """str or None : Session label, without the ``ses-`` prefix."""
        return self.entities.get("ses")

    @property
    def label(self):
        """str : Entity chain identifying the run, e.g. ``sub-01_run-2``."""
        parts = [
            f"{key}-{self.entities[key]}"
            for key in BIDS_ENTITY_ORDER
            if key in self.entities
        ]
        return "_".join(parts) if parts else Path(self.dwi).name

    @property
    def relative_dir(self):
        """str : Dataset-relative directory of the run, e.g. ``sub-01/dwi``."""
        parts = [f"sub-{self.subject}"]
        if self.session:
            parts.append(f"ses-{self.session}")
        parts.append("dwi")
        return os.path.join(*parts)

    @property
    def is_complete(self):
        """bool : True when both gradient files were resolved."""
        return self.bval is not None and self.bvec is not None

    @property
    def sort_key(self):
        """tuple : Ordering key over subject, session, acquisition and run."""
        return (
            self.subject,
            self.entities.get("ses", ""),
            self.entities.get("acq", ""),
            self.entities.get("dir", ""),
            _natural_key(label=self.entities.get("run", "")),
            self.dwi,
        )


def _is_number(*, label):
    """Tell whether a label is a plain ASCII decimal number.

    ``str.isdigit`` also accepts characters such as ``²`` that ``int``
    rejects, hence the stricter test.

    Parameters
    ----------
    label : str
        Entity label.

    Returns
    -------
    bool
        True when ``int(label)`` parses the label.
    """
    return label.isascii() and label.isdecimal()


def _natural_key(*, label):
    """Build a sort key ordering numeric labels by value.

    Parameters
    ----------
    label : str
        Entity label, e.g. ``2`` or ``10``.

    Returns
    -------
    tuple
        ``(0, int(label), label)`` for a numeric label, ``(1, 0, label)``
        otherwise, so that ``run-2`` sorts before ``run-10``.
    """
    label = str(label)
    if _is_number(label=label):
        return (0, int(label), label)
    return (1, 0, label)


def split_extension(*, filename):
    """Split a BIDS file name into its stem and its extension.

    Parameters
    ----------
    filename : str or Path
        File name or path.

    Returns
    -------
    stem : str
        File name without its extension.
    extension : str
        Extension, including the leading dot. ``.nii.gz`` is kept whole.
    """
    name = Path(filename).name
    for extension in (".nii.gz", ".json", ".bval", ".bvals", ".bvec", ".bvecs", ".nii"):
        if name.endswith(extension):
            return name[: -len(extension)], extension
    stem, extension = os.path.splitext(name)
    return stem, extension


def parse_entities(*, filename):
    """Parse the BIDS entities of a file name.

    Unknown entity keys are kept verbatim so that a dataset using entities
    DIPY does not know about still round-trips.

    Parameters
    ----------
    filename : str or Path
        File name or path, e.g. ``sub-01_ses-1_dir-AP_dwi.nii.gz``.

    Returns
    -------
    dict
        Entity keys mapped to their labels, plus ``suffix`` and
        ``extension`` when the name carries them.

    Examples
    --------
    >>> from dipy.io.bids import parse_entities
    >>> sorted(parse_entities(filename="sub-01_dir-AP_dwi.nii.gz").items())
    [('dir', 'AP'), ('extension', '.nii.gz'), ('sub', '01'), ('suffix', 'dwi')]
    """
    stem, extension = split_extension(filename=filename)
    entities = {}
    chunks = stem.split("_")

    for chunk in chunks:
        match = _ENTITY_RE.match(chunk)
        if match:
            entities[match.group(1)] = match.group(2)

    if chunks and not _ENTITY_RE.match(chunks[-1]):
        entities["suffix"] = chunks[-1]
    if extension:
        entities["extension"] = extension

    return entities


def build_entity_name(*, entities, suffix, extension):
    """Build a BIDS file name from entities, a suffix and an extension.

    Parameters
    ----------
    entities : dict
        Entity keys mapped to their labels. ``suffix`` and ``extension``
        keys, if present, are ignored in favour of the arguments.
    suffix : str
        BIDS suffix, e.g. ``dwi``, ``dwimap`` or ``mask``.
    extension : str
        Extension including the leading dot, e.g. ``.nii.gz``.

    Returns
    -------
    str
        File name with entities in the canonical BIDS order, unknown keys
        appended alphabetically.

    Examples
    --------
    >>> from dipy.io.bids import build_entity_name
    >>> build_entity_name(entities={"sub": "01", "desc": "FA"},
    ...                   suffix="dwimap", extension=".nii.gz")
    'sub-01_desc-FA_dwimap.nii.gz'
    """
    known = [f"{key}-{entities[key]}" for key in BIDS_ENTITY_ORDER if key in entities]
    unknown = [
        f"{key}-{value}"
        for key, value in sorted(entities.items())
        if key not in BIDS_ENTITY_ORDER and key not in ("suffix", "extension")
    ]
    return "_".join([*known, *unknown, suffix]) + extension


def validate_dataset(*, root, require_description=True):
    """Check that ``root`` looks like a BIDS dataset.

    Parameters
    ----------
    root : str or Path
        Candidate dataset root.
    require_description : bool, optional
        If True, a missing ``dataset_description.json`` is an error rather
        than a warning.

    Returns
    -------
    dict
        Parsed ``dataset_description.json``, or an empty dict when the file
        is absent and ``require_description`` is False.

    Raises
    ------
    ValueError
        If ``root`` does not exist, is not a directory, holds no subject
        directory, or carries a malformed or (when required) missing
        ``dataset_description.json``.
    """
    root = Path(root)

    if not root.exists():
        raise ValueError(f"BIDS dataset not found: {root}")
    if not root.is_dir():
        raise ValueError(f"BIDS dataset root is not a directory: {root}")

    description_file = root / "dataset_description.json"
    description = {}

    if description_file.exists():
        try:
            with open(description_file, encoding="utf-8") as f:
                description = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Malformed dataset_description.json in {root}: {e}"
            ) from e
    elif require_description:
        raise ValueError(
            f"No dataset_description.json in {root}. Point --bids at the "
            "dataset root, not at a subject or a datatype directory."
        )
    else:
        logger.warning(f"No dataset_description.json in {root}")

    if not any(child.is_dir() for child in root.glob("sub-*")):
        raise ValueError(f"No sub-* directory in {root}")

    return description


def looks_like_dataset(*, path):
    """Tell whether a directory looks like the root of a BIDS dataset.

    Parameters
    ----------
    path : str or Path
        Candidate directory.

    Returns
    -------
    bool
        True when the directory holds a ``dataset_description.json`` or at
        least one ``sub-*`` directory.
    """
    path = Path(path)

    if not path.is_dir():
        return False
    if (path / "dataset_description.json").is_file():
        return True

    return any(child.is_dir() for child in path.glob("sub-*"))


def list_subjects(*, root):
    """List the subject labels of a BIDS dataset.

    Parameters
    ----------
    root : str or Path
        Dataset root.

    Returns
    -------
    list of str
        Sorted subject labels, without the ``sub-`` prefix.
    """
    return sorted(
        path.name[len("sub-") :] for path in Path(root).glob("sub-*") if path.is_dir()
    )


def list_sessions(*, root, subject):
    """List the session labels of one subject.

    Parameters
    ----------
    root : str or Path
        Dataset root.
    subject : str
        Subject label, with or without the ``sub-`` prefix.

    Returns
    -------
    list of str
        Sorted session labels without the ``ses-`` prefix, empty when the
        subject has no session level.
    """
    subject = _strip_prefix(label=subject, prefix="sub")
    return sorted(
        path.name[len("ses-") :]
        for path in (Path(root) / f"sub-{subject}").glob("ses-*")
        if path.is_dir()
    )


def inherited_files(*, root, path, suffix, extension):
    """Apply the BIDS inheritance principle to find applicable files.

    A candidate applies to ``path`` when its entities are a subset of those
    of ``path`` and its suffix and extension match.

    Parameters
    ----------
    root : str or Path
        Dataset root. The walk never goes above it.
    path : str or Path
        File the candidates should apply to.
    suffix : str
        BIDS suffix the candidates must carry, e.g. ``dwi``.
    extension : str
        Extension the candidates must carry, e.g. ``.json``.

    Returns
    -------
    list of str
        Absolute paths of the applicable files, nearest first. Within one
        directory level, the file matching the most entities comes first.
    """
    root = Path(root).absolute()
    path = Path(path).absolute()
    target = parse_entities(filename=path)
    target_entities = {
        key: value
        for key, value in target.items()
        if key not in ("suffix", "extension")
    }

    matches = []
    directory = path.parent

    while True:
        level = []
        for candidate in sorted(directory.glob(f"*{extension}")):
            if candidate == path:
                continue
            found = parse_entities(filename=candidate)
            if found.get("suffix") != suffix:
                continue
            if found.get("extension") != extension:
                continue
            entities = {
                key: value
                for key, value in found.items()
                if key not in ("suffix", "extension")
            }
            if all(
                target_entities.get(key) == value for key, value in entities.items()
            ):
                level.append((-len(entities), str(candidate)))

        matches.extend(candidate for _, candidate in sorted(level))

        if directory == root or root not in directory.parents:
            break
        directory = directory.parent

    return matches


def _read_json(*, path):
    """Read a JSON file, warning and returning an empty dict on failure.

    Parameters
    ----------
    path : str or Path
        File to read.

    Returns
    -------
    dict
        Parsed contents, empty when the file cannot be read or parsed.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Ignoring malformed JSON sidecar {path}: {e}")
        return {}


def read_sidecar(*, root, path):
    """Read the JSON metadata that applies to a file.

    The inheritance chain is merged from the most distant file to the
    nearest one, so that nearer keys win. Within one directory level, the
    file matching the most entities is merged last and wins.

    Parameters
    ----------
    root : str or Path
        Dataset root.
    path : str or Path
        File whose metadata is wanted, e.g. a DWI NIfTI file.

    Returns
    -------
    sidecar : str or None
        Absolute path of the nearest JSON file, None when there is none.
    metadata : dict
        Merged metadata, empty when there is no JSON file.
    """
    suffix = parse_entities(filename=path).get("suffix", "")
    chain = inherited_files(root=root, path=path, suffix=suffix, extension=".json")

    metadata = {}
    for candidate in reversed(chain):
        metadata.update(_read_json(path=candidate))

    return (chain[0] if chain else None), metadata


def _nearest_inherited(*, root, path, suffix, extensions):
    """Return the nearest applicable file among several extensions.

    Parameters
    ----------
    root : str or Path
        Dataset root.
    path : str or Path
        File the candidate must apply to.
    suffix : str
        BIDS suffix the candidate must carry.
    extensions : sequence of str
        Accepted extensions, in order of preference.

    Returns
    -------
    str or None
        Absolute path of the nearest applicable file.
    """
    for extension in extensions:
        matches = inherited_files(
            root=root, path=path, suffix=suffix, extension=extension
        )
        if matches:
            return matches[0]
    return None


def find_t1w(*, root, entities):
    """Find the T1w image of a subject.

    Preference goes to the same session, then to the fewest extra entities,
    then to a name without a ``desc-`` entity, then to the first in
    lexicographic order. A T1w of another session is used, with a warning,
    only when the session of the run has none.

    Parameters
    ----------
    root : str or Path
        Dataset root.
    entities : dict
        Entities of the diffusion run, used for the subject and session.

    Returns
    -------
    str or None
        Absolute path of the chosen T1w image, None when the subject has
        none.
    """
    subject = entities.get("sub")
    if not subject:
        return None

    session = entities.get("ses")
    subject_dir = Path(root).absolute() / f"sub-{subject}"
    candidates = []

    for extension in NIFTI_EXTENSIONS:
        candidates.extend(
            str(path) for path in sorted(subject_dir.glob(f"anat/*_T1w{extension}"))
        )
        candidates.extend(
            str(path)
            for path in sorted(subject_dir.glob(f"ses-*/anat/*_T1w{extension}"))
        )

    if not candidates:
        return None

    def rank(candidate):
        found = parse_entities(filename=candidate)
        same_session = session is not None and found.get("ses") == session
        extra = len(set(found) - {"sub", "ses", "suffix", "extension"})
        return (
            0 if same_session or session is None else 1,
            extra,
            0 if "desc" not in found else 1,
            candidate,
        )

    chosen = min(candidates, key=rank)
    chosen_session = parse_entities(filename=chosen).get("ses")
    if session is not None and chosen_session != session:
        where = f"ses-{chosen_session}" if chosen_session else "no session"
        logger.warning(
            f"No T1w in sub-{subject}/ses-{session}: using the T1w of {where} "
            f"instead ({chosen}); it was acquired in another head position"
        )
    logger.info(f"T1w for sub-{subject}: {chosen}")
    return chosen


def _intended_for_paths(*, metadata):
    """Normalise the ``IntendedFor`` entries of a fieldmap sidecar.

    Parameters
    ----------
    metadata : dict
        Fieldmap JSON metadata.

    Returns
    -------
    list of str
        Entries stripped of the ``bids::`` or ``bids:raw:`` URI prefix, with
        forward slashes.
    """
    intended = metadata.get("IntendedFor", [])
    if isinstance(intended, str):
        intended = [intended]

    paths = []
    for entry in intended:
        normalised = str(entry).replace("\\", "/")
        for prefix in _DATASET_URI_PREFIXES:
            if normalised.startswith(prefix):
                normalised = normalised[len(prefix) :]
                break
        paths.append(normalised.lstrip("/"))

    return paths


def _chosen_fmap(*, fmap, metadata, intended, rule, dwi_relative):
    """Log a fieldmap choice and return it.

    Parameters
    ----------
    fmap : str or Path
        Chosen fieldmap.
    metadata : dict
        Merged metadata of ``fmap``.
    intended : list of str
        Normalised ``IntendedFor`` entries of ``fmap``. Non-empty means they
        name other diffusion runs only, which is warned about.
    rule : str
        Rule that selected the fieldmap, for the log.
    dwi_relative : str
        Dataset-relative path of the diffusion file.

    Returns
    -------
    fmap : str
        Absolute path of the chosen fieldmap.
    metadata : dict
        ``metadata``, unchanged.
    """
    logger.info(f"Fieldmap for {dwi_relative} by {rule}: {fmap}")
    if intended:
        logger.warning(
            f"Fieldmap {fmap} is declared IntendedFor other diffusion runs "
            f"({', '.join(intended)}) but is used for {dwi_relative}"
        )
    return str(fmap), metadata


def find_opposite_pe_fmap(*, root, dwi_path, dwi_metadata, entities):
    """Find the EPI fieldmap acquired with the opposite phase encoding.

    Candidates are ranked by three rules, in order: the fieldmap declares
    the DWI file in ``IntendedFor``; it shares the ``acq-`` entity and
    carries the opposite ``dir-`` entity; its ``PhaseEncodingDirection`` is
    the opposite of the DWI one. The last two rules skip a fieldmap whose
    ``IntendedFor`` is set but lists no diffusion image, since it was
    acquired for another modality (e.g. a BOLD run), and try a fieldmap
    without ``IntendedFor`` before one declared for other diffusion runs
    only. Falling back on the latter logs a warning.

    Parameters
    ----------
    root : str or Path
        Dataset root.
    dwi_path : str or Path
        Absolute path of the diffusion file.
    dwi_metadata : dict
        Metadata of the diffusion file.
    entities : dict
        Entities of the diffusion file.

    Returns
    -------
    fmap : str or None
        Absolute path of the chosen fieldmap.
    metadata : dict
        Merged metadata of the chosen fieldmap, empty when none was found.
    """
    subject = entities.get("sub")
    if not subject:
        return None, {}

    root = Path(root).absolute()
    subject_dir = root / f"sub-{subject}"
    session = entities.get("ses")

    # A fieldmap of another session was acquired in another head position, so
    # the search never leaves the session the diffusion run belongs to.
    search_dirs = [subject_dir / f"ses-{session}"] if session else [subject_dir]
    candidates = []

    for search_dir in search_dirs:
        for extension in NIFTI_EXTENSIONS:
            candidates.extend(sorted(search_dir.glob(f"fmap/*_epi{extension}")))

    if not candidates:
        return None, {}

    dwi_relative = os.path.relpath(Path(dwi_path).absolute(), root).replace(os.sep, "/")
    dwi_subject_relative = os.path.relpath(
        Path(dwi_path).absolute(), subject_dir
    ).replace(os.sep, "/")
    dwi_dir = entities.get("dir")
    dwi_pe = dwi_metadata.get("PhaseEncodingDirection")

    available = []

    for candidate in candidates:
        _, metadata = read_sidecar(root=root, path=candidate)
        intended = _intended_for_paths(metadata=metadata)
        if any(
            entry in (dwi_relative, dwi_subject_relative)
            or entry.endswith(dwi_relative)
            for entry in intended
        ):
            logger.info(f"Fieldmap for {dwi_relative} by IntendedFor: {candidate}")
            return str(candidate), metadata
        if intended and not any(
            parse_entities(filename=entry).get("suffix") == "dwi" for entry in intended
        ):
            logger.debug(
                f"Ignoring fieldmap {candidate}: its IntendedFor lists no "
                "diffusion image"
            )
            continue
        available.append((candidate, metadata, intended))

    available.sort(key=lambda item: bool(item[2]))

    for candidate, metadata, intended in available:
        found = parse_entities(filename=candidate)
        if dwi_dir is None or found.get("dir") != OPPOSITE_PE.get(dwi_dir):
            continue
        if found.get("acq") != entities.get("acq"):
            continue
        return _chosen_fmap(
            fmap=candidate,
            metadata=metadata,
            intended=intended,
            rule="opposite dir- entity",
            dwi_relative=dwi_relative,
        )

    for candidate, metadata, intended in available:
        if dwi_pe is None:
            continue
        if metadata.get("PhaseEncodingDirection") != OPPOSITE_PE.get(dwi_pe):
            continue
        return _chosen_fmap(
            fmap=candidate,
            metadata=metadata,
            intended=intended,
            rule="PhaseEncodingDirection",
            dwi_relative=dwi_relative,
        )

    logger.warning(
        f"No opposite phase-encoding fieldmap found for {dwi_relative} among "
        f"{len(candidates)} fmap candidate(s)"
    )
    return None, {}


def phase_encoding_info(*, run):
    """Collect the acquisition metadata a distortion correction would need.

    Parameters
    ----------
    run : DWIRun
        Resolved diffusion acquisition.

    Returns
    -------
    dict
        The subset of ``METADATA_KEYS`` present in the run metadata.
    """
    info = {key: run.metadata[key] for key in METADATA_KEYS if key in run.metadata}

    for key, value in info.items():
        logger.info(f"{run.label}: {key} = {value}")

    if run.fmap_epi is not None:
        for key in ("PhaseEncodingDirection", "TotalReadoutTime"):
            if key not in info:
                logger.warning(
                    f"{run.label}: fieldmap found but {key} is missing from the "
                    "DWI sidecar"
                )

    return info


def validate_with_pybids(*, root, paths):
    """Warn about file names that the BIDS validator rejects.

    Does nothing when ``bids_validator``, which ships with ``pybids``, is
    not installed. Discovery never depends on this check.

    Parameters
    ----------
    root : str or Path
        Dataset root the paths are relative to.
    paths : sequence of str
        Absolute paths to check.

    Returns
    -------
    list of str
        Dataset-relative paths the validator rejected.
    """
    if not have_bids_validator:
        logger.debug(
            "bids_validator is not installed; skipping BIDS name validation "
            "(pip install pybids to enable it)"
        )
        return []

    validator = bids_validator.BIDSValidator()
    root = Path(root).absolute()
    rejected = []

    for path in paths:
        relative = "/" + os.path.relpath(Path(path).absolute(), root).replace(
            os.sep, "/"
        )
        if not validator.is_bids(relative):
            rejected.append(relative)
            logger.warning(f"Not a valid BIDS file name: {relative}")

    return rejected


def _strip_prefix(*, label, prefix):
    """Remove an entity prefix from a label.

    Parameters
    ----------
    label : str
        Label, with or without its prefix, e.g. ``sub-01`` or ``01``.
    prefix : str
        Entity key, e.g. ``sub``.

    Returns
    -------
    str
        Label without the prefix.
    """
    label = str(label)
    return label[len(prefix) + 1 :] if label.startswith(f"{prefix}-") else label


def _matches_filter(*, value, wanted, prefix, numeric=False):
    """Test one entity value against a user supplied filter.

    Parameters
    ----------
    value : str or None
        Entity value of the candidate.
    wanted : sequence of str or None
        Accepted labels, with or without their prefix. None accepts all.
    prefix : str
        Entity key, used to strip the prefix from ``wanted``.
    numeric : bool, optional
        If True, numeric labels match by value, so that ``01`` matches ``1``.

    Returns
    -------
    bool
        True when the candidate passes the filter.
    """
    if wanted is None:
        return True
    if value is None:
        return False
    if isinstance(wanted, (str, int)):
        wanted = [wanted]
    labels = {_strip_prefix(label=label, prefix=prefix) for label in wanted}
    if value in labels:
        return True
    if numeric and _is_number(label=value):
        return any(
            _is_number(label=label) and int(label) == int(value) for label in labels
        )
    return False


def find_dwi_runs(
    *,
    root,
    participant_label=None,
    session_label=None,
    acquisition_label=None,
    run_label=None,
    require_gradients=False,
    validate=True,
):
    """Resolve every diffusion acquisition of a BIDS dataset.

    Parameters
    ----------
    root : str or Path
        Dataset root.
    participant_label : sequence of str, optional
        Subject labels to keep, with or without the ``sub-`` prefix. None
        keeps every subject.
    session_label : sequence of str, optional
        Session labels to keep, with or without the ``ses-`` prefix.
    acquisition_label : sequence of str, optional
        ``acq-`` labels to keep.
    run_label : sequence of str, optional
        ``run-`` labels to keep. Numeric labels match by value, so ``1``
        selects ``run-01``.
    require_gradients : bool, optional
        If True, runs with a missing b-value or b-vector file are dropped
        instead of being returned with None.
    validate : bool, optional
        If True and ``pybids`` is installed, warn about file names the BIDS
        validator rejects.

    Returns
    -------
    list of DWIRun
        Runs sorted by subject, session, acquisition and run. When both a
        ``.nii.gz`` and a ``.nii`` file share a name, only the ``.nii.gz`` one
        is kept.
    """
    root = Path(root).absolute()
    patterns = [f"sub-*/dwi/*_dwi{extension}" for extension in NIFTI_EXTENSIONS] + [
        f"sub-*/ses-*/dwi/*_dwi{extension}" for extension in NIFTI_EXTENSIONS
    ]

    by_stem = {}
    for path in sorted(
        {str(path) for pattern in patterns for path in root.glob(pattern)}
    ):
        stem, extension = split_extension(filename=path)
        key = os.path.join(os.path.dirname(path), stem)
        previous = by_stem.get(key)
        if previous is None:
            by_stem[key] = path
            continue
        kept = path if extension == ".nii.gz" else previous
        dropped = previous if kept == path else path
        logger.warning(f"Both {kept} and {dropped} exist; ignoring {dropped}")
        by_stem[key] = kept

    dwi_files = sorted(by_stem.values())

    if validate:
        validate_with_pybids(root=root, paths=dwi_files)

    runs = []

    for dwi in dwi_files:
        entities = parse_entities(filename=dwi)

        if not _matches_filter(
            value=entities.get("sub"), wanted=participant_label, prefix="sub"
        ):
            continue
        if not _matches_filter(
            value=entities.get("ses"), wanted=session_label, prefix="ses"
        ):
            continue
        if not _matches_filter(
            value=entities.get("acq"), wanted=acquisition_label, prefix="acq"
        ):
            continue
        if not _matches_filter(
            value=entities.get("run"), wanted=run_label, prefix="run", numeric=True
        ):
            continue

        sidecar, metadata = read_sidecar(root=root, path=dwi)
        bval = _nearest_inherited(
            root=root, path=dwi, suffix="dwi", extensions=(".bval", ".bvals")
        )
        bvec = _nearest_inherited(
            root=root, path=dwi, suffix="dwi", extensions=(".bvec", ".bvecs")
        )

        if require_gradients and (bval is None or bvec is None):
            logger.warning(f"Skipping {dwi}: no b-value or b-vector file found")
            continue

        fmap_epi, fmap_metadata = find_opposite_pe_fmap(
            root=root, dwi_path=dwi, dwi_metadata=metadata, entities=entities
        )

        runs.append(
            DWIRun(
                root=str(root),
                dwi=dwi,
                bval=bval,
                bvec=bvec,
                sidecar=sidecar,
                metadata=metadata,
                t1w=find_t1w(root=root, entities=entities),
                fmap_epi=fmap_epi,
                fmap_metadata=fmap_metadata,
                entities=entities,
            )
        )

    runs.sort(key=lambda run: run.sort_key)
    return runs


def derivative_path(*, deriv_root, run, datatype="dwi"):
    """Build the output directory of a run inside a derivative dataset.

    Parameters
    ----------
    deriv_root : str or Path
        Root of the derivative dataset.
    run : DWIRun
        Resolved diffusion acquisition.
    datatype : str, optional
        BIDS datatype directory to write into.

    Returns
    -------
    str
        Absolute path of ``<deriv_root>/sub-X[/ses-Y]/<datatype>``.
    """
    parts = [str(deriv_root), f"sub-{run.subject}"]
    if run.session:
        parts.append(f"ses-{run.session}")
    parts.append(datatype)
    return os.path.abspath(os.path.join(*parts))


def write_dataset_description(
    *,
    path,
    source_datasets=(),
    name="dipy",
    pipeline_name=None,
    bids_version="1.9.0",
):
    """Write the ``dataset_description.json`` of a derivative dataset.

    Parameters
    ----------
    path : str or Path
        Directory to write into. Created when absent.
    source_datasets : sequence of str, optional
        Roots of the raw datasets the derivatives were computed from. The
        first one is also declared as ``raw`` in ``DatasetLinks``, so that
        ``bids:raw:<path>`` URIs in derivative sidecars resolve to it.
    name : str, optional
        Dataset name.
    pipeline_name : str, optional
        Name of the pipeline, reported in ``GeneratedBy``.
    bids_version : str, optional
        BIDS version the description claims to follow.

    Returns
    -------
    str
        Absolute path of the file that was written.
    """
    path = Path(path).absolute()
    path.mkdir(parents=True, exist_ok=True)

    description = "dipy_auto pipeline"
    if pipeline_name:
        description = f"dipy_auto pipeline '{pipeline_name}'"

    content = {
        "Name": name,
        "BIDSVersion": bids_version,
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "dipy",
                "Version": dipy_version,
                "CodeURL": "https://github.com/dipy/dipy",
                "Description": description,
            }
        ],
        "SourceDatasets": [
            {"URL": Path(source).absolute().as_uri()} for source in source_datasets
        ],
    }
    if content["SourceDatasets"]:
        content["DatasetLinks"] = {"raw": content["SourceDatasets"][0]["URL"]}

    target = path / "dataset_description.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=4)

    logger.info(f"Derivative dataset description written to {target}")
    return str(target)
