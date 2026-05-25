"""Create fixed/moving registration pairs from the OASIS-2 raw layout.

This module contains the dataset-specific logic needed to discover the
intra-session OASIS-2 pairs used for the initial benchmark:

    fixed  = RAW/mpr-1.nifti.hdr
    moving = RAW/mpr-2.nifti.hdr

Example, from benchmarks/registration:

    python datasets/oasis2/make_pairs.py \
        --root /path/to/OAS2_RAW_PART1/OAS2_RAW_PART1 \
        --out data/oasis2_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

SESSION_DIR_RE = re.compile(r"^(OAS2_\d+)_MR(\d+)$")
DEFAULT_FIELDS = [
    "pair_id",
    "fixed_path",
    "moving_path",
]


def has_analyze_pair(hdr_path: Path) -> bool:
    """Return True when the .hdr file and its matching .img file exist."""
    return hdr_path.exists() and hdr_path.with_suffix(".img").exists()


def iter_oasis2_mpr1_mpr2_pairs(root: Path):
    """Yield one mpr-2 to mpr-1 pair for each valid OASIS-2 session folder."""
    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir():
            continue

        match = SESSION_DIR_RE.match(session_dir.name)
        if match is None:
            continue

        raw_dir = session_dir / "RAW"
        fixed_path = raw_dir / "mpr-1.nifti.hdr"
        moving_path = raw_dir / "mpr-2.nifti.hdr"

        if not has_analyze_pair(fixed_path) or not has_analyze_pair(moving_path):
            continue

        yield {
            "pair_id": f"{session_dir.name}_mpr2_to_mpr1",
            "fixed_path": str(fixed_path),
            "moving_path": str(moving_path),
        }


def write_pairs_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build generic registration pairs from OASIS-2 RAW folders."
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path containing OAS2_*_MR* session folders.",
    )
    parser.add_argument(
        "--out",
        default=Path("data/oasis2_pairs.csv"),
        type=Path,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"OASIS-2 root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"OASIS-2 root is not a directory: {root}")

    rows = list(iter_oasis2_mpr1_mpr2_pairs(root))
    write_pairs_csv(rows, args.out)

    print(f"Found {len(rows)} OASIS-2 mpr-2 to mpr-1 pairs")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
