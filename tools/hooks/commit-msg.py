#!/usr/bin/env python3

# This script is borrowed with minor modifications from the equivalent one in
# ITK (https://github.com/InsightSoftwareConsortium/ITK), and as such, it keeps
# its license (see below). Thanks to all ITK contributors and maintainers.
#
# ==========================================================================
#
#   Copyright NumFOCUS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ==========================================================================

import os
import re
import sys
from pathlib import Path


DEFAULT_LINE_LENGTH: int = 78
MIN_SUBJ_LINE_LENGTH: int = 15


def die(message, commit_msg_path):
    print("commit-msg hook failure", file=sys.stderr)
    print("-----------------------", file=sys.stderr)
    print(message, file=sys.stderr)
    print("-----------------------", file=sys.stderr)
    print(
        f"""
To continue editing a message, run the command:
  git commit -e -F "{commit_msg_path}"
(assuming your working directory is at the top).""",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    git_dir_path: Path = Path(os.environ.get("GIT_DIR", ".git")).resolve()
    commit_msg_path: Path = git_dir_path / "COMMIT_MSG"

    if len(sys.argv) < 2:
        die(f"Usage: {sys.argv[0]} <git_commit_message_file>", commit_msg_path)

    input_file: Path = Path(sys.argv[1])
    if not input_file.exists():
        die(
            f"Missing input file {sys.argv[1]} for {sys.argv[0]} processing",
            commit_msg_path,
        )

    original_input_file_lines: list[str] = []
    with open(input_file) as f_in:
        original_input_file_lines = f_in.readlines()

    input_file_lines: list[str] = []
    for test_line in original_input_file_lines:
        test_line = test_line.strip()
        is_empty_line_before_subject: bool = (
            len(input_file_lines) == 0 and len(test_line) == 0
        )
        if test_line.startswith("#") or is_empty_line_before_subject:
            continue
        input_file_lines.append(f"{test_line}\n")

    with open(commit_msg_path, "w") as f_out:
        f_out.writelines(input_file_lines)

    subject_line: str = input_file_lines[0]

    if len(subject_line) < MIN_SUBJ_LINE_LENGTH:
        die(
            f"The first line must be at least {MIN_SUBJ_LINE_LENGTH} characters:\n--------\n{subject_line}\n--------",
            commit_msg_path,
        )
    if (
        len(subject_line) > DEFAULT_LINE_LENGTH
        and not subject_line.startswith("Merge ")
        and not subject_line.startswith("Revert ")
    ):
        die(
            f"The first line may be at most {DEFAULT_LINE_LENGTH} characters:\n"
            + "-" * DEFAULT_LINE_LENGTH
            + f"\n{subject_line}\n"
            + "-" * DEFAULT_LINE_LENGTH,
            commit_msg_path,
        )
    if re.match(r"^[ \t]|[ \t]$", subject_line):
        die(
            f"The first line may not have leading or trailing space:\n[{subject_line}]",
            commit_msg_path,
        )
    if re.match(r"\.$", subject_line):
        die(
            f"The first line may not have a trailing period:\n[{subject_line}]",
            commit_msg_path,
        )
    if not re.match(
        r"^(Merge|Revert|BF:|RF:|NF:|BW:|OPT:|CI:|MNT:|DOC:|TEST:|STYLE:|WIP:)\s", subject_line
    ):
        die(
            f"""Start DIPY commit messages with a standard prefix (and a space):
  BF:     - bug fix
  RF:     - refactoring
  NF:     - new feature
  BW:     - addresses backward-compatibility
  OPT:    - optimization
  CI:     - continuous integration
  MNT:    - maintenance tasks, such as release preparation
  DOC:    - for all kinds of documentation related commits
  TEST:   - for adding or changing tests
  STYLE:  - PEP8 conformance, whitespace changes etc that do not affect function
  WIP:    - Work In Progress not ready for merge
To reference GitHub issue XXXX, add "Issue #XXXX" to the PR message.
If the issue addresses an open issue, add "Closes #XXXX" to the PR message.""",
            commit_msg_path,
        )
    del subject_line

    if len(input_file_lines) > 1:
        second_line: str = input_file_lines[
            1
        ].strip()  # Remove whitespace at beginning and end
        if len(second_line) == 0:
            input_file_lines[1] = "\n"  # Replace line with only newline
        else:
            die(
                f'The second line of the commit message must be empty:\n"{second_line}" with length {len(second_line)}',
                commit_msg_path,
            )
        del second_line


if __name__ == "__main__":
    main()
