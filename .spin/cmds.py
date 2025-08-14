"""Additional Command-line interface for spin."""

import datetime
import os
import re
import shutil
import subprocess

import click
from packaging.version import Version
from spin import util
from spin.cmds import meson


# From scipy: benchmarks/benchmarks/common.py
def _set_mem_rlimit(max_mem=None):
    """Set address space rlimit."""
    import resource

    import psutil

    mem = psutil.virtual_memory()

    if max_mem is None:
        max_mem = int(mem.total * 0.7)
    cur_limit = resource.getrlimit(resource.RLIMIT_AS)
    if cur_limit[0] > 0:
        max_mem = min(max_mem, cur_limit[0])

    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, cur_limit[1]))
    except ValueError:
        # on macOS may raise: current limit exceeds maximum limit
        pass


def _commit_to_sha(commit):
    p = util.run(["git", "rev-parse", commit], output=False, echo=False)
    if p.returncode != 0:
        raise click.ClickException(f"Could not find SHA matching commit `{commit}`")

    return p.stdout.decode("ascii").strip()


def _dirty_git_working_dir():
    # Changes to the working directory
    p0 = util.run(["git", "diff-files", "--quiet"])

    # Staged changes
    p1 = util.run(["git", "diff-index", "--quiet", "--cached", "HEAD"])

    return p0.returncode != 0 or p1.returncode != 0


def _run_asv(cmd):
    # Always use ccache, if installed
    PATH = os.environ["PATH"]
    # EXTRA_PATH = os.pathsep.join([
    #     '/usr/lib/ccache', '/usr/lib/f90cache',
    #     '/usr/local/lib/ccache', '/usr/local/lib/f90cache'
    # ])
    env = os.environ
    env["PATH"] = f"EXTRA_PATH:{PATH}"

    # Control BLAS/LAPACK threads
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    # Limit memory usage
    try:
        _set_mem_rlimit()
    except (ImportError, RuntimeError):
        pass

    util.run(cmd, cwd="benchmarks", env=env)


@click.command()
@click.option(
    "--tests",
    "-t",
    default=None,
    metavar="TESTS",
    multiple=True,
    help="Which tests to run",
)
@click.option(
    "--compare",
    "-c",
    is_flag=True,
    default=False,
    help="Compare benchmarks between the current branch and main "
    "(unless other branches specified). "
    "The benchmarks are each executed in a new isolated "
    "environment.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option(
    "--quick",
    "-q",
    is_flag=True,
    default=False,
    help="Run each benchmark only once (timings won't be accurate)",
)
@click.argument("commits", metavar="", required=False, nargs=-1)
@click.pass_context
def bench(ctx, tests, compare, verbose, quick, commits):
    """🏋 Run benchmarks.

    \b
    Examples:

    \b
    $ spin bench -t bench_lib
    $ spin bench -t bench_random.Random
    $ spin bench -t Random -t Shuffle

    Two benchmark runs can be compared.
    By default, `HEAD` is compared to `main`.
    You can also specify the branches/commits to compare:

    \b
    $ spin bench --compare
    $ spin bench --compare main
    $ spin bench --compare main HEAD

    You can also choose which benchmarks to run in comparison mode:

    $ spin bench -t Random --compare
    """
    if not commits:
        commits = ("main", "HEAD")
    elif len(commits) == 1:
        commits = commits + ("HEAD",)
    elif len(commits) > 2:
        raise click.ClickException("Need a maximum of two revisions to compare")

    bench_args = []
    for t in tests:
        bench_args += ["--bench", t]

    if verbose:
        bench_args = ["-v"] + bench_args

    if quick:
        bench_args = ["--quick"] + bench_args

    if not compare:
        # No comparison requested; we build and benchmark the current version

        click.secho(
            "Invoking `build` prior to running benchmarks:",
            bold=True,
            fg="bright_green",
        )
        ctx.invoke(meson.build)

        meson._set_pythonpath()

        p = util.run(
            ["python", "-c", "import dipy; print(dipy.__version__)"],
            cwd="benchmarks",
            echo=False,
            output=False,
        )
        os.chdir("..")

        dipy_ver = p.stdout.strip().decode("ascii")
        click.secho(
            f"Running benchmarks on DIPY {dipy_ver}", bold=True, fg="bright_green"
        )
        cmd = ["asv", "run", "--dry-run", "--show-stderr", "--python=same"] + bench_args
        _run_asv(cmd)
    else:
        # Ensure that we don't have uncommitted changes
        commit_a, commit_b = [_commit_to_sha(c) for c in commits]

        if commit_b == "HEAD" and _dirty_git_working_dir():
            click.secho(
                "WARNING: you have uncommitted changes --- "
                "these will NOT be benchmarked!",
                fg="red",
            )

        cmd_compare = (
            [
                "asv",
                "continuous",
                "--factor",
                "1.05",
            ]
            + bench_args
            + [commit_a, commit_b]
        )
        _run_asv(cmd_compare)


@click.command()
@click.argument("tutorials", nargs=-1, metavar="[TUTORIAL]...")
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="Run `make clean` before building.",
)
@click.option(
    "--plot/--no-plot",
    "plot",
    default=True,
    help="Execute examples (default: yes). --no-plot skips example execution.",
)
def docs(*, tutorials, clean, plot):
    """📖 Build Sphinx documentation.

    Optionally restrict the build to one or more named tutorials.

    \b
    Examples:

    \b
    $ spin docs                          # full build with examples
    $ spin docs --no-plot                # skip example execution (fast)
    $ spin docs --clean                  # clean before building
    $ spin docs reconst_csa              # build only reconst_csa tutorial
    $ spin docs reconst_csa reconst_dti  # build only those two tutorials
    $ spin docs reconst_csa --no-plot    # build tutorial RST without running it
    """
    if clean:
        _run_shell("make -C doc clean", check=False)

    sphinx_opts = ["-j auto"]
    if not plot:
        sphinx_opts.append("-D plot_gallery=0")
    if tutorials:
        pattern = "|".join(tutorials)
        sphinx_opts.append(f'-D sphinx_gallery_conf.filename_pattern="{pattern}"')

    opts_str = " ".join(sphinx_opts)
    _run_shell(f'make -C doc html SPHINXOPTS="{opts_str}"')


@click.command()
def clean():
    """🧹 Remove build and install folder."""
    build_dir = "build"
    install_dir = "build-install"
    print(f"Removing `{build_dir}`")
    if os.path.isdir(build_dir):
        shutil.rmtree(build_dir)
    print(f"Removing `{install_dir}`")
    if os.path.isdir(install_dir):
        shutil.rmtree(install_dir)


# ---------------------------------------------------------------------------
# Release preparation helpers
# ---------------------------------------------------------------------------


def _run_shell(cmd, *, check=True):
    """Run a shell command, echo it, and print its exit status.

    Parameters
    ----------
    cmd : str
        Shell command to run.
    check : bool
        Whether to raise on non-zero exit code.
    """
    click.secho(f"$ {cmd}", fg="cyan")
    result = subprocess.run(cmd, shell=True, check=check)
    rc = result.returncode
    color = "green" if rc == 0 else "red"
    click.secho(f"--- exit code: {rc} ---", fg=color)


def _get_latest_tag():
    """Fetch and return the latest DIPY release tag.

    Returns
    -------
    str or None
        The latest semver tag, or None if none found.
    """
    try:
        subprocess.run(
            ["git", "fetch", "https://github.com/dipy/dipy.git", "--tags"],
            check=True,
            capture_output=True,
        )
        tags = subprocess.check_output(["git", "tag"]).decode().splitlines()
        pattern = r"^\d+\.\d+\.\d+([a-zA-Z0-9]+)?$"
        release_tags = [t for t in tags if re.match(pattern, t)]
        if not release_tags:
            return None
        release_tags.sort(key=lambda v: Version(v))
        return release_tags[-1]
    except subprocess.CalledProcessError:
        return None


def _update_mailmap(*, last_tag, mailmap_path=".mailmap"):
    """Detect duplicate authors since last_tag and propose .mailmap entries.

    Scans ``git log`` for (name, email) pairs and identifies:

    * Same email with different display names (clear duplicates).
    * Same normalised name (lowercase, no punctuation) with different emails
      (likely the same person using multiple addresses).

    Already-mapped entries are skipped. New proposals are shown to the user
    who can accept or reject each one before they are appended to
    ``.mailmap``.  A full ``git shortlog -nse`` is shown afterwards for a
    final manual review.

    Parameters
    ----------
    last_tag : str
        The previous release tag used to scope the shortlog display.
    mailmap_path : str
        Path to the ``.mailmap`` file.
    """
    click.secho("--- Scanning git log for duplicate authors ---", bold=True)

    # 1. Collect all (canonical_name, canonical_email) pairs ever seen
    raw = (
        subprocess.check_output(["git", "log", "--format=%aN|%aE"])
        .decode()
        .splitlines()
    )

    # map email → set of names, name → set of emails
    email_to_names: dict[str, set[str]] = {}
    name_to_emails: dict[str, set[str]] = {}
    for line in raw:
        if "|" not in line:
            continue
        name, email = line.split("|", 1)
        name, email = name.strip(), email.strip().lower()
        email_to_names.setdefault(email, set()).add(name)
        # normalise: lowercase, keep only alphanum + spaces
        norm = re.sub(r"[^a-z0-9 ]", "", name.lower())
        name_to_emails.setdefault(norm, set()).add(email)

    # 2. Read existing .mailmap so we skip already-handled entries
    existing_lines: list[str] = []
    if os.path.isfile(mailmap_path):
        with open(mailmap_path) as f:
            existing_lines = f.readlines()
    existing_text = "".join(existing_lines)

    # 3. Build proposals
    proposals: list[str] = []

    # Same email → multiple names: pick the longest/most complete name as canonical
    for email, names in email_to_names.items():
        if len(names) <= 1:
            continue
        canonical = max(names, key=len)
        for alias in names:
            if alias == canonical:
                continue
            entry = f"{canonical} <{email}> {alias} <{email}>\n"
            if alias not in existing_text and entry not in existing_text:
                proposals.append(entry)

    # Same normalised name → multiple emails: flag for user to pick canonical
    for norm, emails in name_to_emails.items():
        if len(emails) <= 1:
            continue
        # gather actual display names for this norm
        display_names: set[str] = set()
        for line in raw:
            if "|" not in line:
                continue
            n, e = line.split("|", 1)
            n, e = n.strip(), e.strip().lower()
            if re.sub(r"[^a-z0-9 ]", "", n.lower()) == norm and e in emails:
                display_names.add(n)
        canonical_name = max(display_names, key=len)
        canonical_email = min(emails)  # placeholder; user should confirm
        for email in emails:
            if email == canonical_email:
                continue
            alias_name = canonical_name  # best guess
            entry = f"{canonical_name} <{canonical_email}> {alias_name} <{email}>\n"
            if email not in existing_text and entry not in existing_text:
                proposals.append(entry)

    # 4. Present proposals to the user
    if not proposals:
        click.secho("No new duplicate authors detected.", fg="green")
    else:
        click.secho(
            f"\nFound {len(proposals)} potential duplicate(s) not yet in "
            f"{mailmap_path}:",
            fg="yellow",
        )
        accepted: list[str] = []
        for proposal in proposals:
            click.echo(f"\n  {proposal.rstrip()}")
            answer = (
                click.prompt(
                    "  Add this entry? [yes/no/edit]",
                    default="yes",
                )
                .strip()
                .lower()
            )
            if answer in ("yes", "y"):
                accepted.append(proposal)
            elif answer in ("edit", "e"):
                edited = click.prompt("  Enter corrected entry").strip()
                if edited:
                    accepted.append(edited + "\n")
            # "no" → skip

        if accepted:
            with open(mailmap_path, "a", newline="\n") as f:
                f.writelines(accepted)
            click.secho(f"Added {len(accepted)} entries to {mailmap_path}.", fg="green")
            _run_shell(f"git diff {mailmap_path}", check=False)

    # 5. Show full shortlog for final manual review
    click.secho(
        f"\n--- Full contributor list since {last_tag} (manual review) ---",
        bold=True,
    )
    _run_shell(f"git shortlog -nse {last_tag}..HEAD", check=False)
    input(
        "\nReview the list above. Make any remaining manual edits to .mailmap, "
        "then press Enter to continue..."
    )


def _update_author(*, file_path="AUTHOR"):
    """Regenerate the AUTHOR file from the full git log.

    Parameters
    ----------
    file_path : str
        Path to the AUTHOR file.
    """
    click.secho(f"--- Updating {file_path} ---", bold=True)
    git = subprocess.Popen(["git", "log", "--format=%aN"], stdout=subprocess.PIPE)
    raw = git.stdout.read().decode().split("\n")
    authors = sorted({a for a in raw if a and a != "dependabot[bot]"})
    with open(file_path, "w", newline="\n") as f:
        f.write("\n".join(authors) + "\n")
    _run_shell(f"git diff {file_path}", check=False)


def _check_copyright_year():
    """Update copyright year in LICENSE and doc/conf.py if needed."""
    current_year = datetime.datetime.now().year

    # LICENSE
    click.secho("--- Checking LICENSE copyright year ---", bold=True)
    with open("LICENSE", "r") as f:
        content = f.read()
    pattern = r"(Copyright \(c\) \d{4}-)(\d{4})"
    match = re.search(pattern, content)
    if match:
        if int(match.group(2)) != current_year:
            click.echo(f"Updating LICENSE year to {current_year}")
            updated = re.sub(pattern, rf"\g<1>{current_year}", content)
            with open("LICENSE", "w", newline="\n") as f:
                f.write(updated)
            _run_shell("git diff LICENSE", check=False)
        else:
            click.echo("LICENSE year is up to date.")
    else:
        click.secho(
            "Could not parse copyright year in LICENSE – update manually.", fg="yellow"
        )

    # doc/conf.py
    click.secho("--- Checking doc/conf.py copyright year ---", bold=True)
    conf_path = "doc/conf.py"
    with open(conf_path, "r") as f:
        content = f.read()
    pattern = r"(Copyright \d{4}-)(\d{4})(,DIPY)"
    match = re.search(pattern, content)
    if match:
        if int(match.group(2)) != current_year:
            click.echo(f"Updating doc/conf.py year to {current_year}")
            updated = re.sub(pattern, rf"\g<1>{current_year}\g<3>", content)
            with open(conf_path, "w", newline="\n") as f:
                f.write(updated)
            _run_shell(f"git diff {conf_path}", check=False)
        else:
            click.echo("doc/conf.py copyright year is up to date.")
    else:
        click.secho(
            "Could not parse copyright year in doc/conf.py – update manually.",
            fg="yellow",
        )


def _generate_release_notes(*, last_tag, new_version):
    """Generate release notes via tools/github_stats.py.

    Parameters
    ----------
    last_tag : str
        The previous release tag (e.g. ``"1.11.0"``).
    new_version : str
        The new version string (e.g. ``"1.12.0"``).
    """
    notes_path = f"doc/release_notes/release{new_version}.rst"
    click.secho(f"--- Generating release notes → {notes_path} ---", bold=True)
    _run_shell(f"python3 tools/github_stats.py {last_tag} > {notes_path}", check=False)
    click.secho(
        f"\nPlease open {notes_path} and add a header / highlights section "
        "above the auto-generated stats.",
        fg="yellow",
    )
    input("Press Enter when done editing the release notes...")


def _update_changelog(*, new_version, file_path="Changelog"):
    """Prepend a new entry for new_version in the Changelog.

    Parameters
    ----------
    new_version : str
        The new version string.
    file_path : str
        Path to the Changelog file.
    """
    click.secho(f"--- Updating {file_path} ---", bold=True)
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    entry = f"\n* {new_version} ({today})\n\n- TODO: add highlights here.\n"
    with open(file_path, "r") as f:
        content = f.read()
    marker = "Releases\n~~~~~~~~"
    idx = content.find(marker)
    if idx == -1:
        click.secho(
            f"Could not find 'Releases' section in {file_path} – skipping.", fg="yellow"
        )
        return
    insert_at = idx + len(marker)
    updated = content[:insert_at] + entry + content[insert_at:]
    with open(file_path, "w", newline="\n") as f:
        f.write(updated)
    _run_shell(f"git diff {file_path}", check=False)
    click.secho(f"Please open {file_path} and fill in the highlights.", fg="yellow")
    input("Press Enter when done...")


def _update_index_announcements(*, new_version, index_path="doc/index.rst"):
    """Prepend a release announcement line in doc/index.rst.

    Parameters
    ----------
    new_version : str
        The new version string (e.g. ``"1.12.0"``).
    index_path : str
        Path to the doc/index.rst file.
    """
    click.secho(f"--- Updating Announcements in {index_path} ---", bold=True)
    release_date = datetime.date.today().strftime("%B %d, %Y")
    major_minor = ".".join(new_version.split(".")[:2])
    note_file = f"release{major_minor}"
    entry = (
        f"- :doc:`DIPY {new_version} <release_notes/{note_file}>` "
        f"released {release_date}.\n"
    )
    with open(index_path, "r") as f:
        lines = f.readlines()
    insert_after = None
    for i, line in enumerate(lines):
        if line.strip() == "Announcements":
            for j in range(i, min(i + 5, len(lines))):
                stripped = lines[j].strip()
                if stripped and all(c == "*" for c in stripped):
                    insert_after = j + 1
                    break
            break
    if insert_after is None:
        click.secho(
            f"Could not find Announcements section in {index_path} – skipping.",
            fg="yellow",
        )
        return
    lines.insert(insert_after, entry)
    with open(index_path, "w", newline="\n") as f:
        f.writelines(lines)
    _run_shell(f"git diff {index_path}", check=False)


def _update_pyproject_version(*, new_version, file_path="pyproject.toml"):
    """Set the version field in pyproject.toml.

    Parameters
    ----------
    new_version : str
        The new version string.
    file_path : str
        Path to pyproject.toml.
    """
    click.secho(f"--- Updating {file_path} version → {new_version} ---", bold=True)
    with open(file_path, "r") as f:
        content = f.read()
    updated = re.sub(
        r'^version = ".*?"',
        f'version = "{new_version}"',
        content,
        flags=re.MULTILINE,
    )
    if updated == content:
        click.secho(
            "Could not find version field in pyproject.toml – update manually.",
            fg="yellow",
        )
        return
    with open(file_path, "w", newline="\n") as f:
        f.write(updated)
    _run_shell(f"git diff {file_path}", check=False)


def _confirm(prompt):
    """Loop until the user answers yes.

    Parameters
    ----------
    prompt : str
        Question to display.
    """
    while True:
        answer = input(f"{prompt} [yes/no]: ").strip().lower()
        if answer in ("yes", "y"):
            return
        if answer in ("no", "n"):
            click.echo("Please complete that step before proceeding.")
        else:
            click.echo("Please answer 'yes' or 'no'.")


# ---------------------------------------------------------------------------
# Release preparation spin command
# ---------------------------------------------------------------------------

RELEASE_STEPS = [
    "fetch-tag",
    "mailmap",
    "version",
    "author",
    "copyright",
    "release-notes",
    "changelog",
    "index",
    "pyproject",
    "deprecations",
    "doctest",
    "tests",
    "docs",
]

STEP_HELP = "\n".join(f"  {i + 1:>2}. {s}" for i, s in enumerate(RELEASE_STEPS))


@click.command(name="prepare-release")
@click.option(
    "--from-step",
    default=1,
    metavar="N",
    help=("Resume from step N (1-based). Steps:\n" + STEP_HELP),
    type=click.IntRange(1, len(RELEASE_STEPS)),
)
@click.option(
    "--last-tag", default=None, metavar="TAG", help="Override auto-detected last tag."
)
@click.option(
    "--new-version", default=None, metavar="VER", help="Skip version prompt, use VER."
)
def prepare_release(*, from_step, last_tag, new_version):
    (
        """🚀 Prepare a DIPY release (interactive checklist).

    Runs each preparation step in order. Use --from-step N to resume
    after an interruption.

    \b
    Steps:
    """
        + STEP_HELP
    )  # noqa: E501
    if not os.path.isdir("dipy") or not os.path.isfile("pyproject.toml"):
        raise click.ClickException("Run this command from the root 'dipy' directory.")

    # State shared across steps
    ctx = {"last_tag": last_tag, "new_version": new_version}

    def skip(step_n):
        return step_n < from_step

    # ── Step 1: Fetch latest tag ──────────────────────────────────────────
    step = 1
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Fetch latest tag",
            bold=True,
            fg="green",
        )
        if ctx["last_tag"] is None:
            ctx["last_tag"] = _get_latest_tag()
        if ctx["last_tag"] is None:
            ctx["last_tag"] = click.prompt(
                "Could not detect last tag. Enter it manually (e.g., 1.11.0)"
            )
        click.echo(f"Last release tag: {ctx['last_tag']}")
    elif ctx["last_tag"] is None:
        ctx["last_tag"] = click.prompt(
            "Enter the last release tag (required, e.g., 1.11.0)"
        )

    # ── Step 2: Update .mailmap / contributors ────────────────────────────
    step = 2
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Update .mailmap and contributors",
            bold=True,
            fg="green",
        )
        _update_mailmap(last_tag=ctx["last_tag"])

    # ── Step 3: Determine new version ────────────────────────────────────
    step = 3
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Determine new version",
            bold=True,
            fg="green",
        )
        if ctx["new_version"] is None:
            last_ver = Version(ctx["last_tag"])
            major, minor, patch = last_ver.release
            proposed = f"{major}.{minor + 1}.0"
            while True:
                entered = click.prompt("Enter new version", default=proposed)
                if re.match(r"^\d+\.\d+\.\d+([a-zA-Z0-9]+)?$", entered):
                    ctx["new_version"] = entered
                    break
                click.echo("Version must follow semver (e.g., 1.12.0 or 1.12.0rc1).")
        click.echo(f"New version: {ctx['new_version']}")
    elif ctx["new_version"] is None:
        ctx["new_version"] = click.prompt(
            "Enter the new version (required, e.g., 1.12.0)"
        )

    # ── Step 4: Update AUTHOR file ────────────────────────────────────────
    step = 4
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Update AUTHOR file",
            bold=True,
            fg="green",
        )
        _update_author()

    # ── Step 5: Check copyright years ─────────────────────────────────────
    step = 5
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Check copyright years",
            bold=True,
            fg="green",
        )
        _check_copyright_year()

    # ── Step 6: Generate release notes ────────────────────────────────────
    step = 6
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Generate release notes",
            bold=True,
            fg="green",
        )
        _generate_release_notes(
            last_tag=ctx["last_tag"], new_version=ctx["new_version"]
        )

    # ── Step 7: Update Changelog ──────────────────────────────────────────
    step = 7
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Update Changelog",
            bold=True,
            fg="green",
        )
        _update_changelog(new_version=ctx["new_version"])

    # ── Step 8: Update doc/index.rst announcements ────────────────────────
    step = 8
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Update doc/index.rst announcements",
            bold=True,
            fg="green",
        )
        _update_index_announcements(new_version=ctx["new_version"])

    # ── Step 9: Update pyproject.toml version ─────────────────────────────
    step = 9
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Update pyproject.toml version",
            bold=True,
            fg="green",
        )
        _update_pyproject_version(new_version=ctx["new_version"])

    # ── Step 10: Check deprecations ───────────────────────────────────────
    step = 10
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Check deprecations",
            bold=True,
            fg="green",
        )
        _confirm(
            "Have you checked deprecated functions/modules and removed"
            " those past their cycle?"
        )

    # ── Step 11: Run Cython/extension doctests ────────────────────────────
    step = 11
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Run extension module doctests",
            bold=True,
            fg="green",
        )
        _run_shell("./tools/doctest_extmods.py dipy", check=False)
        _confirm("Did the extension module doctests pass?")

    # ── Step 12: Run test suite ───────────────────────────────────────────
    step = 12
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Run test suite",
            bold=True,
            fg="green",
        )
        _run_shell("pytest -svv --doctest-modules dipy", check=False)
        _confirm("Did the test suite pass?")

    # ── Step 13: Build documentation ─────────────────────────────────────
    step = 13
    if not skip(step):
        click.secho(
            f"\n[Step {step}/{len(RELEASE_STEPS)}] Build documentation",
            bold=True,
            fg="green",
        )
        _run_shell("cd doc && make clean && make html && cd ..", check=False)
        _confirm("Have you reviewed the generated docs (tutorials, figures, API)?")

    v = ctx["new_version"]
    click.secho("\n✅ Release preparation complete!", bold=True, fg="bright_green")
    click.echo(
        f"\nNext steps:\n"
        f"  1. Stage and commit: git commit -m 'REL: set version to {v}'\n"
        f"  2. Open a Pull Request and get it merged.\n"
        f"  3. After merge, tag:  git tag -am 'Public release {v}' {v}\n"
        f"  4. Build source dist: git clean -dfx && python -m build\n"
        f"  5. Upload to PyPI:    twine upload dist/*\n"
        f"  6. Push tag:          git push upstream {v}\n"
        f"  7. Create maint branch: git checkout -b maint/{v}\n"
        f"  8. Bump master version to {v.rsplit('.', 1)[0]}.dev0 in pyproject.toml.\n"
        f"  9. Announce on mailing lists.\n"
    )
