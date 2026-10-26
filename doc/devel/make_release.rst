.. _release-guide:

*********************************
A guide to making a DIPY release
*********************************

A guide for developers who are doing a DIPY release

.. contents:: Contents
   :local:
   :depth: 2

.. _release-checklist:

Major / minor releases (from master)
=====================================

Automated release preparation
-------------------------------

Most of the manual steps below are automated by the ``spin prepare-release``
command.  Run it from the root of the DIPY repository::

    spin prepare-release

The command walks you through each step interactively.  If it is interrupted
you can resume from any step with ``--from-step N``::

    spin prepare-release --from-step 6

You can also override the auto-detected previous tag or skip the version
prompt::

    spin prepare-release --last-tag 1.11.0 --new-version 1.12.0

Available steps (``--from-step`` values):

.. code-block:: text

     1. fetch-tag        – detect the previous release tag
     2. mailmap          – auto-detect duplicate authors, propose .mailmap entries, show full shortlog
     3. version          – choose the new version number
     4. author           – regenerate the AUTHOR file
     5. copyright        – update copyright years in LICENSE and doc/conf.py
     6. release-notes    – generate doc/release_notes/releaseX.Y.Z.rst
     7. changelog        – prepend entry in Changelog
     8. api-changes      – review and update doc/api_changes.rst
     9. index            – add announcement line to doc/index.rst
    10. old-news         – rotate oldest announcements from index.rst → doc/old_news.rst
    11. highlights       – move previous highlights from index.rst → doc/old_highlights.rst
    12. stateoftheart    – add new release to doc/stateoftheart.rst toctree
    13. toolchain        – add new version row to doc/devel/toolchain.rst
    14. version-switcher – promote new version to stable in doc/_static/version_switcher.json
    15. developers       – review and update doc/developers.rst contributor list
    16. pyproject        – set version in pyproject.toml
    17. deprecations     – confirm deprecated code has been removed
    18. doctest          – run Cython / extension module doctests
    19. tests            – run the full test suite
    20. docs             – build HTML documentation (without examples)
    21. tutorials        – build and review individual tutorials with ``spin docs <name>``
    22. website          – deploy docs to docs.dipy.org and verify version switcher

Before running the script, make sure to:

* Review the open list of `dipy issues`_.  Close or label outstanding issues,
  and check whether any should delay the release.

* Verify that no builds are failing on `DIPY Github Actions`_.  The ``PRE``
  build is allowed to fail for PR merges but **must not** fail at release time.

* Check the ``README`` in the root directory and verify all links are still valid.

Notes on individual steps
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 2 – mailmap**
  The script parses ``git log`` to detect duplicate authors automatically
  (same email / different name, or same normalised name / different email).
  Each new candidate entry is shown and you are asked ``yes / no / edit``.
  Accepted entries are appended to ``.mailmap`` and a diff is shown.
  A full ``git shortlog -nse <last_tag>..HEAD`` is printed afterwards for a
  final manual review before you continue.

**Step 6 – release-notes**
  Release notes are generated with ``tools/github_stats.py``::

      python3 tools/github_stats.py <last_tag> > doc/release_notes/release<version>.rst

  You will be paused to add a header and highlights section above the
  auto-generated contributor/PR/issue lists before continuing.

**Step 7 – changelog**
  A skeleton entry is prepended to ``Changelog``.  You are paused to fill in
  the highlights (derived from the release notes) before continuing.

**Step 18 – doctest**
  Runs ``./tools/doctest_extmods.py dipy``.  No output means all doctests
  passed; the exit code is always printed so you can tell the command
  actually ran.  Requires the package to be built in-place
  (``spin build`` or ``pip install --no-build-isolation -e .``).

**Step 19 – tests**
  Runs ``pytest -svv --doctest-modules dipy`` from the repo root.

**Step 20 – docs**
  Runs ``make clean && make html`` inside ``doc/``.  Review the generated
  output for broken tutorials, figures, or API pages before confirming.

  During development you can use ``spin docs`` for faster, selective builds::

      spin docs                      # full build with examples
      spin docs --no-plot            # skip example execution
      spin docs reconst_csa          # build only one tutorial
      spin docs reconst_csa --no-plot  # RST only, no execution

  Tutorial names are matched as regex substrings against the example file
  path; multiple names are joined with ``|`` (union).

Doing the major/minor release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once ``spin prepare-release`` completes successfully:

1. Stage and commit all changes::

    git commit -m "REL: set version to <version>"

2. Open a Pull Request, get it reviewed and merged.

3. After the merge, create an annotated tag on the merge commit::

    git tag -am 'Public release <version>' <version>

4. Build the source distribution::

    git clean -dfx
    python -m build

5. Upload to PyPI::

    pip install twine
    twine upload dist/*

6. Push the tag to trigger wheel builds on CI::

    git push upstream <version>

7. Create a maintenance branch and bump the development version::

    git checkout -b maint/<version>
    # set version to <major>.<minor+1>.0.dev0 in pyproject.toml on master
    git checkout master
    git merge -s ours maint/<version>

8. Announce to the mailing lists.

Release checklist (manual reference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The steps below are the manual equivalent of what ``spin prepare-release``
does, kept here for reference.

* Detect the previous tag::

    git tag | sort -V | tail -1

* Review contributors and update ``.mailmap``::

    git shortlog -nse <last_tag>..HEAD

* Generate release notes::

    python3 tools/github_stats.py <last_tag> > doc/release_notes/release<version>.rst

  Edit the file to add a header and highlights section.

* Prepend an entry to ``Changelog`` with the new version and highlights.

* Add the new announcement line to ``doc/index.rst``.

* Regenerate the ``AUTHOR`` file from git history::

    git log --format='%aN' | sort -u > AUTHOR

* Check copyright years in ``LICENSE`` and ``doc/conf.py``.

* Set the version in ``pyproject.toml``.

* Confirm all deprecated functions past their removal cycle have been removed.

* Run Cython / extension module doctests::

    ./tools/doctest_extmods.py dipy

* Run the full test suite::

    pytest -svv --doctest-modules dipy

* Build the HTML documentation::

    cd doc && make clean && make html && cd ..

* Build and test the DIPY wheels via `DIPY Github Actions`_.  Wheels are
  built automatically on CI when a release tag is pushed.


.. _maint-release-guide:

Patch releases (from a maintenance branch)
===========================================

Patch releases (e.g. ``1.12.1``) are cut from an existing ``maint/X.Y.x``
branch, **not** from master.  They carry only backported bug fixes and require
a much shorter preparation process — most documentation and website steps are
skipped.

Prerequisites
--------------

* The ``maint/X.Y.x`` branch must already exist (created when ``X.Y.0`` was
  released).
* All fixes to be included must already be backported to the branch (use the
  ``backport-maint/X.Y.x`` label on PRs to trigger the automated backport
  workflow, then verify each backport PR was merged).
* Verify that CI is green on ``maint/X.Y.x`` before starting.

Automated preparation
----------------------

Switch to the maintenance branch and run ``spin prepare-release``.  The tool
**auto-detects** that you are on a ``maint/`` branch and activates the reduced
9-step checklist automatically::

    git fetch upstream
    git checkout maint/1.12.x
    git merge --ff-only upstream/maint/1.12.x
    spin prepare-release

You can also pass ``--maint-branch`` explicitly (useful when resuming a
partially-complete run from a detached HEAD or a different branch name)::

    spin prepare-release --maint-branch maint/1.12.x

To resume after an interruption::

    spin prepare-release --from-step 5

Reduced step list (9 steps):

.. code-block:: text

     1. fetch-tag        – detect the previous release tag on this branch
     2. mailmap          – deduplicate authors (.mailmap updated for backport authors)
     3. version          – propose patch bump (X.Y.Z+1); confirm or enter manually
     4. author           – regenerate the AUTHOR file
     5. release-notes    – generate notes filtered to PRs targeting maint/X.Y.x
     6. changelog        – prepend patch entry in Changelog
     7. pyproject        – set version in pyproject.toml
     8. doctest          – run Cython / extension module doctests
     9. tests            – run the full test suite

Notes on maintenance-specific steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 5 – release-notes**
  ``tools/github_stats.py`` is called with ``--branch maint/X.Y.x`` so that
  only pull requests whose *base* (target) branch is the maintenance branch
  are included.  Master-branch PRs do not appear in the list.

  You can also run it manually::

      python3 tools/github_stats.py <last_tag> --branch maint/1.12.x \
          > doc/release_notes/release<version>.rst

  Edit the file to add a short header; no full highlights section is needed
  for a patch release.

**Step 7 – pyproject**
  Changes ``version = "X.Y.Z.dev0"`` to ``version = "X.Y.Z"`` in
  ``pyproject.toml``.

Doing the patch release
------------------------

Once ``spin prepare-release`` finishes, follow these steps:

1. Commit the release preparation changes::

    git add pyproject.toml doc/release_notes/release<version>.rst Changelog .mailmap
    git commit -m "REL: set version to <version>"

2. Push to ``maint/X.Y.x`` (directly or via a short-lived PR targeting that
   branch — **not** master)::

    git push upstream maint/X.Y.x

3. After the commit lands on the branch, create an annotated tag::

    git fetch upstream
    git merge --ff-only upstream/maint/X.Y.x
    git tag -am 'Public release <version>' <version>

4. Build the source distribution::

    git clean -dfx
    python -m build --sdist

5. Upload the sdist to PyPI::

    twine upload dist/dipy-<version>.tar.gz

6. Push the tag::

    git push upstream <version>

7. Trigger wheel builds via ``workflow_dispatch`` on the ``nightly.yml``
   workflow — the ``upload_anaconda`` job will be skipped (it only runs for
   master), but all wheels will be built and saved as artifacts::

    gh workflow run nightly.yml --repo dipy/dipy --field branch_or_tag=<version>

   Wait for all platform builds to finish (~40–60 min), then download and
   upload the wheels::

    gh run download <run-id> --dir dist-wheels/
    twine upload dist-wheels/**/*.whl

8. Create the GitHub release::

    gh release create <version> --repo dipy/dipy --title "DIPY <version>"

9. Bump the maintenance branch to the next patch dev version::

    # edit pyproject.toml: version = "X.Y.(Z+1).dev0"
    git commit -am "REL: bump version to X.Y.(Z+1).dev0"
    git push upstream maint/X.Y.x

10. Announce to the mailing lists.

.. note::

   Do **not** update ``doc/index.rst``, ``doc/stateoftheart.rst``,
   ``doc/devel/toolchain.rst``, or ``doc/_static/version_switcher.json`` for a
   patch release.  These files track major/minor versions only.

Other stuff that needs doing for the release
============================================

* Build and upload the HTML docs to the GitHub Pages website::

    make upload

* Announce to the mailing lists.


.. include:: ../links_names.inc
