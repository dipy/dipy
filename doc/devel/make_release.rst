.. _release-guide:

*********************************
A guide to making a DIPY release
*********************************

A guide for developers who are doing a DIPY release

.. _release-checklist:

Automated release preparation
==============================

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

     1. fetch-tag       – detect the previous release tag
     2. mailmap         – auto-detect duplicate authors, propose .mailmap entries, then show full shortlog for manual review
     3. version         – choose the new version number
     4. author          – regenerate the AUTHOR file
     5. copyright       – update copyright years in LICENSE and doc/conf.py
     6. release-notes   – generate doc/release_notes/releaseX.Y.Z.rst
     7. changelog       – prepend entry in Changelog
     8. index           – add announcement line to doc/index.rst
     9. pyproject       – set version in pyproject.toml
    10. deprecations    – confirm deprecated code has been removed
    11. doctest         – run Cython / extension module doctests (exit code shown; no output means all passed)
    12. tests           – run the full test suite
    13. docs            – build HTML documentation

Before running the script, make sure to:

* Review the open list of `dipy issues`_.  Close or label outstanding issues,
  and check whether any should delay the release.

* Verify that no builds are failing on `DIPY Github Actions`_.  The ``PRE``
  build is allowed to fail for PR merges but **must not** fail at release time.

* Check the ``README`` in the root directory and verify all links are still valid.

Notes on individual steps
--------------------------

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

**Step 11 – doctest**
  Runs ``./tools/doctest_extmods.py dipy``.  No output means all doctests
  passed; the exit code is always printed so you can tell the command
  actually ran.  Requires the package to be built in-place
  (``spin build`` or ``pip install --no-build-isolation -e .``).

**Step 12 – tests**
  Runs ``pytest -svv --doctest-modules dipy`` from the repo root.

**Step 13 – docs**
  Runs ``make clean && make html`` inside ``doc/``.  Review the generated
  output for broken tutorials, figures, or API pages before confirming.

Release checklist (manual reference)
=====================================

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

    git log --format=’%aN’ | sort -u > AUTHOR

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

Doing the release
=================

Once ``spin prepare-release`` completes successfully:

1. Stage and commit all changes::

    git commit -m "REL: set version to <version>"

2. Open a Pull Request, get it reviewed and merged.

3. After the merge, create an annotated tag on the merge commit::

    git tag -am ‘Public release <version>’ <version>

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

Other stuff that needs doing for the release
============================================

* Build and upload the HTML docs to the GitHub Pages website::

    make upload

* Announce to the mailing lists.


.. include:: ../links_names.inc
