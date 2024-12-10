.. _release-guide:

*********************************
A guide to making a DIPY release
*********************************

A guide for developers who are doing a DIPY release

.. _release-checklist:

Release checklist
=================

* Review the open list of `dipy issues`_.  Check whether there are
  outstanding issues that can be closed, and whether there are any issues that
  should delay the release.  Label them !

* Check whether there are no build failing on `DIPY Github Actions`_. Indeed, ``PRE`` build is
  allowed to fail and does not block a PR merge but it should block release !
  So make sure that ``PRE`` build is not failing.

* Generate, Review and update the release notes. Run the following command from
  the root directory of DIPY::

    python3 tools/github_stats.py 1.7.0 > doc/release_notes/release1.8.rst

  where ``1.7.0`` was the last release tag name.

* Review and update the :file:`Changelog` file.  Get a partial list of
  contributors with something like::

      git shortlog -ns 0.1.7.0..

  where ``1.7.0`` was the last release tag name.

  Then manually go over ``git shortlog 0.7.0..`` to make sure the release notes
  are as complete as possible and that every contributor was recognized.

* Use the opportunity to update the ``.mailmap`` file if there are any duplicate
  authors listed from ``git shortlog -nse``.

* Add any new authors to the ``AUTHORS`` file.

* Check the copyright years in ``doc/conf.py`` and ``LICENSE``

* Check the examples - we really need an automated check here.

* Check the ``pyx`` file doctests with::

    ./tools/doctest_extmods.py dipy

  We really need an automated run of these using the buildbots, but we haven't
  done it yet.

* Check the ``README`` in the root directory. Check all the links are still valid.

* Check all the DIPY builds are green on the nipy `DIPY Github Actions`_

* If you have `DIPY Github Actions`_ building set up you might want to push the code in its
  current state to a branch that will build, e.g.::

    git branch -D pre-release-test # in case branch already exists
    git co -b pre-release-test

* Clean and compile::

    make distclean
    git clean -fxd
    python3 -m build  or  pip install --no-build-isolation  -e .

* Make sure all tests pass on your local machine (from the ``<dipy root>`` directory)::

    cd ..
    pytest -svv --with-doctest --pyargs dipy
    cd dipy # back to the root directory

* Check the documentation doctests::

    cd doc
    make doctest
    cd ..

  At the moment this generates lots of errors from the autodoc documentation
  running the doctests in the code, where the doctests pass when run in pytest -
  we should find out why this is at some point, but leave it for now.

* Build and test the DIPY wheels.  See the `wheel builder README
  <https://github.com/MacPython/dipy-wheels>`_ for instructions.  In summary,
  clone the wheel-building repo, edit the ``.github/workflows/wheel.yml``
  text files (if present) with the branch or commit for the release, commit
  and then push back up to github.  This will trigger a wheel build and test
  on macOS, Linux and Windows. Check the build has passed on the Github Actions
  interface at https://github.com/MacPython/dipy-wheels/actions.  You'll need
  commit privileges to the ``dipy-wheels`` repo; ask Matthew Brett or on the
  mailing list if you do not have them.

* The release should now be ready.

Doing the release
=================

Doing the release! This has two steps:

* build and upload the DIPY wheels;
* make and upload the DIPY source release.

The trick here is to get all the testing, pushing to upstream done *before* you
do the final release commit.  There should be only one commit with the release
version number, so you might want to make the release commit on your local
machine, push to `dipy pypi`_, review, fix, rebase, until all is good.  Then and only
then do you push to upstream on github.

* Make the release commit.  Edit :file:`pyproject.toml` and to get the correct version
  number and commit it with a message like REL: set version to <version-number>.
  Don’t push this commit to the DIPY_ repo yet.;

* Finally tag the release locally with git tag <v1.x.y>. Continue with building
  release artifacts (next section). Only push the release commit to the DIPY_
  repo once you have built the sdists and docs successfully.
  Then continue with building wheels. Only push the release tag to the repo once
  all wheels have been built successfully on `DIPY Github Actions`_.

* For the wheel build / upload, follow the `wheel builder README`_
  instructions again.  Edit the ``.github/workflows/wheel.yml`` files (if
  present) to give the release tag to build.  Check the build has passed on
  the Github Interface interface at https://github.com/MacPython/dipy-wheels/actions.
  Now follow the instructions in the page above to download the built wheels to a
  local machine and upload to PyPI.

* Now it's time for the source release. Build the release files::

    make distclean
    git clean -fxd
    make source-release

* Once everything looks good, upload the source release to PyPi::

    pip install twine
    twine upload dist/*

* Check how everything looks on pypi - the description, the packages.  If
  necessary delete the release and try again if it doesn't look right.

* Make an annotated tag for the release with tag of form ``1.8.0``::

    git tag -am 'Public release 1.8.0' 1.8.0

* Set up maintenance / development branches

  If this is a full release you need to set up two branches, one for
  further substantial development (often called 'trunk') and another for
  maintenance releases.

  * Branch to maintenance::

      git co -b maint/1.8.x

    Set ``_version_extra`` back to ``.dev`` and bump ``_version_micro`` by 1.
    Thus the maintenance series will have version numbers like - say - '1.8.1.dev'
    until the next maintenance release - say '1.8.1'.  Commit.

    Push with something like ``git push upstream-rw maint/1.8.x --set-upstream``

  * Start next development series::

      git co main-master

    then restore ``.dev`` to ``_version_extra``, and bump ``_version_minor`` by 1.
    Thus the development series ('trunk') will have a version number here of
    '0.7.0.dev' and the next full release will be '0.7.0'.

    Next merge the maintenance branch with the "ours" strategy.  This just labels
    the maintenance branch `info.py` edits as seen but discarded, so we can
    merge from maintenance in future without getting spurious merge conflicts::

       git merge -s ours maint/1.8.x

    Push with something like ``git push upstream-rw main-master:master``

  If this is just a maintenance release from ``maint/1.8.x`` or similar, just
  tag and set the version number to - say - ``1.8.2.dev``.

* Push the tag with ``git push upstream-rw 1.8.0``

Uploading binary builds for the release
=======================================

By far the easiest way to do this is via the buildbots.

In order to do this, you need first to push the release commit and the release
tag to github, so the buildbots can find the released code and build it.

* In order to trigger the binary builds for the release commit, you need to go
  to the web interface for the binary builder, go to the "Force build" section,
  enter your username and password for the buildbot web service and enter the
  commit tag name in the *revision* field.  For example, if the tag was
  ``1.8.0`` then you would enter ``1.8.0`` in the revision field of the form.
  This builds the exact commit labeled by the tag, which is what we want.

* Trigger binary builds for Windows from the ``https://github.com/MacPython/dipy-wheels``

  Download the builds and upload to pypi.

  You can upload the exe files with the *files* interface for the new DIPY release.
  Obviously you'll need to log in to do this, and you'll need to be an admin for
  the DIPY pypi project.

  For reference, if you need to do binary exe builds by hand, use something
  like::

    make distclean
    git clean -fxd

* Trigger binary builds for macOS from the buildbots ``https://github.com/MacPython/dipy-wheels``.
  Download the eggs and upload to pypi.

  Upload the dmg files with the *files* interface for the new DIPY release.

* Building macOS dmgs from the mpkg builds.

  The buildbot binary builders build ``mpkg`` directories, which are installers
  for macOS.

  These need their permissions to be fixed because the installers should install
  the files as the root user, group ``admin``.  The all need to be converted to
  macOS disk images.  Use the ``./tools/build_dmgs.py``, with something like
  this command line::

    ./tools/build_dmgs "dipy-dist/dipy-1.8.0-py*.mpkg"

  For this to work you'll need several things:

    * An account on a macOS box with sudo (Admin user) on which to run the
      script.
    * ssh access to the buildbot server http://nipy.bic.berkeley.edu (ask
      Matthew or Eleftherios).
    * a development version of ``bdist_mpkg`` installed from
      https://github.com/matthew-brett/bdist_mpkg.  You need this second for the
      script ``reown_mpkg`` that fixes the permissions.

  Upload the dmg files with the *files* interface for the new DIPY release.

Other stuff that needs doing for the release
============================================

* Checkout the tagged release, build the html docs and upload them to
  the github pages website::

    make upload

  You need to checkout the tagged version in order to get the version number
  correct for the doc build.  The version number gets picked up from the
  ``info.py`` version.

* Announce to the mailing lists.  With fear and trembling.


.. include:: ../links_names.inc
