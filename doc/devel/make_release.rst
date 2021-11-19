.. _release-guide:

*********************************
A guide to making a DIPY release
*********************************

A guide for developers who are doing a DIPY release

.. _release-tools:

Release tools
=============

There are some release utilities that come with nibabel_. nibabel should
install these as the ``nisext`` package, and the testing stuff is understandably
in the ``testers`` module of that package. DIPY has Makefile targets for their
use.  The relevant targets are::

    make check-version-info
    make check-files
    make sdist-tests

The first installs the code from a git archive, from the repository, and for
in-place use, and runs the ``get_info()`` function to confirm that installation
is working and information parameters are set correctly.

The second (``sdist-tests``) makes an sdist source distribution archive,
installs it to a temporary directory, and runs the tests of that install.

If you have a version of nibabel trunk past February 11th 2011, there will also
be a functional make target::

    make bdist-egg-tests

This builds an egg (which is a zip file), hatches it (unzips the egg) and runs
the tests from the resulting directory.

.. _release-checklist:

Release checklist
=================

* Review the open list of `dipy issues`_.  Check whether there are
  outstanding issues that can be closed, and whether there are any issues that
  should delay the release.  Label them !

* Check whether there are no build failing on `Travis`. Indeed, ``PRE`` build is
  allowed to fail and does not block a PR merge but it should block release !
  So make sure that ``PRE`` build is not failing. 

* Review and update the release notes.  Review and update the :file:`Changelog`
  file.  Get a partial list of contributors with something like::

      git shortlog -ns 0.6.0..

  where ``0.6.0`` was the last release tag name.

  Then manually go over ``git shortlog 0.6.0..`` to make sure the release notes
  are as complete as possible and that every contributor was recognized.

* Use the opportunity to update the ``.mailmap`` file if there are any duplicate
  authors listed from ``git shortlog -ns``.

* Add any new authors to the ``AUTHORS`` file.  Add any new entries to the
  ``THANKS`` file.

* Check the copyright years in ``doc/conf.py`` and ``LICENSE``

* Check the examples - we really need an automated check here.

* Check the ``pyx`` file doctests with::

    ./tools/doctest_extmods.py dipy

  We really need an automated run of these using the buildbots, but we haven't
  done it yet.

* Check the ``long_description`` in ``dipy/info.py``.  Check it matches the
  ``README`` in the root directory, maybe with ``vim`` ``diffthis`` command.
  Check all the links are still valid.

* Check all the DIPY builds are green on the nipy `buildbots`_

* If you have travis-ci_ building set up you might want to push the code in its
  current state to a branch that will build, e.g.::

    git branch -D pre-release-test # in case branch already exists
    git co -b pre-release-test

* Run the builder and review the output from
  http://nipy.bic.berkeley.edu/builders/dipy-release-checks   This builder does
  *not* check the outputs - they will likely all be green - you have to check the
  ``stdio`` output for each step using the web interface.

  The ``dipy-release-checks`` builder runs these tests::

    make distclean
    python -m compileall .
    make sdist-tests
    make bdist-egg-tests
    make check-version-info
    make check-files

* ``make bdist-egg-tests`` may well fail because of a problem with the script
  tests; if you have a recent (>= March 31 2013) nibabel ``nisext`` package, you
  could try instead doing::

    python -c 'from nisext.testers import bdist_egg_tests; bdist_egg_tests("dipy", label="not slow and not script_test")'

  Eventually we should update the ``bdist-egg-tests`` makefile target.

* ``make check-version-info`` checks how the commit hash is stored in the
  installed files.  You should see something like this::

    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'archive substitution', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/dipy', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}
    /var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/dipy/__init__.pyc
    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'installation', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/dipy', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}
    /Users/mb312/dev_trees/dipy/dipy/__init__.pyc
    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'repository', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/Users/mb312/dev_trees/dipy/dipy', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}

* ``make check-files`` checks if the source distribution is picking up all the
  library and script files.  Look for output at the end about missed files, such
  as::

    Missed script files:  /Users/mb312/dev_trees/dipy/bin/nib-dicomfs, /Users/mb312/dev_trees/dipy/bin/nifti1_diagnose.py

  Fix ``setup.py`` to carry across any files that should be in the distribution.

* Clean and compile::

    make distclean
    git clean -fxd
    python setup.py build_ext --inplace

* Make sure all tests pass on your local machine (from the ``<dipy root>`` directory)::

    cd ..
    pytest -sv --with-doctest dipy
    cd dipy # back to the root directory

* Check the documentation doctests::

    cd doc
    make doctest
    cd ..

  At the moment this generates lots of errors from the autodoc documentation
  running the doctests in the code, where the doctests pass when run in pytest -
  we should find out why this is at some point, but leave it for now.

* Trigger builds of all the binary build testers for DIPY, using the web
  interface. You may need permissions set to do this - contact Matthew or
  Eleftherios if you do.

  At the moment, the useful DIPY binary build testers are:

      * http://nipy.bic.berkeley.edu/builders/dipy-bdist32-35
      * http://nipy.bic.berkeley.edu/builders/dipy-bdist32-27
      * http://nipy.bic.berkeley.edu/builders/dipy-bdist64-27
      * http://nipy.bic.berkeley.edu/builders/dipy-bdist64-35
      * http://nipy.bic.berkeley.edu/builders/dipy-bdist-mpkg-2.6
      * http://nipy.bic.berkeley.edu/builders/dipy-bdist-mpkg-2.7

* Build and test the DIPY wheels.  See the `wheel builder README
  <https://github.com/MacPython/dipy-wheels>`_ for instructions.  In summary,
  clone the wheel-building repo, edit the ``.travis.yml`` and ``appveyor.yml``
  text files (if present) with the branch or commit for the release, commit
  and then push back up to github.  This will trigger a wheel build and test
  on OSX, Linux and Windows. Check the build has passed on on the Travis-CI
  interface at https://travis-ci.org/MacPython/dipy-wheels.  You'll need
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

* Make the release commit.  Edit :file:`dipy/info.py` to set
  ``_version_extra`` to ``''``; commit.  Push.

* For the wheel build / upload, follow the `wheel builder README`_
  instructions again.  Edit the ``.travis.yml`` and ``appveyor.yml`` files (if
  present) to give the release tag to build.  Check the build has passed on on
  the Travis-CI interface at https://travis-ci.org/MacPython/dipy-wheels.  Now
  follow the instructions in the page above to download the built wheels to a
  local machine and upload to PyPI.

* Now it's time for the source release. Build the release files::

    make distclean
    git clean -fxd
    make source-release

* Once everything looks good, upload the source release to PyPi.  See
  `setuptools intro`_::

    python setup.py register
    python setup.py sdist --formats=gztar,zip upload

* Remember you'll need your ``~/.pypirc`` file set up right for this to work.
  See `setuptools intro`_.  The file should look something like this::

    [distutils]
    index-servers =
        pypi

    [pypi]
    username:your.pypi.username
    password:your-password

    [server-login]
    username:your.pypi.username
    password:your-password

* Check how everything looks on pypi - the description, the packages.  If
  necessary delete the release and try again if it doesn't look right.

* Make an annotated tag for the release with tag of form ``0.6.0``::

    git tag -am 'Second public release' 0.6.0

* Set up maintenance / development branches

  If this is this is a full release you need to set up two branches, one for
  further substantial development (often called 'trunk') and another for
  maintenance releases.

  * Branch to maintenance::

      git co -b maint/0.6.x

    Set ``_version_extra`` back to ``.dev`` and bump ``_version_micro`` by 1.
    Thus the maintenance series will have version numbers like - say - '0.6.1.dev'
    until the next maintenance release - say '0.6.1'.  Commit.

    Push with something like ``git push upstream-rw maint/0.6.x --set-upstream``

  * Start next development series::

      git co main-master

    then restore ``.dev`` to ``_version_extra``, and bump ``_version_minor`` by 1.
    Thus the development series ('trunk') will have a version number here of
    '0.7.0.dev' and the next full release will be '0.7.0'.

    Next merge the maintenace branch with the "ours" strategy.  This just labels
    the maintenance branch `info.py` edits as seen but discarded, so we can
    merge from maintenance in future without getting spurious merge conflicts::

       git merge -s ours maint/0.6.x

    Push with something like ``git push upstream-rw main-master:master``

  If this is just a maintenance release from ``maint/0.6.x`` or similar, just
  tag and set the version number to - say - ``0.6.2.dev``.

* Push the tag with ``git push upstream-rw 0.6.0``

Uploading binary builds for the release
=======================================

By far the easiest way to do this is via the buildbots.

In order to do this, you need first to push the release commit and the release
tag to github, so the buildbots can find the released code and build it.

* In order to trigger the binary builds for the release commit, you need to go
  to the web interface for the binary builder, go to the "Force build" section,
  enter your username and password for the buildbot web service and enter the
  commit tag name in the *revision* field.  For example, if the tag was
  ``0.6.0`` then you would enter ``0.6.0`` in the revision field of the form.
  This builds the exact commit labeled by the tag, which is what we want.

* Trigger binary builds for Windows from the buildbots. See builders
  ``dipy-bdist32-26``, ``dipy-bdist32-27``.  The ``exe`` builds will appear in
  http://nipy.bic.berkeley.edu/dipy-dist .  Check that the binary build version
  numbers are release numbers (``dipy-0.6.0.win32.exe`` rather than
  ``dipy-0.6.0.dev.win32.exe``).

  Download the builds and upload to pypi.

  You can upload the exe files with the *files* interface for the new DIPY release.
  Obviously you'll need to log in to do this, and you'll need to be an admin for
  the DIPY pypi project.

  For reference, if you need to do binary exe builds by hand, use something
  like::

    make distclean
    git clean -fxd
    c:\Python26\python.exe setup.py bdist_egg upload
    c:\Python26\python.exe setup.py bdist_wininst --target-version=2.6 register upload

* Trigger binary builds for OSX from the buildbots ``dipy-bdist-mpkg-2.6``,
  ``dipy-bdist-mpkg-2.7``. ``egg`` and ``mpkg`` builds will appear in
  http://nipy.bic.berkeley.edu/dipy-dist .  Download the eggs and upload to
  pypi.

  Upload the dmg files with the *files* interface for the new DIPY release.

* Building OSX dmgs from the mpkg builds.

  The buildbot binary builders build ``mpkg`` directories, which are installers
  for OSX.

  These need their permissions to be fixed because the installers should install
  the files as the root user, group ``admin``.  The all need to be converted to
  OSX disk images.  Use the ``./tools/build_dmgs.py``, with something like this
  command line::

    ./tools/build_dmgs "dipy-dist/dipy-0.6.0-py*.mpkg"

  For this to work you'll need several things:

    * An account on a OSX box with sudo (Admin user) on which to run the script.
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

.. _setuptools intro: http://packages.python.org/an_example_pypi_project/setuptools.html

.. include:: ../links_names.inc