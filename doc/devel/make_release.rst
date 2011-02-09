.. _release-guide:

*********************************
A guide to making a dipy release
*********************************

A guide for developers who are doing a dipy release

* Edit :file:`info.py` and bump the version number

.. _release-tools:

Release tools
=============

In the :file:`tools` directory, among other files, you will find the following
utilities::

    tools/
    |- build_release
    |- release
    |- toollib.py

There are also some release utilities in :file:`nisext/testers.py`, with
makefile targets for their use.  The relevant targets are::

    make check-version-info
    make sdist-tests

The first installs the code from a git archive, from the repository, and for
in-place use, and runs the ``get_info()`` function to confirm that installation
is working and information parameters are set correctly.

The second (``sdist-tests``) makes an sdist source distribution archive,
installs it to a temporary directory, and runs the tests of that install.

.. _release-checklist:

Release checklist
=================

* Review the open list of `issues <http://github.com/Garyfallidis/dipy/issues>`_ .
  Check whether there are outstanding issues that can be closed, and whether
  there are any issues that should delay the release.  Label them !

* Review and update the release notes.  Review and update the :file:`Changelog`
  file.  Get a partial list of contributors with something like::

      git log 0.9.0.. | grep '^Author' | cut -d' ' -f 2- | sort | uniq

  where ``0.9.0`` was the last release tag name.

  Then manually go over the *git log* to make sure the release notes are
  as complete as possible and that every contributor was recognized.

* Make sure all tests pass::

    nosetests --with-doctest dipy

* Make sure all tests pass from sdist::

    make sdist-tests

* First pass run :file:`build_release` from the :file:`tools` directory::

    cd tools
    ./build_release

* The release should now be ready.

* Edit :file:`dipy/info.py` to set ``_version_extra`` to ``''``; commit

* Once everything looks good, run :file:`release` from the
  :file:`tools` directory.

* Tag the release with tag of form ``1.0.0``::

    git tag -am 'First public release' 1.0.0

* Now the version number is OK, push the docs to sourceforge with::

    make upload-htmldoc-mysfusername

  where ``mysfusername`` is obviously your own sourceforge username.

* Set up maintenance / development branches

  If this is this is a full release you need to set up two branches, one for
  further substantial development (often called 'trunk') and another for
  maintenance releases.

  * Branch to maintainance::

      git co -b maint/1.0.x

    Set ``_version_extra`` back to ``.dev`` and bump ``_version_micro`` by 1.
    Thus the maintenance series will have version numbers like - say - '1.0.1.dev'
    until the next maintenance release - say '1.0.1'.  Commit.

  * Start next development series::

      git co main-master

    then restore ``.dev`` to ``_version_extra``, and bump ``_version_minor`` by 1.
    Thus the development series ('trunk') will have a version number here of
    '1.1.0.dev' and the next full release will be '1.1.0'.

  If this is just a maintenance release from ``maint/1.0.x`` or similar, just
  tag and set the version number to - say - ``1.0.2.dev``.

* Make next development release tag

    After each release the master branch should be tagged
    with an annotated (or/and signed) tag, naming the intended
    next version, plus an 'upstream/' prefix and 'dev' suffix.
    For example 'upstream/1.0.0.dev' means "development start
    for upcoming version 1.0.0.

    This tag is used in the Makefile rules to create development snapshot
    releases to create proper versions for those. The version derives its name
    from the last available annotated tag, the number of commits since that, and
    an abbrevated SHA1. See the docs of ``git describe`` for more info.

    Please take a look at the Makefile rules ``devel-src``,
    ``devel-dsc`` and ``orig-src``.

* Make a tarball for the examples, for packagers to get away without having vtk
  or a display on the build machines::

        cd doc
        make examples-tgz

  The command requires pytables_ and python vtk on your machine. It writes an
  archive named for the dipy version and the docs, e.g::

    <dipy root>/dist/dipy-0.5.0.dev-doc-examples.tar.gz

  We need to decided where to put this tarball.

* Announce to the mailing lists.

