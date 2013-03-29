.. _release-guide:

*********************************
A guide to making a dipy release
*********************************

A guide for developers who are doing a dipy release


.. _release-tools:

Release tools
=============

There are some release utilities that come with nibabel_.  nibabel should
install these as the ``nisext`` package, and the testing stuff is understandably
in the ``testers`` module of that package.  Dipy has Makefile targets for their
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

* Review and update the release notes.  Review and update the :file:`Changelog`
  file.  Get a partial list of contributors with something like::

      git log 0.5.0.. | grep '^Author' | cut -d' ' -f 2- | sort | uniq

  where ``0.5.0`` was the last release tag name.

  Then manually go over ``git shortlog 0.5.0..`` to make sure the release notes
  are as complete as possible and that every contributor was recognized.

* Use the opportunity to update the ``.mailmap`` file if there are any duplicate
  authors listed from ``git shortlog -ns``.

* Add any new authors to the ``AUTHORS`` file.  Add any new entries to the
  ``THANKS`` file.

* Check the copyright years in ``doc/conf.py`` and ``LICENSE``

* Check the ``long_description`` in ``dipy/info.py``.  Check it matches the
  ``README`` in the root directory, maybe with ``vim`` ``diffthis`` command.

* Clean and compile::

    make distclean
    python setup.py build_ext --inplace

* Make sure all tests pass (from the dipy root directory)::

    cd ..
    nosetests --with-doctest dipy
    cd dipy # back to the root directory

* Check the documentation doctests::

    cd doc
    make doctest
    cd ..

  At the moment this generates lots of errors from the autodoc documentation
  running the doctests in the code, where the doctests pass when run in nose -
  we should find out why this is at some point, but leave it for now.

* Make sure all tests pass from sdist::

    make sdist-tests

  and bdist_egg::

    make bdist-egg-tests

  and the three ways of installing (from tarball, repo, local in repo)::

    make check-version-info

  The last may not raise any errors, but you should detect in the output
  lines of this form::

    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'archive substitution', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/dipy', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}
    /var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/dipy/__init__.pyc
    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'installation', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/dipy', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}
    Files not taken across by the installation:
    []
    /Users/mb312/dev_trees/dipy/dipy/__init__.pyc
    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'repository', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/Users/mb312/dev_trees/dipy/dipy', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}

* The release should now be ready.

* Edit :file:`dipy/info.py` to set ``_version_extra`` to ``''``; commit

* Build the release files::

    make distclean
    make source-release

* Once everything looks good, upload the source release to PyPi.  See
  `setuptools intro`_::

    python setup.py register
    python setup.py sdist --formats=gztar,zip upload

* Then upload the binary release for the platform you are currently on::

    python setup.py bdist_egg upload

* Do binary builds for any virtualenvs you have::

    workon python25
    python setup.py bdist_egg upload
    deactivate

  etc.  (``workon`` is a virtualenvwrapper command).

  For OSX and python 2.5 only, the installation didn't recognize it was doing a fat (i386 + PPC)
  build, and build with name ``dipy-0.5.0-py2.5-macosx-10.3-i386.egg``.  I tried
  to tell it to use ``fat`` and ``universal`` in the name, but uploading these
  tp pypi didn't result in in easy_install finding them.  In the end did the
  standard::

    python setup.py bdist_egg upload

  which uploaded the 'i386' egg, followed by::

    python setup.py bdist_egg --plat-name macosx-10.3-ppc upload

  which may or may not work to allow easy_install to find the egg for PPC.  It
  does work for easy_install on my Intel machine.  I found the default platform
  name with ``python setup.py bdist_egg --help``.

  When trying to upload in python25, after previously saving my ``~/.pypirc``
  during the initial ``register`` step, I got a configparser error.  I found
  `this python 2.5 pypirc page
  <http://docs.python.org/release/2.5.2/dist/pypirc.html>`_ and so hand edited
  the ``~/.pypirc`` file to have a new section::

    [server-login]
    username:my-username
    password:my-password

  after which python25 upload seemed to go smoothly.

* Building OSX dmgs.  This is a little complicated

  See `MBs OSX setup
  <http://matthew-brett.github.com/pydagogue/develop_mac.html>`_).

  We have builders building the packages on the buildbots.

  The mpkg packages will appear in http://nipy.bic.berkeley.edu/dipy-dist

  The problem is that the buildbots built the packages as the ``buildslave``
  user, but we need to make files have root permissions when installed from the
  installer.

  This can be done using a script ``reown_mpkg`` that is part of the development
  version of ``bdist_mpkg`` : https://github.com/matthew-brett/bdist_mpkg

  Install the development version, and then you can build correct installers
  with something like the following (on a machine to which you have ``sudo``
  permission)::

      mkdir mpkgs
      cd mpkgs/
      scp -r buildbot-master:nibotmi/public_html/dipy-dist/*.mpkg .
      sudo reown_mpkg dipy-0.6.0.dev-py2.6-macosx10.6.mpkg root admin
      sudo reown_mpkg dipy-0.6.0.dev-py2.7-macosx10.6.mpkg root admin
      zip -r dipy-0.6.0.dev-py2.6-macosx10.6.mpkg.zip dipy-0.6.0.dev-py2.6-macosx10.6.mpkg
      zip -r dipy-0.6.0.dev-py2.7-macosx10.6.mpkg.zip dipy-0.6.0.dev-py2.7-macosx10.6.mpkg

* Windows ``.exe`` installers should arrive in the same directory as the mpkg
  builds.  You may need to trigger the relevant builders from the buildbot web
  interface.  You should be able to download them and then upload them to pypi.

  If not, the manual way is something like::

    make distclean
    c:\Python26\python.exe setup.py bdist_egg upload
    c:\Python26\python.exe setup.py bdist_wininst --target-version=2.6 register upload

  Maybe virtualenvs for the different versions of python?  I haven't explored
  that yet.

* Tag the release with tag of form ``0.5.0``::

    git tag -am 'First public release' 0.5.0

* Now the version number is OK, push the docs to sourceforge with::

    make upload-website-mysfusername

  where ``mysfusername`` is obviously your own sourceforge username.

* Set up maintenance / development branches

  If this is this is a full release you need to set up two branches, one for
  further substantial development (often called 'trunk') and another for
  maintenance releases.

  * Branch to maintainance::

      git co -b maint/0.5.x

    Set ``_version_extra`` back to ``.dev`` and bump ``_version_micro`` by 1.
    Thus the maintenance series will have version numbers like - say - '0.5.1.dev'
    until the next maintenance release - say '0.5.1'.  Commit.

  * Start next development series::

      git co main-master

    then restore ``.dev`` to ``_version_extra``, and bump ``_version_minor`` by 1.
    Thus the development series ('trunk') will have a version number here of
    '0.6.0.dev' and the next full release will be '0.6.0'.

  If this is just a maintenance release from ``maint/0.5.x`` or similar, just
  tag and set the version number to - say - ``0.5.2.dev``.

* Make a tarball for the examples, to allow packagers to bypass the need for
  having vtk or pytables or a display on the build machines::

        cd doc
        make upload-examples-mysfusername

  The command requires pytables_ and python vtk on your machine. It writes an
  archive named for the dipy version and the docs, e.g::

    <dipy root>/dist/dipy-0.5.0.dev-doc-examples.tar.gz

  and thence writes the archive to the dipy doc directory on sourceforge.

* Announce to the mailing lists.  With fear and trembling.

.. _setuptools intro: http://packages.python.org/an_example_pypi_project/setuptools.html
.. include:: ../links_names.inc
