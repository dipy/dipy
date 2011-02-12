.. _installation:

############
Installation
############

dipy_ is in active development at the moment. You can install it from our latest
release, but you may find that the release has got well behind the current
development - at least - we hope so - if we're developing fast enough!

.. _python-versions:

***********************
Note on python versions
***********************

Sorry, but dipy_ does not yet work with python 3 - so all the instructions
following instructions apply to python 2.5 or python 2.6 or python 2.7.

On OSX we always use the python binaries available from the python.org
downloads, and not the python that comes with the OSX system.  If you don't have
the python.org python you need to go to http://python.org/downloads, then
download and install the python version you want (2.7 or 2.6 or 2.5).  Check
that you have this version on your path (perhaps after ``. ~/.bash_profile``)
with ``which python``.  This should show something like::

    /Library/Frameworks/Python.framework/Versions/2.6/bin/python

We've compiled dipy against this python, and all our testing on OSX too.

********************
Installing a release
********************

If you are on Debian or Ubuntu Linux we recommend you try
:ref:`install-packages` first. Otherwise please try `install-easy-install`.

.. _install-easy-install:

Using easy_install
==================

See first the :ref:`python-versions`.

In case ``easy_install`` is not installed then please install setuptools_ or
distribute_.

Please install numpy_ and scipy_ using their respective binary installers if you
haven't already.

For windows you can use pythonxy_ to get numpy and scipy and lots of other
useful python packages. This is quite a big package but will install
lots of python stuff that is useful for your scientific endeavors.

When you have numpy and scipy installed then try ::

    easy_install dipy

This command should work under Linux, Mac OS X and Windows.

Then from any python console or script try ::

    >>> import dipy

Does it work? For any problems/suggestions please let us know by sending us an
e-mail to the `nipy mailing list`_ with the subject line starting with
``[dipy]``.

By the way, you might be tempted to try and run the tests after installing with
easy_install.  Unfortunately this doesn't work because of a problem with
easy_install.  To run the tests you need to install from source on windows or
mac, or via a package on Linux (source also works on Linux).

.. _install-packages:

Using packages
==============

Windows
-------

Download and install numpy_ and scipy_ - or install pythonxy_.  Install nibabel_
from `nibabel pypi`_ using the ``.exe`` installer.  Install dipy from `dipy
pypi`_ using the ``.exe`` installer for your version of python.

Then from any python console or script try ::

    >>> import dipy

OSX
---

Download and install numpy_ and scipy_ using the OSX binary packages for your
distribution of python.  Install nibabel_ from `nibabel pypi`_ using the
``.mpkg`` installer for your version of python.  Install dipy from `dipy pypi`_
using the ``.mpkg`` installer for your version of python.

Then from any python console or script try ::

    >>> import dipy

Linux
-----

For Debian and Ubuntu, set up the NeuroDebian_ repositories - see `NeuroDebian
how to`_. Then::

    sudo apt-get install python-dipy

We hope to get packages for the other Linux distributions, but for now, please
try :ref:`install-easy-install` instead.

**********************
Installing from source
**********************

Getting the source
==================

You can get the released source zip file or ``tar.gz`` archive from `dipy
pypi`_.

If you want the latest development source as an archive, go to the `dipy
github`_ page, and click on the Download button.

More likely you will want to get the source repository to be able to follow the
latest changes.  In that case, see :ref:`following-latest`.

After you've unpacked the archive or cloned the repository, you will have a new
directory, containing the dipy ``setup.py`` file, among others.  We'll call this
directory - that contains the ``setup.py`` file - the *dipy source root
directory*.  Sometimes we'll also call it the ``<dipy root>`` directory.

Building and installing
=======================

Windows
-------

pythonxy_ is probably the easiest way to install the dependencies that you need.

Otherwise you will need python_ (obviously). You'll need to install the mingw_
compiler suite if you don't have a c compiler on your machine. We suggest you
run the mingw_ automated installer, and install the developer tools, including
msys_.  Don't forget to put the mingw ``bin`` directory on your path so python
can find the compiler. Install numpy_, scipy_, nibabel_ and cython_ from their
respective binary installers.  All of these come with pythonxy_ . You can also
install them from their Windows binary installers.  You'll find these by
following the links from their home pages.

Start a command shell like ``cmd`` or Powershell_ and change directory into the
*dipy source root directory*.

To install into your system::

    python setup.py install --compiler=mingw32

To install inplace - so that dipy is running out of the source code directory::

    python setup.py develop

(this is the mode we recommend for following the latest source code).  If you
get an error with ``python setup.py develop`` make sure you have installed
`setuptools`_.

If you get an error saying  "unable to find vcvarsall.bat" then you need to
create a file called "pydistutils.cfg" in notepad and give it the contents ::

  [build]
  compiler=mingw32

Save this into your system python ``distutils`` directory as ``distutils.cfg``.
This will be something like ``C:\Python26\Lib\distutils\distutils.cfg``.

OSX
---

See the :ref:`python-versions` for which python you need.

Make sure you have Xcode_ installed.

Download and install numpy_ and scipy_ from their respective download sites.
Chose the version for your versions of OSX and python.  Install cython_.
This is probably most easily done with::

    sudo easy_install cython

Install nibabel_ ::

    sudo easy_install nibabel

From here follow the :ref:`install-source-nix` instructions.

Ubuntu/Debian
-------------

::

    sudo apt-get install python-dev python-setuptools
    sudo apt-get install python-numpy python-scipy

then::

    sudo easy_install cython
    sudo easy_install nibabel

(we need the latest version of these two - hence ``easy_install`` rather than
``apt-get``).

You might want the optional packages too (highly recommended)::

    sudo apt-get install ipython python-tables python-vtk python-matplotlib

Now follow :ref:`install-source-nix`.

Fedora / Mandriva maybe Redhat
------------------------------

Making this up, but::

   yum install gcc-c++
   yum install python-devel
   yum install python-setuptools
   yum install numpy scipy

Then::

    sudo easy_install cython
    sudo easy_install nibabel

Options::

   yum install ipython
   yum install python-matplotlib python-vtk python-tables

Now follow :ref:`install-source-nix`.

.. _install-source-nix:

Install from source for unices (e.g Linux, OSX)
-----------------------------------------------

Change directory into the *dipy source root directory* .

To install for the system::

    python setup.py install

To build in the source tree so you can run the code in the source tree
(recommended for following the latest source) either:

* option 1 - using ``setup.py develop``::

    python setup.py develop

* option 2 - putting dipy into your search path manually.  This is more
  long-winded but a bit easier to understand what's going on::

    python setup.py build_ext --inplace

  and then symlink the ``<dipy-root>/dipy`` directory into a directory on your
  python path (``>>> import sys; print sys.path``) or add the *dipy source root
  directory* into your ``PYTHONPATH`` environment variable. Search google for
  ``PYTHONPATH`` for details or see `python module path`_ for an introduction.

  When adding dipy_ to the ``PYTHONPATH``, we usually add the ``PYTHONPATH`` at
  the end of ``~/.bashrc`` or (OSX) ``~/.bash_profile`` so we don't need to
  retype it every time. This should look something like::

    export PYTHONPATH=/home/user_dir/Devel/dipy:/home/user_dir/Devel/nibabel

  After changing the ``~/.bashrc`` or (OSX) ``~/.bash_profile`` try::

    source ~/.bashrc

  or::

    source ~/.bash_profile

  so that you can have immediate access to dipy_ without needing to
  restart your terminal.

If you want to run the tests::

    sudo easy_install nose

Then (in python or ipython_)::

    >>> import dipy
    >>> dipy.test()

You can also run the examples in ``<dipy root>/doc``.

To build the documentation you will need::

    sudo easy_install -U sphinx

Then change directory to ``<dipy root>`` and::

    make html

to make the html documentation.


