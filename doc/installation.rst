.. _installation:

############
Installation
############

DIPY_ is in active development. You can install it from our latest release, but
you may find that the release has gotten well behind the current development -
at least - we hope so - if we're developing fast enough!

If you want to install the latest and greatest from the bleeding edge of the
development, skip to :ref:`from-source`. If you just want to install a released
version, read on for your platform.

********************
Installing a release
********************

If you are on Debian or Ubuntu Linux we recommend you try
:ref:`install-packages` first. Otherwise please try :ref:`install-pip`.

.. _install-packages:


Using Anaconda:
===============

On all platforms, you can use Anaconda_ to install DIPY. To do so issue the following command in a terminal::

    conda install -c conda-forge dipy

Some of the visualization methods require the FURY_ library and this can be installed separately (for the time being only on Python 3.4+)::

    conda install -c conda-forge fury

Using packages:
===============

Windows
-------

#. First, install the python library dependencies. One easy way to do that is to
   use the Anaconda_ distribution (see below for :ref:`alternatives`).

#. Even with Anaconda_ installed, you will still need to install the nibabel_
   library, which supports reading and writing of neuroimaging data formats. Open
   a terminal and type::

    pip install nibabel

#. Finally, we are ready to install DIPY itself. Same as with `nibabel` above,
   we will type at the terminal shell command line::

    pip install dipy

   When the installation has finished we can check if it is successful in the following way. From a Python console script try::

    >>> import dipy

   This should work with no error.

#. Some of the visualization methods require the FURY_ library and this can be installed by doing ::

    pip install fury


OSX
---

#. To use DIPY_, you need to have some :ref:`dependencies` installed. First of all, make sure that you have installed the Apple Xcode_ developer tools. You'll need those to install all the following dependencies.

#. Next, install the python library dependencies. One easy way to do that is to use the Anaconda_ distribution (see below for :ref:`alternatives`).

#. Even with Anaconda_ installed, you will still need to install the nibabel_ library, which supports the reading and writing of neuroimaging data formats. Open a terminal and type::

    pip install nibabel

#. Finally, we are ready to install DIPY itself. Same as with `nibabel` above, we will type at the terminal shell command line::

    pip install dipy

   When the installation has finished we can check if it is successful in the following way. From a Python console script try::

    >>> import dipy

   This should work with no error.

#. Some of the visualization methods require the FURY_ library and this can be installed by doing::

    pip install fury

Linux
-----

For Debian, Ubuntu and Mint set up the NeuroDebian_ repositories - see
`NeuroDebian how to`_. Then::

    sudo apt-get install python-dipy

We hope to get packages for the other Linux distributions, but for now, please
try :ref:`install-pip` instead.


.. _install-pip:

Using pip:
==========

This method should work under Linux, Mac OS X, and Windows.

Please install numpy_ and scipy_ using their respective binary installers if
you haven't already.

For Windows and Mac OSX you can use Anaconda_ to get numpy, scipy, cython and lots
of other useful python modules. Anaconda_ is a big package but will install many
tools and libraries that are useful for scientific processing.

When you have numpy, scipy and cython installed then try::

    pip install nibabel
    pip install dipy

Then from any python console or script try::

    >>> import dipy


*******
Support
*******

Contact us:
===========

Do these installation instructions work for you? For any problems/suggestions please let us know by sending us an e-mail to the `nipy mailing list`_ with the subject line starting with ``[DIPY]``.

Common problems:
================

Multiple installations
----------------------

Make sure that you have uninstalled all previous versions of DIPY before installing a new one. A simple and general way to uninstall DIPY is by removing the installation directory. You can find where DIPY is installed by using::

    import dipy
    dipy.__file__

and then remove the DIPY directory that contains that file.

.. _alternatives:

Alternatives to Anaconda
-------------------------
If you have problems installing Anaconda_ we recommend using Canopy_ or pythonxy_.

Memory issues
-------------
DIPY can process large diffusion datasets. For this reason, we recommend using a 64bit operating system that can allocate larger memory chunks than 32bit operating systems. If you don't have a 64bit computer that is okay DIPY works with 32bit too.

.. _python-versions:

Note on python versions
-----------------------

Most DIPY functionality can be used with Python versions 2.6 and newer, including Python 3.
However, some visualization functionality depends on FURY, which only supports Python 3 in versions 7 and newer.

.. _from-source:

**********************
Installing from source
**********************

Getting the source
==================

More likely you will want to get the source repository to be able to follow the
latest changes.  In that case, you can use::

    git clone https://github.com/dipy/dipy.git

For more information about this see :ref:`following-latest`.

After you've cloned the repository, you will have a new directory, containing
the DIPY ``setup.py`` file, among others.  We'll call this directory - that
contains the ``setup.py`` file - the *DIPY source root directory*.  Sometimes
we'll also call it the ``<dipy root>`` directory.

Building and installing
=======================

Install from source (all operating systems)
-------------------------------------------

Change directory into the *DIPY source root directory*.

To clean your directory from temporary file, use::

    git clean -fxd

This command will delete all files not present in your github repository.

Then, complete your installation by using this command::

    pip install --user -e .

This command will do the following :
    - remove the old dipy installation if present
    - build dipy (equivalent to `python setup.py build_ext --inplace`)
    - install dipy locally on your user environment

.. _install-source-nix:

Install from source for Unix (e.g Linux, OSX)
---------------------------------------------

Change directory into the *DIPY source root directory*.

To install for the system::

    python setup.py install

To build DIPY in the source tree (locally) so you can run the code in the source tree (recommended for following the latest source) run::

    python setup.py build_ext --inplace

add the *DIPY source root directory* into your ``PYTHONPATH`` environment variable. Search google for ``PYTHONPATH`` for details or see `python module path`_ for an introduction.

When adding dipy_ to the ``PYTHONPATH``, we usually add the ``PYTHONPATH`` at
the end of ``~/.bashrc`` or (OSX) ``~/.bash_profile`` so we don't need to
retype it every time. This should look something like::

  export PYTHONPATH="/home/user_dir/Devel/dipy:\$PYTHONPATH"

After changing the ``~/.bashrc`` or (OSX) ``~/.bash_profile`` try::

  source ~/.bashrc

or::

  source ~/.bash_profile

so that you can have immediate access to DIPY_ without needing to
restart your terminal.


Ubuntu/Debian
-------------

::

    sudo apt-get install python-dev python-setuptools
    sudo apt-get install python-numpy python-scipy
    sudo apt-get install cython

then::

    sudo pip install nibabel

(we need the latest version of this one - hence ``pip`` rather than
``apt-get``).

You might want the optional packages too (highly recommended)::

    sudo apt-get install ipython python-h5py python-vtk python-matplotlib


Now follow :ref:`install-source-nix`.

Fedora / Mandriva maybe Redhat
------------------------------

Same as above but use yum rather than apt-get when necessary.

Now follow :ref:`install-source-nix`.


Windows
-------

Anaconda_ is probably the easiest way to install the dependencies that you need.
To build from source, you will also need to install the exact compiler which is
used with your specific version of python.

For getting this information, type this command in shell like ``cmd`` or Powershell_::

    python -c "import platform;print(platform.python_compiler())"

This command should print information of this form::

    MSC v.1900 64 bit (AMD64)

Now that you find the relevant compiler, you have to install the VisualStudioBuildTools_
by respecting the following table::

    Visual C++ 2008  (9.0)          MSC_VER=1500
    Visual C++ 2010 (10.0)          MSC_VER=1600
    Visual C++ 2012 (11.0)          MSC_VER=1700
    Visual C++ 2013 (12.0)          MSC_VER=1800
    Visual C++ 2015 (14.0)          MSC_VER=1900
    Visual C++ 2017 (15.0)          MSC_VER=1910

After the VisualStudioBuildTools_ installation,  restart a command shell and
change directory into the *DIPY source root directory*.

To install into your system::

    python setup.py install

To install inplace - so that DIPY is running out of the source code directory::

    python setup.py develop

(this is the mode we recommend for following the latest source code).  If you
get an error with ``python setup.py develop`` make sure you have installed
`setuptools`_.

If you get an error saying  "unable to find vcvarsall.bat" then you need to
check your environment variable ``PATH`` or reinstall VisualStudioBuildTools_.
Setuptools should automatically detect the compiler and use it.

OSX
---

Make sure you have Xcode_ and Anaconda_ installed.

From here follow the :ref:`install-source-nix` instructions.

OpenMP with OSX
---------------
OpenMP_ is a standard library for efficient multithreaded applications. This
is used in DIPY for speeding up many different parts of the library (e.g., denoising
and bundle registration). If you do not have an OpenMP-enabled compiler, you can
still compile DIPY from source using the above instructions, but it might not take
advantage of the multithreaded parts of the code. To be able to compile
DIPY from source with OpenMP on Mac OSX, you will have to do a few more things. First
of all, you will need to install the Homebrew_ package manager. Next, you will need
to install and configure the compiler. You have two options: using the GCC compiler
or the CLANG compiler. This depends on your python installation:

Under Anaconda
~~~~~~~~~~~~~~~~

If you are using Anaconda_, you will need to use GCC. The first option is to run the following command::

    conda install gcc

After this installation, gcc will be your default compiler in Anaconda_ environment.

The second option is to install gcc via homebrew. Run the following::

    brew reinstall gcc --without-multilib

This should take about 45 minutes to complete. Then add to your bash
configuration (usually in ``~/.bash_profile``), the following::

    export PATH="/usr/local/Cellar/gcc/5.2.0/bin/gcc-5:\$PATH"


Under Homebrew Python or python.org Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are already using the Homebrew Python, or the standard python.org Python,
you will need to use the CLANG compiler with OMP. Run::

    brew install clang-omp

And then edit the ``setup.py`` file to include the following line (e.g., on line 14,
at the top of the file, but after the initial imports)::

    os.environ['CC'] = '/usr/local/bin/clang-omp'


Building and installing
~~~~~~~~~~~~~~~~~~~~~~~
Whether you are using Anaconda_ or Hombrew/python.org Python, you will need to then
run ``python setup.py install``. When you do that, it should now
compile the code with this OpenMP-enabled compiler, and things should go faster!


Testing
========

If you want to run the tests::

    sudo pip install pytest

Then (in python or ipython_)::

    >>> import dipy
    >>> dipy.test()

You can also run the examples in ``<dipy root>/doc``.

Documentation (Unix only)
=========================

To build the documentation in HTML in your computer you will need to do::

    sudo pip install sphinx

Then change directory to ``<dipy root>`` and::

    cd doc
    make clean
    make html

Tip
---

Building the entire ``DIPY`` documentation takes a few hours. You may want to
skip building the documentation for the examples, which will reduce the
documentation build time to a few minutes. You can do so by executing::

    make -C . html-after-examples

Troubleshooting
---------------

If you encounter the following error when trying to build the documentation::

    tools/build_modref_templates.py dipy reference
    *WARNING* API documentation not generated: Can not import dipy
    tools/docgen_cmd.py dipy reference_cmd
    *WARNING* Command line API documentation not generated: Cannot import dipy
    Build API docs...done.
    cd examples_built && ../../tools/make_examples.py
    Traceback (most recent call last):
      File "../../tools/make_examples.py", line 33, in <module>
        import dipy
    ModuleNotFoundError: No module named 'dipy'

it is probably due to a conflict between the picked ``Sphinx`` version: this
happens when the system's ``Sphinx`` package is used instead of the virtual
environment's ``Sphinx`` package, and the former trying to import a ``DIPY``
version in the system: the ``Sphinx`` package used should correspond to that of
the virtual environment where ``DIPY`` lives. This can be solved by specifying
the path to the ``Sphinx`` package in the virtual environment::

    make html SPHINXBUILD='python <path_to_sphinx>/sphinx-build'

.. include:: links_names.inc
