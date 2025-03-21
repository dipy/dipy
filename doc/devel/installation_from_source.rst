.. _installation-from-source:

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
the DIPY ``pyproject.toml`` file, among others.  We'll call this directory - that
contains the ``pyproject.toml`` file - the *DIPY source root directory*.  Sometimes
we'll also call it the ``<dipy root>`` directory.

Building and installing
=======================

Install from source (all operating systems - recommended)
---------------------------------------------------------

Change directory into the *DIPY source root directory*.

To clean temporary files from your directory, use::

    git clean -fxd

This command will delete all files not present in your github repository.

Then, complete your installation by using this command::

    pip install -r requirements/build.txt
    pip install --no-build-isolation -e .

This command will do the following :
    - install the requirements for building dipy
    - remove the old dipy installation if present
    - build dipy
    - install dipy locally on your user environment

.. _install-source-nix:

Install from source for Unix (e.g Linux, macOS)
-----------------------------------------------

Change directory into the *DIPY source root directory*.

First, install some python build packages::

    pip install -r requirements/build.txt

Then, to install for the system::

    pip install dipy

Or, to build DIPY in the source tree (locally) so you can run the code in the source tree (recommended for following the latest source) run::

    pip install --no-build-isolation -e .

add the *DIPY source root directory* into your ``PYTHONPATH`` environment variable. Search google for ``PYTHONPATH`` for details or see `python module path`_ for an introduction.

When adding dipy_ to the ``PYTHONPATH``, we usually add the ``PYTHONPATH`` at
the end of ``~/.bashrc`` or (macOS) ``~/.bash_profile`` so we don't need to
retype it every time. This should look something like::

  export PYTHONPATH="/home/user_dir/Devel/dipy:\$PYTHONPATH"

After changing the ``~/.bashrc`` or (macOS) ``~/.bash_profile`` try::

  source ~/.bashrc

or::

  source ~/.bash_profile

so that you can have immediate access to DIPY_ without needing to
restart your terminal.


Ubuntu/Debian
-------------

::

    sudo apt-get install python3-dev python3-setuptools
    sudo apt-get install python3-numpy python3-scipy
    sudo apt-get install cython

then::

    sudo pip install nibabel

(we need the latest version of this one - hence ``pip`` rather than
``apt-get``).

You might want the optional packages too (highly recommended)::

    sudo apt-get install ipython python3-h5py python3-vtk python3-matplotlib


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

    python3 -c "import platform;print(platform.python_compiler())"

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
    Visual C++ 2019 (16.0)          MSC_VER=1920
    Visual C++ 2022 (17.0)          MSC_VER=1930

After the VisualStudioBuildTools_ installation,  restart a command shell and
change directory into the *DIPY source root directory*.

Start to install the build tools::

    pip install -r requirements/build.txt

Then to install into your system::

    pip install dipy

To install inplace - so that DIPY is running out of the source code directory::

    pip install --no-build-isolation -e .

(this is the mode we recommend for following the latest source code).

If you get an error saying  "unable to find vcvarsall.bat" then you need to
check your environment variable ``PATH`` or reinstall VisualStudioBuildTools_.
Setuptools should automatically detect the compiler and use it.

macOS
-----

Make sure you have Xcode_ and Anaconda_ installed.

From here follow the :ref:`install-source-nix` instructions.

OpenMP with macOS
-----------------
OpenMP_ is a standard library for efficient multithreaded applications. This
is used in DIPY for speeding up many different parts of the library (e.g., denoising
and bundle registration). If you do not have an OpenMP-enabled compiler, you can
still compile DIPY from source using the above instructions, but it might not take
advantage of the multithreaded parts of the code. To be able to compile
DIPY from source with OpenMP on macOS, you will have to do a few more things. First
of all, you will need to install the Homebrew_ package manager. Next, you will need
to install and configure the compiler. You have two options: using the GCC compiler
or the CLANG compiler. This depends on your python installation:

Under Anaconda
~~~~~~~~~~~~~~~~

We recommend to install llvm via Anaconda_. Run the following::

    conda install -c conda-forge llvm-openmp

In case the compiler is not detected automatically, you can specify the compiler
by using the environment variable ``CC``.

Under Homebrew Python or python.org Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are already using the Homebrew Python, or the standard python.org Python,
you will need to use the llvm compiler with OMP. Run::

    brew install llvm
    brew install libomp
    export CC="/opt/homebrew/opt/llvm/bin/clang"

In case the compiler or OpenMP are not detected , you can specify some environment
variables. For example, you can add the following lines to your ``~/.bash_profile``::

    export CC="/opt/homebrew/opt/llvm/bin/clang"
    export CXX="/opt/homebrew/opt/llvm/bin/clang++"
    export LDFLAGS="-L/opt/homebrew/opt/libomp/lib -L/opt/homebrew/opt/llvm/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/libomp/include -I/opt/homebrew/opt/llvm/include"
    export CFLAGS="-I/opt/homebrew/opt/libomp/include -I/opt/homebrew/opt/llvm/include"

Building and installing
~~~~~~~~~~~~~~~~~~~~~~~
Whether you are using Anaconda_ or Homebrew/python.org Python, you will need to then
run ``pip install dipy``. When you do that, it should now
compile the code with this OpenMP-enabled compiler, and things should go faster!

Testing
========

If you want to run the tests::

    sudo pip install pytest

Then, in the terminal from ``<dipy root>``::

    pytest -svv dipy

You can also run the examples in ``<dipy root>/doc``.

Documentation
=============

To build the documentation in HTML in your computer you will need to do::

    sudo pip install sphinx

Then change directory to ``<dipy root>`` and::

    cd doc
    make clean
    make -C . html

Tip
---

Building the entire ``DIPY`` documentation takes a few hours. You may want to
skip building the documentation for the examples, which will reduce the
documentation build time to a few minutes. You can do so by executing::

    make -C . html-no-examples

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

    make html SPHINXBUILD='python3 <path_to_sphinx>/sphinx-build'


.. include:: ../links_names.inc
