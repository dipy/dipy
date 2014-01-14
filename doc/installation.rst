.. _installation:

############
Installation
############

dipy_ is in active development. You can install it from our latest release, but
you may find that the release has gotten well behind the current development -
at least - we hope so - if we're developing fast enough!

If you want install the latest and greatest from the bleeding edge of the
development, skip to :ref:`from-source`. If you just want to install a released
version, read on for your platform.

********************
Installing a release
********************

If you are on Debian or Ubuntu Linux we recommend you try
:ref:`install-packages` first. Otherwise please try :ref:`install-pip`.

.. _install-packages:

Using packages:
===============

Windows
-------

#. To use dipy_, you will need to install a few :ref:`dependencies`. One
    simple way to get the dependencies on your machine, is to install the
    Anaconda_ python distribution, following the instructions on their
    website. Note that there are alternatives to Anaconda_. See below for :ref:`alternatives`

#. You will still need to install the nibabel_ library, which supports reading
    and writing of neuroimaging data formats. You can install it from a binary
    installer on the `nibabel pypi`_ web page. Download the `.exe` installer in
    the table at the bottom and double-click it to launch the installation
    process.

    .. note ::

#. Similarly, you can install dipy_ itself from the `dipy pypi`_
    web-page. Download the `.exe` installer in the table at the bottom of the page and double-click it to launch the installation.


Now, you should be able to start using dipy_. From a python console or
script try ::

    >>> import dipy

.. note::

    There are other alternatives than the one we have outlined above to
    installing the dependencies and to install nibabel_ and dipy_. In
    particular, you can use `pip`_ or `easy_install`_ from the Anaconda Command
    Prompt (which you can launch from the Start Menu),

    To do that, you would replace the instructions above with typing the
    following at the command prompt ::

        pip install nibabel
	    pip install dipy

    Or ::

        easy_install nibabel
	    easy_install dipy

OSX
---

#. To use dipy_, you need to have some :ref:`dependencies` installed. First of
    all, make sure that you have installed the Apple Xcode_ developer tools. You'll
    need those to install all the following dependencies.

#. Next, install the python library dependencies. One easy way to do that is to
    use the Anaconda_ distribution (see below for :ref:`alternatives`).

#. Even with Anaconda installed, you will still need to install the nibabel_
    library, which supports reading and writing of neuroimaging data formats.
    Open a terminal and type ::

       pip install nibabel

#. Finally, we are ready to install `dipy` itself. Same as with `nibabel`
    above, we will type at the terminal shell command line ::

        pip install dipy

When the installation has finished we can check if it is successful in the
following way. From a Python console script try ::

    >>> import dipy

This should work with no error.

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

This method should work under Linux, Mac OS X and Windows.

Please install numpy_ and scipy_ using their respective binary installers if
you haven't already.

For Windows and Mac OSX you can use Anaconda_ to get numpy, scipy, cython and lots
of other useful python module. Anaconda_ is a big package but will install many
tools and libraries that are useful for scientific processing.

When you have numpy, scipy and cython installed then try ::

    pip install nibabel
    pip install dipy

Then from any python console or script try ::

    >>> import dipy


*******
Support
*******

Contact us:
===========

Do these installation instructions work for you? For any problems/suggestions please let us know by sending us an e-mail to the `nipy mailing list`_ with the subject line starting with ``[dipy]``.

Common problems:
================

Multiple installations
----------------------

Make sure that you have uninstalled all previous versions of Dipy before installing a new one. A simple and general way to uninstall Dipy is by removing the installation directory. You can find where Dipy is installed by using::

    import dipy
    dipy.__file__

and then remove the Dipy directory that contains that file.

.. _alternatives:

Alternatives to Anaconda
-------------------------
If you have problems installing Anaconda_ we recommend using Canopy_ or pythonxy_.

Memory issues
-------------
Dipy can process large diffusion datasets. For this reason we recommend using a 64bit operating system which can allocate larger memory chunks than 32bit operating systems. If you don't have a 64bit computer that is okay Dipy works with 32bit too.

.. _python-versions:

Note on python versions
-----------------------

Dipy supports python 2.6+ and python 3+. We started supporting Python 3 from version dipy.0.7.0.

.. _from-source:

**********************
Installing from source
**********************

Getting the source
==================

More likely you will want to get the source repository to be able to follow the
latest changes.  In that case, you can use::

    git clone https://github.com/nipy/dipy.git

For more information about this see :ref:`following-latest`.

After you've cloned the repository, you will have a new directory, containing
the dipy ``setup.py`` file, among others.  We'll call this directory - that
contains the ``setup.py`` file - the *dipy source root directory*.  Sometimes
we'll also call it the ``<dipy root>`` directory.

Building and installing
=======================

Windows
-------

Anaconda_ is probably the easiest way to install the dependencies that you need.

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

Make sure you have Xcode_ and Anaconda_ installed.

From here follow the :ref:`install-source-nix` instructions.

Ubuntu/Debian
-------------

::

    sudo apt-get install python-dev python-setuptools
    sudo apt-get install python-numpy python-scipy
    sudo apt-get install cython

then::

    sudo pip install nibabel

(we need the latest version of these two - hence ``easy_install`` rather than
``apt-get``).

You might want the optional packages too (highly recommended)::

    sudo apt-get install ipython python-tables python-vtk python-matplotlib

Now follow :ref:`install-source-nix`.

Fedora / Mandriva maybe Redhat
------------------------------

Same as above but use yum rather than apt-get when necessary.

Now follow :ref:`install-source-nix`.

.. _install-source-nix:

Install from source for Unix (e.g Linux, OSX)
---------------------------------------------

Change directory into the *dipy source root directory* .

To install for the system::

    python setup.py install

To build dipy in the source tree (locally) so you can run the code in the source tree (recommended for following the latest source) run::

    python setup.py build_ext --inplace

add the *dipy source root directory* into your ``PYTHONPATH`` environment variable. Search google for ``PYTHONPATH`` for details or see `python module path`_ for an introduction.

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

Testing
========

If you want to run the tests::

    sudo pip install nose

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


.. include:: links_names.inc
