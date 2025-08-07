.. _installation:

############
Installation
############

DIPY_ is in active development. You can install it from our latest release, but
you may find that the release has gotten well behind the current development -
at least - we hope so - if we're developing fast enough!

If you want to install the latest and greatest from the bleeding edge of the
development, skip to :ref:`installation-from-source`. If you just want to install a released
version, read on for your platform.

********************
Installing a release
********************

In general, we recommend you try to install DIPY_ package :ref:`install-pip`.

.. _install-packages:

.. _install-pip:

Using pip:
==========

This method should work under Linux, Mac OS X, and Windows.

For Windows and macOS you can use Anaconda_ to get numpy, scipy, cython and lots
of other useful python modules. Anaconda_ is a big package but will install many
tools and libraries that are useful for scientific processing.

When you have numpy, scipy and cython installed then try::

    pip install dipy

Then from any python console or script try::

    >>> import dipy
    >>> print(dipy.get_info())


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
    >>> print(dipy.__version__)

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

In Fedora DIPY can be installed from the main repositories courtesy of
NeuroFedora_::

    sudo dnf install python3-dipy

We hope to get packages for the other Linux distributions, but for now, please
try :ref:`install-pip` instead.


*******
Support
*******

Contact us:
===========

Do these installation instructions work for you? For any problems/suggestions please let us know by:

- sending us an e-mail to the `dipy mailing list`_   or
- sending us an e-mail to the `nipy mailing list`_ with the subject line starting with ``[DIPY]`` or
- create a discussion thread via https://github.com/dipy/dipy/discussions

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
If you have problems installing Anaconda_ we recommend using Canopy_.

Memory issues
-------------
DIPY can process large diffusion datasets. For this reason, we recommend using a 64bit operating system that can allocate larger memory chunks than 32bit operating systems. If you don't have a 64bit computer that is okay DIPY works with 32bit too.

.. _python-versions:

Note on python versions
-----------------------

Most DIPY functionality can be used with Python versions 2.6 and newer, including Python 3.
However, some visualization functionality depends on FURY, which only supports Python 3 in versions 7 and newer.

.. include:: ../links_names.inc
