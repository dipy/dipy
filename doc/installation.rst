.. _installation:

======================
 Installing dipy
======================

dipy_ is in active development at the moment and we are doing our best
to create a release as soon as we can. 

General usage
---------------

.. _using-on-cmdline:

In case ``easy_install`` is not installed then please install
setuptools_  and then try ::

    easy_install dipy         

This command should work under Linux, Mac OS X and Windows.

Then from any python console or script try ::

    >>> import dipy

Does it work? For any problems/suggestions please let us know by sending us
an e-mail to nipy-devel@neuroimaging.scipy.org with subject starting
with ``[dipy]``.


Ubuntu/Debian development usage
----------------------------------

The primary development repository is `dipy github`_ 

We are describing here the installation for Ubuntu 9.10 or 10.04 or 10.10
assuming that the installation for other Linux, Macosx and Windows 
distributions is straightforward. We know that it is not ;-) Don't panic 
a release is coming soon.

.. _using-on-cmdline:


In case ``easy_install`` is not installed then please install
setuptools_ ::

        sudo apt-get install python-setuptools

Install git using::

        sudo apt-get install git-core

Go to a folder e.g. /home/user_dir/Devel that you want to have dipy installed and try ::
 
        git clone git://github.com/Garyfallidis/dipy.git
        cd dipy/
        sudo python setup.py install

For a local build try::

        python setup.py build_ext --inplace

and add your dipy_ directory in your PYTHONPATH.

dipy_ requires the following packages::

       sudo apt-get install python-numpy python-scipy ipython cython python-dev python-vtk
       sudo easy_install -U sphinx
       
It also requires nibabel for reading medical images::

        cd ..
        sudo easy_install -U pydicom
        git clone git://github.com/hanke/nibabel.git

and then add the ``nibabel/nibabel`` directory to your PYTHONPATH. We usually
add the PYTHONPATH at the end of ~/.bashrc so we don't need to retype it every
time. This should look like::

         export PYTHONPATH=/home/user_dir/Devel/dipy:/home/user_dir/Devel/nibabel

After changing the ~/.bashrc try::

      source ~/.bashrc

so that you can have immediate access to dipy_ without needing to
restart your terminal.


After doing the above execute ipython in the terminal and try::

    >>> import dipy

You can also try to run the python files in the examples directory.

Do they work? For any problems/suggestions please let us know by sending us
an e-mail to nipy-devel@neuroimaging.scipy.org with subject starting
with ``[dipy]``.


.. include:: links_names.txt
