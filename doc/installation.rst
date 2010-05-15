.. _development:

======================
 Installing dipy
======================

dipy_ is in active development at the moment and we are doing our best
to create a release at the soonest possible. However, if you still want
to play with it here is what you need to do.

The primary development repository is `dipy github`_ 

We will describe for now that the installation for Ubuntu 9.10 assuming
that the installation for other Linux, Macosx and Windows distributions is straightforward. We know that it is not ;-) Don't panic a release is coming soon.

.. _using-on-cmdline:

First, install git using::

        sudo apt-get install git

then try::
 
        git clone git://github.com/Garyfallidis/dipy.git
        cd dipy/
        sudo python setup.py install

For a local build try::

        python setup.py build_ext --inplace

but then add you dipy_ directory in your PYTHONPATH.

dipy_ requires the following packages::

       sudo apt-get install python-numpy python-scipy ipython cython python-dev
       sudo easy_install -U sphinx
       sudo easy_install -U pydicom
       
It also requires nibabel for reading medical images::

        git clone git://github.com/hanke/nibabel.git
        cd nibabel
        python setup.py build_ext --inplace

and then add nibabel directory to your PYTHONPATH. We usually add the PYTHONPATH at the end of ~/.bashrc so we don't need to retype it every time. This should look like::
    export PYTHONPATH=/home/user_dir/Devel/dipy:/home/user_dir/Devel/nibabel

After changing the ~/.bashrc try::

      source ~/.bash

so that you can have immediate access to dipy_ without needing to restart your terminal.       

For visualisation *remove all* older mayavi or traits installations and then install the latest Enthought Suite::

    mkdir ets
    cd ets
    svn co https://svn.enthought.com/svn/enthought/ETSProjectTools/trunk ETSProjectTools
    cd ETSProjectTools/
    sudo python setup.py install
    ets -h
    cd ..
    ets co ETS
    sudo apt-get install swig python-qt4 python-qt4-dev libxtst-dev
    cd ETS_3.3.1/
    ets bdist -r
    sudo easy_install -f dist -H dist ets

In case easy_install is not installed then please install _setuptools.

After doing the above execute ipython in the terminal and try::

    >>>import dipy

Does it work? For any problems/suggestions please let us know by sending an e-mail to nipy-devel@neuroimaging.scipy.org with subject starting with [dipy].


.. include:: links_names.txt
