.. _development:

======================
 Installing dipy
======================

dipy_ is in active development at the moment and we are doing our best
to create a release as soon as we can. However, if you still want to
play with it here is what you need to do.

The primary development repository is `dipy github`_ 

We are describing here the installation for Ubuntu 9.10 or 10.04 
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
       sudo easy_install -U pydicom
       
It also requires nibabel for reading medical images::

        cd ..
        git clone git://github.com/hanke/nibabel.git
        cd nibabel
        python setup.py build_ext --inplace

and then add nibabel directory to your PYTHONPATH. We usually add the
PYTHONPATH at the end of ~/.bashrc so we don't need to retype it every
time. This should look like::

         export PYTHONPATH=/home/user_dir/Devel/dipy:/home/user_dir/Devel/nibabel:/home/user_dir/Devel/nipy

After changing the ~/.bashrc try::

      source ~/.bashrc

so that you can have immediate access to dipy_ without needing to
restart your terminal.

Finally, download and install nipy::

     
         mkdir ~/.nipy
         mkdir ~/.nipy/nipy
         cd ~/.nipy/nipy
         wget http://nipy.sourceforge.net/data-packages/nipy-templates-0.2.tar.gz
         tar -xzvf nipy-templates-0.2.tar.gz
         mv nipy-templates-0.2/templates .
         wget http://nipy.sourceforge.net/data-packages/nipy-data-0.2.tar.gz
         tar -xzvf nipy-data-0.2.tar.gz
         mv nipy-data-0.2/data .                 
         sudo easy_install -U sympy
         cd ~/Devel
         git clone git://github.com/nipy/nipy.git
         cd nipy
         python setup.py build_ext --inplace     


After doing the above execute ipython in the terminal and try::

    >>> import dipy

You can also try to run the python files in the examples directory.

Do they work? For any problems/suggestions please let us know by sending us
an e-mail to nipy-devel@neuroimaging.scipy.org with subject starting
with ``[dipy]``.


.. include:: links_names.txt
