.. _development:

======================
 Installing dipy
======================

dipy_ is in active development at the moment and we are doing our best to create a release at the soonest possible. However, if you still want to play with it here is what you need to do. 

The primary development repository is `dipy github`_ 

We will describe here the installation for Debian & Ubuntu.  

Install git using ::

	$sudo apt-get install git

then try::
 
	$git clone git://github.com/Garyfallidis/dipy.git
       	$cd dipy/
	$sudo python setup.py install

in case you want to just build it locally try::

   	$python setup.py build_ext --inplace


For visualisation

remove all mayavi or traits installations if you have any then try

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

if easy_install is not installed then try _setuptools.

For any problems/suggestions please let us know by sending an e-mail at nipy-devel@neuroimaging.scipy.org


.. include:: links_names.txt
