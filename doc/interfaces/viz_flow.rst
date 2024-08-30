.. _viz_flow:

===============================
dMRI Visualization with Horizon
===============================

This section talks about Horizon workflows in DIPY and how to use them.

To follow this tutorial let's make sure we have the latest version of dipy and
fury installed on our system. ::

  pip install dipy --upgrade
  pip install fury --upgrade


Let's explore the options that horizon provides. ::

  dipy_horizon --help


-------------------------
Visualize 3D Brain Image
-------------------------

This tutorial provides a basic example of loading an dMRI image to horizon using
the command line interface.

Using a terminal, let's download a dataset called ``mni_template``. You can 
skip this step if you already have the dataset downloaded. ::

  dipy_fetch mni_template

To see more details about ``dipy_fetch`` you can refer to :ref:`data_fetch`

This command will download the data in your ``.dipy`` folder placed in your home 
directory. 

Let's try to load the image.

**For macOS and Linux** ::

  dipy_horizon ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii ~/.dipy/mni_template/mni_icbm152_t2_tal_nlin_asym_09a.nii

**For Windows(cmd)** ::
  
  dipy_horizon %USERPROFILE%/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii %USERPROFILE%/.dipy/mni_template/mni_icbm152_t2_tal_nlin_asym_09a.nii

**For Windows(powershell)** ::
  
  dipy_horizon "\$home/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii" "\$home/.dipy/mni_template/mni_icbm152_t2_tal_nlin_asym_09a.nii"

You can also use your own data by invoking the command below. ::

  dipy_horizon <your_file_name>.nii


We also support direct visualization of compressed NIFTI files with extension 
``.nii.gz``. ::

  dipy_horizon <your_fist_file_name>.nii.gz


------------------------
Visualize 4D Brain Image
------------------------

This tutorial shows how visualize a 3D image with Volume.

Using a terminal, let's download a dataset called ``stanford_hardi``. You can 
skip this step if you already have the dataset downloaded. ::

  dipy_fetch stanford_hardi

This dataset has a NIFTI file with volumes.

Let's try to load the image.

**For macOS and Linux** ::

  dipy_horizon ~/.dipy/stanford_hardi/HARDI150.nii.gz

**For Windows(cmd)** ::
  
  dipy_horizon %USERPROFILE%/.dipy/stanford_hardi/HARDI150.nii.gz

**For Windows(powershell)** ::

  dipy_horizon "\$home/.dipy/stanford_hardi/HARDI150.nii.gz"


--------------------------
Visualize Brain Tractogram
--------------------------

This tutorial shows how to visualize a tractogram.

Using a terminal, let's download a dataset called ``bundle_atlas_hcp842``. You
can skip this step if you already have the dataset downloaded. ::

  dipy_fetch bundle_atlas_hcp842


Horizon supports below mentioned tractogram formats.

* .trk
* .trx
* .dpy
* .tck
* .vtk
* .vtp
* .fib


Let's try to load the tractogram.

**For macOS and Linux** ::

  dipy_horizon ~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk --cluster

**For Windows(cmd)** ::
  
  dipy_horizon %USERPROFILE%/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk --cluster

**For Windows(powershell)** ::

  dipy_horizon "\$home/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk" --cluster

Using the ``--cluster`` option, we visualize the clusters(bundles) of the 
tractograms. If we do not provide ``--cluster`` it will open up all the 
streamlines and the interaction panel will not be provided. Opening the 
streamlines can be computationally expensive, if a large dataset is provided.

-----------------------
Visualize Brain Surface
-----------------------

This tutorial shows how to visualize surfaces in the Horizon.

Using terminal, let's download brain surface.

**For macOS and Linux** ::

  wget https://github.com/maharshi-gor/dipy_data/raw/surface_data/surfaces/lh.pial

**For Windows(powershell)** ::

  wget https://github.com/maharshi-gor/dipy_data/raw/surface_data/surfaces/lh.pial -O lh.pial

**For macOS users**, if you do not have ``wget`` on your terminal you can setup 
by writing following command ::

  brew install wget

If you are still getting an error you can download the surface by clicking 
`here <https://github.com/maharshi-gor/dipy_data/raw/surface_data/surfaces/lh.pial>`_.

Previous step will download the file into your current directory.


**Downloaded using wget** To load the surface, ::

  dipy_horizon lh.pial

**Downloaded using link** To load the surface, ::

  dipy_horizon <PATH_TO_YOUR_DIRECTORY>/lh.pial
