.. _denoise_flow:

=========
Denoising
=========

This tutorial walks through the steps to denoise diffusion-weighted MR images using DIPY.
Multiple denoising methods are available in DIPY.

You can try these methods using your own data; we will be using the data in DIPY.
You can check how to :ref:`fetch the DIPY data<data_fetch>`.

-------------------------
Denoising using Local PCA
-------------------------

PCA-based denoising algorithms are effective denoising methods because they explore 
the redundancy of the multi-dimensional information of diffusion-weighted datasets.

The algorithm to perform the PCA-based denoising involves the following steps:

* First, we estimate the local noise variance at each voxel.

* Then, we apply PCA in local patches around each voxel over the gradient directions.

* Finally, we threshold the eigenvalues based on the local estimate of sigma and then do a PCA reconstruction

We are using ``stanford_hardi`` dataset. You can also use your own data.

The workflow for the LPCA denoising requires the paths to the diffusion input file,
as well as the b-values and b-vectors files.

You may want to create an output directory to save the denoised data::

    mkdir denoise_lpca_output

Run the following command::

    dipy_denoise_lpca data/stanford_hardi/HARDI150.nii.gz data/stanford_hardi/HARDI150.bval data/stanford_hardi/HARDI150.bvec --out_dir "denoise_lpca_output"

This command will denoise the diffusion image and save it to the directory
``denoise_lpca_output``.

In case, you did not specify ``out_dir``, the denoised diffusion image will 
be saved to a file named ``dwi_lpca.nii.gz`` by default, located in the 
input directory also by default.

Note: Depending on the parameters' values, the effect of the denoising can 
be subtle or even hardly noticeable, apparent or visible, depending on the 
choice. Users are encouraged to carefully choose the parameters.

.. |image1| image:: https://github.com/dipy/dipy_data/blob/master/stanford_hardi_original.png?raw=true
   :scale: 100%
   :align: middle
.. |image2| image:: https://github.com/dipy/dipy_data/blob/master/stanford_hardi_denoise_LPCA.png?raw=true
   :scale: 100%
   :align: middle

+--------------------+--------------------+
|  Before Denoising  |  After Denoising   |
+====================+====================+
|      |image1|      |      |image2|      |
+--------------------+--------------------+

-----------------------------------
Denoising using Marcenko-Pastur PCA
-----------------------------------

The PCA-based denoising algorithm exploits the redundancy across the diffusion-
weighted images. This algorithm has been shown to provide an optimal compromise
between noise suppression and loss of anatomical information for different 
techniques such as DTI, spherical deconvolution and DKI.

The basic idea behind the PCA-based denoising algorithms is to remove the components 
of the data that are classified as noise. The Principal Components classification
can be performed based on prior noise variance estimates or automatically based 
on the Marcenko-Pastur distribution. In addition to noise suppression, the PCA
algorithm can be used to get the standard deviation of the noise.

We will use the ``sherbrooke_3shell`` dataset in DIPY to showcase this denoising method.
As with any other workflow in DIPY, you can also use your own data!

Create a directory where to save the denoised image (e.g.:
``denoise_mppca_output``)::

    mkdir denoise_mppca_output

In order to run the MPPCA denoising method, we need to specify the location of the 
diffusion data file, followed by the optional arguments. In this case, we will be 
specifying the ``patch radius`` value and ``output directory``.

So, we will run the command as::

    dipy_denoise_mppca data/sherbrooke_3shell/HRADI193.nii.gz --patch_radius 10 --out_dir "denoise_mppca_output"

This command will denoise the diffusion image and save it to the specified output directory.

.. |image3| image:: https://github.com/dipy/dipy_data/blob/master/sherbrooke_3shell_original.png?raw=true
   :scale: 70%
   :align: middle
.. |image4| image:: https://github.com/dipy/dipy_data/blob/master/sherbrooke_3shell_denoise_MPPCA.png?raw=true
   :scale: 70%
   :align: middle

+--------------------+--------------------+
|  Before Denoising  |  After Denoising   |
+====================+====================+
|      |image3|      |      |image4|      |
+--------------------+--------------------+

-----------------------
Denoising using NLMEANS
-----------------------

Using the non-local means filter [Coupe08]_ and [Coupe11]_, you can denoise 3D or 4D
images and boost the SNR of your datasets.

We will use the ``cfin_multib`` dataset in DIPY to showcase this denoising method.
As with any other workflow in DIPY, you can also use your own data!

In order to run the NLMEANS denoising method, we need to specify the location of the 
diffusion data file, followed by the optional arguments. In this case, we will be 
specifying the ``sigma`` and ``patch radius`` values and ``output directory``.

Create a directory where to save the denoised image (e.g.:
``denoise_nlmeans_output``)::

Then, we will run the command as::

    dipy_denoise_nlmeans data/cfin_multib/__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii --sigma 2 --patch_radius 2 --out_dir "denoise_nlmeans_output"

The command will denoise the input diffusion volume and write the result to the specified
output directory.

.. |image5| image:: https://github.com/dipy/dipy_data/blob/master/cfin_multib_original.png?raw=true
   :scale: 20%
   :align: middle
.. |image6| image:: https://github.com/dipy/dipy_data/blob/master/cfin_multib_denoise_NLMEANS.png?raw=true
   :scale: 20%
   :align: middle

+--------------------+--------------------+
|  Before Denoising  |  After Denoising   |
+====================+====================+
|      |image5|      |      |image6|      |
+--------------------+--------------------+

-----------------------------
Overview of Denoising Methods
-----------------------------

Note: Users are recommended to zoom (click on each image) to see the denoising effect.

.. |image7| image:: https://github.com/dipy/dipy_data/blob/master/sherbrooke_3shell_original.png?raw=true
   :scale: 100%
   :align: middle
.. |image8| image:: https://github.com/dipy/dipy_data/blob/master/sherbrooke_denoise_LPCA.png?raw=true
   :scale: 100%
   :align: middle
.. |image9| image:: https://github.com/dipy/dipy_data/blob/master/sherbrooke_3shell_denoise_MPPCA.png?raw=true
   :scale: 100%
   :align: middle
.. |image10| image:: https://github.com/dipy/dipy_data/blob/master/sherbrooke_denoise_NLMEANS.png?raw=true
   :scale: 100%
   :align: middle
.. |image11| image:: https://github.com/dipy/dipy_data/blob/master/stanford_hardi_original.png?raw=true
   :scale: 100%
   :align: middle
.. |image12| image:: https://github.com/dipy/dipy_data/blob/master/stanford_hardi_denoise_LPCA.png?raw=true
   :scale: 100%
   :align: middle
.. |image13| image:: https://github.com/dipy/dipy_data/blob/master/stanford_hardi_denoise_MPPCA.png?raw=true
   :scale: 100%
   :align: middle
.. |image14| image:: https://github.com/dipy/dipy_data/blob/master/stanford_hardi_denoise_NLMEANS.png?raw=true
   :scale: 100%
   :align: middle
.. |image15| image:: https://github.com/dipy/dipy_data/blob/master/cfin_multib_original.png?raw=true
   :scale: 100%
   :align: middle
.. |image16| image:: https://github.com/dipy/dipy_data/blob/master/cfin_multib_LPCA.png?raw=true
   :scale: 100%
   :align: middle
.. |image17| image:: https://github.com/dipy/dipy_data/blob/master/cfin_multib_denoise_MPPCA.png?raw=true
   :scale: 100%
   :align: middle
.. |image18| image:: https://github.com/dipy/dipy_data/blob/master/cfin_multib_denoise_NLMEANS.png?raw=true
   :scale: 100%
   :align: middle
.. |image19| image:: https://github.com/dipy/dipy_data/blob/master/stanford_hardi_t1_original.png?raw=true
   :scale: 100%
   :align: middle
.. |image20| image:: https://github.com/dipy/dipy_data/blob/master/stanford_hardi_t1_NLMEANS.png?raw=true
   :scale: 100%
   :align: middle

Diffusion
---------

+--------------------+--------------------+--------------------+--------------------+--------------------+
|      Dataset       |   Original Image   |    Denoise LCPA    |   Denoise MPPCA    |   Denoise NLMEANS  |
+====================+====================+====================+====================+====================+
|  sherbrooke_3shell |      |image7|      |      |image8|      |      |image9|      |      |image10|     |
+--------------------+--------------------+--------------------+--------------------+--------------------+
|  stanford_hardi    |      |image11|     |      |image12|     |      |image13|     |      |image14|     |
+--------------------+--------------------+--------------------+--------------------+--------------------+
|  cfin_multib       |      |image15|     |      |image16|     |      |image17|     |      |image18|     |
+--------------------+--------------------+--------------------+--------------------+--------------------+

Structural
----------

+--------------------+--------------------+--------------------+
|      Dataset       |   Original Image   |  Denoise NLMEANS   |
+====================+====================+====================+
|  stanford_hardi T1 |      |image19|     |      |image20|     |
+--------------------+--------------------+--------------------+


References
----------
.. [Coupe08] P. Coupe, P. Yger, S. Prima, P. Hellier, C. Kervrann, C. Barillot,
    "An Optimized Blockwise Non Local Means Denoising Filter for 3D Magnetic
    Resonance Images", IEEE Transactions on Medical Imaging, 27(4):425-441, 2008
.. [Coupe11] Pierrick Coupe, Jose Manjon, Montserrat Robles, Louis Collins.
    "Adaptive Multiresolution Non-Local Means Filter for 3D MR Image Denoising"
    IET Image Processing, Institution of Engineering and Technology, 2011