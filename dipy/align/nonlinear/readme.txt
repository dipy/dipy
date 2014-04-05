

                          2D Experiment

To run the classic 2D Circle-to-C experiment using the SSD metric (from an ipython session):
In [1]: from SymmetricRegistrationOptimizer import *
In [2]: test_optimizer_monomodal_2d()



                          3D Experiment


To run the 3D Symmetric Normalization with Cross-Correlation (the counterpart to ANTS default), from the command line:

python dipyreg.py $target $reference $affine $warpdir --metric=CC[0.25,3.0,4] --iter=5,10,10

where:
$target is the file name of the target (moving) image (e.g. a .nii.gz file),
$reference is the file name of the reference (fixed) image (e.g. a .nii.gz file),
$affine is the file name of the affine transformation matrix IN ANTS FORMAT to linearly align $target to $reference 
$warpdir is the folder name containing the volumes to be warped using the resulting transformation (this is currently mandatory to avoid writing huge binary transformation files)

The rest of the parameters specified in the above command are the ANTS defaults: Cross-Correlation (CC) metric with 0.25 step size, 3.0 smoothing sigma, and CC neighborhood radious
of 4 voxels. The number of iterations was indicated to be 5 at the 0th (finest) level of the Gaussian pyramid and 10 iterations at each of the coarcer pyramid levels.

The affine transformation matrix can be obtained by executing the following ANTS command:

ANTS 3 -m MI[$reference, $target, 1, 32] -i 0 -o $affine_base

where $affine_base is the same as $afine defined above but without the ".txt" extension. This will generate the aligning matrix and save it in a text file names "$affine_base".txt

