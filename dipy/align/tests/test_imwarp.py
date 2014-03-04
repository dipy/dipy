import numpy as np
from numpy.testing import (assert_equal,
                           assert_array_equal,
                           assert_array_almost_equal)
import matplotlib.pyplot as plt
import dipy.align.imwarp as imwarp
import dipy.align.metrics as metrics 
import dipy.align.vector_fields as vfu
import dipy.align.registration_common as rcommon
from dipy.data import get_data
from dipy.align import floating
import nibabel as nib

def getRotationMatrix(angles):
    ca=np.cos(angles[0])
    cb=np.cos(angles[1])
    cg=np.cos(angles[2])
    sa=np.sin(angles[0])
    sb=np.sin(angles[1])
    sg=np.sin(angles[2])
    return np.array([[cb*cg,-ca*sg+sa*sb*cg,sa*sg+ca*sb*cg],[cb*sg,ca*cg+sa*sb*sg,-sa*cg+ca*sb*sg],[-sb,sa*cb,ca*cb]])

def test_ssd_2d():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration
    '''
    fname_moving = get_data('reg_o')
    fname_fixed = get_data('reg_c')

    moving = plt.imread(fname_moving)
    fixed = plt.imread(fname_fixed)
    moving = moving[:, :, 0].astype(floating)
    fixed = fixed[:, :, 0].astype(floating)
    moving = np.array(moving, dtype = floating)
    fixed = np.array(fixed, dtype = floating)
    moving = (moving-moving.min())/(moving.max() - moving.min())
    fixed = (fixed-fixed.min())/(fixed.max() - fixed.min())
    #Configure and run the Optimizer
    smooth = 4
    inner_iter =5
    step_length = 0.25
    step_type = 0
    similarity_metric = metrics.SSDMetric(2, smooth, inner_iter, step_length, step_type) 
    opt_iter = [20, 100, 100, 100]
    opt_tol = 1e-4
    inv_iter = 40
    inv_tol = 1e-3
    reportStatus = False
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        2, fixed, moving, None, similarity_metric, imwarp.compose_displacements, 
        opt_iter, opt_tol, inv_iter, inv_tol, reportStatus)
    registration_optimizer.optimize()
    energy_profile = registration_optimizer.full_energy_profile
    expected_profile = [302.61125317089767, 297.6436794411608, 293.67260699136824,
    290.60303831315383, 288.16098933682366, 286.1445999573916,
    284.4364649588415, 281.8112893717045, 279.0796365979373,
    276.7102547768021, 274.7369256624365, 273.012162541487,
    702.300837028873, 695.0943626501489, 684.6428323887304,
    674.3030143419778, 663.7864435035457, 653.136524343007,
    643.2242842058321, 633.7112211654177, 624.685760970435,
    615.74946592097, 606.8390237201834, 597.1126106280611,
    179.33400779171248, 166.30313504036184, 142.42238280274464,
    122.65367889871, 106.44073315886713, 95.41108800097213,
    87.96182679672143, 82.61402880855695, 78.19231311503307,
    74.47642022606254, 71.15748591895019, 68.06597585727408]
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile), decimal=3)
    #######################show results#################################
    displacement = registration_optimizer.get_forward()
    direct_inverse = registration_optimizer.get_backward()
    moving_to_fixed = np.array(vfu.warp_image(moving, displacement))
    fixed_to_moving = np.array(vfu.warp_image(fixed, direct_inverse))
    rcommon.overlayImages(moving_to_fixed, fixed, True)
    rcommon.overlayImages(fixed_to_moving, moving, True)
    direct_residual, stats = vfu.compose_vector_fields(displacement,
                                                     direct_inverse)
    direct_residual = np.array(direct_residual)
    rcommon.plotDiffeomorphism(displacement, direct_inverse, direct_residual,
                               'inv-direct', 7)


def test_cc_3d():
    from dipy.data import read_sherbrooke_3shell

    img, gtab = read_sherbrooke_3shell()

    target = np.array(img.get_data()[..., 0], dtype = floating)

    #Warp the S0 with a synthetic rotation
    degrees = np.array([2.0, 3.0, 4.0])
    angles = degrees * (np.pi/180.0)
    rotation = getRotationMatrix(angles).astype(floating)
    new_shape = np.array(target.shape, dtype = np.int32)
    reference = np.asarray(vfu.warp_volume_affine(target, new_shape, rotation))

    #Create the CC metric
    step_length = 0.25
    sigma_diff = 3.0
    radius = 4
    similarity_metric = metrics.CCMetric(3, step_length, sigma_diff, radius)

    #Create the optimizer
    opt_iter = [5, 10, 10]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    reportStatus = False
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        3, reference, target, None, similarity_metric, imwarp.compose_displacements, 
        opt_iter, opt_tol, inv_iter, inv_tol, reportStatus)
    registration_optimizer.optimize()
    energy_profile = 1e-6 * np.array(registration_optimizer.full_energy_profile)
    expected_profile = 1e-6 * np.array([-15763.543499318299, -18746.625000814667, -20160.312070620796, 
    -20951.446057415866, -21680.17488217326, -22354.501210638806, 
    -22683.407001490395, -23244.38786732867, -23786.579623749625, 
    -24171.656863448723, -115548.1069087715, -133171.4764221798, 
    -136956.3675746713, -143931.32627938036, -144240.57626152827, 
    -146812.38023202776, -147219.9288492704, -149772.61647280722, 
    -150492.3160459624, -152611.88737725923])
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile), decimal=5)


if __name__=='__main__':
    #test_ssd_2d()
    test_cc_3d()