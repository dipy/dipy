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
    return np.array([[cb*cg,-ca*sg+sa*sb*cg,sa*sg+ca*sb*cg],
                     [cb*sg,ca*cg+sa*sb*sg,-sa*cg+ca*sb*sg],
                     [-sb,sa*cb,ca*cb]])


def test_ssd_2d():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration
    '''
    fname_moving = get_data('reg_o')
    fname_static = get_data('reg_c')

    moving = plt.imread(fname_moving)
    static = plt.imread(fname_static)
    moving = moving[:, :, 0].astype(floating)
    static = static[:, :, 0].astype(floating)
    moving = np.array(moving, dtype = floating)
    static = np.array(static, dtype = floating)
    moving = (moving-moving.min())/(moving.max() - moving.min())
    static = (static-static.min())/(static.max() - static.min())
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
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, 2, imwarp.compose_displacements, 
        opt_iter, opt_tol, inv_iter, inv_tol, None)
    registration_optimizer.optimize(static, moving, None)
    subsampled_energy_profile = registration_optimizer.full_energy_profile[::10]
    if floating is np.float32:
        expected_profile = [302.6112060546875, 274.73687744140625, 258.37188720703125, 
                            242.0118865966797, 225.96595764160156, 212.8796844482422, 
                            200.59584045410156, 188.8486328125, 178.32041931152344, 
                            165.65579223632812, 702.3003540039062, 606.8388061523438, 
                            511.5794372558594, 417.9437255859375, 329.8865661621094, 
                            242.92117309570312, 165.19544982910156, 99.67949676513672, 
                            56.240074157714844, 39.08817672729492, 179.33363342285156, 
                            71.15731811523438, 51.66040802001953, 43.185237884521484, 
                            37.47501754760742, 34.42680358886719, 32.244903564453125, 
                            29.302459716796875, 28.516944885253906, 26.80443000793457]
    else:
        expected_profile = [302.61125317089767, 274.7369256624365, 258.3718071091768, 
                            242.01193676614497, 225.96598999638158, 212.87967285363396, 
                            200.5959806064401, 188.84863550148992, 178.3204633084462, 
                            165.6558070298394, 702.300837028873, 606.8390237201834, 
                            511.5795215789606, 417.944226511893, 329.88682685347106, 
                            242.92150013784828, 165.1957684235344, 99.67985374850804, 
                            56.24016825599313, 39.088227648263825, 179.33400779171248, 
                            71.15748591895019, 51.66042879906375, 43.18517211651795, 
                            37.47503071707744, 34.426881654216494, 32.24493906419912, 
                            29.302506040713634, 28.516894783752793, 26.804434032428883]
    assert_array_almost_equal(np.array(subsampled_energy_profile), np.array(expected_profile), decimal=6)


def test_cc_3d():
    from dipy.data import read_sherbrooke_3shell

    img, gtab = read_sherbrooke_3shell()

    moving = np.array(img.get_data()[..., 0], dtype = floating)

    #Warp the S0 with a synthetic rotation
    degrees = np.array([2.0, 3.0, 4.0])
    angles = degrees * (np.pi/180.0)
    rotation = getRotationMatrix(angles).astype(floating)
    new_shape = np.array(moving.shape, dtype = np.int32)
    static = np.asarray(vfu.warp_volume_affine(moving, new_shape, rotation))

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
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, 3, imwarp.compose_displacements, 
        opt_iter, opt_tol, inv_iter, inv_tol, None)
    registration_optimizer.optimize(static, moving, None)
    energy_profile = np.array(registration_optimizer.full_energy_profile) 
    if floating is np.float32:
        expected_profile = np.array([-15763.543499318299, -18746.625000814667, -20160.312070620796, 
                                     -20951.446057415866, -21680.17488217326, -22354.501210638806, 
                                     -22683.407001490395, -23244.38786732867, -23786.579623749625, 
                                     -24171.656863448723, -115548.1069087715, -133171.4764221798, 
                                     -136956.3675746713, -143931.32627938036, -144240.57626152827, 
                                     -146812.38023202776, -147219.9288492704, -149772.61647280722, 
                                     -150492.3160459624, -152611.88737725923])
    else:
        expected_profile = np.array([-15763.54423899, -18746.62574691, -20160.31307454,  
                                     -20951.44677417, -21680.17524703, -22354.50221778,
                                     -22683.4081272, -23244.38845643, -23786.58025344, 
                                     -24171.65692437, -115548.11123551, -133175.89330572,
                                     -136955.33769781, -143931.25245346, -144239.90675822, 
                                     -146824.79881482, -147244.51045622, -149816.17947782,
                                     -150527.54499074, -152670.81683311])
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile), decimal=6)


def test_em_3d():
    from dipy.data import read_sherbrooke_3shell

    img, gtab = read_sherbrooke_3shell()

    moving = np.array(img.get_data()[..., 0], dtype = floating)

    #Warp the S0 with a synthetic rotation
    degrees = np.array([2.0, 3.0, 4.0])
    angles = degrees * (np.pi/180.0)
    rotation = getRotationMatrix(angles).astype(floating)
    new_shape = np.array(moving.shape, dtype = np.int32)
    static = np.asarray(vfu.warp_volume_affine(moving, new_shape, rotation))
    moving = (moving - moving.min())/(moving.max() - moving.min())
    static = (static -static.min())/ (static.max() - static.min())
    #Create the EM metric
    smooth=25.0
    inner_iter=20
    step_length=0.25
    q_levels=256
    double_gradient=True
    iter_type='v_cycle'
    similarity_metric = metrics.EMMetric(
        3, smooth, inner_iter, step_length, q_levels, double_gradient, iter_type)

    #Create the optimizer
    opt_iter = [1, 5, 10]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, 3, imwarp.compose_displacements, 
        opt_iter, opt_tol, inv_iter, inv_tol, None)
    registration_optimizer.optimize(static, moving, None)
    energy_profile = registration_optimizer.full_energy_profile
    if floating is np.float32:
        expected_profile =[11.12615966796875, 8.084357261657715, 6.636898040771484,
                            4.629724383354187, 4.004666566848755, 3.1289035081863403,
                            2.2731465697288513, 1.8173362612724304, 2.061128258705139,
                            1.6410276293754578, 31.634721755981445, 24.582207679748535,
                            19.60957908630371, 15.937037467956543, 13.944169521331787]
    else:
        expected_profile =[11.126297989876795, 8.084506642727089, 6.636979472116404, 
                            4.62543551294909, 3.9926128517335844, 3.0231896806152454, 
                            1.929883720362989, 1.562734306076318, 2.069354258402535, 
                            2.044004912659469, 28.434427672995895, 22.07834272698154, 
                            17.817407211769005, 15.205636938768833, 13.310639093692913]
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile), decimal=6)


if __name__=='__main__':
    test_ssd_2d()
    test_cc_3d()
    test_em_3d()