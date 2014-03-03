import numpy as np
from numpy.testing import (assert_equal,
                           assert_array_equal,
                           assert_array_almost_equal)
import matplotlib.pyplot as plt
from dipy.align.imwarp import Composition, SymmetricRegistrationOptimizer
from dipy.align.metrics import SSDMetric
import dipy.align.vector_fields as vfu
import dipy.align.registration_common as rcommon
floating  = np.float32

def test_optimizer_monomodal_2d():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration
    '''
    fname_moving = '../../data/circle.png'
    fname_fixed = '../../data/C.png'
    moving = plt.imread(fname_moving)
    fixed = plt.imread(fname_fixed)
    moving = moving[:, :, 0].astype(floating)
    fixed = fixed[:, :, 0].astype(floating)
    moving = np.array(moving, order = 'C', dtype = floating)
    fixed = np.array(fixed, order = 'C', dtype = floating)
    moving = (moving-moving.min())/(moving.max() - moving.min())
    fixed = (fixed-fixed.min())/(fixed.max() - fixed.min())
    ################Configure and run the Optimizer#####################
    max_iter = [i for i in [20, 100, 100, 100]]
    similarity_metric = SSDMetric(2, {'symmetric':True,
                                'lambda':4.0,
                                'stepType':SSDMetric.GAUSS_SEIDEL_STEP})
    optimizer_parameters = {
        'max_iter':max_iter,
        'inversion_iter':40,
        'inversion_tolerance':1e-3,
        'report_status':True}
    update_rule = Composition()
    registration_optimizer = SymmetricRegistrationOptimizer(2, fixed, moving,
                                                         None, None,
                                                         similarity_metric,
                                                         update_rule, optimizer_parameters)
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
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile))
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



# def test_feature():

#     A = np.ones((3, 3))
#     B = A + 2

#     assert_array_equal(A, B)
if __name__=='__main__':
    test_optimizer_monomodal_2d()
