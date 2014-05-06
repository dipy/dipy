from __future__ import print_function
import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal, 
                           assert_array_equal,
                           assert_array_almost_equal)
import matplotlib.pyplot as plt
import dipy.align.imwarp as imwarp
import dipy.align.metrics as metrics 
import dipy.align.vector_fields as vfu
from dipy.data import get_data
from dipy.align import floating
import nibabel as nib
from dipy.align.imwarp import DiffeomorphicMap


def test_ssd_2d_demons():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of SSD in
    2D using the Demons step, and this test checks that the current energy
    profile matches the saved one.
    '''
    fname_moving = get_data('reg_o')
    fname_static = get_data('reg_c')

    moving = plt.imread(fname_moving)
    static = plt.imread(fname_static)
    moving = moving[:, :, 0].astype(floating)
    static = static[:, :, 0].astype(floating)
    moving = np.array(moving, dtype=floating)
    static = np.array(static, dtype=floating)
    moving = (moving-moving.min())/(moving.max() - moving.min())
    static = (static-static.min())/(static.max() - static.min())
    #Create the SSD metric
    smooth = 4
    step_type = 'demons'
    similarity_metric = metrics.SSDMetric(2, smooth=smooth, step_type=step_type) 

    #Configure and run the Optimizer
    opt_iter = [25, 50, 100, 200]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 40
    inv_tol = 1e-3
    ss_sigma_factor = 0.2
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric, 
        opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    mapping = optimizer.optimize(static, moving, None)
    subsampled_energy_profile = np.array(optimizer.full_energy_profile[::10])
    if floating is np.float32:
        expected_profile = \
            np.array([ 312.6813333 ,  164.59050263,  103.73623002,   82.1164849,
                       63.31888794,   57.02372298,   48.88254136,   45.4015576 ,
                       42.45817589,  174.94422108,   92.43030985,   58.73123347,
                       43.70869018,   15.79207659,   20.30039959,   41.99069232,
                       37.1587315 ,   33.1963267 ,   32.89163671,   87.82289011,
                       78.28761195])
    else:
        expected_profile = \
            np.array([ 312.68133361,  164.59049075,  103.73635218,  82.11638224,
                       63.3188368 ,   57.02375694,   48.88245596,   45.4014475 ,
                       42.4579966 ,  174.94167955,   92.42725191,   58.72655199,
                       43.71955268,   15.78579491,   20.45497118,   41.92597862,
                       37.60531526,   33.25877969,   30.638574  ,   91.49825032,
                       80.524506  ])
    assert_array_almost_equal(subsampled_energy_profile, expected_profile)


def test_ssd_2d_gauss_newton():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of SSD in
    2D using the Gauss Newton step, and this test checks that the current energy 
    profile matches the saved one.
    '''
    fname_moving = get_data('reg_o')
    fname_static = get_data('reg_c')

    moving = plt.imread(fname_moving)
    static = plt.imread(fname_static)
    moving = moving[:, :, 0].astype(floating)
    static = static[:, :, 0].astype(floating)
    moving = np.array(moving, dtype=floating)
    static = np.array(static, dtype=floating)
    moving = (moving-moving.min())/(moving.max() - moving.min())
    static = (static-static.min())/(static.max() - static.min())
    #Create the SSD metric
    smooth = 4
    inner_iter = 5
    step_type = 'gauss_newton'
    similarity_metric = metrics.SSDMetric(2, smooth, inner_iter, step_type) 

    #Configure and run the Optimizer
    opt_iter = [25, 50, 100, 200]
    step_length = 0.5
    opt_tol = 1e-4
    inv_iter = 40
    inv_tol = 1e-3
    ss_sigma_factor = 0.2
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    mapping = optimizer.optimize(static, moving, None)
    subsampled_energy_profile = np.array(optimizer.full_energy_profile[::10])
    if floating is np.float32:
        expected_profile = \
            np.array([ 312.68133316,   79.81322732,   28.37105316,   24.3985506,
                       13.92768078,   11.52267765,    9.11339687,   27.28819896,
                       42.9770759 ,  237.44444211,  153.43258717,  137.2169711])
    else:
        expected_profile = \
            np.array([ 312.68133361,   79.8132289 ,   27.28523819,  24.22883738,
                       56.71942103,   30.20320996,   19.4766414 ,   74.72561337,
                       108.0512537 ,  106.37445697])
    assert_array_almost_equal(subsampled_energy_profile, expected_profile)


def get_synthetic_warped_circle(nslices):
    #get a subsampled circle
    fname_cicle = get_data('reg_o')
    circle = plt.imread(fname_cicle)[::4,::4,0].astype(floating)
    
    #create a synthetic invertible map and warp the circle
    d, dinv = vfu.create_harmonic_fields_2d(64, 64, 0.1, 4)
    d = np.asarray(d, dtype=floating)
    dinv = np.asarray(dinv, dtype=floating)
    mapping = DiffeomorphicMap(2, (64, 64))
    mapping.forward, mapping.backward = d, dinv
    wcircle = mapping.transform(circle)

    if(nslices == 1):
        return circle, wcircle

    #normalize and form the 3d by piling slices
    circle = (circle-circle.min())/(circle.max() - circle.min())
    circle_3d = np.ndarray(circle.shape + (nslices,), dtype=floating)
    circle_3d[...] = circle[...,None]
    circle_3d[...,0] = 0
    circle_3d[...,-1] = 0

    #do the same with the warped circle
    wcircle = (wcircle-wcircle.min())/(wcircle.max() - wcircle.min())
    wcircle_3d = np.ndarray(wcircle.shape + (nslices,), dtype=floating)
    wcircle_3d[...] = wcircle[...,None]
    wcircle_3d[...,0] = 0
    wcircle_3d[...,-1] = 0

    return circle_3d, wcircle_3d


def test_ssd_3d_demons():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with 
    a synthetic diffeomorphism. This test is intended to detect regressions
    only: we saved the energy profile (the sequence of energy values at each
    iteration) of a working version of SSD in 3D using the Demons step, and this
    test checks that the current energy profile matches the saved one. The
    validation of the "working version" was done by registering the 18 manually
    annotated T1 brain MRI database IBSR with each other and computing the
    jaccard index for all 31 common anatomical regions. 
    '''
    moving, static = get_synthetic_warped_circle(20)

    #Create the SSD metric
    smooth = 4
    step_type = 'demons'
    similarity_metric = metrics.SSDMetric(3, smooth=smooth, step_type=step_type) 

    #Create the optimizer
    opt_iter = [5, 10]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    mapping = optimizer.optimize(static, moving, None)
    energy_profile = np.array(optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = \
        np.array([601.17342436,   468.94537817,   418.67267847,   393.0580613,
                  367.8863422 ,   319.61865314,   272.3558511 ,   269.57838565,
                  254.63664301,   266.9605625 ,  2541.47438277,  2033.988534,
                  1779.69793906,  1693.11368711,  1653.95419258])
    else:
        expected_profile = \
            np.array([  601.17344986, 468.97523898, 418.73047322, 393.0534384,
                        367.80005903, 319.44987629, 272.62769902, 268.10394736,
                        254.30487935, 267.7249719, 2547.05251526, 2035.19403818,
                        1780.21839845,  1692.64443559,  1653.6224987 ])
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


def test_ssd_3d_gauss_newton():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with 
    a synthetic diffeomorphism. This test is intended to detect regressions 
    only: we saved the energy profile (the sequence of energy values at each
    iteration) of a working version of SSD in 3D using the Gauss-Newton step,
    and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR 
    with each other and computing the jaccard index for all 31 common anatomical
    regions. 
    '''
    moving, static = get_synthetic_warped_circle(20)

    #Create the SSD metric
    smooth = 4
    inner_iter = 5
    step_type = 'gauss_newton'
    similarity_metric = metrics.SSDMetric(3, smooth, inner_iter, step_type) 

    #Create the optimizer
    opt_iter = [5, 10]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    mapping = optimizer.optimize(static, moving, None)
    energy_profile = np.array(optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = \
            np.array([601.17342027, 398.62217561, 300.29774716, 263.11446187,
                      279.6762572 , 280.86396456, 292.32200297, 288.94831405,
                      296.04799   , 290.72802694, 2723.60750459, 2400.04528908,
                      2235.94249224, 2152.32966366, 2128.3250948])
    else:
        expected_profile = \
            np.array([601.17344986, 398.62218184, 300.29775583, 263.11445705,
                      279.67625651, 280.86396779, 292.32201095, 288.94831954,
                      296.04799876, 290.72802577, 2723.60787772, 2400.0456365,
                      2235.94286635, 2152.33001603, 2128.32545284])
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


def test_cc_2d():
    r'''
    Register two slices from the Sherbrooke database. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of CC in
    2D, and this test checks that the current energy profile matches the saved
    one.
    '''

    moving, static = get_synthetic_warped_circle(1)

    #Configure the metric
    sigma_diff = 3.0
    radius = 4
    metric = metrics.CCMetric(2, sigma_diff, radius)

    #Configure and run the Optimizer
    opt_iter = [10, 20, 40]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, opt_iter)
    mapping = optimizer.optimize(static, moving, None)
    energy_profile = np.array(optimizer.full_energy_profile)
    
    if floating is np.float32:
        expected_profile = \
            [   -435.79559516,  -460.80739355,  -469.88508346,  -486.87396486,
                -486.04298263,  -484.30780055,  -489.19779192,  -484.44738633,
                -489.17020371,  -485.6637196 ,  -488.70801039,  -487.46399496,
                -489.71671264,  -488.09117139,  -490.42271222,  -488.27909614,
                -490.28857064,  -487.60445667,  -490.03035784,  -485.72591888,
                -490.60729319, -1260.19301574, -1327.14719131, -1309.49160837,
                -1342.19150863, -1356.90061164, -1275.25601701, -1317.07887913,
                -1343.0784944 , -1301.45605487, -1336.04013439, -1366.93546512,
                -1328.10275902, -1317.85372622, -1317.62486769, -1274.53697105,
                -1337.79152122, -2801.90904108, -2857.68596628, -2849.56767541,
                -2867.77931765, -2846.8404648 , -2875.67021308, -2851.85228212,
                -2879.43368375, -2861.36274169, -2889.69112071]
    else:
        expected_profile = \
            [   -435.7955967 ,  -460.80739935,  -469.88508352,  -486.87396919,
                -486.0429746 ,  -484.30780608,  -489.19779364,  -484.44739074,
                -489.17020447,  -485.66372153,  -488.7080131 ,  -487.46399372,
                -489.71671982,  -488.09117245,  -490.42271431,  -488.27909883,
                -490.28856556,  -487.60445041,  -490.03035556,  -485.72592274,
                -490.60729406, -1258.19305758, -1358.34000624, -1348.08308818,
                -1376.5332102 , -1361.61634539, -1371.62866869, -1354.9690168 ,
                -1356.56553571, -1365.8866856 , -1308.45095778, -1366.49097861,
                -1330.98891026, -1353.73575477, -2765.92375447, -2871.07572026,
                -2885.22181863, -2873.25158879, -2883.36175689, -2882.74507256,
                -2892.91338306, -2891.84375023, -2894.12822118, -2890.7756098 ]
    expected_profile = np.asarray(expected_profile)
    assert_array_almost_equal(energy_profile, expected_profile)


def test_cc_3d():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with 
    a synthetic diffeomorphism. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of CC in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR 
    with each other and computing the jaccard index for all 31 common anatomical
    regions. The "working version" of CC in 3D obtains very similar results as
    those reported for ANTS on the same database with the same number of
    iterations. Any modification that produces a change in the energy profile
    should be carefully validated to ensure no accuracy loss.
    '''
    moving, static = moving, static = get_synthetic_warped_circle(20)

    #Create the CC metric
    sigma_diff = 2.0
    radius = 4
    similarity_metric = metrics.CCMetric(3, sigma_diff, radius)

    #Create the optimizer
    opt_iter = [5, 10, 20]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    mapping = optimizer.optimize(static, moving, None)
    energy_profile = np.array(optimizer.full_energy_profile)*1e-4
    if floating is np.float32:
        expected_profile = \
            [   -0.23135541, -0.22171793, -0.23767394, -0.24236032, -0.22654608,
                -0.19675488, -0.24164528, -0.24076027, -0.22999321, -0.22685398,
                -0.20686259, -0.23939138, -0.24139779, -1.32298218, -1.37421899,
                -1.37280958, -1.38166606, -1.37794505, -1.38500984, -1.38071534,
                -1.37929357, -1.37501299, -1.38839658, -6.12090669, -6.19221629,
                -6.19314241, -6.13668367, -6.11476345]
    else:
        expected_profile = \
            [   -0.23135541, -0.22171793, -0.23767394, -0.24236032, -0.22654608,
                -0.19675488, -0.24164527, -0.24076027, -0.22999321, -0.22685398,
                -0.20686259, -0.23939137, -0.24139779, -1.32178231, -1.37421862,
                -1.37280946, -1.38166568, -1.37794478, -1.38500996, -1.38071547,
                -1.37928428, -1.37501037, -1.38838905, -6.11733785, -6.4959287,
                -6.63564872, -6.6980932 , -6.74961869]
    expected_profile = np.asarray(expected_profile)
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


def test_em_3d():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with 
    a synthetic diffeomorphism. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of EM in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR 
    with each other and computing the jaccard index for all 31 common anatomical
    regions. The "working version" of EM in 3D obtains very similar results as
    those reported for ANTS on the same database. Any modification that produces
    a change in the energy profile should be carefully validated to ensure no 
    accuracy loss.
    '''
    moving, static = moving, static = get_synthetic_warped_circle(20)

    #Create the EM metric
    smooth=25.0
    inner_iter=20
    step_length=0.25
    q_levels=256
    double_gradient=True
    iter_type='gauss_newton'
    similarity_metric = metrics.EMMetric(
        3, smooth, inner_iter, q_levels, double_gradient, iter_type)

    #Create the optimizer
    opt_iter = [2, 5, 10]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    mapping = optimizer.optimize(static, moving, None)
    energy_profile = np.array(optimizer.full_energy_profile)*1e-3
    if floating is np.float32:
        expected_profile = \
            np.array([  1.43656283e-02,   1.59238212e-02,   1.65779722e-02,
                        1.19135204e-02,   2.27319887e-02,   1.86793964e-02,
                        1.50664627e-02,   1.26429582e-02,   1.24745213e-02,
                        8.65137197e-03,   3.02190182e-01,   1.75392988e-01,
                        1.80810344e-01,   1.78224223e-01,   1.82246495e-01,
                        2.81197199e+01,   2.00932495e+01])
    else:
        expected_profile = \
            np.array([  1.43656285e-02,   1.59238234e-02,   1.65779716e-02,
                        1.19135203e-02,   2.27319848e-02,   1.86793883e-02,
                        1.50664623e-02,   1.26429540e-02,   1.25667716e-02,
                        1.17727709e-02,   2.99312690e-01,   1.76228197e-01,
                        2.40907927e-01,   1.70675610e-01,   1.64373533e-01,
                        2.48934851e+01,   2.14020096e+01])
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


def test_em_2d():
    r'''
    Register two slices from the Sherbrooke database. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of CC in
    2D, and this test checks that the current energy profile matches the saved
    one.
    '''

    moving, static = get_synthetic_warped_circle(1)

    #Configure the metric
    smooth=25.0
    inner_iter=20
    step_length=0.25
    q_levels=256
    double_gradient=False
    iter_type='gauss_newton'
    metric = metrics.EMMetric(
        2, smooth, inner_iter, q_levels, double_gradient, iter_type)

    #Configure and run the Optimizer
    opt_iter = [10, 20, 40]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, opt_iter)
    mapping = optimizer.optimize(static, moving, None)
    energy_profile = np.array(optimizer.full_energy_profile)
    
    if floating is np.float32:
        expected_profile = \
            np.array([   5.07521773, 3.93179134, 3.85313489, 4.16914495,
                         2.90004982, 3.16738105, 2.69627741, 2.59370897,
                         2.5180502 , 2.51041629, 2.69912632, 2.55035876,
                         2.46371317, 2.67660867, 2.49318511, 2.94369408,
                         2.77158745, 2.82379763, 17.11120265, 16.58076545,
                         14.0681015 , 14.54551446, 14.65197388, 13.49130845,
                         13.09843029, 13.35734757, 13.48209453, 13.74433992,
                         13.94080819, 14.07583289, 14.13893569, 14.82990374,
                         14.49082004, 106.04095586, 101.70510575, 99.77728753,
                         102.36543498, 101.1531505 , 102.85582307, 104.86979379,
                         103.71664918, 103.46962525, 103.90057054])
    else:
        expected_profile = \
            np.array([  5.07521878,   3.93179259,   3.85313625,   4.16914645,
                        2.90004626,   3.16737784,   2.69626851,   2.59365884,
                        2.51799588,   2.51044257,   2.69915148,   2.55039255,
                        2.46377307,   2.72547961,   2.56444664,   3.0655473 ,
                        2.77688171,  21.16233049,  17.91300118,  16.03890266,
                        12.85097448,  14.19668193,  13.56659675,  12.38216363,
                        12.32214476,  12.51793663,  12.60944975,  12.64067646,
                        12.41519846,  12.43092523,  12.48408716,  12.25886823,
                        12.36918177,  12.43335928,  12.57647121,  12.67075872,
                        12.76274505,  91.88512699,  86.77080544,  85.61197959,
                        84.63291853,  83.49849488,  84.96890764,  87.67030269,
                        87.3451905 ,  86.08676152,  84.24546416])
    assert_array_almost_equal(energy_profile, expected_profile)


if __name__=='__main__':
    test_ssd_2d_demons()
    test_ssd_2d_gauss_newton()
    test_ssd_3d_demons()
    test_ssd_3d_gauss_newton()
    test_cc_2d()
    test_cc_3d()
    test_em_2d()
    test_em_3d()
