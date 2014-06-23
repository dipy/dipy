from __future__ import print_function

import unittest
import numpy as np

from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_raises

import dipy.core.streamline as dipystreamline


def length_python(xyz, along=False):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] < 2:
        if along:
            return np.array([0])
        return 0
    dists = np.sqrt((np.diff(xyz, axis=0)**2).sum(axis=1))
    if along:
        return np.cumsum(dists)
    return np.sum(dists)


def resample_python(xyz, n_pols=3):
    def _extrap(xyz, cumlen, distance):
        ''' Helper function for extrapolate '''
        ind = np.where((cumlen-distance) > 0)[0][0]
        len0 = cumlen[ind-1]
        len1 = cumlen[ind]
        Ds = distance-len0
        Lambda = Ds/(len1-len0)
        return Lambda*xyz[ind]+(1-Lambda)*xyz[ind-1]

    cumlen = np.zeros(xyz.shape[0])
    cumlen[1:] = length_python(xyz, along=True)
    step = cumlen[-1]/(n_pols-1)

    ar = np.arange(0, cumlen[-1], step)
    if np.abs(ar[-1]-cumlen[-1]) < np.finfo('f4').eps:
        ar = ar[:-1]

    xyz2 = [_extrap(xyz, cumlen, distance) for distance in ar]
    return np.vstack((np.array(xyz2), xyz[-1]))


class TestStreamline(unittest.TestCase):
    def setUp(self):
        self.streamline = np.array([[82.20181274,  91.3650589 ,  43.15737152],
                                    [82.3844223 ,  91.79336548,  43.87036514],
                                    [82.48710632,  92.27861023,  44.56298065],
                                    [82.53310394,  92.7854538 ,  45.24635315],
                                    [82.53793335,  93.26902008,  45.94785309],
                                    [82.48797607,  93.75003815,  46.6493988 ],
                                    [82.35533142,  94.2518158 ,  47.32533264],
                                    [82.15484619,  94.76634216,  47.97451019],
                                    [81.90982819,  95.28792572,  48.6024437 ],
                                    [81.63336945,  95.78153229,  49.23971176],
                                    [81.35479736,  96.24868011,  49.89558792],
                                    [81.08713531,  96.69807434,  50.56812668],
                                    [80.81504822,  97.14285278,  51.24193192],
                                    [80.52591705,  97.56719971,  51.92168427],
                                    [80.26599884,  97.98269653,  52.61848068],
                                    [80.0463562 ,  98.38131714,  53.3385582 ],
                                    [79.8469162 ,  98.77052307,  54.06955338],
                                    [79.57667542,  99.13599396,  54.78985596],
                                    [79.23351288,  99.4320755 ,  55.51065063],
                                    [78.84815979,  99.64141846,  56.24016571],
                                    [78.47383881,  99.77347565,  56.9929924 ],
                                    [78.12837219,  99.81330872,  57.76969528],
                                    [77.80438995,  99.85082245,  58.55574799],
                                    [77.4943924 ,  99.88065338,  59.34777069],
                                    [77.21414185,  99.85343933,  60.15090561],
                                    [76.96416473,  99.82772827,  60.96406937],
                                    [76.74712372,  99.80519104,  61.78676605],
                                    [76.52263641,  99.79122162,  62.60765076],
                                    [76.03757477, 100.08692169,  63.24152374],
                                    [75.44867706, 100.3526535 ,  63.79513168],
                                    [74.78033447, 100.57255554,  64.272789  ],
                                    [74.11605835, 100.7733078 ,  64.76428986],
                                    [73.51222992, 100.98779297,  65.32373047],
                                    [72.97387695, 101.23387146,  65.93502045],
                                    [72.47355652, 101.49151611,  66.57343292],
                                    [71.99834442, 101.72480774,  67.2397995 ],
                                    [71.5690918 , 101.98665619,  67.92664337],
                                    [71.18083191, 102.29483795,  68.61888123],
                                    [70.81879425, 102.63343048,  69.31127167],
                                    [70.47422791, 102.98672485,  70.00532532],
                                    [70.10092926, 103.28502655,  70.70999908],
                                    [69.69512177, 103.51667023,  71.42147064],
                                    [69.27423096, 103.71351624,  72.13452911],
                                    [68.91260529, 103.81676483,  72.89796448],
                                    [68.60788727, 103.81982422,  73.69258118],
                                    [68.34162903, 103.7661972 ,  74.49915314],
                                    [68.08542633, 103.70635223,  75.30856323],
                                    [67.83590698, 103.60187531,  76.11553955],
                                    [67.56822968, 103.4482193 ,  76.90870667],
                                    [67.28399658, 103.25878906,  77.68825531],
                                    [67.00117493, 103.03740692,  78.45989227],
                                    [66.72718048, 102.80329895,  79.23099518],
                                    [66.4619751 , 102.54130554,  79.99622345],
                                    [66.20803833, 102.22305298,  80.7438736 ],
                                    [65.96872711, 101.88980865,  81.48987579],
                                    [65.72864532, 101.59316254,  82.25085449],
                                    [65.47808075, 101.33383942,  83.02194214],
                                    [65.21841431, 101.11295319,  83.80186462],
                                    [64.95678711, 100.94080353,  84.59326935],
                                    [64.71759033, 100.82022095,  85.40114594],
                                    [64.48053741, 100.73490143,  86.21411896],
                                    [64.24304199, 100.65074158,  87.02709198],
                                    [64.01773834, 100.55318451,  87.84204865],
                                    [63.83801651, 100.41996765,  88.66333008],
                                    [63.70982361, 100.25119019,  89.48779297],
                                    [63.60707855, 100.06730652,  90.31262207],
                                    [63.46164322,  99.91001892,  91.13648224],
                                    [63.26287842,  99.78648376,  91.95485687],
                                    [63.03713226,  99.68377686,  92.76905823],
                                    [62.81192398,  99.56619263,  93.58140564],
                                    [62.57145309,  99.42708588,  94.38592529],
                                    [62.32259369,  99.25592804,  95.18167114],
                                    [62.07497787,  99.05770111,  95.97154236],
                                    [61.82253647,  98.83877563,  96.7543869 ],
                                    [61.59536743,  98.59293365,  97.5370636 ],
                                    [61.46530151,  98.30503845,  98.32772827],
                                    [61.39904785,  97.97928619,  99.11172485],
                                    [61.33279419,  97.65353394,  99.89572906],
                                    [61.26067352,  97.30914307, 100.67123413],
                                    [61.19459534,  96.96743011, 101.44847107],
                                    [61.1958046 ,  96.63417053, 102.23215485],
                                    [61.26572037,  96.2988739 , 103.01185608],
                                    [61.39840698,  95.96297455, 103.78307343],
                                    [61.5720787 ,  95.6426239 , 104.55268097],
                                    [61.78163528,  95.35540771, 105.32629395],
                                    [62.06700134,  95.09746552, 106.08564758],
                                    [62.39427185,  94.8572464 , 106.83369446],
                                    [62.74076462,  94.62278748, 107.57482147],
                                    [63.11461639,  94.40107727, 108.30641937],
                                    [63.53397751,  94.20418549, 109.02002716],
                                    [64.00019836,  94.03809357, 109.71183777],
                                    [64.43580627,  93.87523651, 110.42416382],
                                    [64.84857941,  93.69993591, 111.14715576],
                                    [65.26740265,  93.51858521, 111.86515808],
                                    [65.69511414,  93.3671875 , 112.58474731],
                                    [66.10470581,  93.22719574, 113.31711578],
                                    [66.45891571,  93.06028748, 114.07256317],
                                    [66.78582001,  92.90560913, 114.84281921],
                                    [67.11138916,  92.79004669, 115.6204071 ],
                                    [67.44729614,  92.75711823, 116.40135193],
                                    [67.75688171,  92.98265076, 117.16111755],
                                    [68.02041626,  93.28012848, 117.91371155],
                                    [68.25725555,  93.53466797, 118.69052124],
                                    [68.46047974,  93.63263702, 119.51107788],
                                    [68.62039948,  93.62007141, 120.34690094],
                                    [68.76782227,  93.56475067, 121.18331909],
                                    [68.90222168,  93.46326447, 122.01765442],
                                    [68.99872589,  93.30039978, 122.84759521],
                                    [69.04119873,  93.05428314, 123.66156769],
                                    [69.05086517,  92.74394989, 124.45450592],
                                    [69.02742004,  92.40427399, 125.23509979],
                                    [68.95466614,  92.09059143, 126.02339935],
                                    [68.84975433,  91.7967453 , 126.81564331],
                                    [68.72673798,  91.53726196, 127.61715698],
                                    [68.6068573 ,  91.3030014 , 128.42681885],
                                    [68.50636292,  91.12481689, 129.25317383],
                                    [68.39311218,  91.01572418, 130.08976746],
                                    [68.25946808,  90.94654083, 130.92756653]],
                                   dtype=np.float32)

        self.streamline_64bit = self.streamline.astype(np.float64)

        self.streamlines = [self.streamline,
                            self.streamline[::2], self.streamline[::3],
                            self.streamline[::5], self.streamline[::6]]
        self.streamlines_64bit = [self.streamline_64bit,
                                  self.streamline_64bit[::2], self.streamline_64bit[::3],
                                  self.streamline_64bit[::4], self.streamline_64bit[::5]]

    def test_resample(self):
        # Test resampling of only one streamline
        nb_points = 12
        resampled_streamline_cython = dipystreamline.resample(self.streamline, nb_points)
        resampled_streamline_python = resample_python(self.streamline, nb_points)
        assert_equal(len(resampled_streamline_cython), nb_points)
        # Using a 5 digits precision because of streamline is in float32.
        assert_array_almost_equal(resampled_streamline_cython, resampled_streamline_python, 5)

        resampled_streamline_cython = dipystreamline.resample(self.streamline_64bit, nb_points)
        resampled_streamline_python = resample_python(self.streamline_64bit, nb_points)
        assert_equal(len(resampled_streamline_cython), nb_points)
        assert_array_almost_equal(resampled_streamline_cython, resampled_streamline_python)

        res = []
        simple_streamline = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], 'f4')
        for nb_points in range(2, 200):
            resampled_streamline_cython = dipystreamline.resample(simple_streamline, nb_points)
            res.append(nb_points - len(resampled_streamline_cython))

        assert_equal(np.sum(res), 0)

        # Test resampling of multiple streamlines of different nb_points
        nb_points = 12
        resampled_streamlines_cython = dipystreamline.resample(self.streamlines, nb_points)

        for i, s in enumerate(self.streamlines):
            resampled_streamline_python = resample_python(s, nb_points)
            # Using a 5 digits precision because of streamline is in float32.
            assert_array_almost_equal(resampled_streamlines_cython[i], resampled_streamline_python, 5)

        resampled_streamlines_cython = dipystreamline.resample(self.streamlines_64bit, nb_points)

        for i, s in enumerate(self.streamlines_64bit):
            resampled_streamline_python = resample_python(s, nb_points)
            assert_array_almost_equal(resampled_streamlines_cython[i], resampled_streamline_python)

        # Test streamlines with mixed dtype
        streamlines = [self.streamline, self.streamline.astype(np.float64)]
        assert_raises(ValueError, dipystreamline.resample, streamlines, nb_points)

        # Test streamline with shape not Nx3
        assert_raises(ValueError, dipystreamline.resample, self.streamline.T, nb_points)

        # Test streamline with integer dtype
        resampled_streamline = dipystreamline.resample(self.streamline.astype(np.int32))
        assert_true(resampled_streamline.dtype == np.float32)
        resampled_streamline = dipystreamline.resample(self.streamline.astype(np.int64))
        assert_true(resampled_streamline.dtype == np.float64)

        # Test empty list
        assert_equal(dipystreamline.resample([]), [])

        # Test streamline having only one point
        assert_raises(ValueError, dipystreamline.resample, np.array([[1, 2, 3]]))

        # We do not support list of lists, it should be numpy ndarray.
        streamline = [[1, 2, 3], [4, 5, 5], [2, 1, 3], [4, 2, 1]]
        assert_raises(AttributeError, dipystreamline.resample, streamline)

    def test_length(self):
        # Test length of only one streamline
        length_streamline_cython = dipystreamline.length(self.streamline)
        length_streamline_python = length_python(self.streamline)
        assert_equal(length_streamline_cython, length_streamline_python)

        length_streamline_cython = dipystreamline.length(self.streamline_64bit)
        length_streamline_python = length_python(self.streamline_64bit)
        assert_equal(length_streamline_cython, length_streamline_python)

        # Test computing length of multiple streamlines of different nb_points
        length_streamlines_cython = dipystreamline.length(self.streamlines)

        for i, s in enumerate(self.streamlines):
            length_streamline_python = length_python(s)
            assert_array_almost_equal(length_streamlines_cython[i], length_streamline_python)

        length_streamlines_cython = dipystreamline.length(self.streamlines_64bit)

        for i, s in enumerate(self.streamlines_64bit):
            length_streamline_python = length_python(s)
            assert_array_almost_equal(length_streamlines_cython[i], length_streamline_python)

        # Test streamlines having mixed dtype
        streamlines = [self.streamline, self.streamline.astype(np.float64)]
        assert_raises(ValueError, dipystreamline.length, streamlines)

        # Test streamline having a shape not Nx3
        assert_raises(ValueError, dipystreamline.length, self.streamline.T)

        # Test streamline having integer dtype
        length_streamline = dipystreamline.length(self.streamline.astype('int'))
        assert_true(length_streamline.dtype == np.float64)

        # Test empty list
        assert_equal(dipystreamline.length([]), 0.0)

        # Test streamline having only one point
        assert_equal(dipystreamline.length(np.array([[1, 2, 3]])), 0.0)

        # We do not support list of lists, it should be numpy ndarray.
        streamline = [[1, 2, 3], [4, 5, 5], [2, 1, 3], [4, 2, 1]]
        assert_raises(AttributeError, dipystreamline.length, streamline)
