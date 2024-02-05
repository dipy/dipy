import warnings
import types

import numpy as np
from numpy.linalg import norm
import numpy.testing as npt
from dipy.testing.memory import get_type_refcount
from dipy.testing import assert_arrays_equal
from dipy.testing.decorators import set_random_number_generator

from dipy.testing import assert_true
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_raises, assert_allclose,
                           assert_almost_equal, assert_equal)

from dipy.tracking.streamlinespeed import (
    compress_streamlines, length, set_number_of_points,
)
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamline import (relist_streamlines,
                                      unlist_streamlines,
                                      center_streamlines,
                                      transform_streamlines,
                                      select_random_set_of_streamlines,
                                      select_by_rois,
                                      orient_by_rois,
                                      orient_by_streamline,
                                      values_from_volume,
                                      deform_streamlines,
                                      cluster_confidence)


streamline = np.array([[82.20181274,  91.36505890,  43.15737152],
                       [82.38442230,  91.79336548,  43.87036514],
                       [82.48710632,  92.27861023,  44.56298065],
                       [82.53310394,  92.78545380,  45.24635315],
                       [82.53793335,  93.26902008,  45.94785309],
                       [82.48797607,  93.75003815,  46.64939880],
                       [82.35533142,  94.25181580,  47.32533264],
                       [82.15484619,  94.76634216,  47.97451019],
                       [81.90982819,  95.28792572,  48.60244370],
                       [81.63336945,  95.78153229,  49.23971176],
                       [81.35479736,  96.24868011,  49.89558792],
                       [81.08713531,  96.69807434,  50.56812668],
                       [80.81504822,  97.14285278,  51.24193192],
                       [80.52591705,  97.56719971,  51.92168427],
                       [80.26599884,  97.98269653,  52.61848068],
                       [80.04635620,  98.38131714,  53.33855820],
                       [79.84691620,  98.77052307,  54.06955338],
                       [79.57667542,  99.13599396,  54.78985596],
                       [79.23351288,  99.43207550,  55.51065063],
                       [78.84815979,  99.64141846,  56.24016571],
                       [78.47383881,  99.77347565,  56.99299240],
                       [78.12837219,  99.81330872,  57.76969528],
                       [77.80438995,  99.85082245,  58.55574799],
                       [77.49439240,  99.88065338,  59.34777069],
                       [77.21414185,  99.85343933,  60.15090561],
                       [76.96416473,  99.82772827,  60.96406937],
                       [76.74712372,  99.80519104,  61.78676605],
                       [76.52263641,  99.79122162,  62.60765076],
                       [76.03757477, 100.08692169,  63.24152374],
                       [75.44867706, 100.35265350,  63.79513168],
                       [74.78033447, 100.57255554,  64.27278900],
                       [74.11605835, 100.77330780,  64.76428986],
                       [73.51222992, 100.98779297,  65.32373047],
                       [72.97387695, 101.23387146,  65.93502045],
                       [72.47355652, 101.49151611,  66.57343292],
                       [71.99834442, 101.72480774,  67.23979950],
                       [71.56909180, 101.98665619,  67.92664337],
                       [71.18083191, 102.29483795,  68.61888123],
                       [70.81879425, 102.63343048,  69.31127167],
                       [70.47422791, 102.98672485,  70.00532532],
                       [70.10092926, 103.28502655,  70.70999908],
                       [69.69512177, 103.51667023,  71.42147064],
                       [69.27423096, 103.71351624,  72.13452911],
                       [68.91260529, 103.81676483,  72.89796448],
                       [68.60788727, 103.81982422,  73.69258118],
                       [68.34162903, 103.76619720,  74.49915314],
                       [68.08542633, 103.70635223,  75.30856323],
                       [67.83590698, 103.60187531,  76.11553955],
                       [67.56822968, 103.44821930,  76.90870667],
                       [67.28399658, 103.25878906,  77.68825531],
                       [67.00117493, 103.03740692,  78.45989227],
                       [66.72718048, 102.80329895,  79.23099518],
                       [66.46197510, 102.54130554,  79.99622345],
                       [66.20803833, 102.22305298,  80.74387360],
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
                       [61.82253647,  98.83877563,  96.75438690],
                       [61.59536743,  98.59293365,  97.53706360],
                       [61.46530151,  98.30503845,  98.32772827],
                       [61.39904785,  97.97928619,  99.11172485],
                       [61.33279419,  97.65353394,  99.89572906],
                       [61.26067352,  97.30914307, 100.67123413],
                       [61.19459534,  96.96743011, 101.44847107],
                       [61.19580460,  96.63417053, 102.23215485],
                       [61.26572037,  96.29887390, 103.01185608],
                       [61.39840698,  95.96297455, 103.78307343],
                       [61.57207870,  95.64262390, 104.55268097],
                       [61.78163528,  95.35540771, 105.32629395],
                       [62.06700134,  95.09746552, 106.08564758],
                       [62.39427185,  94.85724640, 106.83369446],
                       [62.74076462,  94.62278748, 107.57482147],
                       [63.11461639,  94.40107727, 108.30641937],
                       [63.53397751,  94.20418549, 109.02002716],
                       [64.00019836,  94.03809357, 109.71183777],
                       [64.43580627,  93.87523651, 110.42416382],
                       [64.84857941,  93.69993591, 111.14715576],
                       [65.26740265,  93.51858521, 111.86515808],
                       [65.69511414,  93.36718750, 112.58474731],
                       [66.10470581,  93.22719574, 113.31711578],
                       [66.45891571,  93.06028748, 114.07256317],
                       [66.78582001,  92.90560913, 114.84281921],
                       [67.11138916,  92.79004669, 115.62040710],
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
                       [68.84975433,  91.79674530, 126.81564331],
                       [68.72673798,  91.53726196, 127.61715698],
                       [68.60685730,  91.30300140, 128.42681885],
                       [68.50636292,  91.12481689, 129.25317383],
                       [68.39311218,  91.01572418, 130.08976746],
                       [68.25946808,  90.94654083, 130.92756653]],
                      dtype=np.float32)

streamline_64bit = streamline.astype(np.float64)

streamlines = [streamline[[0, 10]], streamline,
               streamline[::2], streamline[::3],
               streamline[::5], streamline[::6]]
streamlines_64bit = [streamline_64bit[[0, 10]], streamline_64bit,
                     streamline_64bit[::2], streamline_64bit[::3],
                     streamline_64bit[::4], streamline_64bit[::5]]

heterogeneous_streamlines = [streamline_64bit,
                             streamline_64bit.reshape((-1, 6)),
                             streamline_64bit.reshape((-1, 2))]


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


def set_number_of_points_python(xyz, n_pols=3):
    def _extrap(xyz, cumlen, distance):
        """ Helper function for extrapolate """
        ind = np.where((cumlen-distance) > 0)[0][0]
        len0 = cumlen[ind-1]
        len1 = cumlen[ind]
        Ds = distance-len0
        Lambda = Ds/(len1-len0)
        return Lambda*xyz[ind] + (1-Lambda)*xyz[ind-1]

    cumlen = np.zeros(xyz.shape[0])
    cumlen[1:] = length_python(xyz, along=True)
    step = cumlen[-1] / (n_pols-1)

    ar = np.arange(0, cumlen[-1], step)
    if np.abs(ar[-1] - cumlen[-1]) < np.finfo('f4').eps:
        ar = ar[:-1]

    xyz2 = [_extrap(xyz, cumlen, distance) for distance in ar]
    return np.vstack((np.array(xyz2), xyz[-1]))


def test_set_number_of_points():
    # Test resampling of only one streamline
    nb_points = 12
    new_streamline_cython = set_number_of_points(
        streamline, nb_points)
    new_streamline_python = set_number_of_points_python(
        streamline, nb_points)
    assert_equal(len(new_streamline_cython), nb_points)
    # Using a 5 digits precision because of streamline is in float32.
    assert_array_almost_equal(new_streamline_cython,
                              new_streamline_python, 5)

    new_streamline_cython = set_number_of_points(
        streamline_64bit, nb_points)
    new_streamline_python = set_number_of_points_python(
        streamline_64bit, nb_points)
    assert_equal(len(new_streamline_cython), nb_points)
    assert_array_almost_equal(new_streamline_cython,
                              new_streamline_python)

    res = []
    simple_streamline = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], 'f4')
    for nb_points in range(2, 200):
        new_streamline_cython = set_number_of_points(
            simple_streamline, nb_points)
        res.append(nb_points - len(new_streamline_cython))
    assert_equal(np.sum(res), 0)

    # Test resampling of multiple streamlines of different nb_points
    nb_points = 12
    new_streamlines_cython = set_number_of_points(
        streamlines, nb_points)

    for i, s in enumerate(streamlines):
        new_streamline_python = set_number_of_points_python(s, nb_points)
        # Using a 5 digits precision because of streamline is in float32.
        assert_array_almost_equal(new_streamlines_cython[i],
                                  new_streamline_python, 5)

    # ArraySequence
    arrseq = Streamlines(streamlines)
    new_streamlines_as_seq_cython = set_number_of_points(arrseq, nb_points)
    assert_array_almost_equal(new_streamlines_as_seq_cython,
                              new_streamlines_cython)

    new_streamlines_cython = set_number_of_points(
        streamlines_64bit, nb_points)

    for i, s in enumerate(streamlines_64bit):
        new_streamline_python = set_number_of_points_python(s, nb_points)
        assert_array_almost_equal(new_streamlines_cython[i],
                                  new_streamline_python)

    # ArraySequence
    arrseq = Streamlines(streamlines_64bit)
    new_streamlines_as_seq_cython = set_number_of_points(arrseq, nb_points)
    assert_array_almost_equal(new_streamlines_as_seq_cython,
                              new_streamlines_cython)

    # Test streamlines with mixed dtype
    streamlines_mixed_dtype = [streamline,
                               streamline.astype(np.float64),
                               streamline.astype(np.int32),
                               streamline.astype(np.int64)]
    nb_points_mixed_dtype = [len(s) for s in set_number_of_points(
        streamlines_mixed_dtype, nb_points)]
    assert_array_equal(nb_points_mixed_dtype,
                       [nb_points] * len(streamlines_mixed_dtype))

    # Test streamlines with different shape
    new_streamlines_cython = set_number_of_points(
        heterogeneous_streamlines, nb_points)

    for i, s in enumerate(heterogeneous_streamlines):
        new_streamline_python = set_number_of_points_python(s, nb_points)
        assert_array_almost_equal(new_streamlines_cython[i],
                                  new_streamline_python)

    # Test streamline with integer dtype
    new_streamline = set_number_of_points(streamline.astype(np.int32))
    assert_equal(new_streamline.dtype, np.float32)
    new_streamline = set_number_of_points(streamline.astype(np.int64))
    assert_equal(new_streamline.dtype, np.float64)

    # Test empty list
    assert_equal(set_number_of_points([]), [])

    # Test streamline having only one point
    assert_raises(ValueError, set_number_of_points, np.array([[1, 2, 3]]))

    # We do not support list of lists, it should be numpy ndarray.
    streamline_unsupported = [[1, 2, 3], [4, 5, 5], [2, 1, 3], [4, 2, 1]]
    assert_raises(AttributeError,
                  set_number_of_points, streamline_unsupported)

    # Test setting number of points of a numpy with flag WRITABLE=False
    streamline_readonly = streamline.copy()
    streamline_readonly.setflags(write=False)
    assert_equal(len(set_number_of_points(streamline_readonly, nb_points=42)),
                 42)

    # Test setting computing length of a numpy with flag WRITABLE=False
    streamlines_readonly = []
    for s in streamlines:
        streamlines_readonly.append(s.copy())
        streamlines_readonly[-1].setflags(write=False)

    assert_equal(len(set_number_of_points(streamlines_readonly,
                                          nb_points=42)),
                 len(streamlines_readonly))

    streamlines_readonly = []
    for s in streamlines_64bit:
        streamlines_readonly.append(s.copy())
        streamlines_readonly[-1].setflags(write=False)

    assert_equal(len(set_number_of_points(streamlines_readonly,
                                          nb_points=42)),
                 len(streamlines_readonly))

    # Test if nb_points is less than 2
    assert_raises(ValueError, set_number_of_points, [
                  np.ones((10, 3)), np.ones((10, 3))], nb_points=1)


@set_random_number_generator(1234)
def test_set_number_of_points_memory_leaks(rng):
    # Test some dtypes
    dtypes = [np.float32, np.float64, np.int32, np.int64]
    for dtype in dtypes:
        s_rng = np.random.default_rng(1234)
        NB_STREAMLINES = 10000
        streamlines = \
            [s_rng.standard_normal((s_rng.integers(10, 100), 3)).astype(dtype)
             for _ in range(NB_STREAMLINES)]

        list_refcount_before = get_type_refcount()["list"]

        rstreamlines = set_number_of_points(streamlines, nb_points=2)
        list_refcount_after = get_type_refcount()["list"]
        del rstreamlines  # Delete `rstreamlines` because it holds a reference
        #                   to `list`.

        # Calling `set_number_of_points` should increase the refcount of `list`
        #  by one since we kept the returned value.
        assert_equal(list_refcount_after, list_refcount_before+1)

    # Test mixed dtypes
    NB_STREAMLINES = 10000
    streamlines = []
    for i in range(NB_STREAMLINES):
        dtype = dtypes[i % len(dtypes)]
        streamlines.append(
            rng.standard_normal((rng.integers(10, 100), 3)).astype(dtype))

    list_refcount_before = get_type_refcount()["list"]
    rstreamlines = set_number_of_points(streamlines, nb_points=2)
    list_refcount_after = get_type_refcount()["list"]

    # Calling `set_number_of_points` should increase the refcount of `list`
    #  by one since we kept the returned value.
    assert_equal(list_refcount_after, list_refcount_before+1)


def test_length():
    # Test length of only one streamline
    length_streamline_cython = length(streamline)
    length_streamline_python = length_python(streamline)
    assert_almost_equal(length_streamline_cython, length_streamline_python)

    length_streamline_cython = length(streamline_64bit)
    length_streamline_python = length_python(streamline_64bit)
    assert_almost_equal(length_streamline_cython, length_streamline_python)

    # Test computing length of multiple streamlines of different nb_points
    length_streamlines_cython = length(streamlines)

    for i, s in enumerate(streamlines):
        length_streamline_python = length_python(s)
        assert_array_almost_equal(length_streamlines_cython[i],
                                  length_streamline_python)

    length_streamlines_cython = length(streamlines_64bit)

    for i, s in enumerate(streamlines_64bit):
        length_streamline_python = length_python(s)
        assert_array_almost_equal(length_streamlines_cython[i],
                                  length_streamline_python)

    # ArraySequence
    # Test length of only one streamline
    length_streamline_cython = length(streamline_64bit)
    length_streamline_arrseq = length(Streamlines([streamline]))
    assert_almost_equal(length_streamline_arrseq, length_streamline_cython)

    length_streamline_cython = length(streamline_64bit)
    length_streamline_arrseq = length(Streamlines([streamline_64bit]))
    assert_almost_equal(length_streamline_arrseq, length_streamline_cython)

    # Test computing length of multiple streamlines of different nb_points
    length_streamlines_cython = length(streamlines)
    length_streamlines_arrseq = length(Streamlines(streamlines))
    assert_array_almost_equal(length_streamlines_arrseq,
                              length_streamlines_cython)

    length_streamlines_cython = length(streamlines_64bit)
    length_streamlines_arrseq = length(Streamlines(streamlines_64bit))
    assert_array_almost_equal(length_streamlines_arrseq,
                              length_streamlines_cython)

    # Test on a sliced ArraySequence
    length_streamlines_cython = length(streamlines_64bit[::2])
    length_streamlines_arrseq = length(Streamlines(streamlines_64bit)[::2])
    assert_array_almost_equal(length_streamlines_arrseq,
                              length_streamlines_cython)
    length_streamlines_cython = length(streamlines[::-1])
    length_streamlines_arrseq = length(Streamlines(streamlines)[::-1])
    assert_array_almost_equal(length_streamlines_arrseq,
                              length_streamlines_cython)

    # Test streamlines having mixed dtype
    streamlines_mixed_dtype = [streamline,
                               streamline.astype(np.float64),
                               streamline.astype(np.int32),
                               streamline.astype(np.int64)]
    lengths_mixed_dtype = [length(s)
                           for s in streamlines_mixed_dtype]
    assert_array_equal(length(streamlines_mixed_dtype),
                       lengths_mixed_dtype)

    # Test streamlines with different shape
    length_streamlines_cython = length(
        heterogeneous_streamlines)

    for i, s in enumerate(heterogeneous_streamlines):
        length_streamline_python = length_python(s)
        assert_array_almost_equal(length_streamlines_cython[i],
                                  length_streamline_python)

    # Test streamline having integer dtype
    length_streamline = length(streamline.astype('int'))
    assert_equal(length_streamline.dtype, np.float64)

    # Test empty list
    assert_equal(length([]), 0.0)

    # Test streamline having only one point
    assert_equal(length(np.array([[1, 2, 3]])), 0.0)

    # We do not support list of lists, it should be numpy ndarray.
    streamline_unsupported = [[1, 2, 3], [4, 5, 5], [2, 1, 3], [4, 2, 1]]
    assert_raises(AttributeError, length,
                  streamline_unsupported)

    # Test setting computing length of a numpy with flag WRITABLE=False
    streamlines_readonly = []
    for s in streamlines:
        streamlines_readonly.append(s.copy())
        streamlines_readonly[-1].setflags(write=False)

    assert_array_almost_equal(length(streamlines_readonly),
                              [length_python(s) for s in streamlines_readonly])
    streamlines_readonly = []
    for s in streamlines_64bit:
        streamlines_readonly.append(s.copy())
        streamlines_readonly[-1].setflags(write=False)

    assert_array_almost_equal(length(streamlines_readonly),
                              [length_python(s) for s in streamlines_readonly])


@set_random_number_generator(1234)
def test_length_memory_leaks(rng):
    # Test some dtypes
    dtypes = [np.float32, np.float64, np.int32, np.int64]
    for dtype in dtypes:
        s_rng = np.random.default_rng(1234)
        NB_STREAMLINES = 10000
        streamlines = \
            [s_rng.standard_normal((s_rng.integers(10, 100), 3)).astype(dtype)
             for _ in range(NB_STREAMLINES)]

        list_refcount_before = get_type_refcount()["list"]

        # lengths = length(streamlines)
        list_refcount_after = get_type_refcount()["list"]

        # Calling `length` shouldn't increase the refcount of `list`
        # since the return value is a numpy array.
        assert_equal(list_refcount_after, list_refcount_before)

    # Test mixed dtypes
    NB_STREAMLINES = 10000
    streamlines = []
    for i in range(NB_STREAMLINES):
        dtype = dtypes[i % len(dtypes)]
        streamlines.append(
            rng.standard_normal((rng.integers(10, 100), 3)).astype(dtype))

    list_refcount_before = get_type_refcount()["list"]

    # lengths = length(streamlines)
    list_refcount_after = get_type_refcount()["list"]

    # Calling `length` shouldn't increase the refcount of `list`
    # since the return value is a numpy array.
    assert_equal(list_refcount_after, list_refcount_before)


@set_random_number_generator()
def test_unlist_relist_streamlines(rng):
    streamlines = [rng.random((10, 3)),
                   rng.random((20, 3)),
                   rng.random((5, 3))]
    points, offsets = unlist_streamlines(streamlines)
    assert_equal(offsets.dtype, np.dtype('i8'))
    assert_equal(points.shape, (35, 3))
    assert_equal(len(offsets), len(streamlines))

    streamlines2 = relist_streamlines(points, offsets)
    assert_equal(len(streamlines), len(streamlines2))
    for i in range(len(streamlines)):
        assert_array_equal(streamlines[i], streamlines2[i])


def test_transform_streamlines_dtype_in_place():
    identity = np.eye(4)
    streamlines = Streamlines([streamline])
    streamlines._data = streamlines._data.astype(np.float16)
    data_dtype = streamlines._data.dtype
    offsets_dtype = streamlines._offsets.dtype

    transform_streamlines(streamlines, identity, in_place=True)
    assert_equal(data_dtype, streamlines._data.dtype)
    assert_equal(offsets_dtype, streamlines._offsets.dtype)


def test_transform_streamlines_dtype():
    identity = np.eye(4)
    streamlines = Streamlines([streamline])
    streamlines._data = streamlines._data.astype(np.float16)
    data_dtype = streamlines._data.dtype
    offsets_dtype = streamlines._offsets.dtype

    streamlines = transform_streamlines(streamlines, identity, in_place=False)
    assert_equal(data_dtype, streamlines._data.dtype)
    assert_equal(offsets_dtype, streamlines._offsets.dtype)


def test_transform_empty_streamlines():
    identity = np.eye(4)
    streamlines = Streamlines([])

    streamlines = transform_streamlines(streamlines, identity, in_place=False)
    assert_equal(len(streamlines), 0)


@set_random_number_generator()
def test_deform_streamlines(rng):
    # Create Random deformation field
    deformation_field = rng.standard_normal((200, 200, 200, 3))
    stream2grid = np.array([
        [-0.13152201, -0.52553149, -0.06759869, -0.80014208],
        [1.01579851, 0.19840874, 0.18875411, 0.81826065],
        [-0.07047617, -0.9290094, -0.55623385, 0.55165017],
        [0., 0., 0., 1.]])
    grid2world = np.array([
        [0.83354727, 1.33876877, 1.0218087, 0.12809569],
        [0.83571344, 0.63824941, 0.20564267, 0.82740437],
        [-0.26574668, -0.66695577, 0.11636694, -0.02620037],
        [0., 0., 0., 1.]])
    stream2world = np.dot(stream2grid, grid2world)

    # Deform streamlines (let two grid spaces be the same for simplicity)
    new_streamlines = deform_streamlines(streamlines,
                                         deformation_field,
                                         stream2grid,
                                         grid2world,
                                         stream2grid,
                                         grid2world)

    # Interpolate displacements onto original streamlines
    streamlines_in_grid = transform_streamlines(streamlines, stream2grid)
    disps = values_from_volume(deformation_field, streamlines_in_grid,
                               np.eye(4))

    # Put new_streamlines into world space
    new_streamlines_world = transform_streamlines(new_streamlines,
                                                  stream2world)

    # Subtract disps from new_streamlines in world space
    orig_streamlines_world = np.subtract(np.array(new_streamlines_world,
                                                  dtype=object),
                                         np.array(disps, dtype=object))

    # Put orig_streamlines_world into voxmm
    orig_streamlines = transform_streamlines(orig_streamlines_world,
                                             np.linalg.inv(stream2world))
    # All close because of floating pt imprecision
    for o, s in zip(orig_streamlines, streamlines):
        assert_allclose(s, o.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_center_and_transform():
    A = np.array([[1, 2, 3], [1, 2, 3.]])
    streamlines = [A for _ in range(10)]
    streamlines2, center = center_streamlines(streamlines)
    B = np.zeros((2, 3))
    assert_array_equal(streamlines2[0], B)
    assert_array_equal(center, A[0])

    affine = np.eye(4)
    affine[0, 0] = 2
    affine[:3, -1] = - np.array([2, 1, 1]) * center
    streamlines3 = transform_streamlines(streamlines, affine)
    assert_array_equal(streamlines3[0], B)


@set_random_number_generator()
def test_select_random_streamlines(rng):
    streamlines = [rng.random((10, 3)),
                   rng.random((20, 3)),
                   rng.random((5, 3))]
    new_streamlines = select_random_set_of_streamlines(streamlines, 2)
    assert_equal(len(new_streamlines), 2)

    new_streamlines = select_random_set_of_streamlines(streamlines, 4)
    assert_equal(len(new_streamlines), 3)


def compress_streamlines_python(streamline, tol_error=0.01,
                                max_segment_length=10):
    """
    Python version of the FiberCompression found on
    https://github.com/scilus/FiberCompression.
    """
    if streamline.shape[0] <= 2:
        return streamline.copy()

    # Euclidean distance
    def segment_length(p1, p2):
        return np.sqrt(((p1-p2)**2).sum())

    # Projection of a 3D point on a 3D line, minimal distance
    def dist_to_line(p1, p2, p0):
        return norm(np.cross(p2-p1, p0-p2)) / norm(p2-p1)

    nb_points = 0
    compressed_streamline = np.zeros_like(streamline)

    # Copy first point since it is always kept.
    compressed_streamline[0, :] = streamline[0, :]
    nb_points += 1
    p1 = streamline[0]
    prev_id = 0

    for next_id, p2 in enumerate(streamline[2:], start=2):
        # Euclidean distance between last added point and current point.
        if segment_length(p1, p2) > max_segment_length:
            compressed_streamline[nb_points, :] = streamline[next_id-1, :]
            nb_points += 1
            p1 = streamline[next_id-1]
            prev_id = next_id-1
            continue

        # Check that each point is not offset by more than `tol_error` mm.
        for o, p0 in enumerate(streamline[prev_id+1:next_id], start=prev_id+1):
            dist = dist_to_line(p1, p2, p0)

            if np.isnan(dist) or dist > tol_error:
                compressed_streamline[nb_points, :] = streamline[next_id-1, :]
                nb_points += 1
                p1 = streamline[next_id-1]
                prev_id = next_id-1
                break

    # Copy last point since it is always kept.
    compressed_streamline[nb_points, :] = streamline[-1, :]
    nb_points += 1

    # Make sure the array have the correct size
    return compressed_streamline[:nb_points]


def test_compress_streamlines():
    for compress_func in [compress_streamlines_python, compress_streamlines]:
        # Small streamlines (less than two points) are incompressible.
        for small_streamline in [np.array([[]]),
                                 np.array([[1, 1, 1]]),
                                 np.array([[1, 1, 1], [2, 2, 2]])]:
            c_streamline = compress_func(small_streamline)
            assert_equal(len(c_streamline), len(small_streamline))
            assert_array_equal(c_streamline, small_streamline)

        # Compressing a straight streamline that is less than 10mm long
        # should output a two points streamline.
        linear_streamline = np.linspace(0, 5, 100*3).reshape((100, 3))
        c_streamline = compress_func(linear_streamline)
        assert_equal(len(c_streamline), 2)
        assert_array_equal(c_streamline, [linear_streamline[0],
                                          linear_streamline[-1]])

        # The distance of consecutive points must be less or equal than some
        # value.
        max_segment_length = 10
        linear_streamline = np.linspace(0, 100, 100*3).reshape((100, 3))
        linear_streamline[:, 1:] = 0.
        c_streamline = compress_func(linear_streamline,
                                     max_segment_length=max_segment_length)
        segments_length = np.sqrt((np.diff(c_streamline,
                                           axis=0)**2).sum(axis=1))
        assert_true(np.all(segments_length <= max_segment_length))
        assert_equal(len(c_streamline), 12)
        assert_array_equal(c_streamline, linear_streamline[::9])

        # A small `max_segment_length` should keep all points.
        c_streamline = compress_func(linear_streamline,
                                     max_segment_length=0.01)
        assert_array_equal(c_streamline, linear_streamline)

        # Test we can set `max_segment_length` to infinity
        # (like the C++ version)
        compress_func(streamline, max_segment_length=np.inf)

        # Incompressible streamline when `tol_error` == 1.
        simple_streamline = np.array([[0, 0, 0],
                                      [1, 1, 0],
                                      [1.5, np.inf, 0],
                                      [2, 2, 0],
                                      [2.5, 20, 0],
                                      [3, 3, 0]])

        # Because of np.inf, compressing that streamline causes a warning.
        with np.errstate(invalid='ignore'):
            c_streamline = compress_func(simple_streamline, tol_error=1)
            assert_array_equal(c_streamline, simple_streamline)

    # Create a special streamline where every other point is increasingly
    # farther from a straight line formed by the streamline endpoints.
    tol_errors = np.linspace(0, 10, 21)
    orthogonal_line = np.array([[-np.sqrt(2)/2, np.sqrt(2)/2, 0]],
                               dtype=np.float32)
    special_streamline = np.array([range(len(tol_errors)*2+1)] * 3,
                                  dtype=np.float32).T
    special_streamline[1::2] += orthogonal_line * tol_errors[:, None]

    # # Uncomment to see the streamline.
    # import pylab as plt
    # plt.plot(special_streamline[:, 0], special_streamline[:, 1], '.-')
    # plt.axis('equal'); plt.show()

    # Test different values for `tol_error`.
    for i, tol_error in enumerate(tol_errors):
        cspecial_streamline = compress_streamlines(special_streamline,
                                                   tol_error=tol_error+1e-4,
                                                   max_segment_length=np.inf)

        # First and last points should always be the same as the original ones.
        assert_array_equal(cspecial_streamline[0], special_streamline[0])
        assert_array_equal(cspecial_streamline[-1], special_streamline[-1])

        assert_equal(len(cspecial_streamline),
                     len(special_streamline)-((i*2)+1))

        # Make sure Cython and Python versions are the same.
        cstreamline_python = compress_streamlines_python(
            special_streamline,
            tol_error=tol_error+1e-4,
            max_segment_length=np.inf)
        assert_equal(len(cspecial_streamline), len(cstreamline_python))
        assert_array_almost_equal(cspecial_streamline, cstreamline_python)


def test_compress_streamlines_identical_points():

    sl_1 = np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3], [3, 3, 3]])
    sl_2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2]])
    sl_3 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                     [2, 2, 2], [2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]])
    sl_4 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2],
                    [3, 3, 3], [3, 3, 3], [1, 1, 1]])
    new_sl_1 = compress_streamlines(sl_1)
    new_sl_2 = compress_streamlines(sl_2)
    new_sl_3 = compress_streamlines(sl_3)
    new_sl_4 = compress_streamlines(sl_4)
    npt.assert_array_equal(new_sl_1, np.array([[1, 1, 1], [3, 3, 3]]))
    npt.assert_array_equal(new_sl_2, np.array([[1, 1, 1], [2, 2, 2]]))
    npt.assert_array_equal(new_sl_3, new_sl_1)
    npt.assert_array_equal(new_sl_4, np.array([[1, 1, 1], [3, 3, 3],
                                               [1, 1, 1]]))


@set_random_number_generator(1234)
def test_compress_streamlines_memory_leaks(rng):
    # Test some dtypes
    dtypes = [np.float32, np.float64, np.int32, np.int64]
    for dtype in dtypes:
        s_rng = np.random.default_rng(1234)
        NB_STREAMLINES = 10000
        streamlines = \
            [s_rng.standard_normal((s_rng.integers(10, 100), 3)).astype(dtype)
             for _ in range(NB_STREAMLINES)]

        list_refcount_before = get_type_refcount()["list"]

        cstreamlines = compress_streamlines(streamlines)
        list_refcount_after = get_type_refcount()["list"]
        del cstreamlines  # Delete `cstreamlines` because it holds a reference
        #                   to `list`.

        # Calling `compress_streamlines` should increase the refcount of `list`
        # by one since we kept the returned value.
        assert_equal(list_refcount_after, list_refcount_before+1)

    # Test mixed dtypes
    NB_STREAMLINES = 10000
    streamlines = []
    for i in range(NB_STREAMLINES):
        dtype = dtypes[i % len(dtypes)]
        streamlines.append(
            rng.standard_normal((rng.integers(10, 100), 3)).astype(dtype))

    list_refcount_before = get_type_refcount()["list"]
    cstreamlines = compress_streamlines(streamlines)
    list_refcount_after = get_type_refcount()["list"]

    # Calling `compress_streamlines` should increase the refcount of `list` by
    # one since we kept the returned value.
    assert_equal(list_refcount_after, list_refcount_before+1)


def generate_sl(streamlines):
    """
    Helper function that takes a sequence and returns a generator

    Parameters
    ----------
    streamlines : sequence
        Usually, this would be a list of 2D arrays, representing streamlines

    Returns
    -------
    generator
    """
    for sl in streamlines:
        yield sl


def test_select_by_rois():
    streamlines = [np.array([[0, 0., 0.9],
                             [1.9, 0., 0.]]),
                   np.array([[0.1, 0., 0],
                             [0, 1., 1.],
                             [0, 2., 2.]]),
                   np.array([[2, 2, 2],
                             [3, 3, 3]])]

    # Make two ROIs:
    mask1 = np.zeros((4, 4, 4), dtype=bool)
    mask2 = np.zeros_like(mask1)
    mask1[0, 0, 0] = True
    mask2[1, 0, 0] = True

    selection = select_by_rois(streamlines, np.eye(4), [mask1], [True],
                               tol=1)

    assert_arrays_equal(list(selection), [streamlines[0],
                                          streamlines[1]])

    selection = select_by_rois(streamlines, np.eye(4), [mask1, mask2],
                               [True, True], tol=1)

    assert_arrays_equal(list(selection), [streamlines[0],
                                          streamlines[1]])

    selection = select_by_rois(streamlines, np.eye(4), [
                               mask1, mask2], [True, False])

    assert_arrays_equal(list(selection), [streamlines[1]])

    # Setting tolerance too low gets overridden:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selection = select_by_rois(streamlines, np.eye(4), [mask1, mask2],
                                   [True, False], tol=0.1)

        assert_arrays_equal(list(selection), [streamlines[1]])

    selection = select_by_rois(streamlines, np.eye(4), [mask1, mask2],
                               [True, True], tol=0.87)

    assert_arrays_equal(list(selection), [streamlines[1]])

    mask3 = np.zeros_like(mask1)
    mask3[0, 2, 2] = 1
    selection = select_by_rois(streamlines, np.eye(4), [mask1, mask2, mask3],
                               [True, True, False], tol=1.0)

    assert_arrays_equal(list(selection), [streamlines[0]])

    # Select using only one ROI
    selection = select_by_rois(streamlines, np.eye(4), [
                               mask1], [True], tol=0.87)
    assert_arrays_equal(list(selection), [streamlines[1]])

    selection = select_by_rois(streamlines, np.eye(4), [
                               mask1], [True], tol=1.0)
    assert_arrays_equal(list(selection), [streamlines[0],
                                          streamlines[1]])

    # Use different modes:
    selection = select_by_rois(streamlines, np.eye(4), [mask1, mask2, mask3],
                               [True, True, False],
                               mode="all",
                               tol=1.0)
    assert_arrays_equal(list(selection), [streamlines[0]])

    selection = select_by_rois(streamlines, np.eye(4), [mask1, mask2, mask3],
                               [True, True, False],
                               mode="either_end",
                               tol=1.0)
    assert_arrays_equal(list(selection), [streamlines[0]])

    selection = select_by_rois(streamlines, np.eye(4), [mask1, mask2, mask3],
                               [True, True, False],
                               mode="both_end",
                               tol=1.0)
    assert_arrays_equal(list(selection), [streamlines[0]])

    mask2[0, 2, 2] = True
    selection = select_by_rois(streamlines, np.eye(4), [mask1, mask2, mask3],
                               [True, True, False],
                               mode="both_end",
                               tol=1.0)

    assert_arrays_equal(list(selection), [streamlines[0],
                                          streamlines[1]])

    # Test with generator input:
    selection = select_by_rois(generate_sl(streamlines), np.eye(4), [mask1],
                               [True], tol=1.0)
    assert_arrays_equal(list(selection), [streamlines[0],
                                          streamlines[1]])


def test_orient_by_rois():
    streamlines = Streamlines([np.array([[0, 0., 0],
                                         [1, 0., 0.],
                                         [2, 0., 0.]]),
                               np.array([[2, 0., 0.],
                                         [1, 0., 0],
                                         [0, 0,  0.]])])

    # Make two ROIs:
    mask1_vol = np.zeros((4, 4, 4), dtype=bool)
    mask2_vol = np.zeros_like(mask1_vol)
    mask1_vol[0, 0, 0] = True
    mask2_vol[1, 0, 0] = True
    mask1_coords = np.array(np.where(mask1_vol)).T
    mask2_coords = np.array(np.where(mask2_vol)).T

    # If there is an affine, we'll use it:
    affine = np.eye(4)
    affine[:, 3] = [-1, 100, -20, 1]
    # Transform the streamlines:
    x_streamlines = Streamlines([sl + affine[:3, 3] for sl in streamlines])

    # After reorientation, this should be the answer:
    flipped_sl = Streamlines([streamlines[0], streamlines[1][::-1]])
    new_streamlines = orient_by_rois(streamlines, np.eye(4),
                                     mask1_vol,
                                     mask2_vol,
                                     in_place=False,
                                     as_generator=False)
    npt.assert_array_equal(new_streamlines, flipped_sl)

    npt.assert_(new_streamlines is not streamlines)

    # Test with affine:
    x_flipped_sl = Streamlines([s + affine[:3, 3] for s in flipped_sl])
    new_streamlines = orient_by_rois(x_streamlines, affine,
                                     mask1_vol,
                                     mask2_vol,
                                     in_place=False,
                                     as_generator=False)
    npt.assert_array_equal(new_streamlines, x_flipped_sl)
    npt.assert_(new_streamlines is not x_streamlines)

    # Test providing coord ROIs instead of vol ROIs:
    new_streamlines = orient_by_rois(x_streamlines, affine,
                                     mask1_coords,
                                     mask2_coords,
                                     in_place=False,
                                     as_generator=False)
    npt.assert_array_equal(new_streamlines, x_flipped_sl)

    # Test with as_generator set to True
    new_streamlines = orient_by_rois(streamlines, np.eye(4),
                                     mask1_vol,
                                     mask2_vol,
                                     in_place=False,
                                     as_generator=True)

    npt.assert_(isinstance(new_streamlines, types.GeneratorType))
    ll = Streamlines(new_streamlines)
    npt.assert_array_equal(ll, flipped_sl)

    # Test with as_generator set to True and with the affine
    new_streamlines = orient_by_rois(x_streamlines, affine,
                                     mask1_vol,
                                     mask2_vol,
                                     in_place=False,
                                     as_generator=True)

    npt.assert_(isinstance(new_streamlines, types.GeneratorType))
    ll = Streamlines(new_streamlines)
    npt.assert_array_equal(ll, x_flipped_sl)

    # Test with generator input:
    new_streamlines = orient_by_rois(generate_sl(streamlines), np.eye(4),
                                     mask1_vol,
                                     mask2_vol,
                                     in_place=False,
                                     as_generator=True)

    npt.assert_(isinstance(new_streamlines, types.GeneratorType))
    ll = Streamlines(new_streamlines)
    npt.assert_array_equal(ll, flipped_sl)

    # Generator output cannot take a True `in_place` kwarg:
    npt.assert_raises(ValueError, orient_by_rois, *[generate_sl(streamlines),
                                                    np.eye(4),
                                                    mask1_vol,
                                                    mask2_vol],
                      **dict(in_place=True,
                             as_generator=True))

    # But you can input a generator and get a non-generator as output:
    new_streamlines = orient_by_rois(generate_sl(streamlines), np.eye(4),
                                     mask1_vol,
                                     mask2_vol,
                                     in_place=False,
                                     as_generator=False)

    npt.assert_(not isinstance(new_streamlines, types.GeneratorType))
    npt.assert_array_equal(new_streamlines, flipped_sl)

    # Modify in-place:
    new_streamlines = orient_by_rois(streamlines, np.eye(4),
                                     mask1_vol,
                                     mask2_vol,
                                     in_place=True,
                                     as_generator=False)

    npt.assert_array_equal(new_streamlines, flipped_sl)
    # The two objects are one and the same:
    npt.assert_(new_streamlines is streamlines)


def test_orient_by_streamline():
    streamlines = Streamlines([np.array([[0, 0., 0],
                                         [1, 0., 0.],
                                         [2, 0., 0.]]),
                               np.array([[2, 0., 0.],
                                         [1, 0., 0],
                                         [0, 0,  0.]])])

    # If there is an affine, we'll use it:
    affine = np.eye(4)
    affine[:, 3] = [-1, 100, -20, 1]
    # Transform the streamlines:
    x_streamlines = Streamlines([sl + affine[:3, 3] for sl in streamlines])

    standard_streamline = streamlines[0]

    # After reorientation, this should be the answer:
    flipped_sl = Streamlines([streamlines[0], streamlines[1][::-1]])

    new_streamlines = orient_by_streamline(streamlines,
                                           standard_streamline,
                                           n_points=12,
                                           in_place=False)

    npt.assert_array_equal(new_streamlines, flipped_sl)
    npt.assert_(new_streamlines is not streamlines)

    # Test with affine:
    x_flipped_sl = Streamlines([s + affine[:3, 3] for s in flipped_sl])
    new_streamlines = orient_by_streamline(x_streamlines,
                                           standard_streamline,
                                           in_place=False)
    npt.assert_array_equal(new_streamlines, x_flipped_sl)
    npt.assert_(new_streamlines is not x_streamlines)

    # Test with as_generator set to True
    new_streamlines = orient_by_streamline(streamlines,
                                           standard_streamline,
                                           in_place=False,
                                           as_generator=True)

    npt.assert_(isinstance(new_streamlines, types.GeneratorType))
    ll = Streamlines(new_streamlines)
    npt.assert_array_equal(ll, flipped_sl)

    # Test with as_generator set to True and with the affine
    new_streamlines = orient_by_streamline(x_streamlines,
                                           standard_streamline,
                                           in_place=False,
                                           as_generator=True)

    npt.assert_(isinstance(new_streamlines, types.GeneratorType))
    ll = Streamlines(new_streamlines)
    npt.assert_array_equal(ll, x_flipped_sl)

    # Modify in-place:
    new_streamlines = orient_by_streamline(streamlines,
                                           standard_streamline,
                                           in_place=True)

    npt.assert_array_equal(new_streamlines, flipped_sl)
    # The two objects are one and the same:
    npt.assert_(new_streamlines is streamlines)


def test_values_from_volume():
    decimal = 4
    data3d = np.arange(2000).reshape(20, 10, 10)
    # Test two cases of 4D data (handled differently)
    # One where the last dimension is length 3:
    data4d_3vec = np.arange(6000).reshape(20, 10, 10, 3)
    # The other where the last dimension is not 3:
    data4d_2vec = np.arange(4000).reshape(20, 10, 10, 2)
    for dt in [np.float32, np.float64]:
        for data in [data3d, data4d_3vec, data4d_2vec]:
            sl1 = [np.array([[1, 0, 0],
                             [1.5, 0, 0],
                             [2, 0, 0],
                             [2.5, 0, 0]]).astype(dt),
                   np.array([[2, 0, 0],
                             [3.1, 0, 0],
                             [3.9, 0, 0],
                             [4.1, 0, 0]]).astype(dt)]

            ans1 = [[data[1, 0, 0],
                     data[1, 0, 0] + (data[2, 0, 0] - data[1, 0, 0]) / 2,
                     data[2, 0, 0],
                     data[2, 0, 0] + (data[3, 0, 0] - data[2, 0, 0]) / 2],
                    [data[2, 0, 0],
                     data[3, 0, 0] + (data[4, 0, 0] - data[3, 0, 0]) * 0.1,
                     data[3, 0, 0] + (data[4, 0, 0] - data[3, 0, 0]) * 0.9,
                     data[4, 0, 0] + (data[5, 0, 0] - data[4, 0, 0]) * 0.1]]

            vv = values_from_volume(data, sl1, np.eye(4))
            npt.assert_almost_equal(vv, ans1, decimal=decimal)

            vv = values_from_volume(data, np.array(sl1), np.eye(4))
            npt.assert_almost_equal(vv, ans1, decimal=decimal)

            vv = values_from_volume(data, Streamlines(sl1), np.eye(4))
            npt.assert_almost_equal(vv, ans1, decimal=decimal)

            affine = np.eye(4)
            affine[:, 3] = [-100, 10, 1, 1]
            x_sl1 = transform_streamlines(sl1, affine)
            x_sl2 = transform_streamlines(sl1, affine)

            vv = values_from_volume(data, x_sl1, affine)
            npt.assert_almost_equal(vv, ans1, decimal=decimal)

            x_sl1 = transform_streamlines(sl1, affine)
            vv = values_from_volume(data, x_sl1, affine)

            npt.assert_almost_equal(vv, ans1, decimal=decimal)

            # Test that the streamlines haven't mutated:
            l_sl2 = list(x_sl2)
            npt.assert_equal(x_sl1, l_sl2)

            vv = values_from_volume(data, np.array(x_sl1), affine)
            npt.assert_almost_equal(vv, ans1, decimal=decimal)
            npt.assert_equal(np.array(x_sl1), np.array(l_sl2))

            # Test for lists of streamlines with different numbers of nodes:
            sl2 = [sl1[0][:-1], sl1[1]]
            ans2 = [ans1[0][:-1], ans1[1]]
            vv = values_from_volume(data, sl2, np.eye(4))
            for ii, v in enumerate(vv):
                npt.assert_almost_equal(v, ans2[ii], decimal=decimal)

    # We raise an error if the streamlines fed don't make sense. In this
    # case, a tuple instead of a list, generator or array
    nonsense_sl = (np.array([[1, 0, 0],
                             [1.5, 0, 0],
                             [2, 0, 0],
                             [2.5, 0, 0]]),
                   np.array([[2, 0, 0],
                             [3.1, 0, 0],
                             [3.9, 0, 0],
                             [4.1, 0, 0]]))

    npt.assert_raises(RuntimeError, values_from_volume, data,
                      nonsense_sl,
                      np.eye(4))

    # For some use-cases we might have singleton streamlines (with only one
    # node each):
    data3D = np.ones((2, 2, 2))
    streamlines = np.ones((10, 1, 3))
    npt.assert_equal(values_from_volume(data3D, streamlines,
                                        np.eye(4)).shape, (10, 1))
    data4D = np.ones((2, 2, 2, 2))
    streamlines = np.ones((10, 1, 3))
    npt.assert_equal(values_from_volume(data4D, streamlines,
                                        np.eye(4)).shape, (10, 1, 2))


def test_streamlines_generator():
    # Test generator
    streamlines_generator = Streamlines(generate_sl(streamlines))
    npt.assert_equal(len(streamlines_generator), len(streamlines))
    # Nothing should change
    streamlines_generator.append(np.array([]))
    npt.assert_equal(len(streamlines_generator), len(streamlines))

    # Test append error
    npt.assert_raises(ValueError, streamlines_generator.append,
                      np.array(streamlines, dtype=object))

    # Test empty streamlines
    streamlines_generator = Streamlines(np.array([]))
    npt.assert_equal(len(streamlines_generator), 0)


def test_cluster_confidence():
    mysl = np.array([np.arange(10)] * 3, 'float').T

    # a short streamline (<20 mm) should raise an error unless override=True
    test_streamlines = Streamlines()
    test_streamlines.append(mysl)
    assert_raises(ValueError, cluster_confidence, test_streamlines)
    cci = cluster_confidence(test_streamlines, override=True)

    # two identical streamlines should raise an error
    test_streamlines = Streamlines()
    test_streamlines.append(mysl, cache_build=True)
    test_streamlines.append(mysl)
    test_streamlines.finalize_append()
    assert_raises(ValueError, cluster_confidence, test_streamlines)

    # 3 offset collinear streamlines
    test_streamlines = Streamlines()
    test_streamlines.append(mysl, cache_build=True)
    test_streamlines.append(mysl+1)
    test_streamlines.append(mysl+2)
    test_streamlines.finalize_append()

    cci = cluster_confidence(test_streamlines, override=True)

    assert_almost_equal(cci[0], cci[2])
    assert_true(cci[1] > cci[0])

    # 3 parallel streamlines
    mysl = np.zeros([10, 3])
    mysl[:, 0] = np.arange(10)
    mysl2 = mysl.copy()
    mysl2[:, 1] = 1
    mysl3 = mysl.copy()
    mysl3[:, 1] = 2
    mysl4 = mysl.copy()
    mysl4[:, 1] = 4
    mysl5 = mysl.copy()
    mysl5[:, 1] = 5000

    test_streamlines_p1 = Streamlines()
    test_streamlines_p1.append(mysl, cache_build=True)
    test_streamlines_p1.append(mysl2)
    test_streamlines_p1.append(mysl3)
    test_streamlines_p1.finalize_append()
    test_streamlines_p2 = Streamlines()
    test_streamlines_p2.append(mysl, cache_build=True)
    test_streamlines_p2.append(mysl3)
    test_streamlines_p2.append(mysl4)
    test_streamlines_p2.finalize_append()
    test_streamlines_p3 = Streamlines()
    test_streamlines_p3.append(mysl, cache_build=True)
    test_streamlines_p3.append(mysl2)
    test_streamlines_p3.append(mysl3)
    test_streamlines_p3.append(mysl5)
    test_streamlines_p3.finalize_append()

    cci_p1 = cluster_confidence(test_streamlines_p1, override=True)
    cci_p2 = cluster_confidence(test_streamlines_p2, override=True)

    # test relative distance
    assert_array_equal(cci_p1, cci_p2*2)

    # test simple cci calculation
    expected_p1 = np.array([1./1+1./2, 1./1+1./1, 1./1+1./2])
    expected_p2 = np.array([1./2+1./4, 1./2+1./2, 1./2+1./4])
    assert_array_equal(expected_p1, cci_p1)
    assert_array_equal(expected_p2, cci_p2)

    # test power variable calculation (dropoff with distance)
    cci_p1_pow2 = cluster_confidence(test_streamlines_p1, power=2,
                                     override=True)

    expected_p1_pow2 = np.array([np.power(1./1, 2)+np.power(1./2, 2),
                                 np.power(1./1, 2)+np.power(1./1, 2),
                                 np.power(1./1, 2)+np.power(1./2, 2)])

    assert_array_equal(cci_p1_pow2, expected_p1_pow2)

    # test max distance (ignore distant sls)
    cci_dist = cluster_confidence(test_streamlines_p3,
                                  max_mdf=5, override=True)

    expected_cci_dist = np.concatenate([cci_p1, np.zeros(1)])
    assert_array_equal(cci_dist, expected_cci_dist)
