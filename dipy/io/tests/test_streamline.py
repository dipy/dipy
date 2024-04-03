import os
from tempfile import TemporaryDirectory
from urllib.error import URLError, HTTPError

from dipy.data import get_fnames
from dipy.io.streamline import (load_tractogram, save_tractogram,
                                load_trk, save_trk)
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.utils import create_nifti_header
from dipy.io.vtk import save_vtk_streamlines, load_vtk_streamlines
from dipy.tracking.streamline import Streamlines
import numpy as np
import numpy.testing as npt
import pytest

from dipy.utils.optpkg import optional_package
fury, have_fury, setup_module = optional_package('fury', min_version="0.10.0")

FILEPATH_DIX, STREAMLINE, STREAMLINES = None, None, None


def setup_module():
    global FILEPATH_DIX, STREAMLINE, STREAMLINES
    try:
        FILEPATH_DIX, _, _ = get_fnames('gold_standard_tracks')
    except (HTTPError, URLError) as e:
        FILEPATH_DIX, STREAMLINE, STREAMLINES = None, None, None
        error_msg = f'"Tests Data failed to download." Reason: {e}'
        pytest.skip(error_msg, allow_module_level=True)
        return

    STREAMLINE = np.array([[82.20181274,  91.36505891,  43.15737152],
                           [82.38442231,  91.79336548,  43.87036514],
                           [82.48710632,  92.27861023,  44.56298065],
                           [82.53310394,  92.78545381,  45.24635315],
                           [82.53793335,  93.26902008,  45.94785309],
                           [82.48797607,  93.75003815,  46.64939880],
                           [82.35533142,  94.25181581,  47.32533264],
                           [82.15484619,  94.76634216,  47.97451019],
                           [81.90982819,  95.28792572,  48.60244371],
                           [81.63336945,  95.78153229,  49.23971176],
                           [81.35479736,  96.24868011,  49.89558792],
                           [81.08713531,  96.69807434,  50.56812668],
                           [80.81504822,  97.14285278,  51.24193192],
                           [80.52591705,  97.56719971,  51.92168427],
                           [80.26599884,  97.98269653,  52.61848068],
                           [80.04635621,  98.38131714,  53.33855821],
                           [79.84691621,  98.77052307,  54.06955338],
                           [79.57667542,  99.13599396,  54.78985596],
                           [79.23351288,  99.43207551,  55.51065063],
                           [78.84815979,  99.64141846,  56.24016571],
                           [78.47383881,  99.77347565,  56.99299241],
                           [78.12837219,  99.81330872,  57.76969528],
                           [77.80438995,  99.85082245,  58.55574799],
                           [77.49439240,  99.88065338,  59.34777069],
                           [77.21414185,  99.85343933,  60.15090561],
                           [76.96416473,  99.82772827,  60.96406937],
                           [76.74712372,  99.80519104,  61.78676605],
                           [76.52263641,  99.79122162,  62.60765076],
                           [76.03757477, 100.08692169,  63.24152374],
                           [75.44867706, 100.35265351,  63.79513168],
                           [74.78033447, 100.57255554,  64.27278901],
                           [74.11605835, 100.77330781,  64.76428986],
                           [73.51222992, 100.98779297,  65.32373047],
                           [72.97387695, 101.23387146,  65.93502045],
                           [72.47355652, 101.49151611,  66.57343292],
                           [71.99834442, 101.72480774,  67.23979950],
                           [71.56909181, 101.98665619,  67.92664337],
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
                           [66.46197511, 102.54130554,  79.99622345],
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
                           [61.19580461,  96.63417053, 102.23215485],
                           [61.26572037,  96.29887391, 103.01185608],
                           [61.39840698,  95.96297455, 103.78307343],
                           [61.57207871,  95.64262391, 104.55268097],
                           [61.78163528,  95.35540771, 105.32629395],
                           [62.06700134,  95.09746552, 106.08564758],
                           [62.39427185,  94.85724641, 106.83369446],
                           [62.74076462,  94.62278748, 107.57482147],
                           [63.11461639,  94.40107727, 108.30641937],
                           [63.53397751,  94.20418549, 109.02002716],
                           [64.00019836,  94.03809357, 109.71183777],
                           [64.43580627,  93.87523651, 110.42416382],
                           [64.84857941,  93.69993591, 111.14715576],
                           [65.26740265,  93.51858521, 111.86515808],
                           [65.69511414,  93.36718751, 112.58474731],
                           [66.10470581,  93.22719574, 113.31711578],
                           [66.45891571,  93.06028748, 114.07256317],
                           [66.78582001,  92.90560913, 114.84281921],
                           [67.11138916,  92.79004669, 115.62040711],
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
                           [68.84975433,  91.79674531, 126.81564331],
                           [68.72673798,  91.53726196, 127.61715698],
                           [68.60685731,  91.30300141, 128.42681885],
                           [68.50636292,  91.12481689, 129.25317383],
                           [68.39311218,  91.01572418, 130.08976746],
                           [68.25946808,  90.94654083, 130.92756653]],
                          dtype=np.float32)

    STREAMLINES = Streamlines([STREAMLINE[[0, 10]], STREAMLINE,
                               STREAMLINE[::2], STREAMLINE[::3],
                               STREAMLINE[::5], STREAMLINE[::6]])


def teardown_module():
    global FILEPATH_DIX, POINTS_DATA, STREAMLINES_DATA, STREA
    FILEPATH_DIX, POINTS_DATA, STREAMLINES_DATA = None, None, None


def io_tractogram(extension):
    with TemporaryDirectory() as tmp_dir:
        fname = 'test.{}'.format(extension)
        fpath = os.path.join(tmp_dir, fname)

        in_affine = np.eye(4)
        in_dimensions = np.array([50, 50, 50])
        in_voxel_sizes = np.array([2, 1.5, 1.5])
        in_affine = np.dot(in_affine, np.diag(np.r_[in_voxel_sizes, 1]))
        nii_header = create_nifti_header(in_affine, in_dimensions,
                                         in_voxel_sizes)

        sft = StatefulTractogram(STREAMLINES, nii_header, space=Space.RASMM)
        save_tractogram(sft, fpath, bbox_valid_check=False)

        if extension in ['trk', 'trx']:
            reference = 'same'
        else:
            reference = nii_header

        sft = load_tractogram(fpath, reference, bbox_valid_check=False)
        affine, dimensions, voxel_sizes, _ = sft.space_attributes

        npt.assert_array_equal(in_affine, affine)
        npt.assert_array_equal(in_voxel_sizes, voxel_sizes)
        npt.assert_array_equal(in_dimensions, dimensions)
        npt.assert_equal(len(sft), len(STREAMLINES))
        npt.assert_array_almost_equal(sft.streamlines[1], STREAMLINE,
                                      decimal=4)


def test_io_trk():
    io_tractogram('trk')


def test_io_tck():
    io_tractogram('tck')


def test_io_trx():
    io_tractogram('trx')


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_io_vtk():
    io_tractogram('vtk')


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_io_vtp():
    io_tractogram('vtp')


def test_io_dpy():
    io_tractogram('dpy')


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_low_io_vtk():
    with TemporaryDirectory() as tmp_dir:
        fname = os.path.join(tmp_dir, 'test.fib')

        # Test save
        save_vtk_streamlines(STREAMLINES, fname, binary=True)
        tracks = load_vtk_streamlines(fname)
        npt.assert_equal(len(tracks), len(STREAMLINES))
        npt.assert_array_almost_equal(tracks[1], STREAMLINE, decimal=4)


def trk_loader(filename):
    try:
        with TemporaryDirectory() as tmp_dir:
            load_trk(os.path.join(tmp_dir, filename), FILEPATH_DIX['gs.nii'])
        return True
    except ValueError:
        return False


def trk_saver(filename):
    sft = load_tractogram(FILEPATH_DIX['gs.trk'], FILEPATH_DIX['gs.nii'])

    try:
        with TemporaryDirectory() as tmp_dir:
            save_trk(sft, os.path.join(tmp_dir, filename))
        return True
    except ValueError:
        return False


def test_io_trk_load():
    npt.assert_(trk_loader(FILEPATH_DIX['gs.trk']),
                msg='trk_loader should be able to load a trk')
    npt.assert_(not trk_loader('fake_file.TRK'),
                msg='trk_loader should not be able to load a TRK')
    npt.assert_(not trk_loader(FILEPATH_DIX['gs.tck']),
                msg='trk_loader should not be able to load a tck')
    npt.assert_(not trk_loader(FILEPATH_DIX['gs.fib']),
                msg='trk_loader should not be able to load a fib')
    npt.assert_(not trk_loader(FILEPATH_DIX['gs.dpy']),
                msg='trk_loader should not be able to load a dpy')


def test_io_trk_save():
    npt.assert_(trk_saver(FILEPATH_DIX['gs.trk']),
                msg='trk_saver should be able to save a trk')
    npt.assert_(not trk_saver('fake_file.TRK'),
                msg='trk_saver should not be able to save a TRK')
    npt.assert_(not trk_saver(FILEPATH_DIX['gs.tck']),
                msg='trk_saver should not be able to save a tck')
    npt.assert_(not trk_saver(FILEPATH_DIX['gs.fib']),
                msg='trk_saver should not be able to save a fib')
    npt.assert_(not trk_saver(FILEPATH_DIX['gs.dpy']),
                msg='trk_saver should not be able to save a dpy')
