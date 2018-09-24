import numpy as np
import numpy.testing as npt

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory
from dipy.tracking.streamline import Streamlines
from dipy.data import get_data
from dipy.workflows.align import ResliceFlow, SlrWithQbxFlow
from os.path import join as pjoin
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import (set_number_of_points,
                                      select_random_set_of_streamlines)
from dipy.align.streamlinear import BundleMinDistanceMetric


def test_reslice():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_data('small_25')
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()

        reslice_flow = ResliceFlow()
        reslice_flow.run(data_path, [1.5, 1.5, 1.5], out_dir=out_dir)

        out_path = reslice_flow.last_generated_outputs['out_resliced']
        out_img = nib.load(out_path)
        resliced = out_img.get_data()

        npt.assert_equal(resliced.shape[0] > volume.shape[0], True)
        npt.assert_equal(resliced.shape[1] > volume.shape[1], True)
        npt.assert_equal(resliced.shape[2] > volume.shape[2], True)
        npt.assert_equal(resliced.shape[-1], volume.shape[-1])


def test_slr_flow():
    with TemporaryDirectory() as out_dir:
        data_path = get_data('fornix')

        streams, hdr = nib.trackvis.read(data_path)
        fornix = [s[0] for s in streams]

        f = Streamlines(fornix)
        f1 = f.copy()

        f1_path = pjoin(out_dir, "f1.trk")
        save_trk(f1_path, Streamlines(f1), affine=np.eye(4))

        f2 = f1.copy()
        f2._data += np.array([50, 0, 0])

        f2_path = pjoin(out_dir, "f2.trk")
        save_trk(f2_path, Streamlines(f2), affine=np.eye(4))

        slr_flow = SlrWithQbxFlow(force=True)
        slr_flow.run(f1_path, f2_path)

        out_path = slr_flow.last_generated_outputs['out_moved']
        moved_f2, _ = load_trk(out_path)

        BMD = BundleMinDistanceMetric()
        nb_pts = 20
        static = set_number_of_points(f1, nb_pts)
        moving = set_number_of_points(moved_f2, nb_pts)

        BMD.setup(static, moving)
        x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1, 0, 0, 0])  # affine
        bmd_value = BMD.distance(x0.tolist())

        # npt.assert_equal(bmd_value < 1, True)


if __name__ == '__main__':
    for i in range(15):
        npt.run_module_suite()
