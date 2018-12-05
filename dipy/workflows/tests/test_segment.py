import numpy.testing as npt
from os.path import join
import nibabel as nib
import numpy as np
from nibabel.tmpdirs import TemporaryDirectory
from dipy.data import get_fnames
from dipy.segment.mask import median_otsu
from dipy.tracking.streamline import Streamlines
from dipy.workflows.segment import MedianOtsuFlow
from dipy.workflows.segment import RecoBundlesFlow, LabelsBundlesFlow
from dipy.io.streamline import load_trk, save_trk
from os.path import join as pjoin
from dipy.tracking.streamline import (set_number_of_points,
                                      select_random_set_of_streamlines)
from dipy.align.streamlinear import BundleMinDistanceMetric


def test_median_otsu_flow():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames('small_25')
        volume = nib.load(data_path).get_data()
        save_masked = True
        median_radius = 3
        numpass = 3
        autocrop = False
        vol_idx = [0]
        dilate = 0

        mo_flow = MedianOtsuFlow()
        mo_flow.run(data_path, out_dir=out_dir, save_masked=save_masked,
                    median_radius=median_radius, numpass=numpass,
                    autocrop=autocrop, vol_idx=vol_idx, dilate=dilate)

        mask_name = mo_flow.last_generated_outputs['out_mask']
        masked_name = mo_flow.last_generated_outputs['out_masked']

        masked, mask = median_otsu(volume, median_radius,
                                   numpass, autocrop,
                                   vol_idx, dilate)

        result_mask_data = nib.load(join(out_dir, mask_name)).get_data()
        npt.assert_array_equal(result_mask_data, mask)

        result_masked_data = nib.load(join(out_dir, masked_name)).get_data()
        npt.assert_array_equal(result_masked_data, masked)


def test_recobundles_flow():
    with TemporaryDirectory() as out_dir:
        data_path = get_fnames('fornix')
        streams, hdr = nib.trackvis.read(data_path)
        fornix = [s[0] for s in streams]

        f = Streamlines(fornix)
        f1 = f.copy()

        f2 = f1[:15].copy()
        f2._data += np.array([40, 0, 0])

        f.extend(f2)

        f2_path = pjoin(out_dir, "f2.trk")
        save_trk(f2_path, f2, affine=np.eye(4))

        f1_path = pjoin(out_dir, "f1.trk")
        save_trk(f1_path, f, affine=np.eye(4))

        rb_flow = RecoBundlesFlow(force=True)
        rb_flow.run(f1_path, f2_path, greater_than=0, clust_thr=10,
                    model_clust_thr=5., reduction_thr=10, out_dir=out_dir)

        labels = rb_flow.last_generated_outputs['out_recognized_labels']
        recog_trk = rb_flow.last_generated_outputs['out_recognized_transf']

        rec_bundle, _ = load_trk(recog_trk)
        npt.assert_equal(len(rec_bundle) == len(f2), True)

        label_flow = LabelsBundlesFlow(force=True)
        label_flow.run(f1_path, labels)

        recog_bundle = label_flow.last_generated_outputs['out_bundle']
        rec_bundle_org, _ = load_trk(recog_bundle)

        BMD = BundleMinDistanceMetric()
        nb_pts = 20
        static = set_number_of_points(f2, nb_pts)
        moving = set_number_of_points(rec_bundle_org, nb_pts)

        BMD.setup(static, moving)
        x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1, 0, 0, 0])  # affine
        bmd_value = BMD.distance(x0.tolist())

        npt.assert_equal(bmd_value < 1, True)


if __name__ == '__main__':
    npt.run_module_suite()
