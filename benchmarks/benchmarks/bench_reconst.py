"""Benchmarks for``dipy.reconst`` module."""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.core.sphere import unique_edges
from dipy.data import default_sphere, get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, csdeconv
from dipy.reconst.recspeed import local_maxima
from dipy.reconst.vec_val_sum import vec_val_vect
from dipy.sims.voxel import multi_tensor


class BenchRecSpeed:
    def setup(self):
        vertices, faces = default_sphere.vertices, default_sphere.faces
        self.edges = unique_edges(faces)
        self.odf = np.zeros(len(vertices))
        self.odf[1] = 1.0
        self.odf[143] = 143.0
        self.odf[305] = 305.0

    def time_local_maxima(self):
        local_maxima(self.odf, self.edges)


class BenchVecValSum:
    def setup(self):
        def make_vecs_vals(shape):
            return (
                np.random.randn(*shape),
                np.random.randn(*(shape[:-2] + shape[-1:])),
            )

        shape = (10, 12, 3, 3)
        self.evecs, self.evals = make_vecs_vals(shape)

    def time_vec_val_vect(self):
        vec_val_vect(self.evecs, self.evals)


class BenchCSDPrefactorizedP:
    def setup(self):
        _, fbvals, fbvecs = get_fnames(name="small_64D")
        bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
        gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=0)

        mevals = np.array(([0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]))
        signal, _ = multi_tensor(
            gtab,
            mevals,
            S0=1.0,
            angles=[(0, 0), (60, 0)],
            fractions=[50, 50],
            snr=40,
        )
        response = (np.array([0.0015, 0.0003, 0.0003]), 1.0)
        self.model = ConstrainedSphericalDeconvModel(gtab, response)
        self.dwi_signal = signal[~gtab.b0s_mask]

    def time_csdeconv_prefactored_P(self):
        csdeconv(
            self.dwi_signal,
            self.model._X,
            self.model.B_reg,
            tau=self.model.tau,
            convergence=self.model.convergence,
            P=self.model._P,
            P_chol=self.model._P_chol,
        )

    def time_csdeconv_factorize_each_call(self):
        csdeconv(
            self.dwi_signal,
            self.model._X,
            self.model.B_reg,
            tau=self.model.tau,
            convergence=self.model.convergence,
            P=self.model._P,
            P_chol=None,
        )


# class BenchCSD:

#     def setup(self):
#         img, self.gtab, labels_img = read_stanford_labels()
#         data = img.get_fdata()

#         labels = labels_img.get_fdata()
#         shape = labels.shape
#         mask = np.isin(labels, [1, 2])
#         mask.shape = shape

#         center = (50, 40, 40)
#         width = 12
#         a, b, c = center
#         hw = width // 2
#         idx = (slice(a - hw, a + hw), slice(b - hw, b + hw),
#                slice(c - hw, c + hw))

#         self.data_small = data[idx].copy()
#         self.mask_small = mask[idx].copy()
#         voxels = self.mask_small.sum()
#         self.small_gtab = GradientTable(self.gtab.gradients[:75])


# def time_csdeconv_basic(self):
#     # TODO: define response and remove None
#     sh_order_max = 8
#     model = ConstrainedSphericalDeconvModel(self.gtab, None,
#                                             sh_order_max=sh_order_max)
#     model.fit(self.data_small, self.mask_small)

# def time_csdeconv_small_dataset(self):
#      # TODO: define response and remove None
#     # Smaller data set
#     # data_small = data_small[..., :75].copy()
#     sh_order_max = 8
#     model = ConstrainedSphericalDeconvModel(self.small_gtab, None,
#                                             sh_order_max=sh_order_max)
#     model.fit(self.data_small, self.mask_small)

# def time_csdeconv_super_resolution(self):
#      # TODO: define response and remove None
#     # Super resolution
#     sh_order_max = 12
#     model = ConstrainedSphericalDeconvModel(self.gtab, None,
#                                             sh_order_max=sh_order_max)
#     model.fit(self.data_small, self.mask_small)
