"""Benchmarks for ``dipy.denoise`` module."""

import numpy as np

from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma


class BenchNLMeans:
    def setup(self):
        rng = np.random.default_rng(1234)
        
        # 3D Data (Magnitude MRI Simulation)
        clean_3d = np.ones((40, 40, 40)) * 100.0
        self.data_3d = np.abs(clean_3d + rng.standard_normal((40, 40, 40)) * 10)
        self.sigma_3d = estimate_sigma(self.data_3d, N=1)

        # 4D Data (DWI Simulation)
        self.data_4d = np.abs(
            np.ones((20, 20, 20, 10)) * 100.0 + rng.standard_normal((20, 20, 20, 10)) * 10
        )
        self.sigma_4d = np.full(10, self.sigma_3d)
        
    def time_nlmeans_3d_classic(self):
        nlmeans(self.data_3d, self.sigma_3d, rician=True, method="classic")

    def time_nlmeans_3d_blockwise(self):
        nlmeans(self.data_3d, self.sigma_3d, rician=True, method="blockwise")

    def time_nlmeans_4d_blockwise(self):
        nlmeans(self.data_4d, self.sigma_4d, rician=True, method="blockwise")
