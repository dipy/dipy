"""Orientation Distribution Function Fingerprinting (ODF-FP).

ODF-FP :footcite:p:`Baete2019,Filipiak2022` reconstructs the diffusion ODF by
matching the ODF reconstructed from the measured signal against a dictionary of
ODF "fingerprints" simulated from a multi-compartment biophysical model. Each
voxel is aligned so that its main peak points to the pole, normalized, and
matched to the most similar dictionary fingerprint by penalized cosine
similarity. The microstructure parameters of the matched fingerprint are then
assigned to the voxel.

By default both the dictionary ODFs and the measured ODFs are reconstructed
with Generalized Q-Sampling Imaging (GQI), which is fast and keeps the two ODF
estimates consistent.

References
----------
.. footbibliography::
"""

import numpy as np

from dipy.data import get_sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel

DEFAULT_DICT_EDGE = 1.2


def _default_sphere():
    return get_sphere(name="repulsion724")


class OdffpDictionary:
    """Dictionary of ODF fingerprints and their microstructure parameters.

    The fingerprints are simulated from a multi-compartment model (intra- and
    extra-axonal and free water) and reconstructed with ``odf_recon_model``
    (GQI by default) on a symmetric ``sphere``.
    """

    IDX_VOID = 0
    IDX_ISO = 1
    PREDEFINED_IDX_NUM = 2

    MICRO_DA = 0
    MICRO_DE = 1
    MICRO_DR = 2
    MICRO_FIN = 3
    MICRO_PARAMS_NUM = 4

    def __init__(self, gtab, sphere=None, dict_file=None):
        self.gtab = gtab
        self.sphere = sphere if sphere is not None else _default_sphere()
        self.max_peaks_num = 0
        self.odf = None
        self.peak_dirs = None
        self.micro = None
        self.ratio = None
        self.peaks_per_voxel = None
        if dict_file is not None:
            self.load(dict_file)

    # -- signal simulation ----------------------------------------------------

    @staticmethod
    def _interval(values):
        return np.array([np.min(values), np.max(values)])

    def _random_fraction_volumes(self, p_iso, p_fib, peaks_per_voxel):
        fraction_volumes = np.zeros(peaks_per_voxel + 1)

        # Lower bounds are hard limits; variability stays in [0, p_random_max].
        p_random_max = 1 - (p_iso[0] + peaks_per_voxel * p_fib[0])
        p_random = np.hstack(
            (
                np.random.uniform(0, p_iso[1] - p_iso[0]),
                np.random.uniform(0, p_fib[1] - p_fib[0], size=peaks_per_voxel),
            )
        )
        p_random /= np.maximum(1e-8, np.sum(p_random))

        fraction_volumes[1:] = p_fib[0] + p_random_max * p_random[1:]
        fraction_volumes[0] = 1 - np.sum(fraction_volumes[1:])
        return fraction_volumes

    def _random_micro_parameters(
        self,
        f_in,
        D_iso,
        D_a,
        D_e,
        D_r,
        peaks_per_voxel,
        equal_fibers,
        assert_faster_D_a,
        tortuosity_approximation,
    ):
        micro = np.zeros((self.MICRO_PARAMS_NUM, peaks_per_voxel + 1))

        # Free-water compartment: D_a = 0, f_in = 0, and D_e = D_iso.
        micro[self.MICRO_DE, 0] = np.random.uniform(D_iso[0], D_iso[1])

        while True:
            if equal_fibers:
                micro[:, 1:] = np.tile(
                    [
                        [np.random.uniform(D_a[0], D_a[1])],
                        [np.random.uniform(D_e[0], D_e[1])],
                        [np.random.uniform(D_r[0], D_r[1])],
                        [np.random.uniform(f_in[0], f_in[1])],
                    ],
                    peaks_per_voxel,
                )
            else:
                micro[:, 1:] = np.array(
                    [
                        np.random.uniform(D_a[0], D_a[1], size=peaks_per_voxel),
                        np.random.uniform(D_e[0], D_e[1], size=peaks_per_voxel),
                        np.random.uniform(D_r[0], D_r[1], size=peaks_per_voxel),
                        np.random.uniform(f_in[0], f_in[1], size=peaks_per_voxel),
                    ]
                )

            if assert_faster_D_a and np.any(
                micro[self.MICRO_DA, 1:] < micro[self.MICRO_DE, 1:]
            ):
                continue

            if tortuosity_approximation:
                micro[self.MICRO_DR, 1:] = (1 - micro[self.MICRO_FIN, 1:]) * micro[
                    self.MICRO_DA, 1:
                ]
                if np.any(micro[self.MICRO_DR, 1:] < D_r[0]) or np.any(
                    micro[self.MICRO_DR, 1:] > D_r[1]
                ):
                    continue
            break
        return micro

    def _compute_dwi(self, ratio, micro, peak_dirs_idx):
        ratio = np.nan_to_num(ratio)
        micro = np.nan_to_num(micro)

        # Convert the b-values from s/mm^2 to ms/um^2.
        bvals = np.vstack(1e-3 * self.gtab.bvals)

        # Diffusion signal of free water.
        dwi = ratio[0] * np.exp(-bvals * micro[self.MICRO_DE, 0])

        # Add the diffusion signal of each fiber.
        for j in range(len(peak_dirs_idx)):
            dir_prod_sqr = (
                np.dot(self.gtab.bvecs, self.sphere.vertices[peak_dirs_idx[j]].T) ** 2
            )
            dwi_intra = np.exp(-bvals * micro[self.MICRO_DA, j + 1] * dir_prod_sqr)
            dwi_extra = np.exp(
                -bvals
                * (
                    micro[self.MICRO_DE, j + 1] * dir_prod_sqr
                    + micro[self.MICRO_DR, j + 1] * (1 - dir_prod_sqr)
                )
            )
            dwi += ratio[j + 1] * (
                micro[self.MICRO_FIN, j + 1] * dwi_intra
                + (1 - micro[self.MICRO_FIN, j + 1]) * dwi_extra
            )

        return 1e3 * dwi.T

    def _compute_odf_trace(self, odf_recon_model, ratio, micro, peak_dirs_idx):
        dwi = self._compute_dwi(ratio, micro, peak_dirs_idx)
        odf = odf_recon_model.fit(dwi).odf(self.sphere).T
        return odf[: len(self.sphere.vertices) // 2]

    def _peaks_per_voxel_cdf(self, total_dirs_num):
        """CDF of the random variable ``peaks_per_voxel``."""
        # The numbers of directions are in the proportion
        # 1 : 1*(k-1) : 1*(k-1)*(k-2) : ...
        cumulative_dirs_num = np.ones(self.max_peaks_num)
        dirs_per_peak = 1
        for i in range(1, self.max_peaks_num):
            dirs_per_peak *= total_dirs_num - i
            cumulative_dirs_num[i] = cumulative_dirs_num[i - 1] + dirs_per_peak
        return cumulative_dirs_num[:-1] / cumulative_dirs_num[-1]

    # -- generation and persistence -------------------------------------------

    def generate(
        self,
        dict_size=1000000,
        max_peaks_num=3,
        equal_fibers=False,
        p_iso=(0.0, 1.0),
        p_fib=(0.0, 1.0),
        f_in=(0.0, 1.0),
        D_iso=(2.0, 3.0),
        D_a=(1.5, 2.5),
        D_e=(1.5, 2.5),
        D_r=(0.5, 1.5),
        max_chunk_size=10000,
        odf_recon_model=None,
        assert_faster_D_a=False,
        tortuosity_approximation=False,
    ):
        """Randomly generate a dictionary of ODF fingerprints."""
        if odf_recon_model is None:
            odf_recon_model = GeneralizedQSamplingModel(
                self.gtab, sampling_length=DEFAULT_DICT_EDGE
            )

        dict_size = np.maximum(1, dict_size)
        self.max_peaks_num = np.maximum(1, max_peaks_num)
        self.peaks_per_voxel = np.zeros(dict_size, dtype=int)
        p_iso, p_fib = self._interval(p_iso), self._interval(p_fib)
        f_in, D_iso = self._interval(f_in), self._interval(D_iso)
        D_a = self._interval(D_a)
        D_e, D_r = self._interval(D_e), self._interval(D_r)

        # Total number of directions allowed by the tessellation (k).
        total_dirs_num = len(self.sphere.vertices) // 2

        # Unused elements are kept as NaNs for backward compatibility.
        self.peak_dirs = np.nan * np.zeros((2, self.max_peaks_num, dict_size))
        self.ratio = np.nan * np.zeros((self.max_peaks_num + 1, dict_size))
        self.micro = np.nan * np.zeros((4, self.max_peaks_num + 1, dict_size))
        self.odf = np.zeros((total_dirs_num, dict_size))

        # VOID element: empty voxels outside the mask (skipped in matching).
        self.ratio[0, self.IDX_VOID] = 0
        self.micro[self.MICRO_DE, 0, self.IDX_VOID] = 0
        self.peaks_per_voxel[self.IDX_VOID] = -1

        # ISO element: voxels with isotropic (free) water only.
        self.ratio[0, self.IDX_ISO] = 1
        self.micro[self.MICRO_DE, 0, self.IDX_ISO] = 3
        self.peaks_per_voxel[self.IDX_ISO] = 0
        self.odf[:, self.IDX_ISO] = np.squeeze(
            self._compute_odf_trace(
                odf_recon_model,
                self.ratio[:, self.IDX_ISO],
                self.micro[:, :, self.IDX_ISO],
                [],
            )
        )

        chunk_bounds = range(max_chunk_size, dict_size, max_chunk_size)
        for chunk_idx in np.split(
            range(self.PREDEFINED_IDX_NUM, dict_size), chunk_bounds
        ):
            chunk_size = len(chunk_idx)
            peak_dirs_idx = np.zeros((self.max_peaks_num, chunk_size), dtype=int)

            # Draw the numbers of peaks per voxel. Direction [0, 0, 1] is
            # obligatory, hence the leading 1.
            self.peaks_per_voxel[chunk_idx] = 1 + np.sum(
                np.random.uniform(size=(chunk_size, 1))
                > self._peaks_per_voxel_cdf(total_dirs_num),
                axis=1,
            )

            for i, j in zip(range(chunk_size), chunk_idx):
                # Direction [0, 0, 1] has index 0 in the tessellation.
                peak_dirs_idx[1 : self.peaks_per_voxel[j], i] = np.random.choice(
                    range(1, total_dirs_num), self.peaks_per_voxel[j] - 1, replace=False
                )

                self.peak_dirs[:, : self.peaks_per_voxel[j], j] = np.array(
                    [
                        self.sphere.phi[peak_dirs_idx[: self.peaks_per_voxel[j], i]],
                        self.sphere.theta[peak_dirs_idx[: self.peaks_per_voxel[j], i]]
                        - np.pi / 2,
                    ]
                )
                self.ratio[: self.peaks_per_voxel[j] + 1, j] = (
                    self._random_fraction_volumes(p_iso, p_fib, self.peaks_per_voxel[j])
                )
                self.micro[:, : self.peaks_per_voxel[j] + 1, j] = (
                    self._random_micro_parameters(
                        f_in,
                        D_iso,
                        D_a,
                        D_e,
                        D_r,
                        self.peaks_per_voxel[j],
                        equal_fibers,
                        assert_faster_D_a,
                        tortuosity_approximation,
                    )
                )

            self.odf[:, chunk_idx] = self._compute_odf_trace(
                odf_recon_model,
                self.ratio[:, chunk_idx],
                self.micro[:, :, chunk_idx],
                peak_dirs_idx,
            )

            # Sort the peaks of each voxel in descending order and recompute the
            # ODF when the main peak was not the obligatory [0, 0, 1].
            recompute_filter = np.zeros(chunk_size, dtype=bool)
            for i, j in zip(range(chunk_size), chunk_idx):
                if self.peaks_per_voxel[j] < 2:
                    continue
                sorted_idx = np.argsort(
                    -self.odf[peak_dirs_idx[: self.peaks_per_voxel[j], i], j]
                )
                seq_idx = np.arange(self.peaks_per_voxel[j])
                if np.any(sorted_idx != seq_idx):
                    self.micro[:, seq_idx + 1, j] = self.micro[:, sorted_idx + 1, j]
                    self.ratio[seq_idx + 1, j] = self.ratio[sorted_idx + 1, j]
                if sorted_idx[0] != 0:
                    recompute_filter[i] = True

            self.odf[:, chunk_idx[recompute_filter]] = self._compute_odf_trace(
                odf_recon_model,
                self.ratio[:, chunk_idx[recompute_filter]],
                self.micro[:, :, chunk_idx[recompute_filter]],
                peak_dirs_idx[:, recompute_filter],
            )

    def save(self, dict_file="odf_dict.npz"):
        """Save the dictionary to a ``.npz`` file."""
        np.savez(
            dict_file,
            odf=self.odf,
            peak_dirs=self.peak_dirs,
            micro=self.micro,
            ratio=self.ratio,
            peaks_per_voxel=self.peaks_per_voxel,
            max_peaks_num=self.max_peaks_num,
        )

    def load(self, dict_file):
        """Load a dictionary previously saved with :meth:`save`."""
        data = np.load(dict_file)
        self.odf = data["odf"]
        self.peak_dirs = data["peak_dirs"]
        self.micro = data["micro"]
        self.ratio = data["ratio"]
        self.peaks_per_voxel = data["peaks_per_voxel"]
        self.max_peaks_num = int(data["max_peaks_num"])
