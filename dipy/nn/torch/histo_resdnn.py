#!/usr/bin/python
"""Class and helper functions for fitting the Histological ResDNN model."""

import logging

import numpy as np

from dipy.core.gradients import get_bval_indices, unique_bvals_magnitude
from dipy.core.sphere import HemiSphere
from dipy.data import get_fnames, get_sphere
from dipy.nn.utils import set_logger_level
from dipy.reconst.shm import sf_to_sh, sh_to_sf, sph_harm_ind_list
from dipy.testing.decorators import doctest_skip_parser
from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")
if have_torch:
    from torch.nn import Linear, Module
else:

    class Module:
        pass

    logging.warning(
        "This model requires Pytorch.\
                    Please install these packages using \
                    pip."
    )

logging.basicConfig()
logger = logging.getLogger("histo_resdnn")


class DenseModel(Module):
    def __init__(self, sh_size, num_hidden):
        super(DenseModel, self).__init__()
        self.fc1 = Linear(sh_size, 400)
        self.fc2 = Linear(400, num_hidden)
        self.fc3 = Linear(num_hidden, 200)
        self.fc4 = Linear(200, num_hidden)
        self.fc5 = Linear(num_hidden, 200)
        self.fc6 = Linear(200, num_hidden)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.relu(self.fc3(x2))
        x4 = self.fc4(x3)

        # Adding x2 and x4
        res_add = x2 + x4

        x5 = torch.relu(self.fc5(res_add))
        x6 = self.fc6(x5)

        return x6


class HistoResDNN:
    """This class is intended for the ResDNN Histology Network model.

    ResDNN :footcite:p:`Nath2019` is a deep neural network that employs residual
    blocks deep neural network to predict ground truth SH coefficients from SH
    coefficients computed using DWI data. To this end, authors considered
    histology FOD-computed SH coefficients (obtained from ex vivo non-human
    primate acquisitions) as their ground truth, and the DWI-computed SH
    coefficients as their target.

    References
    ----------
    .. footbibliography::
    """

    @doctest_skip_parser
    def __init__(
        self, *, sh_order_max=8, basis_type="tournier07", verbose=False, use_cuda=False
    ):
        """Model initialization

        The model was re-trained for usage with a different basis function
        ('tournier07') like the proposed model in :footcite:p:`Nath2019`.

        To obtain the pre-trained model, use::
        >>> resdnn_model = HistoResDNN() # skip if not have_torch
        >>> fetch_model_weights_path = get_fnames(name='histo_resdnn_torch_weights') # skip if not have_torch
        >>> resdnn_model.load_model_weights(fetch_model_weights_path) # skip if not have_torch

        This model is designed to take as input raw DWI signal on a sphere
        (ODF) represented as SH of order 8 in the tournier basis and predict
        fODF of order 8 in the tournier basis. Effectively, this model is
        mimicking a CSD fit.

        Parameters
        ----------
        sh_order_max : int, optional
            Maximum SH order (l) in the SH fit.  For ``sh_order_max``, there
            will be
            ``(sh_order_max + 1) * (sh_order_max + 2) / 2`` SH coefficients
            for a symmetric basis.
        basis_type : {'tournier07', 'descoteaux07'}, optional
            ``tournier07`` (default) or ``descoteaux07``.
        verbose : bool, optional
            Whether to show information about the processing.
        use_cuda : bool, optional
            Whether to use GPU for processing.
            If False or no CUDA is detected, CPU will be used.

        References
        ----------
        .. footbibliography::
        """  # noqa: E501
        if not have_torch:
            raise torch()

        self.sh_order_max = sh_order_max
        self.sh_size = len(sph_harm_ind_list(sh_order_max)[0])
        self.basis_type = basis_type

        log_level = "INFO" if verbose else "CRITICAL"
        set_logger_level(log_level, logger)
        if self.basis_type != "tournier07":
            logger.warning(
                "Be careful, original weights were obtained "
                "from training on the tournier07 basis, "
                "unless you re-trained the network, do not "
                "change basis!"
            )

        # ResDNN Network Flow
        num_hidden = self.sh_size
        self.model = DenseModel(self.sh_size, num_hidden).type(torch.float64)
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

    def fetch_default_weights(self):
        """Load the model pre-training weights to use for the fitting.
        Will not work if the declared SH_ORDER does not match the weights
        expected input.
        """
        fetch_model_weights_path = get_fnames(name="histo_resdnn_torch_weights")
        self.load_model_weights(fetch_model_weights_path)

    def load_model_weights(self, weights_path):
        """Load the custom pre-training weights to use for the fitting.
        Will not work if the declared SH_ORDER does not match the weights
        expected input.

        The weights for a sh_order of 8 can be obtained via the function:
            get_fnames('histo_resdnn_torch_weights').

        Parameters
        ----------
        weights_path : str
            Path to the file containing the weights (pth, saved by Pytorch)
        """
        try:
            self.model.load_state_dict(
                torch.load(
                    weights_path,
                    weights_only=True,
                    map_location=self.device,
                )
            )
            self.model.eval()
        except RuntimeError as e:
            raise RuntimeError(
                "Expected input for the provided model weights do not match the "
                f"declared model ({self.sh_size})"
            ) from e

    def __predict(self, x_test):
        """Predict fODF (as SH) from input raw DWI signal (as SH)

        Parameters
        ----------
        x_test : np.ndarray
            Array of size (N, M) where M is
            ``(sh_order_max + 1) * (sh_order_max + 2) / 2``.
            N should not be too big as to limit memory usage.

        Returns
        -------
        np.ndarray (N, M)
            Predicted fODF (as SH)
        """
        if x_test.shape[-1] != self.sh_size:
            raise ValueError(
                "Expected input for the provided model weights do not match the "
                f"declared model ({self.sh_size})"
            )

        return self.model(torch.from_numpy(x_test)).detach().numpy()

    def predict(self, data, gtab, *, mask=None, chunk_size=1000):
        """Wrapper function to facilitate prediction of larger dataset.
        The function will mask, normalize, split, predict and 're-assemble'
        the data as a volume.

        Parameters
        ----------
        data : np.ndarray
            DWI signal in a 4D array
        gtab : GradientTable class instance
            The acquisition scheme matching the data (must contain at least
            one b0)
        mask : np.ndarray, optional
            Binary mask of the brain to avoid unnecessary computation and
            unreliable prediction outside the brain.
            Default: Compute prediction only for nonzero voxels (with at least
            one nonzero DWI value).
        chunk_size : int, optional
            Batch size when running model prediction.

        Returns
        -------
        pred_sh_coef : np.ndarray (x, y, z, M)
            Predicted fODF (as SH). The volume has matching shape to the input
            data, but with
            ``(sh_order_max + 1) * (sh_order_max + 2) / 2`` as a last
            dimension.
        """
        if mask is None:
            logger.warning(
                "Mask should be provided to accelerate "
                "computation, and because predictions are "
                "not reliable outside of the brain."
            )
            mask = np.sum(data, axis=-1)
        mask = mask.astype(bool)

        # Extract B0's and obtain a mean B0
        b0_indices = gtab.b0s_mask
        if not len(b0_indices) > 0:
            raise ValueError("b0 must be present for DWI normalization.")
        logger.info(f"b0 indices found are: {np.argwhere(b0_indices).ravel()}")

        mean_b0 = np.mean(data[..., b0_indices], axis=-1)

        # Detect number of b-values and extract a single shell of DW-MRI Data
        unique_shells = np.sort(unique_bvals_magnitude(gtab.bvals))
        logger.info(f"Number of b-values: {unique_shells}")

        # Extract DWI only
        dw_indices = get_bval_indices(gtab.bvals, unique_shells[1])
        dw_data = data[..., dw_indices]
        dw_bvecs = gtab.bvecs[dw_indices, :]

        # Normalize the DW-MRI Data with the mean b0 (voxel-wise)
        norm_dw_data = np.zeros(dw_data.shape)
        for n in range(len(dw_indices)):
            norm_dw_data[..., n] = np.divide(
                dw_data[..., n], mean_b0, where=np.abs(mean_b0) > 0.000001
            )

        # Fit SH to the raw DWI signal
        h_sphere = HemiSphere(xyz=dw_bvecs)
        dw_sh_coef = sf_to_sh(
            norm_dw_data,
            h_sphere,
            smooth=0.0006,
            basis_type=self.basis_type,
            sh_order_max=self.sh_order_max,
        )

        # Flatten and mask the data (N, SH_SIZE) to facilitate chunks
        ori_shape = dw_sh_coef.shape
        flat_dw_sh_coef = dw_sh_coef[mask > 0]
        flat_pred_sh_coef = np.zeros(flat_dw_sh_coef.shape)

        count = len(flat_dw_sh_coef) // chunk_size
        for i in range(count + 1):
            if i % 100 == 0 or i == count:
                logger.info(f"Chunk #{i} out of {count}")
            tmp_sh = self.__predict(
                flat_dw_sh_coef[i * chunk_size : (i + 1) * chunk_size]
            )

            # Removing negative values from the SF
            sphere = get_sphere(name="repulsion724")
            tmp_sf = sh_to_sf(
                sh=tmp_sh,
                sphere=sphere,
                basis_type=self.basis_type,
                sh_order_max=self.sh_order_max,
            )
            tmp_sf[tmp_sf < 0] = 0
            tmp_sh = sf_to_sh(
                tmp_sf,
                sphere,
                smooth=0.0006,
                basis_type=self.basis_type,
                sh_order_max=self.sh_order_max,
            )
            flat_pred_sh_coef[i * chunk_size : (i + 1) * chunk_size] = tmp_sh

        pred_sh_coef = np.zeros(ori_shape)
        pred_sh_coef[mask > 0] = flat_pred_sh_coef

        return pred_sh_coef
