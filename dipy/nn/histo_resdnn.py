"""
This script is intended for the model object
of ResDNN Histology Network.
The model was re-trained for usage with different basis function ('mrtrix') set
as per the proposed model from the paper:
[1] Nath, V., Schilling, K. G., Parvathaneni, P., Hansen,
C. B., Hainline, A. E., Huo, Y., ... & Stepniewska, I. (2019).
Deep learning reveals untapped information for local white-matter
fiber reconstruction in diffusion-weighted MRI.
Magnetic resonance imaging, 62, 220-227.
[2] Nath, V., Schilling, K. G., Hansen, C. B., Parvathaneni,
P., Hainline, A. E., Bermudez, C., ... & Stępniewska, I. (2019).
Deep learning captures more accurate diffusion fiber orientations
distributions than constrained spherical deconvolution.
arXiv preprint arXiv:1911.07927.
"""
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
import logging

from dipy.core.sphere import HemiSphere
from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf, sph_harm_ind_list
import numpy as np

from distutils.version import LooseVersion
from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package('tensorflow')
if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')

logging.basicConfig()
logger = logging.getLogger('histo_resdnn')


def custom_accuracy_sh(y_true, y_pred):
    y_true = y_true
    y_pred = y_pred

    comp_true = tf.math.conj(y_true)
    norm_true = y_true / tf.sqrt(tf.reduce_sum(tf.multiply(y_true, comp_true)))

    comp_pred = tf.math.conj(y_pred)
    norm_pred = y_pred / tf.sqrt(tf.reduce_sum(tf.multiply(y_pred, comp_pred)))

    comp_p2 = tf.math.conj(norm_pred)
    acc = tf.math.real(tf.reduce_sum(tf.multiply(norm_true, comp_p2)))

    return acc


def set_logger_level(log_level):
    """ Change the logger of the HistoResDNN to one on the following:
    DEBUG, INFO, WARNING, CRITICAL, ERROR

    Parameters
    ----------
    log_level : str
        Log level for the HistoResDNN only
    """
    logger.setLevel(level=log_level)


class HistoResDNN(object):
    def __init__(self, sh_order=8, sh_basis='descoteaux07', verbose=False):
        """
        Single Layer Perceptron with Dropout
        """
        self.sh_order = sh_order
        self.sh_size = len(sph_harm_ind_list(sh_order)[0])
        self.sh_basis = sh_basis

        log_level = 'INFO' if verbose else 'CRITICAL'
        set_logger_level(log_level)
        if self.sh_basis != 'descoteaux07':
            logger.warning('Be careful, original weights were obtained '
                           'from training on the descoteaux07 basis, '
                           'unless you re-trained the network, do not '
                           'change basis!')

        # ResDNN Network Flow
        num_hidden = self.sh_size
        inputs = Input(shape=(self.sh_size,))
        x1 = Dense(400, activation='relu')(inputs)
        x2 = Dense(num_hidden, activation='relu')(x1)
        x3 = Dense(200, activation='relu')(x2)
        x4 = Dense(num_hidden, activation='linear')(x3)
        res_add = Add()([x2, x4])
        x5 = Dense(200, activation='relu')(res_add)
        x6 = Dense(num_hidden)(x5)

        model = Model(inputs=inputs, outputs=x6)
        opt_func = RMSprop(learning_rate=0.0001)
        model.compile(optimizer=opt_func,
                      loss='mse',
                      metrics=[custom_accuracy_sh])

        self.model = model

    def load_model_weights(self, weights_path):
        """
        """
        try:
            self.model.load_weights(weights_path)
        except ValueError:
            raise ValueError('Expected input for the provided model weights '
                             'do not match the declared model ({})'
                             .format(self.sh_size))

    def predict(self, x_test):
        return self.model.predict(x_test)

    def fit(self, data, gtab, mask=None):
        """
        """
        if mask is None:
            logger.warning('Mask should be provided to accelerate '
                           'computation, and because predictions are '
                           'not reliable outside of the brain.')
            mask = np.sum(data, axis=-1).astype(bool)

        # Extract B0's and obtain a mean B0
        b0_indices = np.where(gtab.bvals == 0)[0]
        logger.info('b0 indices found are: {}'.format(b0_indices))

        mean_b0 = data[..., b0_indices]
        mean_b0 = np.mean(mean_b0, axis=-1)

        # Detect number of b-values and extract a single shell of DW-MRI Data
        unique_shells = np.sort(np.unique(gtab.bvals))
        logger.info('Number of b-values: {}'.format(unique_shells))

        # Extract DWI only
        dw_indices = np.where(gtab.bvals == unique_shells[1])[0]
        dw_data = data[..., dw_indices]
        dw_bvecs = gtab.bvecs[dw_indices, :]

        # Normalize the DW-MRI Data with the mean b0 (voxel-wise)
        norm_dw_data = np.zeros(dw_data.shape)
        for n in range(len(dw_indices)):
            norm_dw_data[..., n] = np.divide(dw_data[..., n], mean_b0,
                                             where=np.abs(mean_b0) > 0.000001)

        # Fit SH to the raw DWI signal
        h_sphere = HemiSphere(xyz=dw_bvecs)
        dw_sh_coef = sf_to_sh(norm_dw_data, h_sphere, smooth=0.0006,
                              basis_type=self.sh_basis,
                              sh_order=self.sh_order)

        # Flatten and mask the data (N, SH_SIZE) to facilitate chunks
        ori_shape = dw_sh_coef.shape
        flat_dw_sh_coef = dw_sh_coef[mask > 0]
        flat_pred_sh_coef = np.zeros(flat_dw_sh_coef.shape)

        count = len(flat_dw_sh_coef) // 1000
        for i in range(count+1):
            if i % 100 == 0 or i == count:
                logger.info('Chunk #{} out of {}'.format(i, count))
            tmp_sh = self.predict(flat_dw_sh_coef[(i)*1000:(i+1)*1000])

            # Removing negative values from the SF
            sphere = get_sphere('repulsion724')
            tmp_sf = sh_to_sf(sh=tmp_sh, sphere=sphere,
                              basis_type=self.sh_basis,
                              sh_order=self.sh_order)
            tmp_sf[tmp_sf < 0] = 0
            tmp_sh = sf_to_sh(tmp_sf, sphere, smooth=0.0006,
                              basis_type=self.sh_basis,
                              sh_order=self.sh_order)
            flat_pred_sh_coef[(i)*1000:(i+1)*1000] = tmp_sh

        pred_sh_coef = np.zeros(ori_shape)
        pred_sh_coef[mask > 0] = flat_pred_sh_coef

        return pred_sh_coef
