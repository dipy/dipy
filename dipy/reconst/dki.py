"""
Diffusion Kurtosis Imaging

"""
import numpy as np
from scipy.misc import factorial

def monomial_matrix(grad, order):
    """
    Compute the monomial matrix of the gradients up to some order

    Parameters
    ----------
    gradients : ndarray of shape (n, 3)
       The b vectors of the measurement

    order : int
       The order of the monomial matrix

    Returns
    -------
    n*(2+order)!/2(order)!

    Notes
    -----
    A. Barmpoutis and B.C. Vemuri (2010) "A Unified Framework for Estimating
    Diffusion Tensors of any order with Symmetric Positive-Definite
    Constraints", Proc IEEE ISBI, 2010
    """
    monom = np.zeros((grad.shape[0], factorial(2+order)/(2*factorial(order)) ))
    for k in range(grad.shape[0]):
        c=0
        for i in range(order):
            for j in range(order-i):
                monom[k,c]=(grad[k,0]**i)*(grad[k,1]**j)*(grad[k,2]**(order-i-j))
                c+=1

    return monom

class DiffusionKurtosisModel(object):
    """

    """
    def __init__(self, gtab, *args, **kwargs):
        """
        gtab: GradientTable


        References
        ----------


        """

        g_order2 = monomial_matrix(gtab.bvecs, 2)
	g_order4 = monomial_matrix(gtab.bvecs, 4)
	bval_order2 = np.tile(gtab.bvals, (6,1)).transpose();
	bval_order4 = np.tile(gtab.bvals, (15,1)).transpose();

	self.G_matrix = np.concatenate((-bval_order2 * g_order2,
                                     bval_order4 * bval_order4 * g_order4 / 6),
                                     axis=1)

    def fit(self, data, mask=None):
        """
        """
        # If a mask is provided, we will use it to access the data
        if mask is not None:
            # Make sure it's boolean, so that it can be used to mask
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = data[mask]
        else:
            data_in_mask = data

        log_relative_data = np.log(data_in_mask/self.S0)

        # least-square estimation
        if mode == 0 :
            kurtosis = linalg.lstsq(Gbig, logS)[0];
        else :
            # The positive-def estimation here THIS DOES NOT WORK. The
            # kurtosis coefficients can be it's the whole kurtosis that needs
            # to be positive
            kurtosis = optimize.nnls(Gbig, logS)[0];

        K[x,y,z,:] = real(kurtosis);

        params_in_mask = self.fit_method(self.design_matrix, data_in_mask,
                                         *self.args, **self.kwargs)

        dti_params = np.zeros(data.shape[:-1] + (12,))

        dti_params[mask, :] = params_in_mask

        return TensorFit(self, dti_params)



class DiffusionKurtosisFit(object):
    """

    """
    def __init__():
        """
        """
