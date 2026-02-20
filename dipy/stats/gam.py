import numpy as np
import pandas as pd
import statsmodels.api as sm


def gam(y, Xmat, gam_method=None, S=None, C=None, labda=None):
    """
    Fits a Generalized Additive Model (GAM) using GLM and returns model coefficients,
    covariance estimates, and intermediate matrices useful for inference.

    Parameters:
    ----------
    y : np.ndarray
        Response variable vector (dependent variable), shape (n_samples,).

    Xmat : np.ndarray
        Design matrix of predictor variables, shape (n_samples, n_features).

    gam_method : str or None, optional
        Method used to fit the GAM (e.g., 'REML', 'GCV')

    S : list of np.ndarray
        List of smoothing (penalty) matrices, one per term.
        Required when `labda` is not provided.

    C : any or None, optional
        Constraint matrix or related structure.

    labda : list or None, optional
        List of smoothing parameters corresponding to each smoothing matrix in `S`.
        If `None`,default estimation is triggered (not fully implemented in this stub).

    Returns:
    -------
    dict
        A dictionary with the following keys:
            - 'gam': Fitted GLM model object from statsmodels.
            - 'coefficients': Estimated model coefficients.
            - 'Vp': Transformed covariance matrix of the coefficients.
            - 'GinvXT': # generalized inverse of Xᵀ times Z.
            - 'cov_params': Raw covariance matrix of model parameters.
            - 'Z': Identity matrix used as a transformation (placeholder for now).
    """
    # Initialize Z as identity matrix (default case)
    Z = np.eye(Xmat.shape[1])

    if C is not None:
        print("Need to implement")
        # For now, keep Z as identity matrix when constraints are present

    ## comment the next lines if you don't want statsmodel's fit
    # Convert y to numpy array if it's a pandas Series
    if hasattr(y, "values"):
        y = y.values

    data = pd.DataFrame(Xmat)
    data["Y"] = y.reshape(-1, 1)
    endog = data["Y"]
    exog = data.drop(columns=["Y"])
    glm = sm.GLM(endog, exog, family=sm.families.Gaussian())
    alpha = 0.1
    L1_wt = 0
    fitobj = glm.fit(alpha=alpha, L1_wt=L1_wt)
    cov_params = fitobj.cov_params().values
    GinvXT = Z @ np.linalg.solve(Xmat.T @ Xmat, Xmat.T)

    return {
        "gam": fitobj,
        "coefficients": Z @ fitobj.params.values,
        "Vp": Z @ cov_params @ Z.T,
        "GinvXT": GinvXT,
        "cov_params": cov_params,
        "Z": Z,
    }
