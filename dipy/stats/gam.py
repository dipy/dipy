import pandas as pd
import statsmodels.api as sm
import numpy as np

def gam(y, Xmat, gam_method = None, S = None, C = None, labda = None):

    # stopifnot( is.null(lambda) | length(lambda)==n.p ) - write the python version of it but now labda is NULL
    n_p = len(S)

    if C is not None:
        print("Need to implement")
    else:
        Z = np.eye(Xmat.shape[1])
        Xmat_copy = Xmat
        if labda is None:
            S_copy = list(list(S)) ## Need to verify this and make this proper
            
    ## comment the next lines if you don't want statsmodel's fit
    data =pd.DataFrame(Xmat)
    data['Y'] = y.reshape(-1,1)
    endog = data['Y']
    exog = data.drop(columns=['Y'])
    glm = sm.GLM(endog, exog, family = sm.families.Gaussian())
    alpha = 0.1
    L1_wt = 0
    fitobj = glm.fit(alpha=alpha, L1_wt=L1_wt)
    cov_params = fitobj.cov_params().values
    GinvXT = Z@np.linalg.solve(Xmat.T @ Xmat, Xmat.T)

    return {
        'gam': fitobj,
        'coefficients': Z@fitobj.params.values,
        'Vp': Z@cov_params@Z.T,
        'GinvXT': GinvXT,
        'cov_params': cov_params,
        'Z': Z
    }