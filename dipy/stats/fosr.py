import numpy as np

import statsmodels.api as sm

from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis

from skfda.misc.regularization import compute_penalty_matrix
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import compute_penalty_matrix, TikhonovRegularization
from skfda.representation import FDataBasis


import gc

from scipy.sparse import diags
import scipy.sparse as sp

from dipy.stats.gam import gam

def get_covariates(df):
    Y = []
    X = []

    unique_subjects = df['subject'].unique()
    unique_disk = df['disk'].unique()

    no_disk = len(unique_disk)

    for sub in unique_subjects:
        sub_df = df[df['subject']==sub]
        unique_streamline = sub_df['streamline'].unique()
        len_streamlines = len(unique_streamline)
        group = sub_df['group'].unique()[0]
        if 'gender' in sub_df.columns:
            gender = sub_df['gender'].unique()[0]
        print("For subject {} I have {} unique streamlines and group is {}".format(sub, len_streamlines, group))
        if(group==0):
            # continue
            sub_X = np.zeros((len_streamlines,1))
        elif(group==1):
            sub_X = np.ones((len_streamlines,1))
        else:
            print("For subject {} I have a invalid group which is {}".format(sub, group))
        if(gender == "Female"):
            zero_column = np.zeros((len_streamlines, 1))
            sub_X = np.hstack((sub_X, zero_column))
        elif(gender == "Male"):
            one_column = np.ones((len_streamlines, 1))
            sub_X = np.hstack((sub_X, one_column))
        if 'age' in sub_df.columns:
            age = sub_df['age'].unique()[0]
            age_column = age*np.ones((len_streamlines, 1))
            sub_X = np.hstack((sub_X, age_column))
        if(len(X)==0):
            X = sub_X
        else:
            X = np.append(X, sub_X, axis=0)
        sub_Y = np.zeros((len_streamlines,no_disk))
        count_Y = np.zeros((len_streamlines,no_disk))

        for index, row in sub_df.iterrows():
            x = row['streamline']
            y = row['disk']-1
            sub_Y[x][y] += row['fa']
            count_Y[x][y] +=1
        for i in range(len_streamlines):
            for j in range(no_disk):
                if(count_Y[i][j]>0):
                    sub_Y[i][j] = sub_Y[i][j]/count_Y[i][j]
        if(len(Y)==0):
            Y = sub_Y
        else:
            Y = np.vstack((Y, sub_Y))
        print("Printing X rows {}. Printing Y rows {}".format(X.shape[0],Y.shape[0]))
        

    selected_indices = np.random.choice(X.shape[0], min(X.shape[0],12000), replace=False)
    X = np.array(X[selected_indices])
    Y = np.array(Y[selected_indices])
    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((X, ones_column))
    print("Printing X rows {}. Printing Y rows {}".format(X.shape[0],Y.shape[0]))
    return X,Y

def fosr(self,formula=None, Y=None, fdobj=None, data=None, X=None, con = None, argvals = None, method = "OLS", gam_method = "REML", cov_method = "naive", labda = None, nbasis=15, norder=4, pen_order=2, multi_sp = False, pve=.99, max_iter = 1, maxlam = None, cv1 = False, scale = False):

    multi_sp = False if method == "OLS" else True

    ## Handle the case when formula is NULL
    resp_type = "fd" if Y is None else "raw"

    if(argvals==None):
        argvals = np.linspace(0, 1, num=Y.shape[1])
    else:
        print("R equivalent code of seq(min(fdobj$basis$range), max(fdobj$basis$range), length=201)")


    if(method!= "OLS" and len(labda)>1):
        print("Vector-valued lambda allowed only if method = 'OLS'")
        return None
    if(labda!=None and multi_sp):
        print("Fixed lambda not implemented with multiple penalties")
        return None
    if(method == "OLS" and multi_sp):
        print("OLS not implemented with multiple penalties")
        return None

    if(resp_type == "raw"):
        bss = BSplineBasis(domain_range=(0,1) , n_basis=nbasis, order=norder)
        Bmat = bss.evaluate(argvals).reshape(nbasis,100)
        print("Bmat shape ", Bmat.shape)
        Theta = bss.evaluate(argvals).reshape(nbasis,100)
        respmat = Y
    elif(resp_type == "fd"):
        print("See scikit-fda and fosr code to implement this part as well!!!! - This will be more useful in the long run")

    new_fit = None
    U = None
    pca_resid = None

    X_sc = X ## TODO - Change this to standard scalar later - StandardScaler with mean as false and scale as scale
    q = X.shape[1]
    ncurve = respmat.shape[0]

    if(multi_sp):
        print("Look at the R code for this!!!")
    else:
        # Define the differential operator for the penalty
        differential_operator = LinearDifferentialOperator(pen_order)
        regularization_parameter = 1.0
        regularization = TikhonovRegularization(differential_operator)

        # Compute the penalty matrix
        bss_derivative = compute_penalty_matrix(
            basis_iterable=[bss], 
            regularization_parameter=regularization_parameter,
            regularization=regularization
        )
        pen = np.kron(np.eye(q), bss_derivative)

    if(con!=None):
        constr = np.kron(con, np.eye(nbasis))
    else:
        constr = None

    cv = None

    if(method == "OLS"):
        if((labda == None  or len(labda) != 1) or cv1):
            print("Time to use lofocv for hyper parameters, figure it out")

    X_gam = np.kron(X_sc, np.transpose(Bmat))
    Y_gam = respmat.ravel()
    firstfit = gam(Y_gam, X_gam, gam_method = gam_method, S = [pen], C = constr, labda = labda)

    print("printing coefficients shape ", firstfit["coefficients"].shape)

    coefmat = firstfit["coefficients"].reshape(q, firstfit["coefficients"].shape[0]//X.shape[1])
    coefmat_ols = firstfit["coefficients"].reshape(q, firstfit["coefficients"].shape[0]//X.shape[1])
    se = None

    if(method != "OLS"):
        print("Take care of this use case")
    if(method == "OLS" or max_iter == 0):
        resid_vec = (respmat.ravel() - (np.kron(X_sc, Bmat.T)@firstfit["coefficients"])).reshape(-1, 1)
        num_rows = len(resid_vec) // ncurve
        cov_mat = ((ncurve-1)/ncurve) * np.cov(np.reshape(resid_vec, (ncurve, num_rows)), rowvar=False)
        ngrid = cov_mat.shape[0]
        M = ngrid * ncurve
        # Construct the block diagonal matrix
        cov_bdiag = sp.block_diag([cov_mat] * ncurve, format="csc")
        var_b = firstfit["GinvXT"]@cov_bdiag@firstfit["GinvXT"].T

        del cov_bdiag
        gc.collect()
    else:
        var_b = new_fit["Vp"] ## newfit will come frome the first if condition which is yet to be written

    se_func = np.full((len(argvals), q), np.nan)
    for j in range(1,q+1):
        start_idx = nbasis * (j - 1)
        end_idx = nbasis * j
        var_b_submatrix = var_b[start_idx:end_idx, start_idx:end_idx]
        product = Theta.T @ var_b_submatrix * Theta.T
        row_sums = np.sqrt(np.sum(product, axis=1))
        se_func[:, j - 1] = row_sums

    fd = FDataBasis(basis=bss, coefficients=coefmat)
    est_func = np.squeeze(fd.evaluate(argvals)).T
    #print("est func dimensions ",est_func.shape)

    if(method == "mix" and max_iter>0):
        fit = new_fit ## need to implement the newfit
    else:
        fit = firstfit

    roughness = np.diag(coefmat@bss_derivative@coefmat.T)

    if(resp_type == "raw"):
        yhat = X@np.dot(coefmat, Theta)
        
    return {"fd": fd, "pca.resid": pca_resid, "U": U, "yhat": yhat, "est.func" : est_func,  "se.func" : se_func, "argvals": argvals, "fit": fit}