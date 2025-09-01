import gc

import numpy as np
import pandas as pd
import scipy.sparse as sp
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import TikhonovRegularization, compute_penalty_matrix
from skfda.representation import FDataBasis
from skfda.representation.basis import BSplineBasis

from dipy.stats.gam import gam


def get_covariates(df, no_streamlines=12000):
    """
    Constructs the covariate matrix X and response matrix Y using 12000
    stramlines from subject-level diffusion imaging data, incorporating
    group, gender, age, and fractional anisotropy (FA) values.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns:
        - 'subject': Subject identifier
        - 'streamline': Streamline index (within each subject)
        - 'disk': Disk index (used to organize response columns)
        - 'fa': Fractional anisotropy value for each streamline-disk
          combination
        - 'group': Binary group label (0 or 1)
        - 'gender' (optional): 'Male' or 'Female'
        - 'age' (optional): Age of subject — if present, used as a
          numeric covariate

    Returns:
    -------
    X : np.ndarray
        Covariate matrix of shape (n_samples, n_features), with features
        including:
        - Group (0 or 1)
        - Gender (1 if Male, 0 if Female; added only if gender column
          exists)
        - Intercept term (final column of 1s)

    Y : np.ndarray
        Response matrix of shape (n_samples, n_disks), where each row
        contains the averaged FA values across disks for each streamline
        within a subject.

    Notes:
    ------
    - The function ensures equal sampling across subjects by computing
      per-streamline averages.
    - Final output is subsampled to at most 12,000 rows for efficiency.
    - Prints diagnostic information about data shape during processing.
    - Handles edge cases: converts numeric string subjects to int,
      validates gender values, requires FA values.
    """

    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Check required columns
    required_columns = ["subject", "streamline", "disk", "fa", "group"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Handle subject column - convert string numbers to int, but allow
    # non-numeric subject IDs
    if "subject" in df.columns:
        # Check if subjects are numeric strings that can be converted to int
        try:
            # Try to convert to numeric first
            numeric_subjects = pd.to_numeric(df["subject"], errors="coerce")
            # If all subjects can be converted to numeric, convert them to int
            if not numeric_subjects.isna().any():
                df["subject"] = numeric_subjects.astype(int)
                print("Converted string numeric subjects to integers")
        except (ValueError, TypeError):
            # If conversion fails, keep as string (this is valid for subject IDs)
            print("Subject IDs kept as strings (non-numeric identifiers)")

    # Create a working copy to avoid modifying the original DataFrame
    df_working = df.copy()

    # Validate data types for numeric columns
    numeric_columns = ["streamline", "disk", "fa", "group"]
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df_working[col]):
            try:
                df_working[col] = pd.to_numeric(df_working[col], errors="coerce")
                if df_working[col].isna().any():
                    raise ValueError(
                        f"Column '{col}' contains non-numeric values that "
                        f"cannot be converted"
                    )
            except (ValueError, TypeError) as err:
                raise ValueError(f"Column '{col}' must be numeric") from err

    # Validate FA values - throw error if not present
    if df_working["fa"].isna().any():
        missing_fa_count = df_working["fa"].isna().sum()
        raise ValueError(
            f"FA values are missing for {missing_fa_count} rows. "
            f"All FA values must be present."
        )

    if np.isinf(df_working["fa"]).any():
        inf_fa_count = df_working["fa"].isinf().sum()
        raise ValueError(
            f"FA values contain infinite values for {inf_fa_count} rows. "
            f"All FA values must be finite."
        )

    if (df_working["fa"] < 0).any():
        neg_fa_count = (df_working["fa"] < 0).sum()
        raise ValueError(
            f"FA values contain negative values for {neg_fa_count} rows. "
            f"All FA values must be non-negative."
        )

    # Validate group values
    if not df_working["group"].isin([0, 1]).all():
        invalid_groups = df_working[~df_working["group"].isin([0, 1])]["group"].unique()
        raise ValueError(
            f"Invalid group values found: {invalid_groups}. " f"Group must be 0 or 1."
        )

    # Validate gender values if present
    if "gender" in df_working.columns:
        valid_genders = ["Male", "Female"]
        invalid_genders = df_working[~df_working["gender"].isin(valid_genders)][
            "gender"
        ].unique()
        if len(invalid_genders) > 0:
            raise ValueError(
                f"Invalid gender values found: {invalid_genders}. "
                f"Gender must be 'Male' or 'Female'."
            )

    # Validate disk and streamline values
    if (df_working["disk"] <= 0).any():
        raise ValueError("Disk values must be positive integers")

    if (df_working["streamline"] < 0).any():
        raise ValueError("Streamline values must be non-negative integers")

    Y = []
    X = []

    unique_subjects = df_working["subject"].unique()
    unique_disk = df_working["disk"].unique()

    no_disk = len(unique_disk)

    for sub in unique_subjects:
        sub_df = df_working[df_working["subject"] == sub]
        unique_streamline = sub_df["streamline"].unique()
        len_streamlines = len(unique_streamline)

        # Create a mapping from actual streamline values to sequential indices
        # Handle both int and float types by converting to int for consistency
        streamline_mapping = {
            int(val): idx for idx, val in enumerate(unique_streamline)
        }

        group = sub_df["group"].unique()[0]
        gender = None
        if "gender" in sub_df.columns:
            gender = sub_df["gender"].unique()[0]
        print(
            "For subject {} I have {} unique streamlines and group is {}".format(
                sub, len_streamlines, group
            )
        )
        if group == 0:
            # continue
            sub_X = np.zeros((len_streamlines, 1))
        elif group == 1:
            sub_X = np.ones((len_streamlines, 1))
        else:
            raise ValueError(
                "For subject {} we have a invalid group which is {}".format(sub, group)
            )
        if gender is not None:
            if gender == "Female":
                zero_column = np.zeros((len_streamlines, 1))
                sub_X = np.hstack((sub_X, zero_column))
            elif gender == "Male":
                one_column = np.ones((len_streamlines, 1))
                sub_X = np.hstack((sub_X, one_column))
            else:
                # This should not happen due to validation above, but just in case
                raise ValueError(
                    f"Warning: Invalid gender value '{gender}' for subject {sub}. "
                    f"Skipping gender column."
                )
        if "age" in sub_df.columns:
            age = sub_df["age"].unique()[0]
            age_column = age * np.ones((len_streamlines, 1))
            sub_X = np.hstack((sub_X, age_column))
        if len(X) == 0:
            X = sub_X
        else:
            X = np.append(X, sub_X, axis=0)
        sub_Y = np.zeros((len_streamlines, no_disk))
        count_Y = np.zeros((len_streamlines, no_disk))

        for _index, row in sub_df.iterrows():
            streamline_val = row["streamline"]
            x = streamline_mapping[int(streamline_val)]  # Map to sequential index
            y = int(row["disk"] - 1)  # Convert to int for array indexing
            sub_Y[x][y] += row["fa"]  # FA remains as float
            count_Y[x][y] += 1
        for i in range(len_streamlines):
            for j in range(no_disk):
                if count_Y[i][j] > 0:
                    sub_Y[i][j] = sub_Y[i][j] / count_Y[i][j]
        if len(Y) == 0:
            Y = sub_Y
        else:
            Y = np.vstack((Y, sub_Y))
        print("Printing X rows {}. Printing Y rows {}".format(X.shape[0], Y.shape[0]))

    selected_indices = np.random.choice(
        X.shape[0], min(X.shape[0], no_streamlines), replace=False
    )
    X = np.array(X[selected_indices])
    Y = np.array(Y[selected_indices])
    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((X, ones_column))
    print("Printing X rows {}. Printing Y rows {}".format(X.shape[0], Y.shape[0]))
    return X, Y


def fosr(
    formula=None,
    Y=None,
    fdobj=None,
    data=None,
    X=None,
    con=None,
    argvals=None,
    method="OLS",
    gam_method="REML",
    cov_method="naive",
    labda=None,
    nbasis=15,
    norder=4,
    pen_order=2,
    multi_sp=False,
    pve=0.99,
    max_iter=1,
    maxlam=None,
    cv1=False,
    scale=False,
):
    """
    Fits a Function-on-Scalar Regression (FoSR) model using basis expansion
    (B-splines) and optionally penalized smoothing through GAM. Supports
    estimation using Ordinary Least Squares (OLS) or penalized approaches
    like REML.

    Parameters:
    ----------
    formula : str or None
        Model formula (R-style). Currently not implemented in this function.

    Y : np.ndarray or None
        Functional response matrix of shape (n_samples, n_grid_points).
        Required if `fdobj` is not provided.

    fdobj : FDataBasis or None
        Functional data object (from scikit-fda). Used as an alternative to
        raw Y input.

    data : pandas.DataFrame or None
        Dataset containing predictors (used if `formula` is implemented in
        the future).

    X : np.ndarray
        Scalar predictor matrix of shape (n_samples, n_predictors).

    con : np.ndarray or None
        Constraint matrix to be applied on coefficients, e.g., for
        identifiability constraints.

    argvals : np.ndarray or None
        Grid values over which functional response is evaluated. If None,
        defaults to linspace from 0 to 1.

    method : str, default = "OLS"
        Estimation method. Options are:
        - "OLS": Ordinary Least Squares.
        - Others (e.g., REML) not fully implemented.

    gam_method : str, default = "REML"
        Method used for GAM fitting (e.g., REML, GCV).

    cov_method : str, default = "naive"
        Method for estimating covariance. Currently a placeholder.

    labda : float, list, or None
        Smoothing parameter(s). If None, automatic selection is used (not
        fully implemented for all cases).

    nbasis : int, default = 15
        Number of B-spline basis functions used to represent functional
        data.

    norder : int, default = 4
        Order of B-spline basis functions (degree + 1).

    pen_order : int, default = 2
        Order of derivative used for roughness penalty.

    multi_sp : bool, default = False
        Whether to use multiple smoothing parameters (not supported with
        "OLS").

    pve : float, default = 0.99
        Proportion of variance explained (used in PCA step; placeholder
        for now).

    max_iter : int, default = 1
        Maximum number of iterations for penalized fitting (not
        implemented beyond 0 or 1).

    maxlam : float or None
        Maximum lambda value to consider during tuning. Currently not
        used.

    cv1 : bool, default = False
        If True, enables cross-validation for lambda selection. Currently
        a placeholder.

    scale : bool, default = False
        If True, standardizes predictors using variance only (no
        centering). Placeholder for now.

    Returns:
    -------
    dict
        A dictionary containing:
            - 'fd': FDataBasis object for fitted coefficient functions.
            - 'pca.resid': Placeholder for residual PCA (currently None).
            - 'U': Placeholder (currently None).
            - 'yhat': Fitted values (n_samples × n_grid_points).
            - 'est.func': Estimated functional coefficient matrix.
            - 'se.func': Pointwise standard error estimates for each
              coefficient function.
            - 'argvals': Grid of evaluation points used in basis
              expansion.
            - 'fit': Dictionary of model fit from `gam()` (including
              coefficients, covariance, etc.).

    Notes:
    ------
    - Only raw response matrix `Y` is currently supported (not `fdobj`).
    - Multiple smoothing parameters and cross-validation are indicated
      but not implemented.
    - Uses B-spline basis expansion and matrix operations similar to R's
      refund::fosr.
    - Penalized estimation uses Tikhonov regularization with
      differential operators.
    """
    multi_sp = False if method == "OLS" else True

    ## Handle the case when formula is NULL
    resp_type = "fd" if Y is None else "raw"

    if argvals is None:
        argvals = np.linspace(0, 1, num=Y.shape[1])
    else:
        print(
            "R equivalent code of seq(min(fdobj$basis$range), "
            "max(fdobj$basis$range), length=201)"
        )

    if method != "OLS" and len(labda) > 1:
        print("Vector-valued lambda allowed only if method = 'OLS'")
        return None
    if labda is not None and multi_sp:
        print("Fixed lambda not implemented with multiple penalties")
        return None
    if method == "OLS" and multi_sp:
        print("OLS not implemented with multiple penalties")
        return None

    if resp_type == "raw":
        bss = BSplineBasis(domain_range=(0, 1), n_basis=nbasis, order=norder)
        Bmat = bss.evaluate(argvals).reshape(nbasis, 100)
        print("Bmat shape ", Bmat.shape)
        Theta = bss.evaluate(argvals).reshape(nbasis, 100)
        respmat = Y
    elif resp_type == "fd":
        print("Work in progress for when resp_type is fd")

    new_fit = None
    U = None
    pca_resid = None

    X_sc = X
    q = X.shape[1]
    ncurve = respmat.shape[0]

    if multi_sp:
        print("Work in progress for multi_sp not None")
    else:
        # Define the differential operator for the penalty
        differential_operator = LinearDifferentialOperator(pen_order)
        regularization_parameter = 1.0
        regularization = TikhonovRegularization(differential_operator)

        # Compute the penalty matrix
        bss_derivative = compute_penalty_matrix(
            basis_iterable=[bss],
            regularization_parameter=regularization_parameter,
            regularization=regularization,
        )
        pen = np.kron(np.eye(q), bss_derivative)

    if con is not None:
        constr = np.kron(con, np.eye(nbasis))
    else:
        constr = None

    if method == "OLS":
        if (labda is None or len(labda) != 1) or cv1:
            print("Time to use lofocv for hyper parameters")

    X_gam = np.kron(X_sc, np.transpose(Bmat))
    Y_gam = respmat.ravel()
    firstfit = gam(Y_gam, X_gam, gam_method=gam_method, S=[pen], C=constr, labda=labda)

    coefmat = firstfit["coefficients"].reshape(
        q, firstfit["coefficients"].shape[0] // X.shape[1]
    )

    if method != "OLS":
        print("Work in progress for method is not OLS")
    if method == "OLS" or max_iter == 0:
        resid_vec = (
            respmat.ravel() - (np.kron(X_sc, Bmat.T) @ firstfit["coefficients"])
        ).reshape(-1, 1)
        num_rows = len(resid_vec) // ncurve
        cov_mat = ((ncurve - 1) / ncurve) * np.cov(
            np.reshape(resid_vec, (ncurve, num_rows)), rowvar=False
        )
        # Construct the block diagonal matrix
        cov_bdiag = sp.block_diag([cov_mat] * ncurve, format="csc")
        var_b = firstfit["GinvXT"] @ cov_bdiag @ firstfit["GinvXT"].T

        del cov_bdiag
        gc.collect()
    else:
        var_b = new_fit["Vp"]

    se_func = np.full((len(argvals), q), np.nan)
    for j in range(1, q + 1):
        start_idx = nbasis * (j - 1)
        end_idx = nbasis * j
        var_b_submatrix = var_b[start_idx:end_idx, start_idx:end_idx]
        product = Theta.T @ var_b_submatrix * Theta.T
        row_sums = np.sqrt(np.sum(product, axis=1))
        se_func[:, j - 1] = row_sums

    fd = FDataBasis(basis=bss, coefficients=coefmat)
    est_func = np.squeeze(fd.evaluate(argvals)).T

    if method == "mix" and max_iter > 0:
        fit = new_fit
    else:
        fit = firstfit

    if resp_type == "raw":
        yhat = X @ np.dot(coefmat, Theta)

    return {
        "fd": fd,
        "pca.resid": pca_resid,
        "U": U,
        "yhat": yhat,
        "est.func": est_func,
        "se.func": se_func,
        "argvals": argvals,
        "fit": fit,
    }
