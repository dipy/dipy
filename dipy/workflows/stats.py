from glob import glob
import json
import logging
import os
from time import time
import warnings

import numpy as np
from scipy.ndimage import binary_dilation

from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.io.peaks import load_peaks
from dipy.io.streamline import load_tractogram
from dipy.reconst.dti import TensorModel
from dipy.segment.bundles import bundle_shape_similarity
from dipy.segment.mask import bounding_box, segment_from_cfa
from dipy.stats.analysis import anatomical_measures, assignment_map, peak_values
from dipy.testing.decorators import warning_for_keywords
from dipy.tracking.streamline import transform_streamlines
from dipy.utils.optpkg import optional_package
from dipy.workflows.workflow import Workflow

from scipy.stats import norm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from dipy.stats.fosr import get_covariates

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
import pandas as pd

pd, have_pd, _ = optional_package("pandas")
smf, have_smf, _ = optional_package("statsmodels.formula.api")
matplt, have_matplotlib, _ = optional_package("matplotlib")
plt, have_plt, _ = optional_package("matplotlib.pyplot")


class SNRinCCFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "snrincc"

    def run(
        self,
        data_files,
        bvals_files,
        bvecs_files,
        mask_file,
        bbox_threshold=(0.6, 1, 0, 0.1, 0, 0.1),
        out_dir="",
        out_file="product.json",
        out_mask_cc="cc.nii.gz",
        out_mask_noise="mask_noise.nii.gz",
    ):
        """Compute the signal-to-noise ratio in the corpus callosum.

        Parameters
        ----------
        data_files : string
            Path to the dwi.nii.gz file. This path may contain wildcards to
            process multiple inputs at once.
        bvals_files : string
            Path of bvals.
        bvecs_files : string
            Path of bvecs.
        mask_file : string
            Path of a brain mask file.
        bbox_threshold : variable float, optional
            Threshold for bounding box, values separated with commas for ex.
            [0.6,1,0,0.1,0,0.1].
        out_dir : string, optional
            Where the resulting file will be saved.
        out_file : string, optional
            Name of the result file to be saved.
        out_mask_cc : string, optional
            Name of the CC mask volume to be saved.
        out_mask_noise : string, optional
            Name of the mask noise volume to be saved.

        """
        io_it = self.get_io_iterator()

        for (
            dwi_path,
            bvals_path,
            bvecs_path,
            mask_path,
            out_path,
            cc_mask_path,
            mask_noise_path,
        ) in io_it:
            data, affine = load_nifti(dwi_path)
            bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)
            gtab = gradient_table(bvals=bvals, bvecs=bvecs)

            mask, affine = load_nifti(mask_path)

            logging.info("Computing tensors...")
            tenmodel = TensorModel(gtab)
            tensorfit = tenmodel.fit(data, mask=mask)

            logging.info("Computing worst-case/best-case SNR using the CC...")

            if np.ndim(data) == 4:
                CC_box = np.zeros_like(data[..., 0])
            elif np.ndim(data) == 3:
                CC_box = np.zeros_like(data)
            else:
                raise OSError("DWI data has invalid dimensions")

            mins, maxs = bounding_box(mask)
            mins = np.array(mins)
            maxs = np.array(maxs)
            diff = (maxs - mins) // 4
            bounds_min = mins + diff
            bounds_max = maxs - diff

            CC_box[
                bounds_min[0] : bounds_max[0],
                bounds_min[1] : bounds_max[1],
                bounds_min[2] : bounds_max[2],
            ] = 1

            if len(bbox_threshold) != 6:
                raise OSError("bbox_threshold should have 6 float values")

            mask_cc_part, cfa = segment_from_cfa(
                tensorfit, CC_box, bbox_threshold, return_cfa=True
            )

            if not np.count_nonzero(mask_cc_part.astype(np.uint8)):
                logging.warning(
                    "Empty mask: corpus callosum not found."
                    " Update your data or your threshold"
                )

            save_nifti(cc_mask_path, mask_cc_part.astype(np.uint8), affine)
            logging.info(f"CC mask saved as {cc_mask_path}")

            masked_data = data[mask_cc_part]
            mean_signal = 0
            if masked_data.size:
                mean_signal = np.mean(masked_data, axis=0)
            mask_noise = binary_dilation(mask, iterations=10)
            mask_noise[..., : mask_noise.shape[-1] // 2] = 1
            mask_noise = ~mask_noise

            save_nifti(mask_noise_path, mask_noise.astype(np.uint8), affine)
            logging.info(f"Mask noise saved as {mask_noise_path}")

            noise_std = 0
            if np.count_nonzero(mask_noise.astype(np.uint8)):
                noise_std = np.std(data[mask_noise, :])

            logging.info(f"Noise standard deviation sigma= {noise_std}")

            idx = np.sum(gtab.bvecs, axis=-1) == 0
            gtab.bvecs[idx] = np.inf
            axis_X = np.argmin(np.sum((gtab.bvecs - np.array([1, 0, 0])) ** 2, axis=-1))
            axis_Y = np.argmin(np.sum((gtab.bvecs - np.array([0, 1, 0])) ** 2, axis=-1))
            axis_Z = np.argmin(np.sum((gtab.bvecs - np.array([0, 0, 1])) ** 2, axis=-1))

            SNR_output = []
            SNR_directions = []
            for direction in ["b0", axis_X, axis_Y, axis_Z]:
                if direction == "b0":
                    SNR = mean_signal[0] / noise_std if noise_std else 0
                    logging.info(f"SNR for the b=0 image is : {SNR}")
                else:
                    logging.info(
                        f"SNR for direction {direction} {gtab.bvecs[direction]} is: "
                        f"{SNR}"
                    )
                    SNR_directions.append(direction)
                    SNR = mean_signal[direction] / noise_std if noise_std else 0
                SNR_output.append(SNR)

            snr_str = f"{SNR_output[0]} {SNR_output[1]} {SNR_output[2]} {SNR_output[3]}"
            dir_str = f"b0 {SNR_directions[0]} {SNR_directions[1]} {SNR_directions[2]}"
            data = [{"data": snr_str, "directions": dir_str}]

            with open(os.path.join(out_dir, out_path), "w") as myfile:
                json.dump(data, myfile)


@warning_for_keywords()
def buan_bundle_profiles(
    model_bundle_folder,
    bundle_folder,
    orig_bundle_folder,
    metric_folder,
    group_id,
    subject,
    *,
    no_disks=100,
    out_dir="",
):
    """
    Applies statistical analysis on bundles and saves the results
    in a directory specified by ``out_dir``.

    See :footcite:p:`Chandio2020a` for further details about the method.

    Parameters
    ----------
    model_bundle_folder : string
        Path to the input model bundle files. This path may contain
        wildcards to process multiple inputs at once.
    bundle_folder : string
        Path to the input bundle files in common space. This path may
        contain wildcards to process multiple inputs at once.
    orig_bundle_folder : string
        Path to the input bundle files in native space. This path may
        contain wildcards to process multiple inputs at once.
    metric_folder : string
        Path to the input dti metric or/and peak files. It will be used as
        metric for statistical analysis of bundles.
    group_id : integer
        what group subject belongs to either 0 for control or 1 for patient.
    subject : string
        subject id e.g. 10001.
    no_disks : integer, optional
        Number of disks used for dividing bundle into disks.
    out_dir : string, optional
        Output directory.

    References
    ----------
    .. footbibliography::

    """

    t = time()

    mb = glob(os.path.join(model_bundle_folder, "*.trk"))
    print(mb)

    mb.sort()

    bd = glob(os.path.join(bundle_folder, "*.trk"))

    bd.sort()
    print(bd)
    org_bd = glob(os.path.join(orig_bundle_folder, "*.trk"))
    org_bd.sort()
    print(org_bd)
    n = len(mb)

    for io in range(n):
        mbundles = load_tractogram(
            mb[io], reference="same", bbox_valid_check=False
        ).streamlines
        bundles = load_tractogram(
            bd[io], reference="same", bbox_valid_check=False
        ).streamlines
        orig_bundles = load_tractogram(
            org_bd[io], reference="same", bbox_valid_check=False
        ).streamlines

        if len(orig_bundles) > 5:
            indx = assignment_map(bundles, mbundles, no_disks)
            ind = np.array(indx)

            metric_files_names_dti = glob(os.path.join(metric_folder, "*.nii.gz"))

            metric_files_names_csa = glob(os.path.join(metric_folder, "*.pam5"))

            _, affine = load_nifti(metric_files_names_dti[0])

            affine_r = np.linalg.inv(affine)
            transformed_orig_bundles = transform_streamlines(orig_bundles, affine_r)

            for mn in range(len(metric_files_names_dti)):
                ab = os.path.split(metric_files_names_dti[mn])
                metric_name = ab[1]

                fm = metric_name[:-7]
                bm = os.path.split(mb[io])[1][:-4]

                logging.info(f"bm = {bm}")

                dt = {}

                logging.info(f"metric = {metric_files_names_dti[mn]}")

                metric, _ = load_nifti(metric_files_names_dti[mn])

                anatomical_measures(
                    transformed_orig_bundles,
                    metric,
                    dt,
                    fm,
                    bm,
                    subject,
                    group_id,
                    ind,
                    out_dir,
                )

            for mn in range(len(metric_files_names_csa)):
                ab = os.path.split(metric_files_names_csa[mn])
                metric_name = ab[1]

                fm = metric_name[:-5]
                bm = os.path.split(mb[io])[1][:-4]

                logging.info(f"bm = {bm}")
                logging.info(f"metric = {metric_files_names_csa[mn]}")
                dt = {}
                metric = load_peaks(metric_files_names_csa[mn])

                peak_values(
                    transformed_orig_bundles,
                    metric,
                    dt,
                    fm,
                    bm,
                    subject,
                    group_id,
                    ind,
                    out_dir,
                )

    print("total time taken in minutes = ", (-t + time()) / 60)


class BundleAnalysisTractometryFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "ba"

    @warning_for_keywords()
    def run(self, model_bundle_folder, subject_folder, *, no_disks=100, out_dir=""):
        """Workflow of bundle analytics.

        Applies statistical analysis on bundles of subjects and saves the
        results in a directory specified by ``out_dir``.

        See :footcite:p:`Chandio2020a` for further details about the method.

        Parameters
        ----------
        model_bundle_folder : string
            Path to the input model bundle files. This path may
            contain wildcards to process multiple inputs at once.
        subject_folder : string
            Path to the input subject folder. This path may contain
            wildcards to process multiple inputs at once.
        no_disks : integer, optional
            Number of disks used for dividing bundle into disks.
        out_dir : string, optional
            Output directory.

        References
        ----------
        .. footbibliography::

        """

        if os.path.isdir(subject_folder) is False:
            raise ValueError("Invalid path to subjects")

        groups = os.listdir(subject_folder)
        groups.sort()
        for group in groups:
            if os.path.isdir(os.path.join(subject_folder, group)):
                logging.info(f"group = {group}")
                all_subjects = os.listdir(os.path.join(subject_folder, group))
                all_subjects.sort()
                logging.info(all_subjects)
            if group.lower() == "patient":
                group_id = 1  # 1 means patient
            elif group.lower() == "control":
                group_id = 0  # 0 means control
            else:
                print(group)
                raise ValueError("Invalid group. Neither patient nor control")

            for sub in all_subjects:
                logging.info(sub)
                pre = os.path.join(subject_folder, group, sub)
                logging.info(pre)
                b = os.path.join(pre, "rec_bundles")
                c = os.path.join(pre, "org_bundles")
                d = os.path.join(pre, "anatomical_measures")
                buan_bundle_profiles(
                    model_bundle_folder,
                    b,
                    c,
                    d,
                    group_id,
                    sub,
                    no_disks=no_disks,
                    out_dir=out_dir,
                )


class LinearMixedModelsFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "lmm"

    def get_metric_name(self, path):
        """Splits the path string and returns name of anatomical measure
        (eg: fa), bundle name eg(AF_L) and bundle name with metric name
        (eg: AF_L_fa)

        Parameters
        ----------
        path : string
            Path to the input metric files. This path may
            contain wildcards to process multiple inputs at once.
        """

        head_tail = os.path.split(path)
        name = head_tail[1]
        count = 0
        i = len(name) - 1
        while i > 0:
            if name[i] == ".":
                count = i
                break
            i = i - 1

        for j in range(len(name)):
            if name[j] == "_":
                if name[j + 1] != "L" and name[j + 1] != "R" and name[j + 1] != "F":
                    return name[j + 1 : count], name[:j], name[:count]

        return " ", " ", " "

    def save_lmm_plot(self, plot_file, title, bundle_name, x, y):
        """Saves LMM plot with segment/disk number on x-axis and
        -log10(pvalues) on y-axis in out_dir folder.

        Parameters
        ----------
        plot_file : string
            Path to the plot file. This path may
            contain wildcards to process multiple inputs at once.
        title : string
            Title for the plot.
        bundle_name : string
            Bundle name.
        x : list
            list containing segment/disk number for x-axis.
        y : list
            list containing -log10(pvalues) per segment/disk number for y-axis.

        """

        n = len(x)
        dotted = np.ones(n)
        dotted[:] = 2
        c1 = np.random.rand(1, 3)

        y_pos = np.arange(n)

        (l1,) = plt.plot(
            y_pos,
            dotted,
            color="red",
            marker=".",
            linestyle="solid",
            linewidth=0.6,
            markersize=0.7,
            label="p-value < 0.01",
        )

        (l2,) = plt.plot(
            y_pos,
            dotted + 1,
            color="black",
            marker=".",
            linestyle="solid",
            linewidth=0.4,
            markersize=0.4,
            label="p-value < 0.001",
        )

        first_legend = plt.legend(handles=[l1, l2], loc="upper right")

        axes = plt.gca()
        axes.add_artist(first_legend)
        axes.set_ylim([0, 6])

        l3 = plt.bar(y_pos, y, color=c1, alpha=0.5, label=bundle_name)
        plt.legend(handles=[l3], loc="upper left")
        plt.title(title.upper())
        plt.xlabel("Segment Number")
        plt.ylabel("-log10(Pvalues)")
        plt.savefig(plot_file)
        plt.clf()

    @warning_for_keywords()
    def run(self, h5_files, *, no_disks=100, out_dir=""):
        """Workflow of linear Mixed Models.

        Applies linear Mixed Models on bundles of subjects and saves the
        results in a directory specified by ``out_dir``.

        Parameters
        ----------
        h5_files : string
            Path to the input metric files. This path may
            contain wildcards to process multiple inputs at once.
        no_disks : integer, optional
            Number of disks used for dividing bundle into disks.
        out_dir : string, optional
            Output directory.

        """

        io_it = self.get_io_iterator()

        for file_path in io_it:
            logging.info(f"Applying metric {file_path}")

            file_name, bundle_name, save_name = self.get_metric_name(file_path)
            logging.info(f" file name = {file_name}")
            logging.info(f"file path = {file_path}")

            pvalues = np.zeros(no_disks)
            warnings.filterwarnings("ignore")
            # run mixed linear model for every disk
            for i in range(no_disks):
                disk_count = i + 1
                df = pd.read_hdf(file_path, where="disk=disk_count")

                logging.info(f"read the dataframe for disk number {disk_count}")
                # check if data has significant data to perform LMM
                if len(df) < 10:
                    raise ValueError("Dataset for Linear Mixed Model is too small")

                criteria = file_name + " ~ group"
                md = smf.mixedlm(criteria, df, groups=df["subject"])

                mdf = md.fit()

                pvalues[i] = mdf.pvalues[1]

            x = list(range(1, len(pvalues) + 1))
            y = -1 * np.log10(pvalues)

            save_file = os.path.join(out_dir, save_name + "_pvalues.npy")
            np.save(save_file, pvalues)

            save_file = os.path.join(out_dir, save_name + "_pvalues_log.npy")
            np.save(save_file, y)

            save_file = os.path.join(out_dir, save_name + ".png")
            self.save_lmm_plot(save_file, file_name, bundle_name, x, y)

class FOSRFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "fosr"
    
    def gam(self, y, Xmat, gam_method = None, S = None, C = None, labda = None):

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

    # def get_covariates(self, df):
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
            gender = None
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
            if(gender != None):
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

    def fosr(self, formula=None, Y=None, fdobj=None, data=None, X=None, con = None, argvals = None, method = "OLS", gam_method = "REML", cov_method = "naive", labda = None, nbasis=15, norder=4, pen_order=2, multi_sp = False, pve=.99, max_iter = 1, maxlam = None, cv1 = False, scale = False):

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
        firstfit = self.gam(Y_gam, X_gam, gam_method = gam_method, S = [pen], C = constr, labda = labda)

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
                
    @warning_for_keywords()
    def run(self, hd5_dir, *, no_disks=100, out_dir=""):
        """Workflow of Functional on Scalar Regression models.

        Applies functional on Scalar Regression models on bundles of subjects.

        Parameters
        ----------
        h5_files : string
            Path to the input metric files. This path may
            contain wildcards to process multiple inputs at once.
        no_disks : integer, optional
            Number of disks used for dividing bundle into disks.
        out_dir : string, optional
            Output directory.
        """
        h5_list = os.listdir(hd5_dir)
        std_error_list = []
        for hd_file in h5_list:
            print("Running fosr for ", hd_file)
            df = pd.read_hdf(os.path.join(hd5_dir,hd_file))
            X,Y = get_covariates(df)
    
            print("Getting the shape of X", X.shape)
            print("Getting the shape of Y", Y.shape)

            fosr_output = self.fosr(Y = Y, X = X) 

            beta_1 = fosr_output["est.func"][:,0]
            std_error = fosr_output["se.func"][:,0]
            std_error_list.append(np.array(std_error))

            beta_1_lower = beta_1 - 1.96*std_error
            beta_1_upper = beta_1 + 1.96*std_error

            z_scores = (beta_1/std_error)

            p_values = norm.sf(abs(z_scores)) * 2


class BundleShapeAnalysis(Workflow):
    @classmethod
    def get_short_name(cls):
        return "BS"

    @warning_for_keywords()
    def run(self, subject_folder, *, clust_thr=(5, 3, 1.5), threshold=6, out_dir=""):
        """Workflow of bundle analytics.

        Applies bundle shape similarity analysis on bundles of subjects and
        saves the results in a directory specified by ``out_dir``.

        See :footcite:p:`Chandio2020a` for further details about the method.

        Parameters
        ----------
        subject_folder : string
            Path to the input subject folder. This path may contain
            wildcards to process multiple inputs at once.
        clust_thr : variable float, optional
            list of bundle clustering thresholds used in QuickBundlesX.
        threshold : float, optional
            Bundle shape similarity threshold.
        out_dir : string, optional
            Output directory.

        References
        ----------
        .. footbibliography::

        """
        rng = np.random.default_rng()
        all_subjects = []
        if os.path.isdir(subject_folder):
            groups = os.listdir(subject_folder)
            groups.sort()
        else:
            raise ValueError("Not a directory")

        for group in groups:
            if os.path.isdir(os.path.join(subject_folder, group)):
                subjects = os.listdir(os.path.join(subject_folder, group))
                subjects.sort()
                logging.info(
                    "first "
                    + str(len(subjects))
                    + " subjects in matrix belong to "
                    + group
                    + " group"
                )

                for sub in subjects:
                    dpath = os.path.join(subject_folder, group, sub)
                    if os.path.isdir(dpath):
                        all_subjects.append(dpath)

        N = len(all_subjects)

        bundles = os.listdir(os.path.join(all_subjects[0], "rec_bundles"))
        for bun in bundles:
            # bundle shape similarity matrix
            ba_matrix = np.zeros((N, N))
            i = 0
            logging.info(bun)
            for sub in all_subjects:
                j = 0

                bundle1 = load_tractogram(
                    os.path.join(sub, "rec_bundles", bun),
                    reference="same",
                    bbox_valid_check=False,
                ).streamlines

                for subi in all_subjects:
                    logging.info(subi)

                    bundle2 = load_tractogram(
                        os.path.join(subi, "rec_bundles", bun),
                        reference="same",
                        bbox_valid_check=False,
                    ).streamlines

                    ba_value = bundle_shape_similarity(
                        bundle1, bundle2, rng, clust_thr=clust_thr, threshold=threshold
                    )

                    ba_matrix[i][j] = ba_value

                    j += 1
                i += 1
            logging.info("saving BA score matrix")
            np.save(os.path.join(out_dir, bun[:-4] + ".npy"), ba_matrix)

            cmap = matplt.colormaps["Blues"]
            plt.title(bun[:-4])
            plt.imshow(ba_matrix, cmap=cmap)
            plt.colorbar()
            plt.clim(0, 1)
            plt.savefig(os.path.join(out_dir, f"SM_{bun[:-4]}"))
            plt.clf()
