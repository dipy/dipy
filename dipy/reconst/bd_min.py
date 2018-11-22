import numpy as np


def TrainModel(data, varargin):

    # simulation parameters
    N = 1   # number of coils (noisetype)
    nt = 1 * 10 ^ 4  # number of training samples
    ncross = [1, 2, 30]  # number of crossing fibers

    # model parameters
    lmax = 2  # SH expansion degree
    reg = 0 * 10 ^ -10  # Tikhonov regularizer for Bayes model
    includeb0 = True  # sensitivity to b0
    # bayesmodel = 'poly'  # bayes model type (fourier or poly)
    order = 3  # complexity of the model
    # qspace = 'qball'  # radial sampling ('multishell' or 'qball')
    # qball parameters, r_k(b) = b^k exp(-D0 b) with 0<=k<=nmax
    nmax = 3
    D0 = 1

    # setting the general parameters
    verbose = True  # show correlations during learing
    force_retraining = 0  # train even if saved model is found
    # type of noise estimation ('estlocal','estglobal',SNRmap,nz)
    noise = 'estlocal'

    for k in range(len(varargin)):
        for k in varargin:
            eval(varargin)

    ten = data.tensor / 1000
    b = np.squeeze(ten)
