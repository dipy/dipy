!/usr/bin/python
# Created by Christopher Nguyen
# 5/17/2010

#import modules
import time
import sys, os, traceback, optparse
import numpy as np
import scipy as sp
from copy import copy, deepcopy


def bootstrap(pdf, statistic = np.std, num_boot = 1000, percentile_ci = 0.95):
    """
    Bootstrap resampling _[1] to accurately estimate the standard error and 
    confidence interval of a desired statistic of a probability distribution 
    function (pdf).

    Parameters
    ----------
    pdf : ndarray (N, 1)
        Probability distribution function to resample. N should be reasonably
        large.
    statistic : method (optional)
        Method to calculate the desired statistic. (Default: calculate 
        standard error)
    num_boot : integer (optional)
        Total number of bootstrap resamples in bootstrap pdf. (Default: 1000)
    ci_percentile : float (optional)
        Percentile for confidence interval of the statistic. (Default: 0.95)
    
    Returns
    -------
    bs_pdf : ndarray (M, 1)
        Jackknife probabilisty distribution function of the statistic.
    se : float
        Standard error of the statistic.
    ci : ndarray (2, 1)
        Confidence interval of the statistic.

    See Also
    --------
    numpy.std, numpy.random.random

    Notes
    -----
    Bootstrap resampling is non parametric. It is quite powerful in
    determining the standard error and the confidence interval of a sample
    distribution. The key characteristics of bootstrap is:

    1) uniform weighting among all samples (1/n)
    2) resampling with replacement
    
    In general, the sample size should be large to ensure accuracy of the 
    estimates. The number of bootstrap resamples should be large as well as 
    that will also influence the accuracy of the estimate.
    
    References
    ----------
    ..  [1] Efron, B., 1979. 1977 Rietz lecture--Bootstrap methods--Another
        look at the jackknife. Ann. Stat. 7, 1-26.
    """
    pass

def jackknife(pdf, statistic = np.std, M = np.round(0.10 * len(pdf))):
    """
    Jackknife resampling _[1] to quickly estimate the bias and standard 
    error of a desired statistic in a probability distribution function (pdf).

    Parameters
    ----------
    pdf : ndarray (N, 1)
        Probability distribution function to resample. N should be reasonably
        large.
    statistic : method (optional)
        Method to calculate the desired statistic. (Default: calculate 
        standard error)
    M : integer (M < N)
        Total number of samples in jackknife pdf. (Default: 10% of N)
    
    Returns
    -------
    jk_pdf : ndarray (M, 1)
        Jackknife probabilisty distribution function of the statistic.
    bias : float
        Bias of the mean of the statistic.
    se : float
        Standard error of the statistic.

    See Also
    --------
    numpy.std, numpy.mean, numpy.random.random

    Notes
    -----
    Jackknife resampling like bootstrap resampling is non parametric. However,
    it requires a large distribution to be accurate and in some ways can be 
    considered deterministic (if one removes the same set of samples, 
    then one will get the same estimates of the bias and variance). 
    
    In the context of this implementation, the sample size should be at least 
    larger than the asymptotic convergence of the statistic (ACstat); 
    preferably, larger than ACstat + np.greater(ACbias, ACvar)
    
    References
    ----------
    ..  [1] Efron, B., 1979. 1977 Rietz lecture--Bootstrap methods--Another
        look at the jackknife. Ann. Stat. 7, 1-26.
    """ 
    N = len(pdf)
    pdf_mask = np.ones((N,),dtype='int16')
    jk_pdf = np.empty((M,))
    
    for ii in M:
        #choose a unique random sample to remove
        while pdf_mask[rand_index] == 0 :
            rand_index = np.round(np.random.random(1) * N)
        #set mask to zero for chosen random index
        pdf_mask[rand_index] = 0
        jk_pdf[ii] = statistic(pdf[pdf_mask > 0]) 
    
    return jk_pdf, np.mean(pdf) - np.mean(jk_pdf), np.std(jk_pdf)

def residual_bootstrap(data):
    pass

def repetition_bootstrap(data):
    pass

