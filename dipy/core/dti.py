#!/usr/bin/python
# Created by Christopher Nguyen
# 5/17/2010

#import modules
import time
import sys, os, traceback, optparse
import numpy as np
import scipy as sp

#dipy modules
from dipy.core.maskedview import MaskedView

class tensor(object):
    """
    Tensor object that when initialized calculates single self diffusion 
    tensor[1]_ in each voxel using selected fitting algorithm 
    (DEFAULT: weighted least squares[1]_)

    Requires a given gradient table, b value for each diffusion-weighted 
    gradient vector, and image data given all as numpy ndarrays.

    Parameters
    ----------
    data : ndarray (V,g)
        The image data needs at least 2 dimensions where the first dimension
        holds the set of voxels that WLS_fit will perform on and second 
        dimension holds the diffusion weighted signals.
    gtab : ndarray (3,g)
        Diffusion gradient table found in DICOM header as a numpy ndarray.
    bval : ndarray (g,1)
        Diffusion weighting factor b for each vector in gtab.
    mask : ndarray (0<V,), optional
        Mask of data that WLS_fit will NOT perform on. If mask is not boolean,
        then WLS_fit will operate where mask > 0
    thresh : data.dtype (0<=data.max()), optional
        Simple threshold to exclude voxels from WLS_fit. Default value for 
        threshold is 0.

    Attributes
    ----------
    evals : ndarray (V,3) 
        Cached eigenvalues of self diffusion tensor for given index. 
        (eval1,eval2,eval3)
    evecs : ndarray (V,3,3)
        Cached associated eigenvectors of self diffusion tensor for given 
        index. (evec1,evec2,evec3)

    Methods
    -------
    ADC : ndarray (V,1)
        Calculates the apparent diffusion coefficient [2]_. 
    FA : ndarray (V,1)
        Calculates fractional anisotropy [2]_.
    MD : ndarray (V,1)
        Calculates the mean diffusitivity [2]_. 
        Note: [units ADC] ~ [units b value]*10**-1
    
    Examples
    --------
    NEED TO WORK ON THIS SECTION
    >>> voxel
    >>> tensor = dti.tensor(voxel, gtab, bvals)
    >>> tensor.

    See Also
    --------
    dipy.io.bvectxt.read_bvec_file, WLS_tensor, design_matrix, 
    dipy.core.qball.ODF
    
    Notes
    -----
    Due to the fact that diffusion MRI entails large volumes (e.g. [256,256,
    50,64]), memory can be an issue. Therefore, only the following parameters 
    of the self diffusion tensor are cached for each voxel:

        -All three eigenvalues
        -Primary and secondary eigenvectors

    From these cached parameters, one can presumably construct any desired
    parameter.

    References
    ----------
    ..  [1] Basser, P.J., Mattiello, J., LeBihan, D., 1994. Estimation of 
        the effective self-diffusion tensor from the NMR spin echo. J Magn 
        Reson B 103, 247-254.
    ..  [2] Basser, P., Pierpaoli, C., 1996. Microstructural and physiological
        features of tissues elucidated by quantitative diffusion-tensor MRI. 
        Journal of Magnetic Resonance 111, 209-219.
    
    """
    ### Shape Property ###
    def _getshape(self):
        return self._D().shape[:-2]
    
    shape = property(_getshape, doc="Shape of tensor array")

    ### Ndim Property ###
    def _getndim(self):
        return self._D().ndim - 2
    
    ndim = property(_getndim, doc="Number of dimensions in tensor array")

    ### Getitem Property ###    
    def __getitem__(self,index):
        if type(index) is not tuple:
            index = (index,)
        if len(index) > self.ndim:
            raise IndexError('invalid index')
        for ii in index:
            if ii is Ellipsis:
                index = index + (slice(None),)
                break
        new_tensor = copy(self)
        new_tensor.evals = self.evals[index]
        new_tensor.evecs = self.evecs[index]
        return new_tensor
    
    ### Eigenvalues Property ###
    def _getevals(self):
        return self._evals
    
    def _setevals(self,evals):
        self._evals = evals

    evals = property(_getevals, _setevals, 
                                doc="Eigenvalues of self diffusion tensor")

    ### Eigenvectors Property ###
    def _getevecs(self):
        evs = np.empty((self.evals.shape[0], 3, 3))
        #Calculate 3rd eigenvector from cached eigenvectors
        for ii,p_s_evecs in enumerate(self._evecs): 
            evs[ii,0:2] = p_s_evecs
            evs[ii,2] = np.cross(p_s_evecs[0,:], p_s_evecs[1,:]) #time 26.9 us
        return evs

    def _setevecs(self,evs):
        self._evecs = evs[...,0:2,:]
    
    evecs = property(_getevecs, _setevecs, 
                                doc="Eigenvectors of self diffusion tensor")

    def __init__(self, data, grad_table, b_values, mask = True, thresh = 0,
                 verbose = False):
        
        dims = data.shape
        #data[mask].shape == data.flat[:].shape
        #data = MaskedView(mask,data[mask],fill_value=0) 
        
        #64 bit design matrix makes for faster pinv
        B = design_matrix(grad_table, b_values)

        #Define total mask from thresh and mask
        tot_mask = (mask > 0) & (data[...,0] > thresh)

        #Perform WLS fit on masked data
        self.evals, self.evecs = WLS_fit_tensor(B, data[tot_mask])


    def _D(self):
        D = np.empty((self.evals.shape[0],3,3))
        for ii,eval in enumerate(self.evals): 
            L = np.diag(eval)
            Q = self.evecs[ii,:,:]
            D[ii,:,:] = np.dot(np.dot(Q,L),Q.T) #timeit = 11.5us
        return D

    def ADC(self):
        """
        Apparent diffusion coefficient (ADC) calculated from diagonal elements
        of calculated self diffusion tensor. 
        
        Returns
        -------
        ADC : ndarray (V,1)
            Calculated ADC.

        Notes
        -----
        ADC is calculated with the following equation:

        .. math:: ADC = \frac{D_xx+D_yy+D_zz}{3}

        Need to generate the self diffusion tensor
        """

    def FA(self):
        """
        Fractional anisotropy (FA) calculated from cached eigenvalues. 
        
        Returns
        -------
        FA : ndarray (V,1)
            Calculated FA. Note: range is 0 <= FA <= 1.

        Notes
        -----
        FA is calculated with the following equation:

        .. math::

        FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                    \lambda_3)^2+(\lambda_2-lambda_3)^2}{\lambda_1^2+
                    \lambda_2^2+\lambda_3^2} }
        """
        adc = self.calc_adc()
        ev1 = self.evals[:,0]
        ev2 = self.evals[:,1]
        ev3 = self.evals[:,2]
        ss_ev = ev1**2+ev2**2+ev3**2
        
        fa = np.zeros(ev1.shape)#,dtype='float32') #'int16')
        fa = np.sqrt(1.5 * ((ev1 - adc)**2 + (ev2 - adc)**2 + (ev3 - adc)**2)
                      / ss_ev)
        fa[ss_ev == 0] = 0
        return fa 

    def MD(self):
        """
        Mean diffusitivity (MD) calculated from cached eigenvalues. 
        
        Returns
        -------
        MD : ndarray (V,1)
            Calculated MD.

        Notes
        -----
        MD is calculated with the following equation:

        .. math:: ADC = \frac{\lambda_1+\lambda_2+\lambda_3}{3}
        """
        #adc = (ev1+ev2+ev3)/3
        return (self.evals[:,0] + self.evals[:,1] + 
                self.evals[:,2]) / 3


def WLS_fit_tensor(design_matrix, data):
    """
    Computes weighted least squares (WLS) fit to calculate self-diffusion 
    tensor using a linear regression model. (Basser et al., 1994a)
    
    Parameters
    ----------
    design_matrix : ndarray (g,g)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : ndarray or MaskedView (X,Y,Z,...,g)
        Data or response variables holding the data. Note that the last 
        dimension should contain the data. It makes no copies of data.

    Returns
    -------
    eigvals : ndarray (V,3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : ndarray (V,3,3)
        Associated eigenvectors from eigen decomposition of the tensor.

    See Also
    --------
    decompose_tensor
    """

    data_flat = data.reshape((-1, data.shape[-1]))
    U,S,V = np.linalg.svd(design_matrix, False)
    #math: SI = B*inv(B.T*B)*B.T
    SI = np.dot(U, U.T)

    eigenvals = np.empty((len(data_flat), 3))
    eigenvecs = np.empty((len(data_flat), 3, 3))

    #wi=np.exp(w[0])
    for ii, sig in enumerate(data):
        log_s = np.log(sig)
        w=np.exp(np.dot(SI, log_s))
        D = np.dot(np.linalg.pinv(design_matrix*w[:,None]), w*log_s)
        eigenvals[ii], eigenvecs[ii,:,:] = decompose_tensor(D)
    
    eigenvals.shape = data.shape[:-1]+(3,)
    eigenvecs.shape = data.shape[:-1]+(3,3)
    return eigenvals, eigenvecs
    
def decompose_tensor(D):
    """
    Computes tensor eigen decomposition to calculate eigenvalues and 
    eigenvectors of self-diffusion tensor. (Basser et al., 1994a)

    Parameters
    ----------
    D : ndarray (7,)
        Array holding the six unique diffusitivities and log(S_o) 
        (Dxx,Dyy,Dzz,Dxy,Dxz,Dyz,log(S_o)). Assumes D has units 
        on order of ~ 10^-4 mm^2/s

    Results
    -------
    eigvals : ndarray (3,)
        Eigenvalues from eigen decomposition of the tensor. Negative
        eigenvalues are forced to zero.
    eigvecs : ndarray (3,3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with 
        eigvals[j])

    See Also
    --------
    numpy.linalg.eig
    """

    tensor = np.empty((3,3),dtype=D.dtype)
    tensor[0,0] = D[0]  #Dxx
    tensor[1,1] = D[1]  #Dyy
    tensor[2,2] = D[2]  #Dzz
    tensor[1,0] = tensor[0,1] = D[3]  #Dxy
    tensor[2,0] = tensor[0,2] = D[4]  #Dxz
    tensor[2,1] = tensor[1,2] = D[5]  #Dyz

    #outputs multiplicity as well so need to unique
    eigenvals, eigenvecs = np.linalg.eig(tensor)

    #need to sort the eigenvalues and associated eigenvectors
    order = eigenvals.argsort()[::-1]
    eigenvecs = eigenvecs[:,order]
    eigenvals = eigenvals[order]

    #Forcing negative eigenvalues to 0
    eigenvals = np.maximum(eigenvals, 0)
    # b ~ 10^3 s/mm^2 and D ~ 10^-4 mm^2/s
    # eigenvecs: each vector is columnar
    
    return eigenvals, eigenvecs


def design_matrix(gtab, bval, dtype=None):
    """
    Constructs design matrix for DTI weighted least squares or least squares 
    fitting. (Basser et al., 1994a)

    Parameters
    ----------
    gtab : ndarray with shape (3, g)
        Diffusion gradient table found in DICOM header as a numpy ndarray.
    bval : ndarray with shape (g,)
        Diffusion weighting factor b for each vector in gtab.
    dtype : string
        Parameter to control the dtype of returned designed matrix

	Return
	------
	design_matrix : ndarray (g,7)
		Design matrix or B matrix assuming Gaussian distributed tensor model.
		Note: design_matrix[j,:] = (Bxx,Byy,Bzz,Bxy,Bxz,Byz,dummy)

    """
    
    B = np.zeros((bval.size,7), dtype=dtype)
    G = gtab
    if gtab.shape[1] != bval.shape[0]:
        raise ValueError('The number of b values and gradient directions must'+
                         ' be the same')

    B[:,0] = G[0,:]*G[0,:]*1.*bval   #Bxx
    B[:,1] = G[1,:]*G[1,:]*1.*bval   #Byy
    B[:,2] = G[2,:]*G[2,:]*1.*bval   #Bzz
    B[:,3] = G[0,:]*G[1,:]*2.*bval   #Bxy
    B[:,4] = G[0,:]*G[2,:]*2.*bval   #Bxz
    B[:,5] = G[1,:]*G[2,:]*2.*bval   #Byz
    B[:,6] = np.ones(np.size(bval))
    
    #Need to return [g by 7]
    return -B


#######################################################################################
#The following are/will be DEPRECATED
#######################################################################################


def __WLS_fit (data,gtab,bval,verbose=False):    
    """
    Computes weighted least squares (WLS) fit to calculate self-diffusion tensor. 
    (Basser et al., 1994a)

    NB: This function is the deprecated version of WLS_fit ... it works fine but
    is not as robust as WLS_fit.

    Parameters
    ----------
    data : ndarray (X,Y,Z,g) OR Maskedview (X,Y,Z)
        The image data will be masked if ndarray is given with threshold = 25.
    gtab : ndarray (3,g)
        Diffusion gradient table found in DICOM header as a numpy ndarray.
    bval : ndarray (g,1)
        Diffusion weighting factor b for each vector in gtab.
    verbose : boolean
        Boolean to indicate verbose output such as timing.

    Returns
    -------
    eig_decomp : ndarray (X,Y,Z,12)
        Eigenvalues and eigenvectors from eigen decomposition of the tensor
    design_mat : ndarray (g,7)
        DTI design matrix to reconstruct fitted data if desired

    """
    #timer for verbose
    start_time = time.time()
    
    #hold original shape
    dims = data.shape
        
    ### Prepare data for analysis    
    if isinstance(data,np.ndarray):#len(dims) == 4:
        #Create conservative mask; also makes sure log behaves
        mask = data[:,:,:,0] > 25

        #turn data into maskedview
        data = MaskedView(mask,data[mask],fill_value=0)
    elif not isinstance(data,MaskedView):
        raise ValueError('input must be of type ndarray or MaskedView')
    
    #reshape data to be (X*Y*Z,1)
    data.shape = (-1,1)

    ###Create log of signal and reshape it to be x:y:z by grad
    data = np.maximum(data,1)
    log_s = np.log(data) #type(log_s) == MaskedView
    
    ###Construct design matrix
    #For DTI this is the so called B matrix
    # X matrix from Chris' paper
    B = design_matrix(gtab,bval) # [g by 7]
	
    ###Weighted Least Squares (WLS) to solve "linear" regression
    # Y hat OLS from Chris' paper
    #  ( [x*y*z by g] [g by 7] [7 by g ] ) = [x*y*z by g]
    log_s_ols = np.dot(log_s, np.dot(B, np.linalg.pinv(B))) #type(log_s_ols) = ndarray
    del log_s #freeing up memory

    #Setting these arrays later to allow the previous step to have all memory
    eig_decomp = np.zeros((log_s_ols.shape[0],12),dtype='float32')#'int16')
    
    time_diff = list((0,0))
    time_iter = time.time()
    # This step is because we cannot vectorize diagonal vector and tensor fit
    for ii,data_ii in enumerate(data): 
        #Split up weighting vector into little w to perform pinv
        w = np.exp(log_s_ols[ii,:])[:,np.newaxis]

        #pointwise broadcasting to avoid diagonal matrix multiply!
        D = np.dot(np.linalg.pinv(B*w), w.ravel()*np.log(data_ii)) #log_s[i,:]
        
        ###Obtain eigenvalues and eigenvectors
        eig_decomp[ii,:] = decompose_tensor(D[0:6])
        
        #Check every check_percent%
        ch_percent=0.05
        if verbose and ii % np.round(ch_percent*log_s_ols.shape[0]) == 0:
            percent = 100.*ii/log_s_ols.shape[0]
            time_diff.append(time.time()-time_iter)
            min = np.mean(time_diff[2:len(time_diff)])/60.0/ch_percent
            sec = np.round((min - np.fix(min)) * 60.0)
            min = np.fix(min)
            print str(np.round(percent)) + '% ... time left: ' + str(min) + ' MIN ' \
                + str(sec) + ' SEC ... memory: ' + str(np.round(memory()/1024.)) + 'MB'
            time_iter=time.time()

    #clear variables not needed to save memory
    del log_s_ols

    # Reshape the output
    data.shape = dims
    eig_decomp = MaskedView(data.mask,eig_decomp,fill_value=0)
    
    #Report how long it took to make the fit  
    if verbose:
        min = (time.time() - start_time) / 60.0
        sec = (min - np.fix(min)) * 60.0
        print 'TOTAL TIME: ' + str(np.fix(min)) + ' MIN ' + str(np.round(sec)) + ' SEC'

    return(eig_decomp.filled(), B)

def __save_scalar_maps(scalar_maps, img=None, coordmap=None):
    """
    Deprecated version of writing scalar maps out to disk. Do not use.
    """

    #for io of writing and reading nifti images
    from nipy import load_image, save_image
    from nipy.core.api import fromarray #data --> image
    
    #For writing out with save_image with appropriate affine matrix
    if img != None:
        coordmap = get_coord_4D_to_3D(img.affine)
        header = img.header.copy()

    ###Save scalar maps if requested
    print ''
    print 'Saving t2di map ... '+out_root+'_t2di.nii.gz'
        
    #fyi the dtype flag for save image does not appear to work right now...
    t2di_img = fromarray(data[:,:,:,0],'ijk','xyz',coordmap=coordmap)
    if img != []: 
        t2di_img.header = header
    save_image(t2di_img,out_root+'_t2di.nii.gz',dtype=np.int16)

        
    scalar_fnames = ('ev1','ev2','ev3','adc','fa','ev1p','ev1f','ev1s')
    for i in range(np.size(scalar_maps,axis=3)):
        #need to send in 4 x 4 affine matrix for 3D image not 5 x 5 from original 4D image
        print 'Saving '+ scalar_fnames[i] + ' map ... '+out_root+'_'+scalar_fnames[i]+'.nii.gz'
        scalar_img = fromarray(np.int16(scalar_maps[:,:,:,i]),'ijk' ,'xyz',coordmap=coordmap)
        if img != []:
            scalar_img.header = header
        save_image(scalar_img,out_root+'_'+scalar_fnames[i]+'.nii.gz',dtype=np.int16)

    print ''
    print 'Saving D = [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz] map ... '+out_root+'_self_diffusion.nii.gz'
    #Saving 4D matrix holding diffusion coefficients
    if img != [] :
        coordmap = img.coordmap
        header = img.header.copy()
    tensor_img = fromarray(tensor_data,'ijkl','xyzt',coordmap=coordmap)
    tensor_img.header = header
    save_image(tensor_img,out_root+'_self_diffusion.nii.gz',dtype=np.int16)

    print

    return
