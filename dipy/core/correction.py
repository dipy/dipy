''' Motion and eddy-current correction using affine registration.
'''
import numpy as np
import nibabel as ni
import dipy as dp

def add_padding(data,pad=10,value=0):
    ''' some times the registration could take you a bit out of the
    field of view of the image. If that is the case zero-padding can be
    helpfull.

    Parameters
    ----------
    data: array, shape (X,Y,Z,...)
    pad: int, number extra margin voxels on each side
    value: scalar, value to padd    

    Example
    -------
    >>> data=np.ones((30,30,30))
    >>> ndata=add_padding(data,10)
    >>> ndata.shape
    (50,50,50)
    >>> data=rm_padding(ndata,10)
    >>> data.shape
    (30,30,30)
    
    '''
    new_data=value*np.ones((data.shape[0]+2*pad,data.shape[1]+2*pad,data.shape[2]+2*pad),dtype=data.dtype)
    new_data[pad:pad+data.shape[0],pad:pad+data.shape[1],pad:pad+data.shape[2]]=data
    return new_data
        

def rm_padding(data,pad=10):
    ''' well if you added padding before the registration you should
    better remove it at the end to return to the proper size of the
    volume.

    Example
    -------
    >>> data=np.ones((30,30,30))
    >>> ndata=add_padding(data,10)
    >>> ndata.shape
    (50,50,50)
    >>> data=rm_padding(ndata,10)
    >>> data.shape
    (30,30,30)

    See also
    --------
    add_padding
    
    '''
    return data[pad:data.shape[0]-pad,pad:data.shape[1]-pad,pad:data.shape[2]-pad]


def eliminate_negatives(data):
    ''' some times the registration can return small negative values if
    that is the case it is better to eliminate them from the data
    '''
    data[np.where(data<0)]=0
    return data


def bvecs_correction(bvecs,mats):
    ''' after motion correction it might be necessary to correct the
    bvecs.

    Parameters
    ----------

    bvecs: array,shape (N,3)
    mats : dictionary, len(N), containing rotation information. This
    function is returned by motion_correction

    Returns
    ------

    nbvecs: array, shape (N,3), new corrected bvecs
    
    Notes
    -----
    
    Only the rotation component of the affine transformation is
    necessary.

    See also
    --------
    motion_correction
    
    
    '''
    nbvecs=bvecs

    for m in mats:

        try:
            a=mats[m].shape
        except:
            mi=mats[m].inv()
            R=dp._rotation_vec2mat(mi.vec12[3:6])
            b=bvecs[m]
            b.shape=(3,1)
            #print b.shape, R.shape, b, R
            #print R
            nb=np.dot(R,b)
            nbvecs[m]=nb.ravel()

    
    return nbvecs
    

def slight_rotation(image):
    ''' When you try to correct the diffusion weighted volumes you might
    get stuck on the initial transform due to a known issue known as
    "interpolation artifacts" in the registration literature. A
    classical fix is to apply a slight initial rotation to avoid
    registration starting with perfectly aligned image grids.
    '''
    #translation [0:3], rotation [3:6], scaling [6:9], shearing [9:12]    
    A=np.array([0,0,0,.11,.12,.13,0,0,0,0,0,0])
    A=dp._affine(A)
    return dp.volume_transform(image, A, reference=image,interp_order=0), A

def preprocess_volumes(data,options='same'):
    ''' changing the values of your data could possibly change the
    result of the registration.

    '''

    if options=='same':
        return data

    if options=='binary':        
        data[data>30]=255
        data[data<=30]=0

    return data

    

def motion_correction(data,affine,ref=0,similarity='cr',interp='tri',subsampling=None,search='affine',optimizer='powel',order=0):
    ''' Correct for motion and eddy currents using the nipy affine
    registration tools.    
    
    Parameters
    ----------
    data: array, shape (x,y,z,w), where w is the volume number, volume
    to be registered to target

    affine: array, shape (4,4)

    ref: reference number for the target in data i.e. which w is the
    volume number for the reference

    similarity: 'cr','cc','crl1', 'mi', je', 'ce', 'nmi', 'smi'.

    interp: 'tri' or 'pv'.

    subsampling: None is identical with [4,4,4] i.e. if you had an image
    of size 256x256x256 the it would be subsampled to shape of [64,64,64]

    search: 'affine' 12, 'rigid' 6, 'similarity' etc.

    optimizer: 'cg', 'powel' etc.

    order: 0 (nearest),1,2 or 3, order of interpolation for the final
    transformation. The previous parameter called interp is used for
    the registration part not for the final transformation.

    Returns
    -------
    data_corr: array, shape (x,y,z,w), motion corrected data

    affine_mats: sequence, with all affine transformation matrices applied
        
    '''

    #preprocess volumes
    data=preprocess_volumes(data,'binary')

    #target image
    T=ni.Nifti1Image(data[...,ref],affine)

    #copy initial reference image
    init_T=T
    
    #apply an small rotation to debloke 
    T,SR=slight_rotation(T)
    
    #ni.save(T,'/tmp/18620_0006_slight_volume0.nii.gz')

    #corrected final image
    ND=np.zeros(data.shape)

    #ND[...,ref]=T.get_data()    
    #ND[...,ref]=data[...,ref]
    ND[...,ref]=dp.volume_transform(T, SR.inv(), reference=init_T,interp_order=order).get_data()
    

    #dictionary for the applied transformations
    A_mats={ref:np.eye(4)} #change that to affine
    
    for s in range(data.shape[-1]):
        if s!=ref:
            print('Working on volume number =========>  %d  '% (s))

            S=ni.Nifti1Image(data[...,s],affine)

            #register S to T
            #volume_register is doing the interpolation on the target
            #space and in the joint histogram after every tranform 
            A=dp.volume_register(S,T,similarity,\
                       interp,subsampling,search,optimizer)
            
            #transform volume 
            ST=dp.volume_transform(S, SR.inv().__mul__(A.inv()), reference=init_T,interp_order=order)            
            ND[...,s]=ST.get_data()
            #A_mats[s]=A#._get_param()
            A_mats[s]=SR.inv().__mul__(A.inv())

            #USE dp.rotation_vec2mat to get matrix from rotation perhaps
            #you probably need the inverse first
        
    return ND, A_mats


#dname = '/home/eg01/Data_Backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
#dname =  '/home/eg01/Data_Backup/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI'
#data,affine,bvals,gradients=dp.load_dcm_dir(dname)

'''

img=ni.load('/tmp/testing_features/18620_0006.nii')
data=img.get_data().astype('uint16')
affine=img.get_affine()
data_corr,mats=motion_correction(data,affine,similarity='cr',order=3)

img=ni.Nifti1Image(data_corr,affine)
ni.save(img,'/tmp/18620_0006_dipy_corr_binary.nii.gz')

bvecs=np.loadtxt('/tmp/testing_features/18620_0006.bvecs').T
nbvecs=bvecs_correction(bvecs,mats)

'''

