''' Motion and eddy-current correction using affine registration.
'''
import numpy as np
import nibabel as ni
import dipy as dp

def add_padding(data,even):
    ''' some times the registration could take you a bit out of the
    field of view of the image. If that is the case zero-padding can be helpfull.
    '''
    pass

def remove_padding(data,even):
    ''' well if you added padding before the registration you should
    better remove it at the end to return to your proper size of the volume.
    '''
    pass


def eliminate_negatives(data):
    ''' some times the registration can return small negative values if
    that is the case it is better to eliminate them from the data
    '''
    data[np.where(data<0)]=0
    return data


def bvecs_correction(bvecs,mats):
    ''' after motion correction it is crucial to correct the bvecs
    '''
    pass


def motion_correction(data,affine,ref=0,similarity='cr',interp='tri',subsampling=None,search='affine',optimizer='powel',order=0):
    ''' Correct for motion and eddy current correction using the nipy
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
    T=ni.Nifti1Image(data[...,ref],affine)

    ND=np.zeros(data.shape)
    ND[...,ref]=data[...,ref]

    A_mats={ref:np.eye(4)}
    
    for s in range(data.shape[-1]):
        if s!=ref:
            print('Working on volume number =========>  %d  '% (s))
            S=ni.Nifti1Image(data[...,s],affine)            
            A=dp.volume_register(S,T,similarity,\
                       interp,subsampling,search,optimizer)
            #transform volume
            ST=dp.volume_transform(S, A.inv(), reference=T,interp_order=order)            
            ND[...,s]=ST.get_data()
            A_mats[s]=A#._get_param()

            #USE dp.rotation_vec2mat to get matrix from rotation perhaps
            #you need the inverse first
        
    return ND, A_mats
       

#dname = '/home/eg01/Data_Backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
#dname =  '/home/eg01/Data_Backup/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI'
#data,affine,bvals,gradients=dp.load_dcm_dir(dname)

img=ni.load('/tmp/testing_features/18620_0006.nii')
data=img.get_data().astype('uint16')
affine=img.get_affine()
data_corr,mats=motion_correction(data,affine,similarity='mi',order=3)

img=ni.Nifti1Image(data_corr,affine)
ni.save(img,'/tmp/saved5.nii.gz')


