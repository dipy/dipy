import numpy as np
import dipy as dp
import nibabel as ni
import resources
import time

#Registration options
#similarity 'cc', 'cr', 'crl1', 'mi', je', 'ce', 'nmi', 'smi'.  'cr'
similarity='mi'
#interp 'pv', 'tri'
interp =  'tri'
#subsampling None or sequence (3,)
subsampling=None
#search 'affine', 'rigid', 'similarity' or ['rigid','affine']
search='rigid'
#optimizer 'simplex', 'powell', 'steepest', 'cg', 'bfgs' or
#sequence of optimizers
optimizer= 'powell'


def eddy_current_correction(data,affine,target=None,target_affine=None):
    result=[]
    
    no_dirs=data.shape[-1]

    if target==None and target_affine==None:
        target=ni.Nifti1Image(data[:,:,:,0],affine)
    else:
        target=ni.Nifti1Image(target,target_affine)
        
    for i in range(1,no_dirs):        
        
        source=ni.Nifti1Image(data[:,:,:,i],affine)        
        T=dp.volume_register(source,target,similarity,\
                                 interp,subsampling,search,optimizer)
        sourceT=dp.volume_transform(source, T.inv(), reference=target)
        print i, sourceT.get_data().shape, sourceT.get_affine().shape
        result.append(sourceT)

    result.insert(0,target)
    print 'no of images',len(result)
    return ni.concat_images(result)

def register_source_2_target(source_data,source_affine,target_data,target_affine):

    #subsampling=target_data.shape[:3]

    target=ni.Nifti1Image(target_data,target_affine)
    source=ni.Nifti1Image(source_data,source_affine)
    T=dp.volume_register(source,target,similarity,\
                              interp,subsampling,search,optimizer)
    sourceT=dp.volume_transform(source, T.inv(), reference=target)

    return sourceT

def save_volumes_as_mosaic(fname,volume_list):

    import Image
    
    vols=[]
    for vol in volume_list:
            
        vol=np.rollaxis(vol,2,1)
        sh=vol.shape
        arr=vol.reshape(sh[0],sh[1]*sh[2])
        arr=np.interp(arr,[arr.min(),arr.max()],[0,255])
        arr=arr.astype('ubyte')

        print 'arr.shape',arr.shape
        
        vols.append(arr)

    mosaic=np.concatenate(vols)
    Image.fromarray(mosaic).save(fname)


if __name__ == '__main__':

    #compare FA of grid versus shell acquisitions using STEAM

    dname_grid=resources.get_paths('DSI STEAM 101 Trio')[2]
    dname_shell=resources.get_paths('DTI STEAM 114 Trio')[2]
    fname_T1=resources.get_paths('MPRAGE nifti Trio')[2]

    data_gr,affine_gr,bvals_gr,gradients_gr=dp.load_dcm_dir(dname_grid)
    data_sh,affine_sh,bvals_sh,gradients_sh=dp.load_dcm_dir(dname_shell)
    imT1=ni.load(fname_T1)
    data_T1=imT1.get_data()
    data_T1=data_T1.astype('uint16')
    affine_T1=imT1.get_affine()
    
    #data_T1,affine_T1,bvals_NaN,gradients_NaN=dp.load_dcm_dir(dname_T1)      
    #assuming that these are already eddy_current corrected 

    s0_gr_T1=register_source_2_target(data_gr[...,0],affine_gr,data_T1,affine_T1) 
    s0_sh_T1=register_source_2_target(data_sh[...,0],affine_sh,data_T1,affine_T1)

    data_s0_gr=s0_gr_T1.get_data()
    data_s0_sh=s0_sh_T1.get_data()
    
    save_volumes_as_mosaic('/tmp/mosaic_rigid2.png',[data_T1,data_s0_gr,data_s0_sh])
    
    '''
    #dicom directories
    dname_grid_101=resources.get_paths('DSI STEAM 101 Trio')[2]
    dname_shell_114=resources.get_paths('DTI STEAM 114 Trio')[2]

    
    #dname_shell_114=resources.get_paths('DTI STEAM 96 Trio')[2] 

    data_101,affine_101,bvals_101,gradients_101=dp.load_dcm_dir(dname_grid_101)    
    data_114,affine_114,bvals_114,gradients_114=dp.load_dcm_dir(dname_shell_114)

    print data_101.shape,data_114.shape    
    
    img_101T=register_source_2_target(data_101[...,0],affine_101,data_114[...,0],affine_114)    
    target_101T=img_101T.get_data()
    affine_101T=img_101T.get_affine()

    save_volumes_as_mosaic('/tmp/mosaic_101.png',[data_101[...,0]])
    save_volumes_as_mosaic('/tmp/mosaic_101T.png',[target_101T])

    img_114T=register_source_2_target(data_114[...,0],affine_114,data_101[...,0],affine_101)    
    target_114T=img_114T.get_data()
    affine_114T=img_114T.get_affine()

    save_volumes_as_mosaic('/tmp/mosaic_114.png',[data_114[...,0]])
    save_volumes_as_mosaic('/tmp/mosaic_114T.png',[target_114T])
    
    img_101Tall=eddy_current_correction(data_101,affine_101,target=target_101T,target_affine=affine_101T)

    img_101Tall_data=img_101Tall.get_data()
    img_101Tall_data[img_101Tall_data<0]=255
    
    ten_101T=dp.Tensor( img_101Tall_data,bvals_101,gradients_101,thresh=50)
    ten_114=dp.Tensor(data_114,bvals_114,gradients_114,thresh=50)
        
    save_volumes_as_mosaic('/tmp/mosaic_mi.png',[data_114[...,0],img_101T.get_data(),ten_114.FA,ten_101T.FA])
    
    #save_volumes_as_mosaic('/tmp/mosaic_extra2.png',[data_114[...,0]])#,data_114[...,0]])

    '''

    
