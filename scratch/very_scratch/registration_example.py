import os
import numpy as np
import dipy as dp
import nibabel as ni
import resources
import time
from subprocess import Popen,PIPE 


#Registration options
#similarity 'cc', 'cr', 'crl1', 'mi', je', 'ce', 'nmi', 'smi'.  'cr'
similarity='cr'
#interp 'pv', 'tri'
interp =  'tri'
#subsampling None or sequence (3,)
subsampling=[1,1,1]
#search 'affine', 'rigid', 'similarity' or ['rigid','affine']
search='affine'
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
        print i, sourceT.get_data().shape, sourceT.affine.shape
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


def haircut_dwi_reference(nii,nii_hair):
    cmd='bet '+nii+' '+ nii_hair + ' -f .2 -g 0'
    print cmd
    p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE)
    sto=p.stdout.readlines()
    ste=p.stderr.readlines()

    print sto
    print ste
   

def register_FA_same_subj_diff_sessions(dname_grid,dname_shell):


    print('create temporary directory')
    tmp_dir='/tmp'

    print('load dicom data')
    data_gr,affine_gr,bvals_gr,gradients_gr=dp.load_dcm_dir(dname_grid)
    data_sh,affine_sh,bvals_sh,gradients_sh=dp.load_dcm_dir(dname_shell)

    print('save DWI reference as nifti')
    tmp_grid=os.path.join(tmp_dir,os.path.basename(dname_grid)+'_ref.nii')
    tmp_shell=os.path.join(tmp_dir,os.path.basename(dname_shell)+'_ref.nii')    
    ni.save(ni.Nifti1Image(data_gr[...,0],affine_gr),tmp_grid)    
    ni.save(ni.Nifti1Image(data_sh[...,0],affine_sh),tmp_shell)

    print('prepare filenames for haircut (bet)')
    tmp_grid_bet=os.path.join(os.path.dirname(tmp_grid),\
                                  os.path.splitext(os.path.basename(dname_grid))[0]+\
                                  '_ref_bet.nii.gz')    
    tmp_shell_bet=os.path.join(os.path.dirname(tmp_shell),\
                                   os.path.splitext(os.path.basename(dname_shell))[0]+\
                                   '_ref_bet.nii.gz')

    print('bet is running')
    haircut_dwi_reference(tmp_grid,tmp_grid_bet)
    haircut_dwi_reference(tmp_shell,tmp_shell_bet)

    print('load nii.gz reference (s0) volumes')
    img_gr_bet=ni.load(tmp_grid_bet)
    img_sh_bet=ni.load(tmp_shell_bet)
    
    print('register the shell reference to the grid reference')
    source=img_sh_bet
    target=img_gr_bet    
    T=dp.volume_register(source,target,similarity,\
                              interp,subsampling,search,optimizer)

    print('apply the inverse of the transformation matrix')
    sourceT=dp.volume_transform(source, T.inv(), reference=target)    
    #ni.save(sourceT,'/tmp/result.nii.gz')

    print('calculate FA for grid and shell data')
    FA_grid=dp.Tensor( data_gr,bvals_gr,gradients_gr,thresh=50).FA
    FA_shell=dp.Tensor(data_sh,bvals_sh,gradients_sh,thresh=50).FA

    print('create an FA nibabel image for shell')
    FA_shell_img=ni.Nifti1Image(FA_shell,affine_sh)

    print('transform FA_shell')
    FA_shell_imgT=dp.volume_transform(FA_shell_img,T.inv(),reference=target)    

    return ni.Nifti1Image(FA_grid,affine_gr),FA_shell_imgT

def flirt(in_nii, ref_nii,out_nii,transf_mat):
    cmd='flirt -in ' + in_nii + ' -ref ' + ref_nii + ' -out ' \
        + out_nii +' -dof 6 -omat ' + transf_mat
    print(cmd)
    pipe(cmd)
    
def flirt_apply_transform(in_nii, target_nii, out_nii, transf_mat):
    cmd='flirt -in ' + in_nii + ' -ref ' + target_nii + ' -out ' \
        + out_nii +' -init ' + transf_mat +' -applyxfm'
    print(cmd)
    pipe(cmd)

def test_registration():


    S012='/tmp/compare_12_with_32_Verio_directly/18620_0004.nii_S0.nii.gz'
    S032='/tmp/compare_12_with_32_Verio_directly/18620_0006.nii_S0.nii.gz'
    S012T='/tmp/compare_12_with_32_Verio_directly/S0_reg.nii.gz'
    MP='/tmp/compare_12_with_32_Verio_directly/MPRAGE.nii'
    D114=resources.get_paths('DTI STEAM 114 Trio')[2]
    data,affine,bvals,gradients=dp.load_dcm_dir(D114)
    D114i=ni.Nifti1Image(data[...,0],affine)

    D101=resources.get_paths('DSI STEAM 101 Trio')[2]
    data,affine,bvals,gradients=dp.load_dcm_dir(D101)
    D101i=ni.Nifti1Image(data[...,0],affine)
    ni.save(D101i,'/tmp/compare_12_with_32_Verio_directly/S0_101_reg.nii.gz')
        
    #source=ni.load(S012)
    source=D114i
    #target=D101i
    #target=ni.load(S032)
    target=ni.load(MP)

    target._data=np.squeeze(target._data)
    #target._affine= np.dot(np.diag([-1, -1, 1, 1]), target._affine)
    
    similarity='cr'
    interp =  'tri'    
    subsampling=None
    search='affine'    
    optimizer= 'powell'

    T=dp.volume_register(source,target,similarity,\
                              interp,subsampling,search,optimizer)

    print('Transformation matrix')
    print(T.inv())
        
    sourceT=dp.volume_transform(source,T.inv(),reference=target,interp_order=0)     

    sourceTd=sourceT.get_data()
    sourceTd[sourceTd<0]=0

    sourceT._data=sourceTd

    ni.save(sourceT,S012T)

    sourced=source.get_data()
    targetd=target.get_data()
    sourceTd=sourceT.get_data()
    
    print 'source info',sourced.min(), sourced.max()
    print 'target info',targetd.min(), targetd.max()
    print 'sourceT info',sourceTd.min(), sourceTd.max()
    
    #save_volumes_as_mosaic('/tmp/mosaic_S0_MP_cr_pv_powell.png',\
    #                           [sourced,sourceTd,targetd])

    # RAS to LPS np.dot(np.diag([-1, -1, 1, 1]), A)
    # LPS to RAS


    

if __name__ == '__main__':


    '''
    print('Goal is to compare FA of grid versus shell acquisitions using STEAM')

    print('find filenames for grid and shell data')    
    dname_grid=resources.get_paths('DSI STEAM 101 Trio')[2]
    dname_shell=resources.get_paths('DTI STEAM 114 Trio')[2]
    #print('find filenames for T1')
    #fname_T1=resources.get_paths('MPRAGE nifti Trio')[2]   

    FA_grid_img,FA_shell_imgT=register_FA_same_subj_diff_sessions(dname_grid,dname_shell)
        
    #FA_shell_data=FA_shell_imgT.get_data()
    #FA_shell_data[FA_shell_data<0]=0
    
    print('tile volumes')
    save_volumes_as_mosaic('/tmp/mosaic_fa.png',\
                               [FA_grid_img.get_data(),FA_shell_imgT.get_data()])

    '''

    




    
