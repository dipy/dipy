import os

def get_sphere(name='symmetric362'):    
    ''' provide triangulated spheres
    
    Parameters
    ----------
    name: str, which sphere
        'symmetric362' or 
        'symmetric642'
    
    Examples
    --------
     
        
    '''
    
    if name=='symmetric362':
        return os.path.join(os.path.dirname(__file__),'evenly_distributed_sphere_362.npz')
    if name=='symmetric642':
        return os.path.join(os.path.dirname(__file__),'evenly_distributed_sphere_642.npz')
    
def get_data(name='small_64D'):
    ''' provides test datasets
      
    Parameters
    ----------
    name: str, the filename/s of which dataset to return
    
    Examples
    --------
    
    
    '''    
    if name=='small_64D':
        fbvals=os.path.join(os.path.dirname(__file__),'small_64D.bvals.npy')
        fbvecs=os.path.join(os.path.dirname(__file__),'small_64D.gradients.npy')
        fimg =os.path.join(os.path.dirname(__file__),'small_64D.nii')        
        return fimg,fbvals, fbvecs
    if name=='55dir_grad.bvec':
        return os.path.join(os.path.dirname(__file__),'55dir_grad.bvec')
    if name=='small_101D':
        fbvals=os.path.join(os.path.dirname(__file__),'small_101D.bval')
        fbvecs=os.path.join(os.path.dirname(__file__),'small_101D.bvec')
        fimg=os.path.join(os.path.dirname(__file__),'small_101D.nii.gz')
        return fimg,fbvals, fbvecs
    
        
    
    