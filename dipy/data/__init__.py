import os

def get_sphere(name='symmetric362'):    
    if name=='symmetric362':
        return os.path.join(os.path.dirname(__file__),'evenly_distributed_sphere_362.npz')
    if name=='symmetric642':
        return os.path.join(os.path.dirname(__file__),'evenly_distributed_sphere_642.npz')
    
def get_data(name='small_64D'):    
    if name=='small_64D':
        fbvals=os.path.join(os.path.dirname(__file__),'small_64D.bvals.npy')
        fbvecs=os.path.join(os.path.dirname(__file__),'small_64D.gradients.npy')
        fimg =os.path.join(os.path.dirname(__file__),'small_64D.nii')        
        return fimg,fbvals, fbvecs
    if name=='55dir_grad.bvec':
        return os.path.join(os.path.dirname(__file__),'55dir_grad.bvec')
    
        
    
    