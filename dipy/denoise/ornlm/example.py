import dipy
import numpy as np
import nibabel as nib
from dipy.denoise.ornlm.ornlm_module import ornlmpy
from dipy.denoise.ornlm.hsm import hsm
from dipy.denoise.ornlm.ascm import ascm
from dipy.data import get_data

fimg, fbval, fbvec = get_data('small_101D')
nib_image = nib.load(fimg)
image=nib_image.get_data().astype(np.double)

fima1=np.empty_like(image, order='F')#Filter using ornlm, search-volume size=3, block size=1
fima2=np.empty_like(image, order='F')#Filter using ornlm, search-volume size=3, block size=2
fima3=np.empty_like(image, order='F')#Filter using hsm
fima4=np.empty_like(image, order='F')#Filter using ascm
#filter each volume (corresponding to each gradient) separately
for i in xrange(image.shape[3]):
    print "Filtering volume",i+1,"/",image.shape[3]
    mv=image[:,:,:,i].max()
    h=0.05*mv #Parameter: the amount of filtering, it is set to 5% of the maximum value in the current volume
    fima1[:,:,:,i]=np.array(ornlmpy(image[:,:,:,i], 3, 1, h))
    fima2[:,:,:,i]=np.array(ornlmpy(image[:,:,:,i], 3, 2, h))
    fima3[:,:,:,i]=np.array(hsm(fima1[:,:,:,i],fima2[:,:,:,i]))
    fima4[:,:,:,i]=np.array(ascm(image[:,:,:,i], fima1[:,:,:,i],fima2[:,:,:,i], h))

