import numpy as np
import nibabel as nib
import numpy.linalg as npl
from dipy.io.dpy import Dpy


def flirt2aff(mat, in_img, ref_img):
    """ Transform from `in_img` voxels to `ref_img` voxels given `matfile`

    Parameters
    ----------
    matfile : (4,4) array
        contents (as array) of output ``-omat`` transformation file from flirt
    in_img : img
        image passed (as filename) to flirt as ``-in`` image
    ref_img : img
        image passed (as filename) to flirt as ``-ref`` image

    Returns
    -------
    aff : (4,4) array
        Transform from voxel coordinates in ``in_img`` to voxel coordinates in
        ``ref_img``
    """
    in_hdr = in_img.header
    ref_hdr = ref_img.header
    # get_zooms gets the positive voxel sizes as returned in the header
    in_zoomer = np.diag(in_hdr.get_zooms() + (1,))
    ref_zoomer = np.diag(ref_hdr.get_zooms() + (1,))
    # The in_img voxels to ref_img voxels as recorded in the current affines
    current_in2ref = np.dot(ref_img.affine, in_img.affine)
    if npl.det(current_in2ref) < 0:
        raise ValueError('Negative determinant to current affine mapping - bailing out')
    return np.dot(npl.inv(ref_zoomer), np.dot(mat, in_zoomer))


def flirt2aff_files(matfile, in_fname, ref_fname):
    """ Map from `in_fname` image voxels to `ref_fname` voxels given `matfile`

    Parameters
    ----------
    matfile : str
        filename of output ``-omat`` transformation file from flirt
    in_fname : str
        filename for image passed to flirt as ``-in`` image
    ref_fname : str
        filename for image passed to flirt as ``-ref`` image

    Returns
    -------
    aff : (4,4) array
        Transform from voxel coordinates in image for ``in_fname`` to voxel
        coordinates in image for ``ref_fname``
    """
    mat = np.loadtxt(matfile)
    in_img = nib.load(in_fname)
    ref_img = nib.load(ref_fname)
    return flirt2aff(mat, in_img, ref_img)

#d101='/home/eg309/Data/TEST_MR10032/subj_10/101/'
d101='/home/eg309/Data/PROC_MR10032/subj_10/101/'

ffa=d101+'1312211075232351192010092912092080924175865ep2dadvdiffDSI10125x25x25STs005a001_bet_FA.nii.gz'
fdis=d101+'1312211075232351192010092912092080924175865ep2dadvdiffDSI10125x25x25STs005a001_nonlin_displacements.nii.gz'
ffareg=d101+'1312211075232351192010092912092080924175865ep2dadvdiffDSI10125x25x25STs005a001_bet_FA_reg.nii.gz'
flirtaff=d101+'1312211075232351192010092912092080924175865ep2dadvdiffDSI10125x25x25STs005a001_affine_transf.mat'
ftrack=d101+'1312211075232351192010092912092080924175865ep2dadvdiffDSI10125x25x25STs005a001_QA_native.dpy'
froi='/home/eg309/Data/PROC_MR10032/NIFTI_ROIs/AnatomicalROIs/ROI01_GCC.nii'
froi2='/home/eg309/Data/PROC_MR10032/NIFTI_ROIs/AnatomicalROIs/ROI02_BCC.nii'
#froi3='/home/eg309/Data/PROC_MR10032/NIFTI_ROIs/AnatomicalROIs/ROI03_SCC.nii'
froi3='/home/eg309/Downloads/SCC_analyze.nii'

ref_fname = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'

dpr=Dpy(ftrack,'r')

print dpr.track_no

T=dpr.read_indexed([0,1,2,3,2000,1000000])

for t in T:
    print t.shape
dpr.close()
track=T[4]
im2im = flirt2aff_files(flirtaff, ffa, ref_fname) #ref_name to be replaced by ffareg
print im2im
from dipy.core.track_metrics import length
print len(track)
print length(track)
#ntrack=np.dot(im2im[:3,:3],track.T)+im2im[:3,[3]]
ntrack=np.dot(track,im2im[:3,:3].T)+im2im[:3,3]
print length(ntrack)
#print length(ntrack.T)
print length(ntrack)/length(track)
#print npl.det(im2im)**(1/3.)
disimg=nib.load(fdis)
ddata=disimg.get_data()
daff=disimg.affine

from scipy.ndimage.interpolation import map_coordinates as mc
di=ddata[:,:,:,0]
dj=ddata[:,:,:,1]
dk=ddata[:,:,:,2]
mci=mc(di,ntrack.T)
mcj=mc(dj,ntrack.T)
mck=mc(dk,ntrack.T)

wtrack=ntrack+np.vstack((mci,mcj,mck)).T
np.set_printoptions(2)   
print np.hstack((wtrack,ntrack))
print length(wtrack),length(ntrack),length(track)

imgroi=nib.load(froi)    
roidata=imgroi.get_data()
roiaff=imgroi.affine
roiaff=daff
I=np.array(np.where(roidata>0)).T    
wI=np.dot(roiaff[:3,:3],I.T).T+roiaff[:3,3]
print wI.shape
wI=wI.astype('f4')

imgroi2=nib.load(froi2)    
roidata2=imgroi2.get_data()
roiaff2=imgroi2.affine
roiaff2=daff
I2=np.array(np.where(roidata2>0)).T    
wI2=np.dot(roiaff2[:3,:3],I2.T).T+roiaff2[:3,3]
print wI2.shape
wI2=wI2.astype('f4')

imgroi3=nib.load(froi3)    
roidata3=imgroi3.get_data()
roiaff3=imgroi3.affine
roiaff3=daff
I3=np.array(np.where(roidata3>0)).T    
wI3=np.dot(roiaff3[:3,:3],I3.T).T+roiaff3[:3,3]
print wI3.shape
wI3=wI3.astype('f4')


 
dpr=Dpy(ftrack,'r')    
print dpr.track_no    

from time import time
t1=time()    
iT=np.random.randint(0,dpr.track_no,10*10**2)
T=dpr.read_indexed(iT)    
dpr.close()
t2=time()
print t2-t1,len(T)

Tfinal=[]

'''
for (i,track) in enumerate(T):
    print i
    ntrack=np.dot(track,im2im[:3,:3].T)+im2im[:3,3]
    mci=mc(di,ntrack.T)
    mcj=mc(dj,ntrack.T)
    mck=mc(dk,ntrack.T)
    wtrack=ntrack+np.vstack((mci,mcj,mck)).T
    Tfinal.append(np.dot(wtrack,daff[:3,:3].T)+daff[:3,3])
'''

lengths=[len(t) for t in T]
lengths.insert(0,0)
offsets=np.cumsum(lengths)

caboodle=np.concatenate(T,axis=0)
ntrack=np.dot(caboodle,im2im[:3,:3].T)+im2im[:3,3]
mci=mc(di,ntrack.T,order=1)
mcj=mc(dj,ntrack.T,order=1)
mck=mc(dk,ntrack.T,order=1)
wtrack=ntrack+np.vstack((mci,mcj,mck)).T
caboodlew=np.dot(wtrack,daff[:3,:3].T)+daff[:3,3]
#caboodlew=np.dot(wtrack,roiaff[:3,:3].T)+roiaff[:3,3]

Tfinal=[]
for i in range(len(offsets)-1):
    s=offsets[i]
    e=offsets[i+1]
    Tfinal.append(caboodlew[s:e])

#ref_fname = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
ref_fname = '/usr/share/fsl/data/standard/FMRIB58_FA-skeleton_1mm.nii.gz'
imgref=nib.load(ref_fname)
refdata=imgref.get_data()
refaff=imgref.affine

'''
refI=np.array(np.where(refdata>5000)).T    
wrefI=np.dot(refaff[:3,:3],refI.T).T+refaff[:3,3]
print wrefI.shape
wrefI=wrefI.astype('f4')
'''

    
from dipy.viz import fos
    
froi='/home/eg309/Data/ICBM_Wmpm/ICBM_WMPM.nii'
    
def get_roi(froi,no):
    imgroi=nib.load(froi)    
    roidata=imgroi.get_data()
    roiaff=imgroi.affine    
    I=np.array(np.where(roidata==no)).T    
    wI=np.dot(roiaff[:3,:3],I.T).T+roiaff[:3,3]
    wI=wI.astype('f4')
    return wI
    


from dipy.viz import fos

r=fos.ren()
#fos.add(r,fos.point(wI,fos.blue))
#fos.add(r,fos.point(wI2,fos.yellow))
#fos.add(r,fos.point(wI3,fos.green))
#fos.add(r,fos.point(wrefI,fos.cyan))
#fos.add(r,fos.point(wrefI,fos.yellow))
fos.add(r,fos.point(get_roi(froi,3),fos.blue))
fos.add(r,fos.point(get_roi(froi,4),fos.yellow))
fos.add(r,fos.point(get_roi(froi,5),fos.green))

fos.add(r,fos.line(Tfinal,fos.red))
fos.show(r)

print roiaff
print roiaff2
print roiaff3
print daff





##load roi image
#roiimg=ni.load(froi)
#roidata=roiimg.get_data()
#roiaff=roiimg.affine
#print 'roiaff',roiaff,roidata.shape
#
##load FA image
#faimg=ni.load(ffa)
#data=faimg.get_data()
#aff=faimg.affine
##aff[0,:]=-aff[0,:]
##aff[0,0]=-aff[0,0]
##aff=np.array([[2.5,0,0,-2.5*48],[0,2.5,0,-2.5*39],[0,0,2.5,-2.5*23],[0,0,0,1]])
#
#print 'aff',aff, data.shape
#
##cube =  np.array([v for v in np.ndindex(5,5,5)]).T + np.array([[47,47,27]]).T
#cube =  np.array([v for v in np.ndindex(data.shape[0],data.shape[1],data.shape[2])]).T
#
##from image space(image coordinates) to native space (world coordinates)
#cube_native = np.dot(aff[:3,:3],cube)+aff[:3,[3]]
##print cube_native.T
#
##load flirt affine
#laff=np.loadtxt(flirtaff)
##laff[0,:]=-laff[0,:]
##laff=np.linalg.inv(laff)
##laff[:3,3]=0
#print 'laff',laff
##print 'inverting laff'
#
#
##from native space(world coordinates) to mni space(world coordinates)
#cube_mni = np.dot(laff[:3,:3],cube_native)+laff[:3,[3]]
##print cube_mni.T
#
#dis=ni.load(fdis)
#disdata=dis.get_data()
#mniaff=dis.affine
#print 'mniaff',mniaff
#
##invert disaff 
#mniaffinv=  np.linalg.inv(mniaff)

##from mni space(world coordinates) to image mni space (image coordinates)
#cube_mni_grid = np.dot(mniaffinv[:3,:3],cube_mni)+mniaffinv[:3,[3]]
#print cube_mni_grid.shape
#
#cube_mni_grid_nearest=np.round(cube_mni_grid).astype(np.int)
#
#print np.max(cube_mni_grid[0,:])
#print np.max(cube_mni_grid[1,:])
#print np.max(cube_mni_grid[2,:])
#
#print np.max(cube_mni_grid_nearest[0,:])
#print np.max(cube_mni_grid_nearest[1,:])
#print np.max(cube_mni_grid_nearest[2,:])
#
#d0,d1,d2,junk = disdata.shape
#
#cube_mni_grid_nearest[np.where(cube_mni_grid_nearest<0)]=0
#cube_mni_grid_nearest[np.where(cube_mni_grid_nearest>181)]=0                               
#
#n0=cube_mni_grid_nearest[0,:]
#n1=cube_mni_grid_nearest[1,:]
#n2=cube_mni_grid_nearest[2,:]

'''
n0 = np.min(np.max(cube_mni_grid_nearest[0,:],0),d0)
n1 = np.min(np.max(cube_mni_grid_nearest[1,:],0),d1)
n2 = np.min(np.max(cube_mni_grid_nearest[2,:],0),d2)
'''


#cube_mni_data=np.zeros(disdata.shape[:-1],dtype=np.float32)

#cube_mni_data[n0,n1,n2]=1

'''
D=disdata[n0,n1,n2]

'''

#from dipy.viz import fos
#r=fos.ren()
##fos.add(r,fos.point(cube.T,fos.red))
##fos.add(r,fos.point(cube_native.T,fos.yellow))
#fos.add(r,fos.point(cube_mni.T,fos.green))
#fos.add(r,fos.sphere(np.array([0,0,0]),10))
#
##fos.add(r,fos.point(cube_mni_grid_nearest.T,fos.red))
###fos.add(r,fos.point(cube.T,fos.green))
###fos.add(r,fos.point(cube_mni_grid.T,fos.red))
###fos.add(r,fos.point(cube.T,fos.yellow))
#fos.show(r)


#
#def map_to_index(grid,shape):
#    x=grid[0,:]
#    y=grid[1,:]
#    z=grid[2,:]
#    xmin=x.min()
#    ymin=y.min()
#    zmin=z.min()
#    xmax=x.max()
#    ymax=y.max()
#    zmax=z.max()
#    i=(x-xmin)/(xmax-xmin)*shape[0]
#    j=(y-ymin)/(ymax-ymin)*shape[1]
#    k=(z-zmin)/(zmax-zmin)*shape[2]
#    return i,j,k
#    
#i,j,k=map_to_index(cube_mni_grid,(182,218,182))
#
#from scipy.ndimage import map_coordinates

#FA_MNI_IMG = map_coordinates(data,np.c_[i, j, k].T)

#from dipy.viz import fos
#r=fos.ren()
#fos.add(r,fos.point(cube_mni.T,fos.blue))
#fos.add(r,fos.point(cube_native.T,fos.green))
#fos.add(r,fos.point(cube_mni_grid.T,fos.red))
#fos.add(r,fos.point(cube.T,fos.yellow))
#fos.show(r)

###corner =  cube[:,:].astype(np.int).T
#print corner
###print data[corner[:,0:27],corner[:,0:27],corner[:,0:27]]

#def func(x,y):
#    return (x+y)*np.exp(-5.*(x**2+y**2))
#
#def map_to_index(x,y,bounds,N,M):    
#    xmin,xmax,ymin,ymax=bounds
#    i1=(x-xmin)/(xmax-xmin)*N
#    i2=(y-ymin)/(ymax-ymin)*M
#    return i1,i2
#
#x,y=np.mgrid[-1:1:10j,-1:1:10j]
#fvals=func(x,y)
#
#xn,yn=np.mgrid[-1:1:100j,-1:1:100j]
#i1,i2 = map_to_index(xn,yn,[-1,1,-1,1],*x.shape)
#
#from scipy.ndimage import map_coordinates
#
#fn = map_coordinates(fvals,[i1,i2])
#true = func(xn,yn)





def test_flirt2aff():
    from os.path import join as pjoin
    from nose.tools import assert_true
    import scipy.ndimage as ndi
    import nibabel as nib
    
    '''
    matfile = pjoin('fa_data',
                    '1312211075232351192010092912092080924175865ep2dadvdiffDSI10125x25x25STs005a001_affine_transf.mat')
    in_fname = pjoin('fa_data',
                     '1312211075232351192010092912092080924175865ep2dadvdiffDSI10125x25x25STs005a001_bet_FA.nii.gz')
    '''
    
    matfile=flirtaff
    in_fname = ffa
    
    ref_fname = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
    res = flirt2aff_files(matfile, in_fname, ref_fname)
    mat = np.loadtxt(matfile)
    in_img = nib.load(in_fname)
    ref_img = nib.load(ref_fname)
    assert_true(np.all(res == flirt2aff(mat, in_img, ref_img)))
    # mm to mm transform
    mm_in2mm_ref =  np.dot(ref_img.affine,
                           np.dot(res, npl.inv(in_img.affine)))
    # make new in image thus transformed
    in_data = in_img.get_data()
    ires = npl.inv(res)
    in_data[np.isnan(in_data)] = 0
    resliced_data = ndi.affine_transform(in_data,
                                         ires[:3,:3],
                                         ires[:3,3],
                                         ref_img.shape)
    resliced_img = nib.Nifti1Image(resliced_data, ref_img.affine)
    nib.save(resliced_img, 'test.nii')






