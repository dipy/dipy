import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
from dipy.align import floating

def renormalizeImage(image):
    r'''
    Changes the dynamic range of image linearly to [0, 127]
    '''
    m=np.min(image)
    M=np.max(image)
    if(M-m<1e-8):
        return image
    image-=m
    image*=127.0
    image/=(M-m)
    return np.array(image, dtype = floating)

def pyramid_gaussian_3D(image, max_layer, mask=None):
    r'''
    Generates a 3D Gaussian Pyramid of max_layer+1 levels from image
    '''
    yield image.copy().astype(floating)
    for i in range(max_layer):
        newImage=np.array(sp.ndimage.filters.gaussian_filter(image, 2.0/3.0)[::2,::2,::2], dtype = floating)
        if(mask!=None):
            mask=mask[::2,::2,::2]
            newImage*=mask
        image=newImage.copy()
        yield newImage

def pyramid_gaussian_2D(image, max_layer, mask=None):
    r'''
    Generates a 3D Gaussian Pyramid of max_layer+1 levels from image
    '''
    yield image.copy().astype(floating)
    for i in range(max_layer):
        newImage=np.empty(shape=((image.shape[0]+1)//2, (image.shape[1]+1)//2), dtype=floating)
        newImage[...]=sp.ndimage.filters.gaussian_filter(image, 2.0/3.0)[::2,::2]
        if(mask!=None):
            mask=mask[::2,::2]
            newImage*=mask
        image=newImage
        yield newImage

def overlayImages(img0, img1, createFig=True):
    r'''
    Plots three images: img0 to the left, img1 to the right and the overlaid
    images at the center drawing img0 to the red channel and img1 to the green 
    channel
    '''
    colorImage=np.zeros(shape=(img0.shape)+(3,), dtype=np.int8)
    colorImage[...,0]=renormalizeImage(img0)
    colorImage[...,1]=renormalizeImage(img1)
    fig=None
    if(createFig):
        fig=plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img0, cmap=plt.cm.gray)
    plt.title('Img0 (red)')
    plt.subplot(1,3,2)
    plt.imshow(colorImage)
    plt.title('Overlay')
    plt.subplot(1,3,3)
    plt.imshow(img1, cmap=plt.cm.gray)
    plt.title('Img1 (green)')
    return fig

def drawLattice2D(nrows, ncols, delta):
    lattice=np.ndarray((1+(delta+1)*nrows, 1+(delta+1)*ncols), dtype=floating)
    lattice[...]=127
    for i in range(nrows+1):
        lattice[i*(delta+1), :]=0
    for j in range(ncols+1):
        lattice[:, j*(delta+1)]=0
    return lattice

def warpImage(image, displacement):
    sh=image.shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]
    warped=ndimage.map_coordinates(image, [X0+displacement[...,0], X1+displacement[...,1]], prefilter=False)
    return warped

def computeJacobianField(displacement):
    g00,g01=sp.gradient(displacement[...,0])
    g10,g11=sp.gradient(displacement[...,1])
    return (1+g00)*(1+g11)-g10*g01

def plotDiffeomorphism(GT, GTinv, GTres, titlePrefix, delta=10):
    nrows=GT.shape[0]
    ncols=GT.shape[1]
    X1,X0=np.mgrid[0:GT.shape[0], 0:GT.shape[1]]
    lattice=drawLattice2D((nrows+delta)/(delta+1), (ncols+delta)/(delta+1), delta)
    lattice=lattice[0:nrows,0:ncols]
    gtLattice=warpImage(lattice, np.array(GT))
    gtInvLattice=warpImage(lattice, np.array(GTinv))
    gtResidual=warpImage(lattice, np.array(GTres))
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(gtLattice, cmap=plt.cm.gray)
    plt.title(titlePrefix+'[Deformation]')
    plt.subplot(2, 3, 2)
    plt.imshow(gtInvLattice, cmap=plt.cm.gray)
    plt.title(titlePrefix+'[Inverse]')
    plt.subplot(2, 3, 3)
    plt.imshow(gtResidual, cmap=plt.cm.gray)
    plt.title(titlePrefix+'[residual]')
    #plot jacobians and residual norm
    detJacobian=computeJacobianField(GT)
    plt.subplot(2, 3, 4)
    plt.imshow(detJacobian, cmap=plt.cm.gray)
    CS=plt.contour(X0,X1,detJacobian,levels=[0.0], colors='r')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('det(J(d))')    
    detJacobianInverse=computeJacobianField(GTinv)
    plt.subplot(2, 3, 5)
    plt.imshow(detJacobianInverse, cmap=plt.cm.gray)
    CS=plt.contour(X0,X1,detJacobianInverse,levels=[0.0], colors='r')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('det(J(d^-1))')    
    nrm=np.sqrt(np.sum(np.array(GTres)**2,2))
    plt.subplot(2, 3, 6)
    plt.imshow(nrm, cmap=plt.cm.gray)
    plt.title('||residual||_2')    
    g00, g01=sp.gradient(GTinv[...,0])
    g10, g11=sp.gradient(GTinv[...,1])
    #priorEnergy=g00**2+g01**2+g10**2+g11**2
    return [gtLattice, gtInvLattice, gtResidual, detJacobian]




def readAntsAffine(fname):
    '''
    readAntsAffine('IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt')
    '''
    try:
        with open(fname) as f:
            lines=[line.strip() for line in f.readlines()]
    except IOError:
        print 'Can not open file: ', fname
        return
    if not (lines[0]=="#Insight Transform File V1.0"):
        print 'Unknown file format'
        return
    if lines[1]!="#Transform 0":
        print 'Unknown transformation type'
        return
    A=np.zeros((3,3))
    b=np.zeros((3,))
    c=np.zeros((3,))
    for line in lines[2:]:
        data=line.split()
        if data[0]=='Transform:':
            if data[1]!='MatrixOffsetTransformBase_double_3_3':
                print 'Unknown transformation type'
                return
        elif data[0]=='Parameters:':
            parameters=np.array([float(s) for s in data[1:]], dtype=floating)
            A=parameters[:9].reshape((3,3))
            b=parameters[9:]
        elif data[0]=='FixedParameters:':
            c=np.array([float(s) for s in data[1:]], dtype=floating)
    T=np.ndarray(shape=(4,4), dtype=floating)
    T[:3,:3]=A[...]
    T[3,:]=0
    T[3,3]=1
    T[:3,3]=b+c-A.dot(c)
    ############This conversion is necessary for compatibility between itk and nibabel#########
    conversion=np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], dtype=floating)
    T=conversion.dot(T.dot(conversion))
    ###########################################################################################
    return T
