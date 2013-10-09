#distutils: language = c++
#distutils: sources = ornlm.cpp upfirdn.cpp
cimport cython
from cython.view cimport memoryview
from cython.view cimport array as cvarray
import numpy as np
import math
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
########################################################################
#################### porting upfirdn from matlab########################
########################################################################
def firdn_vector_pyx(double[:] f, double[:] h, double[:] out):
    cdef int n=len(f)
    cdef int klen=len(h)
    cdef int outLen=(n+klen)//2
    cdef double ss
    cdef int i, k, limInf, limSup, x=0, ox=0, ks=0
    for i in range(outLen):
        ss=0
        limInf=int_max(0, x-klen+1)
        limSup=1+int_min(n-1, x)
        ks=limInf
        for k in range(limInf, limSup):
            ss+=f[ks]*h[x-k]
            ks+=1
        out[ox]=ss
        x+=2
        ox+=1

def upfir_vector_pyx(double[:] f, double[:] h, double[:] out):
    cdef int n=f.shape[0]
    cdef int klen=h.shape[0]
    cdef int outLen=2*n+klen-2
    cdef int x, limInf, limSup, k, ks
    cdef double ss
    for x in range(outLen):
        limInf=int_max(0, x-klen+1);
        if(limInf%2==1):
            limInf+=1
        limSup=int_min(2*(n-1), x)
        if(limSup%2==1):
            limSup-=1
        ss=0
        k=limInf
        ks=limInf//2
        while(k<=limSup):
            ss+=f[ks]*h[x-k];
            k+=2;
            ks+=1
        out[x]=ss

def firdn_matrix_pyx(double[:,:] F, double[:] h, double[:,:] out):
    cdef int n=F.shape[0]
    cdef int m=F.shape[1]
    cdef int j
    for j in range(m):
        firdn_vector_pyx(F[:,j], h, out[:,j])

def upfir_matrix_pyx(double[:,:] F, double[:] h, double[:,:] out):
    cdef int n=F.shape[0]
    cdef int m=F.shape[1]
    for j in range(m):
        upfir_vector_pyx(F[:,j], h, out[:,j]);

cpdef firdnpy(double[:,:] image, double[:] h):
    nrows=image.shape[0]
    ncols=image.shape[1]
    ll=h.shape[0]
    cdef double[:,:] filtered=np.zeros(shape=((nrows+ll)//2, ncols), dtype=np.double)
    firdn_matrix_pyx(image, h, filtered)
    return filtered

cpdef upfirpy(double[:,:] image, double[:] h):
    nrows=image.shape[0]
    ncols=image.shape[1]
    ll=h.shape[0]
    cdef double[:,:] filtered=np.zeros(shape=(2*nrows+ll-2, ncols), dtype=np.double)
    upfir_matrix_pyx(image, h, filtered)
    return filtered

########################################################################
############### Reimplementation of ORNLM routines #####################
########################################################################

def Average_block_pyx(double[:,:,:] ima, int x, int y, int z, double[:,:,:] average, double weight):
    cdef int a, b, c, x_pos, y_pos, z_pos
    cdef int is_outside
    cdef int count = 0
    cdef int neighborhoodsize=average.shape[0]//2
    for a in range(average.shape[0]):
        for b in range(average.shape[1]):
            for c in range(average.shape[2]):
                x_pos = x+a-neighborhoodsize
                y_pos = y+b-neighborhoodsize
                z_pos = z+c-neighborhoodsize
                is_outside=0;
                if ((x_pos < 0) or (x_pos >= ima.shape[1])):
                    is_outside = 1
                if ((y_pos < 0) or (y_pos >= ima.shape[0])):
                    is_outside = 1
                if ((z_pos < 0) or (z_pos >= ima.shape[2])):
                    is_outside = 1
                if (is_outside==1):
                    average[a,b,c]+= weight*(ima[y,x,z]**2)
                else:
                    average[a,b,c]+= weight*(ima[y_pos,x_pos,z_pos]**2)

def Value_block_pyx(double[:,:,:] Estimate, double[:,:,:] Label, int x, int y, int z, double[:,:,:] average, double global_sum, double hh):
    cdef int is_outside, a, b, c, x_pos, y_pos, z_pos, count=0
    cdef double value = 0.0
    cdef double denoised_value =0.0
    cdef double label = 0.0
    cdef int neighborhoodsize=average.shape[0]//2
    for a in range(average.shape[0]):
        for b in range(average.shape[1]):
            for c in range(average.shape[2]):
                is_outside = 0
                x_pos = x+a-neighborhoodsize
                y_pos = y+b-neighborhoodsize
                z_pos = z+c-neighborhoodsize
                if ((x_pos < 0) or (x_pos >= Estimate.shape[1])):
                    is_outside = 1
                if ((y_pos < 0) or (y_pos >= Estimate.shape[0])):
                    is_outside = 1
                if ((z_pos < 0) or (z_pos >= Estimate.shape[2])):
                    is_outside = 1
                if (is_outside==0):
                    value = Estimate[y_pos, x_pos, z_pos];
                    denoised_value  = (average[a,b,c]/global_sum) - hh;
                    if (denoised_value > 0):
                        denoised_value = np.sqrt(denoised_value)
                    else:
                        denoised_value = 0.0
                    value += denoised_value
                    label = Label[y_pos, x_pos, z_pos]
                    Estimate[y_pos, x_pos, z_pos] = value
                    Label[y_pos, x_pos, z_pos] = label +1

# Computes the distance between two square subpatches of image located at p and q, respectively.
# If the centered squares lie beyond the boundaries of image, they are mirrored.
def distance_pyx(double[:,:,:] image, int x, int y, int z, int nx, int ny, int nz, int f):
    cdef double d, acu, distancetotal
    cdef int i, j, k, ni1, nj1, ni2, nj2, nk1, nk2
    cdef int sx=image.shape[1], sy=image.shape[0], sz=image.shape[2]
    acu=0
    distancetotal=0
    for i in range(-f, f+1):
        for j in range(-f, f+1):
            for k in range(-f, f+1):
                ni1=x+i
                nj1=y+j
                nk1=z+k
                ni2=nx+i
                nj2=ny+j
                nk2=nz+k
                if(ni1<0):ni1=-ni1
                if(nj1<0):nj1=-nj1
                if(ni2<0):ni2=-ni2
                if(nj2<0):nj2=-nj2
                if(nk1<0):nk1=-nk1
                if(nk2<0):nk2=-nk2
                if(ni1>=sx):ni1=2*sx-ni1-1
                if(nj1>=sy):nj1=2*sy-nj1-1
                if(nk1>=sz):nk1=2*sz-nk1-1
                if(ni2>=sx):ni2=2*sx-ni2-1
                if(nj2>=sy):nj2=2*sy-nj2-1
                if(nk2>=sz):nk2=2*sz-nk2-1
                distancetotal+=(image[nj1, ni1, nk1]-image[nj2, ni2, nk2])**2
                acu=acu + 1
    d=distancetotal/acu
    return d

def localMean(double [:,:,:]ima, int x, int y, int z):
    cdef double ss=0
    cdef int px, py, pz, dx, dy, dz, nx, ny, nz
    for px in range(x-1,x+2):
        for py in range(y-1,y+2):
            for pz in range(z-1,z+2):
                px=(-px if px<0 else (2*ima.shape[0]-px-1 if px>=ima.shape[0] else px))
                py=(-py if py<0 else (2*ima.shape[1]-py-1 if py>=ima.shape[1] else py))
                pz=(-pz if pz<0 else (2*ima.shape[2]-pz-1 if pz>=ima.shape[2] else pz))
                ss+=ima[px,py,pz]
    return ss/27.0

def localVariance(double[:,:,:] ima, double mean, int x, int y, int z):
    cdef int cnt=0
    cdef double ss=0
    cdef int dx, dy, dz, nx, ny, nz
    for px in range(x-1,x+2):
        for py in range(y-1,y+2):
            for pz in range(z-1,z+2):
                if px>=0 and py>=0 and pz>0 and px<ima.shape[0] and py<ima.shape[1] and pz<ima.shape[2]:
                    ss+=(ima[px,py,pz]-mean)*(ima[px,py,pz]-mean)
                    cnt+=1
    return ss/(cnt-1)

def ornlm_pyx(double [:,:,:]ima, int v, int f, double h):
    cdef int[:] dims=cvarray((3,), itemsize=sizeof(int), format="i")
    dims[0]=ima.shape[0]
    dims[1]=ima.shape[1]
    dims[2]=ima.shape[2]
    cdef double hh=2*h*h
    cdef int Ndims=(2*f+1)**3
    cdef int nvox=dims[0]*dims[1]*dims[2]
    cdef double[:,:,:] average=np.zeros((2*f+1,2*f+1,2*f+1), dtype=np.float64)
    cdef double[:,:,:] fima=np.zeros_like(ima)
    cdef double[:,:,:] means=np.zeros_like(ima)
    cdef double[:,:,:] variances=np.zeros_like(ima)
    cdef double[:,:,:] Estimate=np.zeros_like(ima)
    cdef double[:,:,:] Label=np.zeros_like(ima)
    cdef int i,j,k, ni, nj, nk
    cdef double mm
    for k in range(dims[2]):
        for i in range(dims[1]):
            for j in range(dims[0]):
                mm=localMean(ima,j,i,k)
                means[j,i,k]=mm
                variances[j,i,k]=localVariance(ima, mm, j, i, k)
    cdef double epsilon = 0.00001
    cdef double mu1 = 0.95
    cdef double var1 = 0.5+1e-7
    cdef double totalWeight, wmax, d, w
    for k in range(0, ima.shape[2], 2):
        for i in range(0, ima.shape[1], 2):
            for j in range(0, ima.shape[0], 2):
                average[...]=0
                totalWeight=0
                if (means[j,i,k]<=epsilon) or (variances[j,i,k]<=epsilon):
                    wmax=1.0
                    Average_block_pyx(ima, i, j, k, average, wmax)
                    totalWeight+=wmax
                    Value_block_pyx(Estimate, Label, i, j, k, average, totalWeight, hh)
                else:
                    wmax=0
                    for nk in range(k-v,k+v+1):
                        for ni in range(i-v,i+v+1):
                            for nj in range(j-v,j+v+1):
                                if((ni==i)and(nj==j)and(nk==k)):
                                    continue
                                if (ni<0) or (nj<0) or (nk<0) or (nj>=ima.shape[0]) or (ni>=ima.shape[1]) or (nk>=ima.shape[2]):
                                    continue;
                                if (means[nj,ni,nk]<=epsilon) or (variances[nj,ni,nk]<=epsilon):
                                    continue
                                t1 = (means[j,i,k])/(means[nj,ni,nk])
                                t2 = (variances[j,i,k])/(variances[nj,ni,nk])
                                
                                if ((t1>mu1) and (t1<(1/mu1)) and (t2>var1) and (t2<(1/var1))):
                                    d=distance_pyx(ima, i, j, k, ni, nj, nk, f)
                                    w=math.exp(-d/(h*h))
                                    if(w>wmax):
                                        wmax = w
                                    Average_block_pyx(ima, ni, nj, nk, average, w)
                                    totalWeight+=w
                    if(wmax==0.0):#FIXME
                        wmax=1.0
                    Average_block_pyx(ima, i, j, k, average, wmax)
                    totalWeight+=wmax
                    if(totalWeight != 0.0):
                        Value_block_pyx(Estimate, Label, i, j, k, average, totalWeight, hh)
    for k in range(0, ima.shape[2]):
        for i in range(0, ima.shape[1]):
            for j in range(0, ima.shape[0]):
                if(Label[j,i,k]==0.0):
                    fima[j,i,k]=ima[j,i,k]
                else:
                    fima[j,i,k]=Estimate[j,i,k]/Label[j,i,k]
    return fima
