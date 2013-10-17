cimport cython
from cython.view cimport array as cvarray
import numpy as np
import math
cdef inline int _int_max(int a, int b): return a if a >= b else b
cdef inline int _int_min(int a, int b): return a if a <= b else b

cdef double bessi0(double x):
    '''
    Returns the modified Bessel function I0(x) for any real x.
    '''
    cdef double ax,ans,a,y
    ax=np.abs(x)
    if(ax<3.75):
        y=x/3.75
        y*=y
        ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))))
    else:
        y=3.75/ax
        ans=(np.exp(ax)/np.sqrt(ax))
        a=y*(0.916281e-2+y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1+y*0.392377e-2))))
        ans=ans*(0.39894228 + y*(0.1328592e-1 +y*(0.225319e-2+y*(-0.157565e-2+a))))
    return ans

cdef double bessi1(double x):
    '''
    Returns the modified Bessel function I1(x) for any real x.
    '''
    cdef double ax,ans,y
    ax=np.abs(x)
    if(ax < 3.75):
        y=x/3.75
        y*=y
        ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934+y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))))
    else:
        y=3.75/ax
        ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1-y*0.420059e-2))
        ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2+y*(0.163801e-2+y*(-0.1031555e-1+y*ans))))
        ans *= (np.exp(ax)/np.sqrt(ax))
    if(x<0):
        return -ans
    return ans

cdef double Epsi(double snr):
    cdef double pi=3.1415926535
    cdef double val
    val=(2 + snr*snr - 
        (pi/8)*np.exp(-(snr*snr)/2)*((2+snr*snr)*bessi0((snr*snr)/4) + 
        (snr*snr)*bessi1((snr*snr)/4))*((2+snr*snr)*bessi0((snr*snr)/4) + 
        (snr*snr)*bessi1((snr*snr)/4)))
    if(val<0.001):
        val=1
    if(val>10):
        val=1    
    return val

cdef _average_block(double[:,:,:] ima, int x, int y, int z, 
                   double[:,:,:] average, double weight, int rician):
    cdef int a, b, c, x_pos, y_pos, z_pos
    cdef int is_outside
    cdef int neighborhoodsize=average.shape[0]//2
    for c in range(average.shape[2]):
        for b in range(average.shape[1]):
            for a in range(average.shape[0]):
                x_pos = x+a-neighborhoodsize
                y_pos = y+b-neighborhoodsize
                z_pos = z+c-neighborhoodsize
                is_outside=0
                if ((x_pos < 0) or (x_pos >= ima.shape[0])):
                    is_outside = 1
                if ((y_pos < 0) or (y_pos >= ima.shape[1])):
                    is_outside = 1
                if ((z_pos < 0) or (z_pos >= ima.shape[2])):
                    is_outside = 1
                if(rician==1):
                    if (is_outside==1):
                        average[a,b,c]+= weight*(ima[x,y,z]**2)
                    else:
                        average[a,b,c]+= weight*(ima[x_pos,y_pos,z_pos]**2)
                else:
                    if (is_outside==1):
                        average[a,b,c]+= weight*(ima[x,y,z])
                    else:
                        average[a,b,c]+= weight*(ima[x_pos,y_pos,z_pos])

cdef _value_block(double[:,:,:] estimate, double[:,:,:] Label, int x, int y, 
                 int z, double[:,:,:] average, double global_sum):
    cdef int is_outside, a, b, c, x_pos, y_pos, z_pos, count=0
    cdef double value = 0.0
    cdef double denoised_value =0.0
    cdef double label = 0.0
    cdef int neighborhoodsize=average.shape[0]//2
    for c in range(average.shape[2]):
        for b in range(average.shape[1]):
            for a in range(average.shape[0]):
                is_outside = 0
                x_pos = x+a-neighborhoodsize
                y_pos = y+b-neighborhoodsize
                z_pos = z+c-neighborhoodsize
                if ((x_pos < 0) or (x_pos >= estimate.shape[0])):
                    is_outside = 1
                if ((y_pos < 0) or (y_pos >= estimate.shape[1])):
                    is_outside = 1
                if ((z_pos < 0) or (z_pos >= estimate.shape[2])):
                    is_outside = 1
                if (is_outside==0):
                    value = estimate[x_pos, y_pos, z_pos]+(average[a,b,c]/global_sum)
                    label = Label[x_pos, y_pos, z_pos]
                    estimate[x_pos, y_pos, z_pos] = value
                    Label[x_pos, y_pos, z_pos] = label +1

cdef double _distance(double[:,:,:] image, int x, int y, int z, 
              int nx, int ny, int nz, int f):
    '''
    Computes the distance between two square subpatches of image located at 
    p and q, respectively. If the centered squares lie beyond the boundaries 
    of image, they are mirrored.
    '''
    cdef double d, acu, distancetotal
    cdef int i, j, k, ni1, nj1, ni2, nj2, nk1, nk2
    cdef int sx=image.shape[0], sy=image.shape[1], sz=image.shape[2]
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
                distancetotal+=(image[ni1, nj1, nk1]-image[ni2, nj2, nk2])**2
                acu=acu + 1
    d=distancetotal/acu
    return d

cdef double _distance2(double[:,:,:] image, double[:,:,:] medias, int x, int y, int z, 
              int nx, int ny, int nz, int f):
    cdef double d, acu, distancetotal
    cdef int i, j, k, ni1, nj1, ni2, nj2, nk1, nk2
    cdef int sx=image.shape[0], sy=image.shape[1], sz=image.shape[2]
    acu=0
    distancetotal=0
    for k in range(-f, f+1):
        for j in range(-f, f+1):
            for i in range(-f, f+1):
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
                d=(image[ni1, nj1, nk1]-medias[ni1, nj1, nk1])-(image[ni2, nj2, nk2]-medias[ni2, nj2, nk2])
                distancetotal+=d**2
                acu=acu + 1
    d=distancetotal/acu
    return d

cdef void _regularize(double[:,:,:] imgIn, double[:,:,:] imgOut, int r):
    cdef double acu
    cdef int ind,i,j,k,ni,nj,nk,ii,jj,kk
    cdef double[:,:,:] temp=np.zeros_like(imgIn, order='F')
    cdef int[:] sh=cvarray((3,), itemsize=sizeof(int), format="i")
    sx=imgIn.shape[0]
    sy=imgIn.shape[1]
    sz=imgIn.shape[2]
    
    #separable convolution
    for k in range(sz):
        for j in range(sy):
            for i in range(sx):
                if(imgIn[i,j,k]==0):#FIXME:shouldn't test for equality
                    continue
                acu=0
                ind=0
                for ii in range(-r, r+1):
                    ni=i+ii
                    if(ni<0):
                        ni=-ni
                    if(ni>=sx):
                        ni=2*sx-ni-1
                    if(imgIn[ni, j, k]>0):
                        acu+=imgIn[ni,j,k]
                        ind+=1
                if(ind==0):
                    ind=1
                imgOut[i,j,k]=acu/ind
    for k in range(sz):
        for j in range(sy):
            for i in range(sx):
                if(imgOut[i, j, k]==0):#FIXME:shouldn't test for equality
                    continue
                acu=0
                ind=0
                for jj in range(-r, r+1):
                    nj=j+jj
                    if(nj<0):
                        nj=-nj
                    if(nj>=sy):
                        nj=2*sy-nj-1
                    if(imgOut[i, nj, k]>0):
                        acu+=imgOut[i, nj, k]
                        ind+=1
                if(ind==0):
                    ind=1
                temp[i,j,k]=acu/ind
    for k in range(sz):
        for j in range(sy):
            for i in range(sx):
                if(temp[i, j, k]==0):
                    continue
                acu=0
                ind=0
                for kk in range(-r, r+1):
                    nk=k+kk
                    if(nk<0):
                        nk=-nk
                    if(nk>=sz):
                        nk=2*sz-nk-1
                    if(temp[i, j, nk]>0):
                        acu+=temp[i, j, nk]
                        ind+=1
                if(ind==0):
                    ind=1
                imgOut[i, j, k]=acu/ind

cdef _local_mean(double [:,:,:]ima, int x, int y, int z):
    cdef int[:] dims=cvarray((3,), itemsize=sizeof(int), format="i")
    dims[0]=ima.shape[0]
    dims[1]=ima.shape[1]
    dims[2]=ima.shape[2]
    cdef double ss=0
    cdef int px, py, pz, dx, dy, dz, nx, ny, nz
    for px in range(x-1,x+2):
        for py in range(y-1,y+2):
            for pz in range(z-1,z+2):
                px=(-px if px<0 else (2*dims[0]-px-1 if px>=dims[0] else px))
                py=(-py if py<0 else (2*dims[1]-py-1 if py>=dims[1] else py))
                pz=(-pz if pz<0 else (2*dims[2]-pz-1 if pz>=dims[2] else pz))
                ss+=ima[px,py,pz]
    return ss/27.0

cdef _local_variance(double[:,:,:] ima, double mean, int x, int y, int z):
    cdef int[:] dims=cvarray((3,), itemsize=sizeof(int), format="i")
    dims[0]=ima.shape[0]
    dims[1]=ima.shape[1]
    dims[2]=ima.shape[2]
    cdef int cnt=0
    cdef double ss=0
    cdef int dx, dy, dz, nx, ny, nz
    for px in range(x-1,x+2):
        for py in range(y-1,y+2):
            for pz in range(z-1,z+2):
                if ((px>=0 and py>=0 and pz>0) and 
                            (px<dims[0] and py<dims[1] and pz<dims[2])):
                    ss+=(ima[px,py,pz]-mean)*(ima[px,py,pz]-mean)
                    cnt+=1
    return ss/(cnt-1)

cdef _firdn_vector(double[:] f, double[:] h, double[:] out):
    cdef int n=len(f)
    cdef int klen=len(h)
    cdef int outLen=(n+klen)//2
    cdef double ss
    cdef int i, k, limInf, limSup, x=0, ox=0, ks=0
    for i in range(outLen):
        ss=0
        limInf=_int_max(0, x-klen+1)
        limSup=1+_int_min(n-1, x)
        ks=limInf
        for k in range(limInf, limSup):
            ss+=f[ks]*h[x-k]
            ks+=1
        out[ox]=ss
        x+=2
        ox+=1

cdef _upfir_vector(double[:] f, double[:] h, double[:] out):
    cdef int n=f.shape[0]
    cdef int klen=h.shape[0]
    cdef int outLen=2*n+klen-2
    cdef int x, limInf, limSup, k, ks
    cdef double ss
    for x in range(outLen):
        limInf=_int_max(0, x-klen+1)
        if(limInf%2==1):
            limInf+=1
        limSup=_int_min(2*(n-1), x)
        if(limSup%2==1):
            limSup-=1
        ss=0
        k=limInf
        ks=limInf//2
        while(k<=limSup):
            ss+=f[ks]*h[x-k]
            k+=2
            ks+=1
        out[x]=ss

cdef _firdn_matrix(double[:,:] F, double[:] h, double[:,:] out):
    cdef int n=F.shape[0]
    cdef int m=F.shape[1]
    cdef int j
    for j in range(m):
        _firdn_vector(F[:,j], h, out[:,j])

cdef _upfir_matrix(double[:,:] F, double[:] h, double[:,:] out):
    cdef int n=F.shape[0]
    cdef int m=F.shape[1]
    for j in range(m):
        _upfir_vector(F[:,j], h, out[:,j])

def firdn(double[:,:] image, double[:] h):
    '''
    Applies the filter given by the convolution kernel 'h' columnwise to 
    'image', then subsamples by 2. This is a special case of the matlab's
    'upfirdn' function, ported to python. Returns the filtered image.
    Parameters
    ----------
        image:  the input image to be filtered
        h:      the convolution kernel
    '''
    nrows=image.shape[0]
    ncols=image.shape[1]
    ll=h.shape[0]
    cdef double[:,:] filtered=np.zeros(shape=((nrows+ll)//2, ncols))
    _firdn_matrix(image, h, filtered)
    return filtered

def upfir(double[:,:] image, double[:] h):
    '''
    Upsamples the columns of the input image by 2, then applies the 
    convolution kernel 'h' (again, columnwise). This is a special case of the 
    matlab's 'upfirdn' function, ported to python. Returns the filtered image.
    Parameters
    ----------
        image:  the input image to be filtered
        h:      the convolution kernel
    '''
    nrows=image.shape[0]
    ncols=image.shape[1]
    ll=h.shape[0]
    cdef double[:,:] filtered=np.zeros(shape=(2*nrows+ll-2, ncols))
    _upfir_matrix(image, h, filtered)
    return filtered

def aonlm(double[:,:,:] ima, int v, int f, int rician):
    cdef double totalweight,t1,t1i,t2,d,w,distanciaminima
    cdef int i,j,k,rc,ii,jj,kk,ni,nj,nk
    cdef int cols=ima.shape[0]
    cdef int rows=ima.shape[1]
    cdef int slices=ima.shape[2]
    cdef double[:,:,:] average=np.zeros((2*f+1,2*f+1,2*f+1), dtype=np.float64, order='F')
    cdef double[:,:,:] fima=np.zeros_like(ima, order='F')
    cdef double[:,:,:] means=np.zeros_like(ima, order='F')
    cdef double[:,:,:] variances=np.zeros_like(ima, order='F')
    cdef double[:,:,:] Estimate=np.zeros_like(ima, order='F')
    cdef double[:,:,:] Label=np.zeros_like(ima, order='F')
    cdef double[:,:,:] bias=np.zeros_like(ima, order='F')
    cdef double epsilon = 0.00001
    cdef double mu1 = 0.95
    cdef double var1 = 0.5+1e-7
    cdef double init = 0
    cdef int Ndims = (2*f+1)*(2*f+1)*(2*f+1)
    cdef double wmax=0.0
    cdef double globalMax=0
    for k in range(slices):
        for i in range(cols):
            for j in range(rows):
                if(globalMax<ima[i,j,k]):
                    globalMax=ima[i,j,k]
                mm=_local_mean(ima,i,j,k)
                means[i,j,k]=mm
                variances[i,j,k]=_local_variance(ima, mm, i, j, k)
    for k in range(0, slices, 2):
        for j in range(0, rows, 2):
            for i in range(0, cols, 2):
                average[...]=0.0
                totalweight=0.0												
                distanciaminima=100000000000000
                if(ima[i,j,k]<=0 or means[i,j,k]<=epsilon or variances[i,j,k]<=epsilon):
                    wmax=1.0
                    _average_block(ima, i, j, k, average, wmax, rician)
                    totalweight = totalweight + wmax
                    _value_block(Estimate, Label, i, j, k, average, totalweight)
                else:
                    #calculate minimum distance
                    for kk in range(-v, v+1):
                        nk=k+kk
                        for jj in range(-v, v+1):
                            nj=j+jj
                            for ii in range(-v, v+1):
                                ni=i+ii														
                                if(ii==0 and jj==0 and kk==0):
                                    continue
                                if(ni>=0 and nj>=0 and nk>=0 and ni<cols and nj<rows and nk<slices):
                                    if (ima[ni, nj, nk]>0 and (means[ni, nj, nk])> epsilon and (variances[ni, nj, nk]>epsilon)):
                                        t1 = (means[i, j, k])/(means[ni, nj, nk])
                                        t1i= (globalMax-means[i, j, k])/(globalMax-means[ni, nj, nk])
                                        t2 = (variances[i, j, k])/(variances[ni, nj, nk])
                                        if( (t1>mu1 and t1<(1/mu1)) or (t1i>mu1 and t1i<(1/mu1)) and t2>var1 and t2<(1/var1)):
                                            d=_distance2(ima, means, i, j, k, ni, nj, nk, f)
                                            if(d<distanciaminima):
                                                distanciaminima=d
                    if(distanciaminima==0):
                        distanciaminima=1
                    #rician correction
                    if(rician==1):
                        for kk in range(-f, f+1):
                            nk=k+kk
                            for ii in range(-f, f+1):
                                ni=i+ii
                                for jj in range(-f, f+1):
                                    nj=j+jj
                                    if(ni>=0 and nj>=0 and nk>=0 and ni<cols and nj<rows and nk<slices):
                                        if(distanciaminima==100000000000000):
                                            bias[ni, nj, nk]=0
                                        else:
                                            bias[ni, nj, nk]=distanciaminima
                    #block filtering
                    for kk in range(-v, v+1):
                        nk=k+kk
                        for jj in range(-v, v+1):
                            nj=j+jj
                            for ii in range(-v, v+1):
                                ni=i+ii
                                if(ii==0 and jj==0 and kk==0):
                                    continue
                                if(ni>=0 and nj>=0 and nk>=0 and ni<cols and nj<rows and nk<slices):
                                    if(ima[ni, nj, nk]>0 and (means[ni, nj, nk]> epsilon) and (variances[ni, nj, nk]>epsilon)):
                                        t1 = (means[i, j, k])/(means[ni, nj, nk])
                                        t1i= (globalMax-means[i, j, k])/(globalMax-means[ni, nj, nk])
                                        t2 = (variances[i, j, k])/(variances[ni, nj, nk])
                                        if( (t1>mu1 and t1<(1/mu1)) or (t1i>mu1 and t1i<(1/mu1)) and t2>var1 and t2<(1/var1)):
                                            d=_distance(ima, i, j, k, ni, nj, nk, f)
                                            if(d>3*distanciaminima):
                                                w=0
                                            else:
                                                w = np.exp(-d/distanciaminima)
                                            if(w>wmax):
                                                wmax = w
                                            if(w>0):
                                                _average_block(ima, ni, nj, nk, average, w, rician)
                                                totalweight = totalweight + w
                    if(wmax==0.0):
                        wmax=1.0
                    _average_block(ima, i, j, k, average, wmax, rician)					
                    totalweight = totalweight + wmax
                    _value_block(Estimate, Label,i, j, k, average, totalweight)
    if(rician):
        r=np.min([5, slices, rows, cols])
        _regularize(bias, variances, r)
        for k in range(slices):
            for j in range(rows):
                for i in range(cols):
                    if(variances[i,j,k]>0):
                        SNR=means[i,j,k]/np.sqrt(variances[i,j,k])
                        bias[i,j,k]=2*(variances[i,j,k]/Epsi(SNR))
                        if(np.isnan(bias[i,j,k])):
                            bias[i]=0
    #Aggregation of the estimators (i.e. means computation)
    label = 0.0;
    estimate = 0.0;
    for i in range(cols):
        for j in range(rows):
            for k in range(slices):
                label = Label[i,j,k]
                if(label == 0.0):
                    fima[i,j,k] = ima[i,j,k]
                else:
                    estimate = Estimate[i,j,k]
                    estimate = (estimate/label)
                    if(rician):
                        estimate=np.max([0, estimate-bias[i,j,k]])
                        fima[i,j,k]=np.sqrt(estimate)
                    else:
                        fima[i,j,k]=estimate
    return fima