import numpy as np
cimport cython
from FusedTypes cimport floating
cdef enum:
    SI = 0
    SI2 = 1
    SJ = 2
    SJ2 = 3
    SIJ = 4
    CNT = 5

cdef inline int _int_max(int a, int b): return a if a >= b else b
cdef inline int _int_min(int a, int b): return a if a <= b else b

def precompute_cc_factors_3d(floating[:,:,:] fixed, floating[:,:,:] moving, int radius):
    cdef int side = 2 * radius + 1
    cdef int ns=fixed.shape[0]
    cdef int nr=fixed.shape[1]
    cdef int nc=fixed.shape[2]
    cdef int s,r,c, k, i, j, t, q, qq, firstc, lastc, firstr, lastr
    cdef floating Imean, Jmean
    cdef floating[:,:,:,:] factors=np.ndarray((ns, nr, nc, 5), dtype=cython.typeof(fixed[0,0,0]))
    cdef floating[:,:] lines=np.zeros((6, side), dtype=cython.typeof(fixed[0,0,0]))
    cdef floating[:] sums=np.zeros((6,), dtype=cython.typeof(fixed[0,0,0]))
    for r in range(nr):
        firstr=_int_max(0, r-radius)
        lastr=_int_min(nr-1, r+radius)
        for c in range(nc):
            firstc=_int_max(0, c-radius);
            lastc=_int_min(nc-1, c+radius);
            #compute factors for line [:,r,c]
            sums[...] = 0
            #Compute all slices and set the sums on the fly
            for k in range(ns):#compute each slice [k, i={r-radius..r+radius}, j={c-radius, c+radius}]
                q=k%side
                for t in range(6):
                    sums[t]-=lines[t,q]
                    lines[t,q]=0
                for i in range(firstr, lastr+1):
                    for j in range(firstc, lastc+1):
                        lines[SI,q]  += fixed[k,i,j]
                        lines[SI2,q] += fixed[k,i,j]*fixed[k,i,j]
                        lines[SJ,q]  += moving[k,i,j]
                        lines[SJ2,q] += moving[k,i,j]*moving[k,i,j]
                        lines[SIJ,q] += fixed[k,i,j]*moving[k,i,j]
                        lines[CNT,q] += 1
                sums[...] = 0
                for t in range(6):
                    for qq in range(side):
                        sums[t]+=lines[t, qq]
                if(k>=radius):
                    s=k-radius#s is the voxel that is affected by the cube with slices [s-radius..s+radius, :, :]
                    Imean=sums[SI]/sums[CNT]
                    Jmean=sums[SJ]/sums[CNT]
                    factors[s,r,c,0] = fixed[s,r,c] - Imean
                    factors[s,r,c,1] = moving[s,r,c] - Jmean
                    factors[s,r,c,2] = sums[SIJ] - Jmean * sums[SI] - Imean * sums[SJ] + sums[CNT] * Jmean * Imean
                    factors[s,r,c,3] = sums[SI2] - Imean * sums[SI] - Imean * sums[SI] + sums[CNT] * Imean * Imean
                    factors[s,r,c,4] = sums[SJ2] - Jmean * sums[SJ] - Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean
            #Finally set the values at the end of the line
            for s in range(ns-radius, ns):
                k=s+radius#this would be the last slice to be processed for voxel [s,r,c], if it existed
                q=k%side
                for t in range(6):
                    sums[t]-=lines[t,q]
                Imean=sums[SI]/sums[CNT];
                Jmean=sums[SJ]/sums[CNT];
                factors[s,r,c,0] = fixed[s,r,c] - Imean
                factors[s,r,c,1] = moving[s,r,c] - Jmean
                factors[s,r,c,2] = sums[SIJ] - Jmean * sums[SI] - Imean * sums[SJ] + sums[CNT] * Jmean * Imean
                factors[s,r,c,3] = sums[SI2] - Imean * sums[SI] - Imean * sums[SI] + sums[CNT] * Imean * Imean
                factors[s,r,c,4] = sums[SJ2] - Jmean * sums[SJ] - Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean
    return factors

def compute_cc_forward_step_3d(floating[:,:,:,:] gradFixed, floating[:,:,:,:] gradMoving, floating[:,:,:,:] factors):
    cdef int ns=gradFixed.shape[0]
    cdef int nr=gradFixed.shape[1]
    cdef int nc=gradFixed.shape[2]
    cdef floating energy = 0
    cdef floating Ii, Ji, sfm, sff, smm, localCorrelation, temp
    cdef floating[:,:,:,:] out=np.zeros((ns, nr, nc, 3), dtype=cython.typeof(gradFixed[0,0,0,0]))
    for s in range(ns):
        for r in range(nr):
            for c in range(nc):
                Ii  = factors[s,r,c,0]
                Ji  = factors[s,r,c,1]
                sfm = factors[s,r,c,2]
                sff = factors[s,r,c,3]
                smm = factors[s,r,c,4]
                if(sff==0.0 or smm==0.0):
                    continue
                localCorrelation = 0
                if(sff*smm>1e-5):
                    localCorrelation=sfm*sfm/(sff*smm)
                if(localCorrelation<1):#avoid bad values...
                    energy-=localCorrelation
                temp = 2.0 * sfm / (sff * smm) * ( Ji - sfm / sff * Ii )
                for qq in range(3):
                    out[s,r,c,qq] -= temp*gradFixed[s,r,c,qq]
    return out, energy

def compute_cc_backward_step_3d(floating[:,:,:,:] gradFixed, floating[:,:,:,:] gradMoving, floating[:,:,:,:] factors):
    cdef int ns=gradFixed.shape[0]
    cdef int nr=gradFixed.shape[1]
    cdef int nc=gradFixed.shape[2]
    cdef floating energy = 0
    cdef floating Ii, Ji, sfm, sff, smm, localCorrelation, temp
    cdef floating[:,:,:,:] out=np.zeros((ns, nr, nc, 3), dtype=cython.typeof(gradFixed[0,0,0,0]))
    for s in range(ns):
        for r in range(nr):
            for c in range(nc):
                Ii  = factors[s,r,c,0]
                Ji  = factors[s,r,c,1]
                sfm = factors[s,r,c,2]
                sff = factors[s,r,c,3]
                smm = factors[s,r,c,4]
                if(sff==0.0 or smm==0.0):
                    continue
                localCorrelation = 0
                if(sff*smm>1e-5):
                    localCorrelation=sfm*sfm/(sff*smm)
                if(localCorrelation<1):#avoid bad values...
                    energy-=localCorrelation
                temp = 2.0 * sfm / (sff * smm) * ( Ii - sfm / smm * Ji )
                for qq in range(3):
                    out[s,r,c,qq] -= temp*gradMoving[s,r,c,qq]
    return out, energy
