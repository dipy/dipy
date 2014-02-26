import numpy as np
cimport cython
from FusedTypes cimport floating, integral, number
cdef extern from "math.h":
    int isinf(double)
    double sqrt(double x) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void solve2DSymmetricPositiveDefiniteSystem(floating[:] A, floating[:] y, floating[:] out):
    r'''
    Solves the symmetric positive-definite linear system Mx = y given by 
    M=[[A[0], A[1]],
       [A[1], A[2]]]. 
    Returns the result in out
    '''
    cdef floating den=(A[0]*A[2]-A[1]*A[1])
    out[1]=(A[0]*y[1]-A[1]*y[0])/den
    out[0]=(y[0]-A[1]*out[1])/A[0]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void solve3DSymmetricPositiveDefiniteSystem(floating[:] A, floating[:] y, floating[:] out):
    r'''
    Solves the symmetric positive-definite linear system Mx = y given by 
    M=[[A[0], A[1], A[2]],
       [A[1], A[3], A[4]],
       [A[2], A[4], A[5]]]. 
    Returns the result in out
    '''
    cdef floating a=A[0]
    cdef floating b=A[1]
    cdef floating c=A[2]
    cdef floating d=(a*A[3]-b*b)/a
    cdef floating e=(a*A[4]-b*c)/a
    cdef floating f=(a*A[5]-c*c)/a - (e*e*a)/(a*A[3]-b*b)
    cdef floating y0=y[0]
    cdef floating y1=(y[1]*a-y0*b)/a
    cdef floating y2=(y[2]*a-A[2]*y0)/a - (e*(y[1]*a-b*y0))/(a*A[3]-b*b)
    out[2]=y2/f
    out[1]=(y1-e*out[2])/d
    out[0]=(y0-b*out[1]-c*out[2])/a

@cython.boundscheck(False)
@cython.wraparound(False)
def iterate_residual_displacement_field_SSD2D(floating[:,:] deltaField, floating[:,:] sigmaField, floating[:,:,:] gradientField,  floating[:,:,:] target, floating lambdaParam, floating[:,:,:] displacementField):
    cdef int NUM_NEIGHBORS = 4
    cdef int[:] dRow = np.array([-1, 0, 1,  0], dtype=np.int32)
    cdef int[:] dCol = np.array([ 0, 1, 0, -1], dtype=np.int32)
    cdef int nrows=deltaField.shape[0]
    cdef int ncols=deltaField.shape[1]
    cdef int r,c, dr, dc, nn, k
    cdef floating[:] b = np.ndarray(shape=(2,), dtype=cython.typeof(deltaField[0,0]))
    cdef floating[:] d = np.ndarray(shape=(2,), dtype=cython.typeof(deltaField[0,0]))
    cdef floating[:] y = np.ndarray(shape=(2,), dtype=cython.typeof(deltaField[0,0]))
    cdef floating[:] A = np.ndarray(shape=(3,), dtype=cython.typeof(deltaField[0,0]))
    cdef floating xx,yy, opt, nrm2, delta, sigma, maxDisplacement
    maxDisplacement=0
    for r in range(nrows):
        for c in range(ncols):
            delta=deltaField[r,c]
            sigma = sigmaField[r,c] if sigmaField!=None else 1
            if(target==None):
                b[0]=deltaField[r,c]*gradientField[r,c,0]
                b[1]=deltaField[r,c]*gradientField[r,c,1]
            else:
                b[0]=target[r,c,0]
                b[1]=target[r,c,1]
            nn=0
            y[:] = 0
            for k in range(NUM_NEIGHBORS):
                dr=r+dRow[k]
                if((dr<0) or (dr>=nrows)):
                    continue
                dc=c+dCol[k]
                if((dc<0) or (dc>=ncols)):
                    continue
                nn+=1
                y[0]+=displacementField[dr, dc, 0]
                y[1]+=displacementField[dr, dc, 1]
            if(isinf(sigma)):
                xx=displacementField[r,c,0]
                yy=displacementField[r,c,1]
                displacementField[r,c,0]=y[0]/nn;
                displacementField[r,c,1]=y[1]/nn;
                xx-=displacementField[r,c,0]
                yy-=displacementField[r,c,1]
                opt=xx*xx+yy*yy
                if(maxDisplacement<opt):
                    maxDisplacement=opt
            elif(sigma==0):
                nrm2=gradientField[r,c,0]**2+gradientField[r,c,1]**2
                if(nrm2==0):
                    displacementField[r,c,:] = 0
                else:
                    displacementField[r,c,0]=(b[0])/nrm2
                    displacementField[r,c,1]=(b[1])/nrm2
            else:
                y[0]=b[0] + sigma*lambdaParam*y[0]
                y[1]=b[1] + sigma*lambdaParam*y[1]
                A[0]=gradientField[r,c,0]**2 + sigma*lambdaParam*nn
                A[1]=gradientField[r,c,0]*gradientField[r,c,1]
                A[2]=gradientField[r,c,1]**2 + sigma*lambdaParam*nn
                xx=displacementField[r,c,0]
                yy=displacementField[r,c,1]
                solve2DSymmetricPositiveDefiniteSystem(A,y,d)
                displacementField[r,c,0]=d[0]
                displacementField[r,c,1]=d[1]
                xx-=d[0]
                yy-=d[1]
                opt=xx*xx+yy*yy
                if(maxDisplacement<opt):
                    maxDisplacement=opt
    return sqrt(maxDisplacement)

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_energy_SSD2D(floating[:,:] deltaField, floating[:,:] sigmaField, floating[:,:,:] gradientField,  floating lambdaParam, floating[:,:,:] displacementField):
    cdef int nrows=deltaField.shape[0]
    cdef int ncols=deltaField.shape[1]
    cdef floating energy = 0
    for r in range(nrows):
        for c in range(ncols):
            energy+=deltaField[r,c]**2
    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
def iterate_residual_displacement_field_SSD3D(floating[:,:,:] deltaField, floating[:,:,:] sigmaField, floating[:,:,:,:] gradientField,  floating[:,:,:,:] target, floating lambdaParam, floating[:,:,:,:] displacementField):
    cdef int NUM_NEIGHBORS = 6 
    cdef int[:] dSlice = np.array([-1,  0, 0, 0,  0, 1], dtype=np.int32)
    cdef int[:] dRow = np.array([0, -1, 0, 1,  0, 0], dtype=np.int32)
    cdef int[:] dCol = np.array([0,  0, 1, 0, -1, 0], dtype=np.int32)
    cdef int nslices=deltaField.shape[0]
    cdef int nrows=deltaField.shape[1]
    cdef int ncols=deltaField.shape[2]
    cdef int s,r,c,ds, dr, dc, nn
    cdef floating[:] b = np.ndarray(shape=(3,), dtype=cython.typeof(deltaField[0,0,0]))
    cdef floating[:] d = np.ndarray(shape=(3,), dtype=cython.typeof(deltaField[0,0,0]))
    cdef floating[:] y = np.ndarray(shape=(3,), dtype=cython.typeof(deltaField[0,0,0]))
    cdef floating[:] A = np.ndarray(shape=(6,), dtype=cython.typeof(deltaField[0,0,0]))
    cdef floating xx,yy,zz, opt, nrm2, delta, sigma, maxDisplacement
    maxDisplacement=0
    for s in range(nslices):
        for r in range(nrows):
            for c in range(ncols):
                delta=deltaField[s,r,c]
                sigma = sigmaField[s,r,c] if sigmaField!=None else 1
                if(target==None):
                    b[0]=deltaField[s,r,c]*gradientField[s,r,c,0]
                    b[1]=deltaField[s,r,c]*gradientField[s,r,c,1]
                    b[2]=deltaField[s,r,c]*gradientField[s,r,c,2]
                else:
                    b[0]=target[s,r,c,0]
                    b[1]=target[s,r,c,1]
                    b[2]=target[s,r,c,2]
                nn=0
                y[:] = 0
                for k in range(NUM_NEIGHBORS):
                    ds=s+dSlice[k]
                    if((ds<0) or (ds>=nslices)):
                        continue
                    dr=r+dRow[k]
                    if((dr<0) or (dr>=nrows)):
                        continue
                    dc=c+dCol[k]
                    if((dc<0) or (dc>=ncols)):
                        continue
                    nn+=1
                    y[0]+=displacementField[ds, dr, dc, 0]
                    y[1]+=displacementField[ds, dr, dc, 1]
                    y[2]+=displacementField[ds, dr, dc, 2]
                if(isinf(sigma)):
                    xx=displacementField[s,r,c,0]
                    yy=displacementField[s,r,c,1]
                    zz=displacementField[s,r,c,2]
                    displacementField[s,r,c,0]=y[0]/nn;
                    displacementField[s,r,c,1]=y[1]/nn;
                    displacementField[s,r,c,2]=y[2]/nn;
                    xx-=displacementField[s,r,c,0]
                    yy-=displacementField[s,r,c,1]
                    zz-=displacementField[s,r,c,2]
                    opt=xx*xx+yy*yy+zz*zz
                    if(maxDisplacement<opt):
                        maxDisplacement=opt
                elif(sigma==0):
                        nrm2=gradientField[s,r,c,0]**2+gradientField[s,r,c,1]**2+gradientField[s,r,c,2]**2
                        if(nrm2==0):
                            displacementField[s,r,c,:] = 0
                        else:
                            displacementField[s,r,c,0]=(b[0])/nrm2
                            displacementField[s,r,c,1]=(b[1])/nrm2
                            displacementField[s,r,c,2]=(b[2])/nrm2
                else:
                    y[0]=b[0] + sigma*lambdaParam*y[0]
                    y[1]=b[1] + sigma*lambdaParam*y[1]
                    y[2]=b[2] + sigma*lambdaParam*y[2]
                    A[0]=gradientField[s,r,c,0]*gradientField[s,r,c,0] + sigma*lambdaParam*nn
                    A[1]=gradientField[s,r,c,0]*gradientField[s,r,c,1]
                    A[2]=gradientField[s,r,c,0]*gradientField[s,r,c,2]
                    A[3]=gradientField[s,r,c,1]*gradientField[s,r,c,1] + sigma*lambdaParam*nn
                    A[4]=gradientField[s,r,c,1]*gradientField[s,r,c,2]
                    A[5]=gradientField[s,r,c,2]**2 + sigma*lambdaParam*nn
                    xx=displacementField[s,r,c,0]
                    yy=displacementField[s,r,c,1]
                    zz=displacementField[s,r,c,2]
                    solve3DSymmetricPositiveDefiniteSystem(A,y,d)
                    displacementField[s,r,c,0] = d[0]
                    displacementField[s,r,c,1] = d[1]
                    displacementField[s,r,c,2] = d[2]
                    xx-=displacementField[s,r,c,0]
                    yy-=displacementField[s,r,c,1]
                    zz-=displacementField[s,r,c,2]
                    opt=xx*xx+yy*yy+zz*zz;
                    if(maxDisplacement<opt):
                        maxDisplacement=opt
    return sqrt(maxDisplacement)

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_energy_SSD3D(floating[:,:,:] deltaField, floating[:,:,:] sigmaField, floating[:,:,:,:] gradientField,  floating lambdaParam, floating[:,:,:,:] displacementField):
    cdef int nslices=deltaField.shape[0]
    cdef int nrows=deltaField.shape[1]
    cdef int ncols=deltaField.shape[2]
    cdef floating energy = 0
    for s in range(nslices):
        for r in range(nrows):
            for c in range(ncols):
                energy+=deltaField[s,r,c]**2
    return energy

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_residual_displacement_field_SSD3D(floating[:,:,:] deltaField, floating[:,:,:] sigmaField, floating[:,:,:,:] gradientField,  floating[:,:,:,:] target, floating lambdaParam, floating[:,:,:,:] displacementField, floating[:,:,:,:] residual):
    cdef int NUM_NEIGHBORS = 6 
    cdef int[:] dSlice = np.array([-1,  0, 0, 0,  0, 1], dtype=np.int32)
    cdef int[:] dRow = np.array([0, -1, 0, 1,  0, 0], dtype=np.int32)
    cdef int[:] dCol = np.array([0,  0, 1, 0, -1, 0], dtype=np.int32)
    cdef floating[:] b = np.ndarray(shape=(3,), dtype=cython.typeof(deltaField[0,0,0]))
    cdef floating[:] y = np.ndarray(shape=(3,), dtype=cython.typeof(deltaField[0,0,0]))
    cdef int nslices=deltaField.shape[0]
    cdef int nrows=deltaField.shape[1]
    cdef int ncols=deltaField.shape[2]
    cdef floating delta, sigma, dotP
    cdef int s,r,c,ds,dr,dc
    if residual==None:
        residual=np.empty(shape=(nslices, nrows, ncols, 3), dtype=np.double)
    for s in range(nslices):
        for r in range(nrows):
            for c in range(ncols):
                delta=deltaField[s,r,c]
                sigma=sigmaField[s,r,c] if sigmaField!=None else 1
                if(target==None):
                    b[0]=delta*gradientField[s,r,c,0]
                    b[1]=delta*gradientField[s,r,c,1]
                    b[2]=delta*gradientField[s,r,c,2]
                else:
                    b[0]=target[s,r,c,0]
                    b[1]=target[s,r,c,1]
                    b[2]=target[s,r,c,2]
                y[...]=0
                for k in range(NUM_NEIGHBORS):
                    ds=s+dSlice[k];
                    if((ds<0) or (ds>=nslices)):
                        continue
                    dr=r+dRow[k]
                    if((dr<0) or (dr>=nrows)):
                        continue
                    dc=c+dCol[k]
                    if((dc<0) or (dc>=ncols)):
                        continue
                    y[0]+=displacementField[s,r,c,0]-displacementField[ds, dr, dc, 0]
                    y[1]+=displacementField[s,r,c,1]-displacementField[ds, dr, dc, 1]
                    y[2]+=displacementField[s,r,c,2]-displacementField[ds, dr, dc, 2]
                if(isinf(sigma)):
                    residual[s,r,c,0]=-lambdaParam*y[0]
                    residual[s,r,c,1]=-lambdaParam*y[1]
                    residual[s,r,c,2]=-lambdaParam*y[2]
                else:
                    dotP=gradientField[s,r,c,0]*displacementField[s,r,c,0]+gradientField[s,r,c,1]*displacementField[s,r,c,1]+gradientField[s,r,c,2]*displacementField[s,r,c,2]
                    residual[s,r,c,0]=b[0]-(gradientField[s,r,c,0]*dotP+sigma*lambdaParam*y[0])
                    residual[s,r,c,1]=b[1]-(gradientField[s,r,c,1]*dotP+sigma*lambdaParam*y[1])
                    residual[s,r,c,2]=b[2]-(gradientField[s,r,c,2]*dotP+sigma*lambdaParam*y[2])
    return residual

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_residual_displacement_field_SSD2D(floating[:,:] deltaField, floating[:,:] sigmaField, floating[:,:,:] gradientField,  floating[:,:,:] target, floating lambdaParam, floating[:,:,:] displacementField, floating[:,:,:] residual):
    cdef int NUM_NEIGHBORS = 4
    cdef int[:] dRow = np.array([-1, 0, 1,  0], dtype=np.int32)
    cdef int[:] dCol = np.array([ 0, 1, 0, -1], dtype=np.int32)
    cdef floating[:] b = np.ndarray(shape=(2,), dtype=cython.typeof(deltaField[0,0]))
    cdef floating[:] y = np.ndarray(shape=(2,), dtype=cython.typeof(deltaField[0,0]))
    cdef int nrows=deltaField.shape[0]
    cdef int ncols=deltaField.shape[1]
    cdef floating delta, sigma, dotP
    cdef int r,c,dr,dc
    if residual==None:
        residual=np.empty(shape=(nrows, ncols, 2), dtype=np.double)
    for r in range(nrows):
        for c in range(ncols):
            delta=deltaField[r,c]
            sigma=sigmaField[r,c] if sigmaField!=None else 1
            if(target==None):
                b[0]=delta*gradientField[r,c,0]
                b[1]=delta*gradientField[r,c,1]
            else:
                b[0]=target[r,c,0]
                b[1]=target[r,c,1]
            y[...]=0
            for k in range(NUM_NEIGHBORS):
                dr=r+dRow[k]
                if((dr<0) or (dr>=nrows)):
                    continue
                dc=c+dCol[k]
                if((dc<0) or (dc>=ncols)):
                    continue
                y[0]+=displacementField[r,c,0]-displacementField[dr, dc, 0]
                y[1]+=displacementField[r,c,1]-displacementField[dr, dc, 1]
            if(isinf(sigma)):
                residual[r,c,0]=-lambdaParam*y[0]
                residual[r,c,1]=-lambdaParam*y[1]
            else:
                dotP=gradientField[r,c,0]*displacementField[r,c,0]+gradientField[r,c,1]*displacementField[r,c,1]
                residual[r,c,0]=b[0]-(gradientField[r,c,0]*dotP+sigma*lambdaParam*y[0])
                residual[r,c,1]=b[1]-(gradientField[r,c,1]*dotP+sigma*lambdaParam*y[1])
    return residual

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_demons_step2D(floating[:,:] deltaField, floating[:,:,:] gradientField, floating maxStepSize):
    cdef int nrows=deltaField.shape[0]
    cdef int ncols=deltaField.shape[1]
    cdef int r,c
    cdef floating nrm2, delta, den, factor, maxDisplacement
    cdef floating[:,:,:] demonsStep=np.zeros(shape=(nrows, ncols, 3), dtype=cython.typeof(deltaField[0,0]))
    maxDisplacement=0
    for r in range(nrows):
        for c in range(ncols):
            nrm2=gradientField[r,c,0]**2+gradientField[r,c,1]*2
            delta=deltaField[r,c]
            den=(nrm2 + delta*delta)
            factor=0;
            if(den!=0):
                factor=delta/den
            demonsStep[r,c,0]=factor*gradientField[r,c,0]
            demonsStep[r,c,1]=factor*gradientField[r,c,1]
            nrm2=demonsStep[r,c,0]**2+demonsStep[r,c,1]**2
            if(maxDisplacement<nrm2):
                maxDisplacement=nrm2
    maxDisplacement=sqrt(maxDisplacement)
    factor=maxStepSize/maxDisplacement
    for r in range(nrows):
        for c in range(ncols):
            demonsStep[r,c,0]*=factor
            demonsStep[r,c,1]*=factor
    return demonsStep

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_demons_step3D(floating[:,:,:] deltaField, floating[:,:,:,:] gradientField,  floating maxStepSize):
    cdef int nslices=deltaField.shape[0]
    cdef int nrows=deltaField.shape[1]
    cdef int ncols=deltaField.shape[2]
    cdef int s,r,c
    cdef floating nrm2, delta, den, factor, maxDisplacement
    cdef floating[:,:,:,:] demonsStep=np.zeros(shape=(nslices, nrows, ncols, 3), dtype=cython.typeof(deltaField[0,0,0]))
    maxDisplacement=0
    for s in range(nslices):
        for r in range(nrows):
            for c in range(ncols):
                nrm2=gradientField[s,r,c,0]**2+gradientField[s,r,c,1]*2+gradientField[s,r,c,2]**2
                delta=deltaField[s,r,c]
                den=(nrm2 + delta*delta)
                factor=0;
                if(den!=0):
                    factor=delta/den
                demonsStep[s,r,c,0]=factor*gradientField[s,r,c,0]
                demonsStep[s,r,c,1]=factor*gradientField[s,r,c,1]
                demonsStep[s,r,c,2]=factor*gradientField[s,r,c,2]
                nrm2=demonsStep[s,r,c,0]**2+demonsStep[s,r,c,1]**2+demonsStep[s,r,c,2]**2
                if(maxDisplacement<nrm2):
                    maxDisplacement=nrm2
    maxDisplacement=sqrt(maxDisplacement)
    factor=maxStepSize/maxDisplacement
    for s in range(nslices):
        for r in range(nrows):
            for c in range(ncols):
                demonsStep[s,r,c,0]*=factor
                demonsStep[s,r,c,1]*=factor
                demonsStep[s,r,c,2]*=factor
    return demonsStep
