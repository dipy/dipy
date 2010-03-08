#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import dicom
except ImportError:
    ImportError('Pydicom is not installed.')
    ImportError('http://code.google.com/p/pydicom/')

try:
    from numpy import linalg as lg
    import numpy as np
except ImportError:
    ImportError('numpy is not installed.')
    
import os
import string
import struct


def list_files(dpath,filt='*'):
  ''' List files under directory path e.g. dpath='/tmp' with a specified filter e.g filt='.nii' .

  Parameters:
  -----------

  dpath: string, 
       contains the name of the directory path
  
  filt: string, 
       file filter i.e. '.dcm' is equivalent with *.dcm* in bash.
  
  Returns:
  --------

  lfiles: sequence,
       with the filenames off all files containing with the specified filter ``filt`` in the specified directory ``dpath``. 
  
  Examples:
  ---------
  Check for *.py* files in current directory

  >>> from dipy.io import bmatrix
  >>> bmatrix.list_files('.','.py')
  ['./build_helpers.py', './setup.py', './build_helpers.pyc']

  Check for any file in the current directory

  >>> bmatrix.list_files('.','*')
  ['./build_helpers.py',
  './build',
  './setup.py',
  './history.bak_sphinx',
  './.git',
  './doc',
  './dipy',
  './build_helpers.pyc',
  './Makefile',
  './.gitignore']
  

  Notes:
  ------
  I am sure that there are better ways to list files in a directory. Any ideas? 
  The filter can become more restricted. For example the filter can become more restricted so it allows only *.py files and not *.py*.

  '''

  dirList=os.listdir(dpath)
  lfiles=[]

  if filt=='*':
    for fname in dirList:
        lfiles.append(dpath+'/'+fname)
  
  else:
    for fname in dirList:
        if fname.rfind(filt) > -1:
            lfiles.append(dpath+'/'+fname)

  return lfiles


def loadbinfodcm(filename,spm_converted=1):
    
    ''' Load B-value and B-vector information from the Dicom Header of a file. This assumes that the scanner is Siemens.
    The needed information is under the CSA Image Information Header in the Dicom header. At the moment only version SV10 is supported.
    
    This was inspired by the work of Williams & Brett using the following matlab script http://imaging.mrc-cbu.cam.ac.uk/svn/utilities/devel/cbu_dti_params.m
    However here we are using pydicom and then read directly from the CSA header.
    
    Parameters:
    -----------
    filename: string,
        Dicom full filename.

    spm_converted: ? flipping ?

    Returns:
    --------
    B_value (stored in the dicom), 
    B_vec (B_vector calculated from stored B_matrix), 
    G_direction(gradient direction stored in dicom), 
    B_value_B_matrix (B value calculated from the stored in dcm B_matrix after using eigenvalue decomposition).
    
    Example:
    ---------
    
    >>> B_value, B_vec, G_direction, B_value_B_matrix =  loadbinfodcm(fname)
    
    '''

    #filename = '/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/1.3.12.2.1107.5.2.32.35119.2009022715012276195181703.dcm'
    #filename = '/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/1.3.12.2.1107.5.2.32.35119.2009022715073976305795724.dcm'
    
    if os.path.isfile(filename)==False:
        
        print('Filename does not exist')
        
        return
        
    data=dicom.read_file(filename)

    if spm_converted:
        y_flipper=np.diag([1, -1, 1])
    else:
        y_flipper=np.eye(3)
           
    #print 'y_flipper',y_flipper
    
    orient=data.ImageOrientationPatient
    orient=np.transpose(np.reshape(orient,(2,3)))
       
    v1=np.array([orient[0,0],orient[1,0],orient[2,0]])
    v2=np.array([orient[0,1],orient[1,1],orient[2,1]])   
    v3=np.cross(v1,v2)
    
    #print 'v3',v3
    
    orient=np.column_stack((v1.transpose(),v2.transpose(),v3.transpose()))      

    if lg.det(orient<0):
        #print('Det orient < 0')
        print 'Negative determinant.'
        orient[:,2]=-orient[:,2]

    #print 'orient',orient

    vox_to_dicom = np.dot(orient, y_flipper)
    
    #print 'vox_to_dicom',vox_to_dicom
    mat = lg.inv(vox_to_dicom)
    
    #print 'mat',mat    
    #print vox_to_dicom*vox_to_dicom
    
    csainfo=data[0x029,0x1010]
    
    #print csainfo[0:4]
    
    if csainfo[0:4]!='SV10':
        print 'No SV10'
        
        B_vec=np.array([np.NaN,np.NaN,np.NaN])
        B_value=0
        G_direction=np.array([0.0,0.0,0.0])
        B_value_B_matrix=0
                
        return B_value, B_vec, G_direction, B_value_B_matrix        
                
    start,stop=8,12

    #print 'CSA Image Info'
    
    n=struct.unpack('I',csainfo[start:stop])
    n=n[0]
    print 'n:',n
        
    B_value=-1
    B_matrix=[]
    G_direction=[]
    no_mosaic=-1
    
        
    #Read B-related Info
    start=16
    for i in xrange(n):
        
        rec=[]
        
        stop=start+64+4+4+4+4+4
        name=struct.unpack('64ci4ciii',csainfo[start:stop])
        nitems=int(name[-2])
        startstore=start
        start =stop       
        
        #print(''.join(name[0:64]))
        #print(''.join(name[0:25]))
        matstart=0
        valstart=0
        diffgradstart=0
        mosaicstart=0
        
        if ''.join(name[0:8])=='B_matrix':
            matstart=startstore

        if ''.join(name[0:7])=='B_value':
            valstart=startstore        
        
        if ''.join(name[0:26])== 'DiffusionGradientDirection':
            diffgradstart=startstore
                
        if ''.join(name[0:22])== 'NumberOfImagesInMosaic':
            mosaicstart=startstore
            
        for j in xrange(nitems):
            
            xx=struct.unpack('4i',csainfo[start:start+4*4])
            length=int(xx[1])    
           
            value =struct.unpack(str(length)+'c',csainfo[start+4*4:start+4*4+length])                 
            
                           
            if matstart > 0:
                if len(value)>0:
                    B_matrix.append(float(''.join(value[:-1] )))
                else :
                    B_matrix.append(0.0)
                           
            if valstart > 0 :
                if len(value)>0:
                    B_value=float(''.join(value[:-1] ))

            if diffgradstart > 0 :
                if len(value)>0 :
                    G_direction.append(float(''.join(value[:-1] )))
                    
            if mosaicstart>0:
                if len(value)>0:
                    no_mosaic=int(''.join(value[:-1] ))
                        
            stop=start+4*4+length+(4-length%4)%4
            start=stop                 
       
    if B_value >0: 
        
        B_mat=np.array([[B_matrix[0],B_matrix[1],B_matrix[2]], [B_matrix[1],B_matrix[3],B_matrix[4]], [B_matrix[2],B_matrix[4],B_matrix[5]]])
        [vals, vecs]=lg.eigh(B_mat)       
       
        dbvec = vecs[:,2]
        
        if dbvec[0] < 0:
            dbvec = dbvec * -1
            
        B_vec=np.transpose(np.dot(mat, dbvec))         
        B_value_B_matrix=vals.max()
        
    else:
        
        B_vec=np.array([0.0,0.0,0.0])
        B_value=0
        G_direction=np.array([0.0,0.0,0.0])
        B_value_B_matrix=0
        no_mosaic=0
                
    return B_value, B_vec, G_direction, B_value_B_matrix, no_mosaic

    
def loadbinfodir(dirname):
    '''
    Load information about b-values and b-vectors from a directory with multiple dicom files. Returns a list (binfo) were every node holds 
    the following info B_value, B_vec[0], B_vec[1], B_vec[2], G_direction[0], G_direction[1], G_direction[2], B_value_B_matrix.
    
    Examples:
    --------
    
    dirname='/backup/Data/Eleftherios/CBU090134_METHODS/20090227_154122/Series_003_CBU_DTI_64D_iso_1000'
    binfo=loadbinfodir(dirname)
    savebinfo(binfo=binfo) 
    

    Notes:
    ------

    This snippet outputs three files binfo.txt, bvecs.txt, bvals.txt
    '''
    if os.path.isdir(dirname):
        pass
    else:
        print 'No Directory found'
    
    lfiles=list_files(dirname,filt='.dcm')
    lfiles.sort()
    binfo=[]
    row=[]
    
    for fname in lfiles:
        #print fname
        #print loadbinfodcm(fname)
        B_value, B_vec, G_direction, B_value_B_matrix = loadbinfodcm(fname)
        row=[B_value, B_vec[0], B_vec[1], B_vec[2], G_direction[0], G_direction[1], G_direction[2], B_value_B_matrix]
        
        binfo.append(row)

    return binfo

def savebinfo(filename='binfo.txt',binfo=[],fnamebvecs='bvecs.txt',fnamebvals='bvals.txt'):
    '''
    Every row in binfo has the following information
    [B_value, B_vec[0], B_vec[1], B_vec[2], G_direction[0], G_direction[1], G_direction[2], B_value_B_matrix]
    '''
    if binfo==[]:
        print 'Binfo list is needed.'
        return
    
    f = open(filename,'w')
    f2= open(fnamebvecs,'w')    
    f3= open(fnamebvals,'w')
    
    for i in binfo:

        f.write(str(i[0]));  f.write(' ');  f.write(str(i[1])); f.write(' ');  f.write(str(i[2])); f.write(' ');  
        f.write(str(i[3]));  f.write(' ');  f.write(str(i[4])); f.write(' ');  f.write(str(i[5])); f.write(' ');
        f.write(str(i[6]));  f.write(' ');  f.write(str(i[7])); f.write('\n')   
       
        
        f3.write(str(i[7]))
        f3.write('   ')
    
    f3.write('\n')    
    
    f.close()
    f3.close()
    
    Binfo=np.array(binfo)
    bvecsX=Binfo[:,1]
    bvecsY=Binfo[:,2]
    bvecsZ=Binfo[:,3]
    
    for i in bvecsX:
        f2.write(str(i)); f2.write('   ')
    for i in bvecsY:
        f2.write(str(i)); f2.write('   ')
    for i in bvecsZ:
        f2.write(str(i)); f2.write('   ')
        
    f2.write('\n')
    
    f2.close()
        
    return    

def loadbvals(filename):
    '''
    Loads B-values from a txt file named usually bvals
    '''

    bvals = []
    for line in open(filename, 'rt'):
        bvals =[float(val) for val in line.split()]            
    bvals=np.array(bvals)
    return bvals
    
def loadbvecs(filename):    
    '''
    Loads B-vecs from a txt file named bvecs
    '''
    
    bvecs = []
    for line in open(filename, 'rt'):
        bvecs.append([float(val) for val in line.split()])
    bvecs=np.array(bvecs)
    
    return bvecs

def loaddata(filename):    
    '''
    Loads row col data
    '''
    
    vecs = []
    for line in open(filename, 'rt'):
        vecs.append([float(val) for val in line.split()])
    vecs=np.array(vecs)
    
    return vecs

  

if __name__ == "__main__":    

    dirname='/backup/Data/Eleftherios/CBU090134_METHODS/20090227_154122/Series_003_CBU_DTI_64D_iso_1000'
    binfo=loadbinfodir(dirname)
    savebinfo(binfo=binfo) 



    
