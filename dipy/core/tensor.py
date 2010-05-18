#!/usr/bin/python

"""

USAGE

    As a script ... (::WARNING NOT REALLY SUGGESTED AND NOT SUPPORTED::)
        tensor [-h,--help] [-v,--verbose] [--version] <data> <gtab> <bvalue>
    
    As a imported module ...
        fit_data,scalars_map = tensor(data,gtab,bvalue,img=img,out_root=out_root,scalars=scalars)

DESCRIPTION

    tensor does...

    INPUTS:
        <data>          = 4D image matrix
        <gtab>      = gradient table [3 x N]
        <bval>      = b value vector [N x 1]
        <scalars>   = 1 : save scalars

    OUTPUTS
        fit_data    = fitted 4D image matrix
        scalars_map = 4D image matrix [x,y,z, scalars]

EXAMPLES



AUTHOR(S)

    Christopher Nguyen <christopher.nguyen@radiology.ucsf.edu>


LICENSE


VERSION

    1.0

"""

#import modules
import time
import sys, os, traceback, optparse


def main (args):
    #global options, args
    #This is to control external call of module from command line

    tensor(args[0],args[1],args[2])    

### TO DO CHANGE THIS TO APPROPRIATE NAME
#This is what is used for import
def tensor (data,gtab,bval,scalars=0,thresh=25,img=[],out_root='noroot'):

    #Import modules here
    import time
    import numpy as np
    import scipy as sp
    
    #for io of writing and reading nifti images
    from nipy import load_image, save_image
    from nipy.core.api import fromarray #data --> image
    
    #HBCAN utilities
    sys.path.append('/home/cnguyen/scripts/HBCAN/utils')
    from multi_dot import multi_dot as multi_dot
    from io import get_coord_4D_to_3D

    #Make sure all input is correct
    try:

        start_time = time.time()
        print 'out_root: ' + out_root
        print 'thresh: ' + str(thresh)
        print ''

        if data == []:
            raise ValueError('You need to send an image matrix, grad table, and b value vector)')

        if gtab == []:
            raise ValueError('Please provide a gradient table')

        if bval == []:
            raise ValueError('Please provide a b value vector')

        if np.size(gtab)/3 != np.size(bval):
            raise ValueError('Gradient table [3 x N] must be consistent with given b value vector [N x 1]')
    except:
        raise IOError('You need to send an image matrix, grad table, and b value vector')

        
    ####main part of code
    fit_data = np.zeros(np.shape(data))
    dims = np.shape(data)

    print 'Performing tensor model fit ... '
    print ''

    ###Create log of signal and reshape it to be x:y:z by grad
    fit_dim = (dims[0]*dims[1]*dims[2],dims[3]) 
    fit_data = np.reshape(fit_data,fit_dim )
    data = np.reshape(data,fit_dim)
    if scalars == 1:
        scalar_maps = np.zeros((np.size(fit_data,axis=0),8))
    tensor_data = np.zeros((np.size(fit_data,axis=0),6))

    # Y matrix from Chris' paper
    # [g by x*y*z]
    data[np.where(data <= 0)] = 1
    log_s = np.transpose(np.log(data))

    ###Construct design matrix
    #For DTI this is the so called B matrix
    # X matrix from Chris' paper
    print 'Constructing tensor design matrix (B matrix) ...'
    print ''
    B = design_matrix(gtab,bval) # [g by 7]
    B_t = np.transpose(B)        # [7 by g]
    
    ###Weighted Least Squares (WLS) to solve "linear" regression
    
    # Y hat OLS from Chris' paper
    # ORIG IDL CODE: log_s_ols = B ## invert( transpose(B) ## B ,/double) ## transpose(B) ## log_s
    # log_s_ols = np.dot(B, np.dot(np.linalg.inv(np.dot(B_t,B)), np.dot(B_t,log_s)))
    #  [g by 7] [7 by 7 ] [7 by g] [ g by x*y*z ] = [g by x*y*z]
    log_s_ols = multi_dot((B, np.linalg.inv(np.dot(B_t,B)), B_t, log_s)) 
   
    ##inv_B_t_W_B = np.zeros((fit_dim[0],7,7))
    ##for i in range(np.size(log_s,axis=1):
    ##    #Weighting factor from WLS
    ##    # should be [x*y*z by g by g]
    ##    W = sp.diag(np.exp(log_s_ols[:,i])**2)
    ##
    ##    #inv[[7 by g] [g by g] [g by 7]] = [7 by 7] 
    ##    inv_B_t_W_B[i,:,:] = np.linalg.inv(multidot((B_t,W,B)))
    ##
    ## Beta matrix from Chris' paper
    ## D = [x*y*z by 6] : the 2nd dim 6 parameters are [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz]
    ## [x*y*z by 7 by 7] [7 by g] [g by g] [g by x*y*z] = [7 by x*y*z]
    #D = multi_dot((inv_B_t_W_B, B_t, W, log_s))
    ##
    ## Y hat matrix from Chris' paper
    ##fit_data = np.dot(B, D)

    # This step is because we cannot vectorize diagonal vector and tensor fit
    print 'Calculating voxelwise diagonal weighting matrix and tensor fit ... '
    print
    time_diff = list((0,0))
    for i in range(np.size(log_s,axis=1)):
        #check every 5 slices (pretty robust time left...prob make into separate module if have time)
        if np.mod(i,dims[0]*dims[1]*5) == 0:
            slice = i/dims[0]/dims[1]+1.
            time_diff.append(time.time()-start_time)
            min = np.mean(time_diff[2:len(time_diff)]) / 60.0 /5 *(dims[2] - slice)
            sec = np.round((min - np.fix(min)) * 60.0/5)
            min = np.fix(min)
            percent = 100.*slice/dims[2]
            print str(np.round(percent)) + '% ... Approx time left ' + str(min) + ' MIN ' + str(sec) + ' SEC'

        if data[i,0] < thresh:
            continue
        #if not finite move on
        if not(np.unique(np.isfinite(log_s[:,i]))[0]) :
            continue

        #Weighting factor from WLS
        #[g by g]
        W = sp.diag(np.exp(log_s_ols[:,i])**2)

        # Beta matrix from Chris' paper
        # D = [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz,dummy]
        # inv[[7 by g] [g by g] [g by 7]] [7 by g] [g by g] [g by 1] = [7 by 1]
        #ORIG IDL CODE: D = invert( transpose(B) ## W ## B ,/double) ## transpose(B) ## W ## log_s[i,:]
        D = multi_dot((np.linalg.inv(multi_dot((B_t,W,B))), B_t, W, log_s[:,i]))  

        # Y hat matrix from Chris' paper
        fit_data[i,:] = np.dot(B, D)
        tensor_data[i,:] = D[0:6]*1000    
        
        ###Calculate scalar maps
        if scalars == 1:
            scalar_maps[i,:] = calc_dti_scalars(D)

    # Reshape the output images
    fit_data = np.reshape(fit_data,dims)
    data = np.reshape(data,dims)
    tensor_data = np.reshape(tensor_data,(dims[0],dims[1],dims[2],6))
    
    #If requesting to save scalars ...
    if scalars == 1:
        #Reshape the scalar map array
        scalar_maps = np.reshape(scalar_maps,(dims[0],dims[1],dims[2],8))

        #For writing out with save_image with appropriate affine matrix
        coordmap = []
    
        if img != []:
            coordmap = get_coord_4D_to_3D(img.affine)
            header = img.header.copy()

        ###Save scalar maps if requested
        print ''
        print 'Saving t2di map ... '+out_root+'_t2di.nii.gz'
        
        #fyi the dtype flag for save image does not appear to work right now...
        t2di_img = fromarray(data[:,:,:,0],'ijk','xyz',coordmap=coordmap)
        if img != []: 
            t2di_img.header = header
        save_image(t2di_img,out_root+'_t2di.nii.gz',dtype=np.int16)

        
        scalar_fnames = ('ev1','ev2','ev3','adc','fa','ev1p','ev1f','ev1s')
        for i in range(np.size(scalar_maps,axis=3)):
            #need to send in 4 x 4 affine matrix for 3D image not 5 x 5 from original 4D image
            print 'Saving '+ scalar_fnames[i] + ' map ... '+out_root+'_'+scalar_fnames[i]+'.nii.gz'
            scalar_img = fromarray(np.int16(scalar_maps[:,:,:,i]),'ijk' ,'xyz',coordmap=coordmap)
            if img != []:
                scalar_img.header = header
            save_image(scalar_img,out_root+'_'+scalar_fnames[i]+'.nii.gz',dtype=np.int16)

    print ''
    print 'Saving D = [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz] map ... '+out_root+'_self_diffusion.nii.gz'
    #Saving 4D matrix holding diffusion coefficients
    if img != [] :
        coordmap = img.coordmap
        header = img.header.copy()
    tensor_img = fromarray(tensor_data,'ijkl','xyzt',coordmap=coordmap)
    tensor_img.header = header
    save_image(tensor_img,out_root+'_self_diffusion.nii.gz',dtype=np.int16)

    print
    min = (time.time() - start_time) / 60.0
    sec = (min - np.fix(min)) * 60.0
    print 'TOTAL TIME: ' + str(np.fix(min)) + ' MIN ' + str(np.round(sec)) + ' SEC'

    return(fit_data, scalar_maps)


def calc_dti_scalars(D):
    import numpy as np

    tensor = np.zeros((3,3))
    tensor[0,0] = D[0]  #Dxx
    tensor[1,1] = D[1]  #Dyy
    tensor[2,2] = D[2]  #Dzz
    tensor[1,0] = D[3]  #Dxy
    tensor[2,0] = D[4]  #Dxz
    tensor[2,1] = D[5]  #Dyz
    tensor[0,1] = tensor[1,0] #Dyx
    tensor[0,2] = tensor[2,0] #Dzx
    tensor[1,2] = tensor[2,1] #Dzy

    #outputs multiplicity as well so need to unique
    # eigenvecs[:,i] corresponds to eigenvals[i]
    eigenvals, eigenvecs = np.linalg.eig(tensor)

    if np.size(eigenvals) != 3:
        raise ValueError('not 3 eigenvalues : ' + str(eigenvals))

    ev1 = eigenvals[0]
    ev2 = eigenvals[1]
    ev3 = eigenvals[2]
    #eigenvecs = np.transpose(eigenvecs)
    
	#calculate scalars
    adc = (ev1+ev2+ev3)/3
    ss_ev = ev1**2+ev2**2+ev3**2
    fa = 0
    if ss_ev > 0 :
        fa  = np.sqrt( 1.5 * ( (ev1-adc)**2+(ev2-adc)**2+(ev3-adc)**2 ) / ss_ev )
    evt = (ev2+ev3)/2
    tsi = 0.5
    if ev1 != ev3 :
        tsi = (ev1-ev2)/(ev1-ev3)

    # fyi b ~ 10^3 s/mm^2 and D ~ 10^-4 mm^2/s
    dti_parameters = np.round([	ev1*10000, ev2*10000, ev3*10000, adc*10000,fa*1000,
					eigenvecs[0,0]*100+1000, eigenvecs[1,0]*100+1000,eigenvecs[2,0]*100+1000]) 
                    #,evt,tsi]
    return(dti_parameters)

def design_matrix(gtab,bval):
    import numpy as np

    #from CTN legacy IDL we start with [7 by g] ... sorry :(
    B = np.zeros((7,np.size(bval)))
    G = gtab
    
    if np.size(gtab,axis=1) < np.size(bval) :
        print 'Gradient table size is not consistent with bval vector... could be b/c of b0 images'
        print 'Will try to set nonzero bval index with gradient table to construct B matrix'
        
        G = np.zeros((3,np.size(bval)))
        G[:,np.where(bval > 0)]=gtab
    
    B[0,:] = G[0,:]*G[0,:]*1.*bval   #Bxx
    B[1,:] = G[1,:]*G[1,:]*1.*bval   #Byy
    B[2,:] = G[2,:]*G[2,:]*1.*bval   #Bzz
    B[3,:] = G[0,:]*G[1,:]*2.*bval   #Bxy
    B[4,:] = G[0,:]*G[2,:]*2.*bval   #Bxz
    B[5,:] = G[1,:]*G[2,:]*2.*bval   #Byz
    B[6,:] = np.ones(np.size(bval))
    
    #Need to return [g by 7]
    return (np.transpose(-B))



#ARGUMENT HANDLING ETC ...
if __name__ == '__main__':
    try:
        start_time = time.time()
        parser = optparse.OptionParser(formatter=optparse.TitledHelpFormatter(), 
                                        usage=globals()['__doc__'], version='1.0')
        parser.add_option ('-v', '--verbose', action='store_true', default=False,
                                        help='verbose output')

        (options, args) = parser.parse_args()
        if len(args) < 1:
            parser.error ('missing argument')
        if options.verbose: print time.asctime()
        
        main(args)

        if options.verbose: print time.asctime()
        if options.verbose: print 'TOTAL TIME IN MINUTES:',
        if options.verbose: print (time.time() - start_time) / 60.0
        sys.exit(0)
    except KeyboardInterrupt, e: # Ctrl-C
        raise e
    except SystemExit, e: # sys.exit()
        raise e
    except Exception, e:
        print 'ERROR, UNEXPECTED EXCEPTION'
        print str(e)
        traceback.print_exc()
        os._exit(1)
                                                            
