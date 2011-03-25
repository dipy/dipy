import nibabel
import os
import numpy as np
import dipy as dp
#import dipy.core.generalized_q_sampling as dgqs
import dipy.reconst.gqi as dgqs
import dipy.reconst.dti as ddti
import dipy.reconst.recspeed as rp

import dipy.io.pickles as pkl
import scipy as sp
from matplotlib.mlab import find
#import dipy.core.sphere_plots as splots
import dipy.core.sphere_stats as sphats
import dipy.core.geometry as geometry
import get_vertices as gv

#old SimData files
'''
results_SNR030_1fibre
results_SNR030_1fibre+iso
results_SNR030_2fibres_15deg
results_SNR030_2fibres_30deg
results_SNR030_2fibres_60deg
results_SNR030_2fibres_90deg
results_SNR030_2fibres+iso_15deg
results_SNR030_2fibres+iso_30deg
results_SNR030_2fibres+iso_60deg
results_SNR030_2fibres+iso_90deg
results_SNR030_isotropic
'''
#fname='/home/ian/Data/SimData/results_SNR030_1fibre'
''' file  has one row for every voxel, every voxel is repeating 1000
times with the same noise level , then we have 100 different
directions. 1000 * 100 is the number of all rows.

The 100 conditions are given by 10 polar angles (in degrees) 0, 20, 40, 60, 80,
80, 60, 40, 20 and 0, and each of these with longitude angle 0, 40, 80,
120, 160, 200, 240, 280, 320, 360. 
'''

#new complete SimVoxels files
simdata = ['fibres_2_SNR_80_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_60_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_40_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_40_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_20_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_100_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_20_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_40_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_60_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_100_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_60_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_80_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_100_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_100_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_80_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_60_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_40_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_80_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_20_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_60_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_100_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_100_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_20_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_1_SNR_20_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_40_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_20_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_80_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_80_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_20_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_60_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_100_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_80_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_60_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_20_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_100_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_20_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_80_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_80_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_100_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_40_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_1_SNR_60_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_40_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_60_angle_30_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_40_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_60_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_80_angle_15_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_1_SNR_40_angle_00_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_100_angle_60_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00',
 'fibres_2_SNR_40_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_1_diso_0.7',
 'fibres_2_SNR_20_angle_90_l1_1.4_l2_0.35_l3_0.35_iso_0_diso_00']


simdir = '/home/ian/Data/SimVoxels/'

def gq_tn_calc_save():

    for simfile in simdata:
    
        dataname = simfile
        print dataname

        sim_data=np.loadtxt(simdir+dataname)

        marta_table_fname='/home/ian/Data/SimData/Dir_and_bvals_DSI_marta.txt'
        b_vals_dirs=np.loadtxt(marta_table_fname)
        bvals=b_vals_dirs[:,0]*1000
        gradients=b_vals_dirs[:,1:]

        gq = dgqs.GeneralizedQSampling(sim_data,bvals,gradients)
        gqfile = simdir+'gq/'+dataname+'.pkl'
        pkl.save_pickle(gqfile,gq)

        '''
        gq.IN               gq.__doc__          gq.glob_norm_param
        gq.QA               gq.__init__         gq.odf              
        gq.__class__        gq.__module__       gq.q2odf_params
        '''

        tn = ddti.Tensor(sim_data,bvals,gradients)
        tnfile = simdir+'tn/'+dataname+'.pkl'
        pkl.save_pickle(tnfile,tn)


        '''
        tn.ADC               tn.__init__          tn._getevals
        tn.B                 tn.__module__        tn._getevecs
        tn.D                 tn.__new__           tn._getndim
        tn.FA                tn.__reduce__        tn._getshape
        tn.IN                tn.__reduce_ex__     tn._setevals
        tn.MD                tn.__repr__          tn._setevecs
        tn.__class__         tn.__setattr__       tn.adc
        tn.__delattr__       tn.__sizeof__        tn.evals
        tn.__dict__          tn.__str__           tn.evecs
        tn.__doc__           tn.__subclasshook__  tn.fa
        tn.__format__        tn.__weakref__       tn.md
        tn.__getattribute__  tn._evals            tn.ndim
        tn.__getitem__       tn._evecs            tn.shape
        tn.__hash__          tn._getD             
        '''

        ''' file  has one row for every voxel, every voxel is repeating 1000
        times with the same noise level , then we have 100 different
        directions. 100 * 1000 is the number of all rows.

        At the moment this module is hardwired to the use of the EDS362
        spherical mesh. I am assumung (needs testing) that directions 181 to 361
        are the antipodal partners of directions 0 to 180. So when counting the
        number of different vertices that occur as maximal directions we wll map
        the indices modulo 181.
        '''

def analyze_maxima(indices, max_dirs, subsets):
    '''This calculates the eigenstats for each of the replicated batches
    of the simulation data
    '''

    results = []


    for direction in subsets:

        batch = max_dirs[direction,:,:]

        index_variety = np.array([len(set(np.remainder(indices[direction,:],181)))])

        #normed_centroid, polar_centroid, centre, b1 = sphats.eigenstats(batch)
        centre, b1 = sphats.eigenstats(batch)
        
        # make azimuth be in range (0,360) rather than (-180,180) 
        centre[1] += 360*(centre[1] < 0)
            
        #results.append(np.concatenate((normed_centroid, polar_centroid, centre, b1, index_variety)))
        results.append(np.concatenate((centre, b1, index_variety)))

    return results

#dt_first_directions = tn.evecs[:,:,0].reshape((100,1000,3))
# these are the principal directions for the full set of simulations


#gq_tn_calc_save()

#eds=np.load(os.path.join(os.path.dirname(dp.__file__),'core','matrices','evenly_distributed_sphere_362.npz'))
from dipy.data import get_sphere

odf_vertices,odf_faces=get_sphere('symmetric362')

#odf_vertices=eds['vertices']

def run_comparisons(sample_data=35):
    for simfile in [simdata[sample_data]]:
    
        dataname = simfile
        print dataname
    
        sim_data=np.loadtxt(simdir+dataname)
    
        gqfile = simdir+'gq/'+dataname+'.pkl'
        gq =  pkl.load_pickle(gqfile)
        tnfile = simdir+'tn/'+dataname+'.pkl'
        tn =  pkl.load_pickle(tnfile)
    
    
        dt_first_directions_in=odf_vertices[tn.IN]
    
        dt_indices = tn.IN.reshape((100,1000))
        dt_results = analyze_maxima(dt_indices, dt_first_directions_in.reshape((100,1000,3)),range(10,90))
    
        gq_indices = np.array(gq.IN[:,0],dtype='int').reshape((100,1000))
    
        gq_first_directions_in=odf_vertices[np.array(gq.IN[:,0],dtype='int')]
    
        #print gq_first_directions_in.shape
    
        gq_results = analyze_maxima(gq_indices, gq_first_directions_in.reshape((100,1000,3)),range(10,90))
    
        #for gqi see example dicoms_2_tracks gq.IN[:,0]
    
        np.set_printoptions(precision=3, suppress=True, linewidth=200, threshold=5000)
    
        out = open('/home/ian/Data/SimVoxels/Out/'+'***_'+dataname,'w')
    
        #print np.vstack(dt_results).shape, np.vstack(gq_results).shape
        
        results = np.hstack((np.vstack(dt_results), np.vstack(gq_results)))
        #print results.shape
        #results = np.vstack(dt_results)
    
        print >> out, results[:,:]
    
        out.close()
    
    
        #up = dt_batch[:,2]>= 0
    
        #splots.plot_sphere(dt_batch[up], 'batch '+str(direction))
    
        #splots.plot_lambert(dt_batch[up],'batch '+str(direction), centre)
        
        #spread = gq.q2odf_params e,v = np.linalg.eigh(np.dot(spread,spread.transpose())) effective_dimension = len(find(np.cumsum(e) > 0.05*np.sum(e))) #95%
    
        #rotated = np.dot(dt_batch,evecs)
    
        #rot_evals, rot_evecs =  np.linalg.eig(np.dot(rotated.T,rotated)/rotated.shape[0])
    
        #eval_order = np.argsort(rot_evals)
    
        #rotated = rotated[:,eval_order]
    
        #up = rotated[:,2]>= 0
    
        #splot.plot_sphere(rotated[up],'first1000')
    
        #splot.plot_lambert(rotated[up],'batch '+str(direction))

def run_gq_sims(sample_data=[35,23,46,39,40,10,37,27,21,20]):

    results = []

    out = open('/home/ian/Data/SimVoxels/Out/'+'npa+fa','w')

    for j in range(len(sample_data)):
        
        sample = sample_data[j]

        simfile = simdata[sample]
    
        dataname = simfile
        print dataname
    
        sim_data=np.loadtxt(simdir+dataname)
    
        marta_table_fname='/home/ian/Data/SimData/Dir_and_bvals_DSI_marta.txt'
        b_vals_dirs=np.loadtxt(marta_table_fname)
        bvals=b_vals_dirs[:,0]*1000
        gradients=b_vals_dirs[:,1:]


        for j in np.vstack((np.arange(100)*1000,np.arange(100)*1000+1)).T.ravel():
        # 0,1,1000,1001,2000,2001,...
        
            s = sim_data[j,:]

            gqs = dp.GeneralizedQSampling(s.reshape((1,102)),bvals,gradients,Lambda=3.5)
            tn = dp.Tensor(s.reshape((1,102)),bvals,gradients,fit_method='LS')
    
            t0, t1, t2, npa = gqs.npa(s, width = 5)
            
            print >> out, dataname, j, npa, tn.fa()[0]
            
            '''
            for (i,o) in enumerate(gqs.odf(s)):
                print i,o
            
            for (i,o) in enumerate(gqs.odf_vertices):
                print i,o
            '''
            #o = gqs.odf(s)
            #v = gqs.odf_vertices
            #pole = v[t0[0]]
            #eqv = dgqs.equatorial_zone_vertices(v, pole, 5)
            #print 'Number of equatorial vertices: ', len(eqv)
            #print np.max(o[eqv]),np.min(o[eqv])
            #cos_e_pole = [np.dot(pole.T, v[i]) for i in eqv]
            #print np.min(cos1), np.max(cos1)
            #print 'equatorial max in equatorial vertices:', t1[0] in eqv
            #x =  np.cross(v[t0[0]],v[t1[0]])
            #x = x/np.sqrt(np.sum(x**2))
            #print x
            #ptchv = dgqs.patch_vertices(v, x, 5)
            #print len(ptchv)
            #eqp = eqv[np.argmin([np.abs(np.dot(v[t1[0]].T,v[p])) for p in eqv])]
            #print (eqp, o[eqp])
            #print t2[0] in ptchv, t2[0] in eqv
            #print np.dot(pole.T, v[t1[0]]), np.dot(pole.T, v[t2[0]])
            #print ptchv[np.argmin([o[v] for v in ptchv])]
                                       
            #gq_indices = np.array(gq.IN[:,0],dtype='int').reshape((100,1000))
        
            #gq_first_directions_in=odf_vertices[np.array(gq.IN[:,0],dtype='int')]
        
            #print gq_first_directions_in.shape
        
            #gq_results = analyze_maxima(gq_indices, gq_first_directions_in.reshape((100,1000,3)),range(100))
        
            #for gqi see example dicoms_2_tracks gq.IN[:,0]
        
            #np.set_printoptions(precision=6, suppress=True, linewidth=200, threshold=5000)

            #out = open('/home/ian/Data/SimVoxels/Out/'+'+++_'+dataname,'w')
        
            #results = np.hstack((np.vstack(dt_results), np.vstack(gq_results)))
            #results = np.vstack(dt_results)
        
            #print >> out, results[:,:]
        
    out.close()
    
def run_small_data():
    
    #smalldir = '/home/ian/Devel/dipy/dipy/data/'
    smalldir = '/home/eg309/Devel/dipy/dipy/data/'
#    from os.path import join as opj

#    bvals=np.load(opj(os.path.dirname(__file__), \
#                          'data','small_64D.bvals.npy'))
    bvals=np.load(smalldir+'small_64D.bvals.npy')
#    gradients=np.load(opj(os.path.dirname(__file__), \
#                              'data','small_64D.gradients.npy'))    
    gradients=np.load(smalldir+'small_64D.gradients.npy')
#    img =ni.load(os.path.join(os.path.dirname(__file__),\
#                                  'data','small_64D.nii'))
    img=nibabel.load(smalldir+'small_64D.nii')
    small_data=img.get_data()    

    print 'real_data', small_data.shape
    gqsmall = dgqs.GeneralizedQSampling(small_data,bvals,gradients)
    tnsmall = ddti.Tensor(small_data,bvals,gradients)

    x,y,z,a,b=tnsmall.evecs.shape
    evecs=tnsmall.evecs
    xyz=x*y*z
    evecs = evecs.reshape(xyz,3,3)
    #vs = np.sign(evecs[:,2,:])
    #print vs.shape
    #print np.hstack((vs,vs,vs)).reshape(1000,3,3).shape
    #evecs = np.hstack((vs,vs,vs)).reshape(1000,3,3)
    #print evecs.shape
    evals=tnsmall.evals
    evals = evals.reshape(xyz,3)
    #print evals.shape

    

    #print('GQS in %d' %(t2-t1))
        
    '''
    eds=np.load(opj(os.path.dirname(__file__),\
                        '..','matrices',\
                        'evenly_distributed_sphere_362.npz'))
    '''
    from dipy.data import get_sphere

    odf_vertices,odf_faces=get_sphere('symmetric362')

    
    #odf_vertices=eds['vertices']
    #odf_faces=eds['faces']

    #Yeh et.al, IEEE TMI, 2010
    #calculate the odf using GQI

    scaling=np.sqrt(bvals*0.01506) # 0.01506 = 6*D where D is the free
    #water diffusion coefficient 
    #l_values sqrt(6 D tau) D free water
    #diffusion coefficiet and tau included in the b-value

    tmp=np.tile(scaling,(3,1))
    b_vector=gradients.T*tmp
    Lambda = 1.2 # smoothing parameter - diffusion sampling length
    
    q2odf_params=np.sinc(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi)
    #implements equation no. 9 from Yeh et.al.

    S=small_data.copy()

    x,y,z,g=S.shape
    S=S.reshape(x*y*z,g)
    QA = np.zeros((x*y*z,5))
    IN = np.zeros((x*y*z,5))
    FA = tnsmall.fa().reshape(x*y*z)

    fwd = 0
    
    #Calculate Quantitative Anisotropy and find the peaks and the indices
    #for every voxel

    summary = {}

    summary['vertices'] = odf_vertices
    v = odf_vertices.shape[0]
    summary['faces'] = odf_faces
    f = odf_faces.shape[0]

    for (i,s) in enumerate(S):

        #print 'Volume %d' % i

        istr = str(i)

        summary[istr] = {}

        t0, t1, t2, npa = gqsmall.npa(s, width = 5)
        summary[istr]['triple']=(t0,t1,t2)
        summary[istr]['npa']=npa

        odf = Q2odf(s,q2odf_params)
        peaks,inds=rp.peak_finding(odf,odf_faces)
        fwd=max(np.max(odf),fwd)
        #peaks = peaks - np.min(odf)
        n_peaks=min(len(peaks),5)
        peak_heights = [odf[i] for i in inds[:n_peaks]]
        #QA[i][:l] = peaks[:n_peaks]
        IN[i][:n_peaks] = inds[:n_peaks]

        summary[istr]['odf'] = odf
        summary[istr]['peaks'] = peaks
        summary[istr]['inds'] = inds
        summary[istr]['evecs'] = evecs[i,:,:]
        summary[istr]['evals'] = evals[i,:]
        summary[istr]['n_peaks'] = n_peaks
        summary[istr]['peak_heights'] = peak_heights
#        summary[istr]['fa'] = tnsmall.fa()[0]
        summary[istr]['fa'] = FA[i]
    '''
    QA/=fwd
    QA=QA.reshape(x,y,z,5)    
    IN=IN.reshape(x,y,z,5)
    '''
    
    peaks_1 = [i for i in range(1000) if summary[str(i)]['n_peaks']==1]
    peaks_2 = [i for i in range(1000) if summary[str(i)]['n_peaks']==2]
    peaks_3 = [i for i in range(1000) if summary[str(i)]['n_peaks']==3]
    #peaks_2 = [i for i in range(1000) if len(summary[str(i)]['inds'])==2]
    #peaks_3 = [i for i in range(1000) if len(summary[str(i)]['inds'])==3]

    print '#voxels with 1, 2, 3 peaks', len(peaks_1),len(peaks_2),len(peaks_3)

    return FA, summary

def Q2odf(s,q2odf_params):
    ''' construct odf for a voxel '''
    odf=np.dot(s,q2odf_params)
    return odf

    
#run_comparisons()
#run_gq_sims()
FA, summary = run_small_data()
peaks_1 = [i for i in range(1000) if summary[str(i)]['n_peaks']==1]
peaks_2 = [i for i in range(1000) if summary[str(i)]['n_peaks']==2]
peaks_3 = [i for i in range(1000) if summary[str(i)]['n_peaks']==3]
fa_npa_1 = [[summary[str(i)]['fa'], summary[str(i)]['npa'], summary[str(i)]['peak_heights']] for i in peaks_1]
fa_npa_2 = [[summary[str(i)]['fa'], summary[str(i)]['npa'], summary[str(i)]['peak_heights']] for i in peaks_2]
fa_npa_3 = [[summary[str(i)]['fa'], summary[str(i)]['npa'], summary[str(i)]['peak_heights']] for i in peaks_3]
