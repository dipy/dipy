''' Learning algorithms for tractography'''

import numpy as np
from dipy.core import track_metrics as tm
import dipy.core.track_performance as pf
from scipy import ndimage as nd
import itertools
import time
import numpy.linalg as npla



def larch(tracks,
          split_thrs=[50.**2,20.**2,10.**2],
          ret_atracks=False,
          info=False):
    ''' LocAl Rapid Clusters for tractograpHy

    Parameters
    ----------
    tracks : sequence
       of tracks as arrays, shape (N1,3) .. (Nm,3)
    split_thrs : sequence
       of 3 floats with the squared distances
    approx_tracks: bool 
       if True return an approximation of the initial tracks
    info: bool 
       print some information

    Returns
    --------
    C : dict
       a tree graph containing the clusters
    atracks : sequence
       of approximated tracks the approximation preserves initial shape.
    '''


    '''
    t1=time.clock()    
    print 'Reducing to 3-point approximate tracks...'
    tracks3=[tm.downsample(t,3) for t in tracks]

    t2=time.clock()
    print 'Done in ', t2-t1, 'secs'
    
    print 'Reducing to n-point approximate tracks...'
    atracks=[pf.approx_polygon_track(t) for t in tracks]

    t3=time.clock()
    print 'Done in ', t3-t2, 'secs'
        
    print('Starting larch_preprocessing...')
    C=pf.larch_preproc(tracks3,split_thrs,info)   
    
    t4=time.clock()
    print 'Done in ', t4-t3, 'secs'

    print('Finding most similar tracks in every cluster ...')
    for c in C:
        
        local_tracks=[atracks[i] for i in C[c]['indices']]        
        #identify the most similar track in the cluster C[c] and return the index of
        #the track and the distances of this track with all other tracks
        msi,distances=pf.most_similar_track_mam(local_tracks,metric='avg')
        
        C[c]['repz']=atracks[C[c]['indices'][msi]]
        C[c]['repz_dists']=distances
    
    print 'Done in ', time.clock()-t4
    
    if ret_atracks:
        return C,atracks
    else:
        return C
    
    '''

    return


def detect_corresponding_tracks(indices,tracks1,tracks2):
    ''' Detect corresponding tracks from 1 to 2
    
    Parameters
    ----------
    indices : sequence
       of indices of tracks1 that are to be detected in tracks2
    tracks1 : sequence 
       of tracks as arrays, shape (N1,3) .. (Nm,3)
    tracks2 : sequence 
       of tracks as arrays, shape (M1,3) .. (Mm,3)
            
    Returns
    -------
    track2track : array
       of int showing the correspondance
    '''
    li=len(indices)
    
    track2track=np.zeros((li,3))
    cnt=0
    for i in indices:        
        
        rt=[pf.zhang_distances(tracks1[i],t,'avg') for t in tracks2]
        rt=np.array(rt)               

        track2track[cnt-1]=np.array([cnt,i,rt.argmin()])        
        cnt+=1
        
    return track2track.astype(int)

def detect_corresponding_tracks_extended(indices,tracks1,indices2,tracks2):
    ''' Detect corresponding tracks from 1 to 2
    
    Parameters:
    ----------------
    indices: sequence
            of indices of tracks1 that are to be detected in tracks2
    
    tracks1: sequence 
            of tracks as arrays, shape (N1,3) .. (Nm,3)
    
    indices2: sequence
            of indices of tracks2 in the initial brain            
                
    tracks2: sequence 
            of tracks as arrays, shape (M1,3) .. (Mm,3)
            
    Returns:
    -----------
    track2track: array of int
            showing the correspondance
    
    '''
    li=len(indices)
    
    track2track=np.zeros((li,3))
    cnt=0
    for i in indices:        
        
        rt=[pf.zhang_distances(tracks1[i],t,'avg') for t in tracks2]
        rt=np.array(rt)               

        track2track[cnt-1]=np.array([cnt,i,indices2[rt.argmin()]])        
        cnt+=1
        
    return track2track.astype(int)


def rm_far_ends(ref,tracks,dist=25):
    ''' rm tracks with far endpoints
    
    Parameters
    ----------
    ref : array, shape (N,3)
       xyz points of the reference track
    tracks : sequence 
       of tracks as arrays, shape (N1,3) .. (Nm,3)
    dist : float
       endpoint distance threshold
        
    Returns
    -------
    tracksr : sequence
       reduced tracks
    indices : sequence
       indices of tracks
    '''
    indices=[i for (i,t) in enumerate(tracks) if tm.max_end_distances(t,ref) <= dist]
    
    tracksr=[tracks[i] for i in indices]
    
    return tracksr,indices
 

def rm_far_tracks(ref,tracks,dist=25,down=False):
    ''' Remove tracks which are far away using as a distance metric the average euclidean distance of the 
    following three points start point, midpoint and end point.

    Parameters
    ----------
    ref : array, shape (N,3)
       xyz points of the reference track
    tracks : sequence 
       of tracks as arrays, shape (N1,3) .. (Nm,3)
    dist : float
       average distance threshold
    down: bool {True, False}
       if down = True then ref and tracks are already downsampled
       if down = False then downsample them
    
    Returns
    -------
    tracksr : sequence
            reduced tracks
    indices : sequence
            indices of tracks
    '''

    if down==False:
        
        tracksd=[tm.downsample(t,3) for t in tracks]
        refd=tm.downsample(ref,3) 
        
        indices=[i for (i,t) in enumerate(tracksd) if np.mean(np.sqrt(np.sum((t-refd)**2,axis=1))) <= dist]
        
        tracksr=[tracks[i] for i in indices]
        return tracksr, indices
    
    if down==True:
        
        indices=[i for (i,t) in enumerate(tracks) if np.mean(np.sqrt(np.sum((t-ref)**2,axis=1))) <= dist]
        tracksr=[tracks[i] for i in indices]
        return tracksr,indices
 

def missing_tracks(indices1,indices2):
    ''' Missing tracks in bundle1 but not bundle2
    
    Parameters:
    ------------------
    indices1: sequence 
            of indices of tracks in bundle1
                
    indices2: sequence 
            of indices of tracks in bundle2
                
    Returns:
    -----------
    indices: sequence of indices
            of tracks in bundle1 absent from bundle2
            
    Example:
    -------------
    >>> tracksar,indar=rm_far_tracks(ref,tracksa,dist=20)
    >>> fornix_ind=G[5]['indices']
    >>> len(missing_tracks(fornix_ind, indar)) = 5
    
    >>> tracksar,indar=rm_far_tracks(ref,tracksa,dist=25)
    >>> fornix_ind=G[5]['indices']
    >>> len(missing_tracks(fornix_ind, indar)) = 0
    
    '''
    
    return list(set(indices1).difference(set(indices2)))    

def skeletal_tracks(tracks,rand_selected=1000,ball_radius=5,neighb_no=50):
    ''' Filter out unnescessary tracks and keep only a few good ones.  
    Aka the balls along a track method.  
    
    Parameters:
    ----------------
    tracks: sequence
            of tracks
    rand_selected: int
            number of initially selected fibers
    ball_radius: float
            balls along tracks radii
    neighb_no: int
            lowest threshold for the number of tracks included 
    
    Returns:
    -----------
    reps: sequence
            of indices of representative aka skeletal tracks. They should be <= rand_selected
    
    
    '''
    trackno=len(tracks)
    #select 1000 random tracks
    random_indices=(trackno*np.random.rand(rand_selected)).astype(int)
    
    tracks3points=[tm.downsample(t,3) for t in tracks]
    
    #store representative tracks
    representative=[]       
    representative_indices=[]       
    
    #store indices of already visited tracks i.e. which already have a representative track
    visited=[]
    
    import time
    t1=time.clock()
    
    # for every index of the possible representative tracks
    for (i,t) in enumerate(random_indices):        
        
        #if track is not already classified 
        if i not in visited:
            
            print(i,t)
            
            #rm far tracks
            tracksr,indices=rm_far_tracks(tracks3points[t],tracks3points,dist=25,down=True)
                    
            cnt_neighb=0            
            just_visited=[]
            
            #for every possible neighbour track tr with index tri
            for tri in indices:                   
                
                cnt_intersected_balls=0
                
                #for every point of the possible representative track 
                for p in tracks[t]:
                    
                    #if you intersect the sphere surrounding the point of the random track increase a counter
                    if tm.inside_sphere(tracks[tri],p,ball_radius): cnt_intersected_balls+=1
                
                #if all spheres are covered then accept this track as your neighbour
                if cnt_intersected_balls ==len(tracks[t]): 
                    
                    cnt_neighb+=1                
                    just_visited.append(tri)
            
            #if the number of possible neighbours is above threshold then accept track[t] as a representative fiber
            if cnt_neighb>=neighb_no: 
                representative.append(t)                
                visited=visited+just_visited
    
    print 'Time:',time.clock()-t1
    
    return representative

def detect_corpus_callosum(tracks,plane=91,ysize=217,zsize=181,width=1.0,use_atlas=0,use_preselected_tracks=0,ball_radius=5):
    ''' Detect corpus callosum in a mni registered dataset of shape (181,217,181)   
    
    Parameters:
    ----------------
    tracks: sequence 
            of tracks
    
    Returns:
    ----------
    cc_indices: sequence
            with the indices of the corpus_callosum tracks
    
    left_indices: sequence
            with the indices of the rest of the brain
       
    '''

    cc=[]

    #for every track
    for (i,t) in enumerate(tracks):
        
        #for every index of any point in the track
        for pi in range(len(t)-1):
           
            #if track segment is cutting the plane (assuming the plane is at the x-axis X=plane)
            if (t[pi][0] <= plane and t[pi+1][0] >= plane) or (t[pi+1][0] <= plane and t[pi][0] >= plane) :
                                
                v=t[pi+1]-t[pi]
                k=(plane-t[pi][0])/v[0]                
                
                hit=k*v+t[pi]
                
                #report the index of the track and the point of intersection with the plane
                cc.append((i,hit))
    
    #indices
    cc_i=[c[0] for c in cc]
    
    print 'Number of tracks cutting plane Before',len(cc_i)
    
    #hit points
    cc_p=np.array([c[1] for c in cc])
    
    p_neighb=len(cc_p)*[0]
    
    cnt=0
    #imaging processing from now on
    
    im=np.zeros((ysize,zsize))
    im2=np.zeros((ysize,zsize))
    
    im_track={}
    
    cnt=0
    for p in cc_p:
        
        p1=int(round(p[1]))
        p2=int(round(p[2]))
                
        im[p1,p2]=1
        im2[p1,p2]=im2[p1,p2]+1
        
        try:
            im_track[(p1,p2)]=im_track[(p1,p2)]+[cc_i[cnt]]
        except:
            im_track[(p1,p2)]=[cc_i[cnt]]
            
        cnt+=1
            
        
    #create a cross structure
    cross=np.array([[0,1,0],[1,1,1],[0,1,0]])
    
    im=(255*im).astype('uint8')
    im2=(np.interp(im2,[0,im2.max()],[0,255])).astype('uint8')
    
    #erosion
    img=nd.binary_erosion(im,structure=cross)    
    
    #and another one erosion
    #img=nd.binary_erosion(img,structure=cross)
    #im2g=nd.grey_erosion(im2,structure=cross)   
    #im2g2=nd.grey_erosion(im2g,structure=cross)
    
    indg2=np.where(im2==im2.max())
    p1max=indg2[0][0]
    p2max=indg2[1][0]
    
    #label objects    
    imgl=nd.label(img)
    no_labels=imgl[1]
    imgl=imgl[0]
    
    #find the biggest objects the second biggest should be the cc the biggest should be the background
    '''
    find_big=np.zeros(no_labels)
    
    for i in range(no_labels):
        
        ind=np.where(imgl==i)
        find_big[i]=len(ind[0])
        
    print find_big
    
    find_bigi=np.argsort(find_big)
    '''
    cc_label=imgl[p1max,p2max]
    
    imgl2=np.zeros((ysize,zsize))
    
    #cc is found and copied to a new image here
    #imgl2[imgl==int(find_bigi[-2])]=1    
    imgl2[imgl==int(cc_label)]=1
    
    imgl2=imgl2.astype('uint8')
        
    #now do another dilation to recover some cc shape from the previous erosion    
    imgl2d=nd.binary_dilation(imgl2,structure=cross)    
    #and another one
    #imgl2d=nd.binary_dilation(imgl2d,structure=cross)    
    
    imgl2d=imgl2d.astype('uint8')
    
    #get the tracks back
    cc_indices=[]
    indcc=np.where(imgl2d>0)
    for i in range(len(indcc[0])):
        p1=indcc[0][i]
        p2=indcc[1][i]
        cc_indices=cc_indices+im_track[(p1,p2)]
        
    print 'After', len(cc_indices)
        
    #export also the rest of the brain
    indices=range(len(tracks))    
    left=set(indices).difference(set(cc_indices))
    left_indices=[l for l in left]    
    
    #return im,im2,imgl2d,cc_indices,left_indices
    return cc_indices,left_indices




def track_indices_for_a_value_in_atlas(atlas,value,tes,tracks):
    
    ind=np.where(atlas==value)
    indices=set([])

    for i in range(len(ind[0])):
        try:
            tmp=tes[(ind[0][i], ind[1][i], ind[2][i])]
            indices=indices.union(set(tmp))
        except:
            pass
    
    #bundle=[tracks[i] for i in list(indices)]        
    #return bundle,list(indices)
    return list(indices)


def relabel_by_atlas_value_and_mam(atlas_tracks,atlas,tes,tracks,tracksd,zhang_thr):
    
    emi=emi_atlas()
    
    brain_relabeled={}
    
    for e in range(1,9): #from emi:
        
        print emi[e]['bundle_name']
        indices=emi[e]['init_ref']+emi[e]['selected_ref']+emi[e]['apr_ref']        
        tmp=detect_corresponding_tracks(indices,atlas_tracks,tracks)
        corresponding_indices=tmp[:,2]
                
        corresponding_indices=list(set(corresponding_indices))
                
        value_indices=[]
        for value in emi[e]['value']:            
            value_indices+=track_indices_for_a_value_in_atlas(atlas,value,tes,tracks)
        
        value_indices=list(set(value_indices))
        
        print 'len corr_ind',len(corresponding_indices)
        
        #check if value_indices do not have anything in common with corresponding_indices and expand
        if list(set(value_indices).intersection(set(corresponding_indices)))==[]:            
            #value_indices=corresponding_indices
            print 'len corr_ind',len(corresponding_indices)
            for ci in corresponding_indices:            
                print 'koukou',ci
                ref=tracksd[ci]
                brain_rf, ind_fr = rm_far_tracks(ref,tracksd,dist=10,down=True)
                value_indices+=ind_fr
                
            
            value_indices=list(set(value_indices))
            print 'len vi',len(value_indices)
        
        value_indices_new=[]
        #reduce value_indices which are far from every corresponding fiber
        for vi in value_indices:            
            dist=[]
            for ci in corresponding_indices:       
                dist.append(pf.zhang_distances(tracks[vi],tracks[ci],'avg'))                                    
                
            for d in dist:
                if d <= zhang_thr[e-1]:
                    value_indices_new.append(vi)
                
        value_indices=list(set(value_indices_new))
        #store value indices
        brain_relabeled[e]={}
        brain_relabeled[e]['value_indices']=value_indices
        brain_relabeled[e]['corresponding_indices']=corresponding_indices        
        brain_relabeled[e]['color']=emi[e]['color']
        brain_relabeled[e]['bundle_name']=emi[e]['bundle_name'][0]
        
        
        
    return brain_relabeled    


def threshold_hitdata(hitdata, divergence_threshold=0.25, fibre_weight=0.8):
    ''' [1] Removes hits in hitdata which have divergence above threshold.
       [2] Removes fibres in hitdata whose fraction of remaining hits is below
        the required weight.

    Parameters:
    ----------------
    ref:  array, shape (N,5)
       xyzrf hit data from cut_planes
    
    divergence_threshold: float
            if radial coefficient of divergence is above this then drop the hit
    
    fibre_weight: float
            the number of remaing hits on a fibre as a fraction of len(trackdata),
            which is the maximum number possible
    
    Returns:
    -----------    
    reduced_hitdata: array, shape (M, 5)
    light_weight_fibres: list of integer track indices
    '''
    # first pass: remove hits with r>divergence_threshold
    firstpass = [[[x,y,z,r,f] for (x,y,z,r,f) in plane  if r<=divergence_threshold] for plane in hitdata]

    # second pass: find fibres hit weights
    fibrecounts = {}
    for l in [[f,r] for (x,y,z,r,f) in itertools.chain(*firstpass)]:
        f = l[0].astype('int')
        try:
            fibrecounts[f] += 1
        except:
            fibrecounts[f] = 1
    
    weight_thresh = len(hitdata)*fibre_weight
    heavy_weight_fibres = [f for f in fibrecounts.keys() if fibrecounts[f]>=weight_thresh]

    # third pass

    reduced_hitdata = [np.array([[x,y,z,r,f] for (x,y,z,r,f) in plane if fibrecounts[f.astype('int')] >= weight_thresh]) for plane in firstpass]
   
    return reduced_hitdata, heavy_weight_fibres

def neck_finder(hitdata, ref):
    '''
    To identify regions of concentration of fibres related by hitdata to a reference fibre
    '''
    
    #typically len(hitdata) = len(ref)-2 at present, though it should ideally be
    # len(ref)-1 which is the number of segments in ref
    # We will assume that hitdata[i] relates to the segment from ref[i] to ref[i+1]
    
    #xyz=[]
    #rcd=[]
    #fibres=[]

    weighted_mean_rcd = []
    unweighted_mean_rcd = []
    weighted_mean_dist = []
    unweighted_mean_dist = []
    hitcount = []
    
    for (p, plane) in enumerate(hitdata):
        xyz = plane[:,:3]
        rcd =plane[:,3]
        fibres = plane[:,4]
    
        hitcount +=[len(plane)]
    
        radial_distances=np.sqrt(np.diag(np.inner(xyz-ref[p],xyz-ref[p])))

        unweighted_mean_rcd += [np.average(1-rcd)]

        weighted_mean_rcd += [np.average(1-rcd, weights=np.exp(-radial_distances))]
    
        unweighted_mean_dist += [np.average(np.exp(-radial_distances))]

        weighted_mean_dist += [np.average(np.exp(-radial_distances), weights=1-rcd)]

    return np.array(hitcount), np.array(unweighted_mean_rcd), np.array(weighted_mean_rcd), \
        np.array(unweighted_mean_dist), np.array(weighted_mean_dist)

def max_concentration(plane_hits,ref):
    '''
    calculates the log determinant of the concentration matrix for the hits in planehits    
    '''
    dispersions = [np.prod(np.sort(npla.eigvals(np.cov(p[:,0:3].T)))[1:2]) for p in plane_hits]
    index = np.argmin(dispersions)
    log_max_concentration = -np.log2(dispersions[index])
    centre = ref[index+1]
    return index, centre, log_max_concentration

def refconc(brain, ref, divergence_threshold=0.3, fibre_weight=0.7):
    '''
    given a reference fibre locates the parallel fibres in brain (tracks)
    with threshold_hitdata applied to cut_planes output then follows
    with concentration to locate the locus of a neck
    '''
    
    hitdata = pf.cut_plane(brain, ref)
    reduced_hitdata, heavy_weight_fibres = threshold_hitdata(hitdata, divergence_threshold, fibre_weight)
    #index, centre, log_max_concentration = max_concentration(reduced_hitdata, ref)
    index=None
    centre=None
    log_max_concentration=None
    
    return heavy_weight_fibres, index, centre

def bundle_from_refs(brain,braind, refs, divergence_threshold=0.3, fibre_weight=0.7,far_thresh=25,zhang_thresh=15, end_thresh=10):
    '''
    '''
    bundle = set([])
    centres = []
    indices = []

    for ref in refs:        
        
        refd=tm.downsample(ref,3)         
        brain_rf, ind_fr = rm_far_tracks(refd,braind,dist=far_thresh,down=True)        
        brain_rf=[brain[i] for i in ind_fr]        
        #brain_rf,ind_fr = rm_far_tracks(ref,brain,dist=far_thresh,down=False)        
        heavy_weight_fibres, index, centre = refconc(brain_rf, ref, divergence_threshold, fibre_weight)        
        heavy_weight_fibres_z = [i for i in heavy_weight_fibres if pf.zhang_distances(ref,brain_rf[i],'avg')<zhang_thresh]        
        #heavy_weight_fibres_z_e = [i for i in heavy_weight_fibres_z if tm.max_end_distances(brain_rf[i],ref)>end_thresh]        
        hwfind = set([ind_fr[i] for i in heavy_weight_fibres_z])        
        bundle = bundle.union(hwfind)

    bundle_med = []
    
    for i in bundle:        
        minmaxdist = 0.
        for ref in refs:
            minmaxdist=min(minmaxdist,tm.max_end_distances(brain[i],ref))
        if minmaxdist<=end_thresh:
            bundle_med.append(i)            
        #centres.append(centre)        
        #indices.append(index)
    
    #return list(bundle), centres, indices
    return bundle_med





class FACT_Delta():
    ''' Generates tracks with termination criteria defined by a
    delta function [1]_ and it has similarities with FACT algorithm [2]_.

    Can be used with any reconstruction method as DTI,DSI,QBI,GQI which can
    calculate an orientation distribution function and find the local peaks of
    that function. For example a single tensor model can give you only
    one peak a dual tensor model 2 peaks and quantitative anisotropy
    method as used in GQI can give you 3,4,5 or even more peaks.    
    
    The parameters of the delta function are checking thresholds for the
    direction propagation magnitude and the angle of propagation.

    A specific number of seeds is defined randomly and then the tracks
    are generated for that seed if the delta function returns true.

    Trilinear interpolation is being used for defining the weights of
    the propagation.

    References
    ----------

    .. [1] Yeh. et al. Generalized Q-Sampling Imaging, TMI 2010.

    .. [2] Mori et al. Three-dimensional tracking of axonal projections
    in the brain by magnetic resonance imaging. Ann. Neurol. 1999.
    

    '''

    def __init__(self,qa,ind,seeds_no=1000,odf_vertices=None,qa_thr=0.0239,step_sz=0.5,ang_thr=60.):
        '''
        Parameters
        ----------

        qa: array, shape(x,y,z,Np), magnitude of the peak (QA) or
        shape(x,y,z) a scalar volume like FA.

        ind: array, shape(x,y,z,Np), indices of orientations of the QA
        peaks found at odf_vertices used in QA or, shape(x,y,z), ind

        seeds_no: number of random seeds

        odf_vertices: sphere points which define a discrete
        representation of orientations for the peaks, the same for all voxels

        qa_thr: float, threshold for QA(typical 0.023)  or FA(typical 0.2) 
        step_sz: float, propagation step

        ang_thr: float, if turning angle is smaller than this threshold
        then tracking stops.        

        Returns
        -------

        tracks: sequence of arrays

        '''

        if len(qa.shape)==3:
            qa.shape=qa.shape+(1,)
            ind.shape=ind.shape+(1,)

        #store number of maximum peacks
        self.Np=qa.shape[-1]

        x,y,z,g=qa.shape
        tlist=[]       

        if odf_vertices==None:
            eds=np.load(os.path.join(os.path.dirname(__file__),'matrices',\
                        'evenly_distributed_sphere_362.npz'))
            odf_vertices=eds['vertices']

        self.seed_list=[]
            
        for i in range(seeds_no):
            rx=(x-1)*np.random.rand()
            ry=(y-1)*np.random.rand()
            rz=(z-1)*np.random.rand()            
            seed=np.array([rx,ry,rz])

            #print 'init seed', seed            
            #self.seed_list.append(seed.copy())            
            track=self.propagation(seed.copy(),qa,ind,odf_vertices,qa_thr,ang_thr,step_sz)

            if track == None:
                pass
            else:
                self.seed_list.append(seed.copy())
                tlist.append(track)
        
        self.tracks=tlist
            


    def trilinear_interpolation(self,X):
        '''
        Parameters
        ----------
        X: array, shape(3,), a point

        Returns
        --------
        W: array, shape(8,2) weights, think of them like the 8
        subvolumes of a unit cube surrounding the seed.

        IN: array, shape(8,2), the corners of the unit cube

        '''

        Xf=np.floor(X)        
        #d holds the distance from the (floor) corner of the voxel
        d=X-Xf
        #nd holds the distance from the opposite corner
        nd = 1-d
        #filling the weights
        W=np.array([[ nd[0] * nd[1] * nd[2] ],
                    [  d[0] * nd[1] * nd[2] ],
                    [ nd[0] *  d[1] * nd[2] ],
                    [ nd[0] * nd[1] *  d[2] ],
                    [  d[0] *  d[1] * nd[2] ],
                    [ nd[0] *  d[1] *  d[2] ],
                    [  d[0] * nd[1] *  d[2] ],
                    [  d[0] *  d[1] *  d[2] ]])

        IN=np.array([[ Xf[0]  , Xf[1]  , Xf[2] ],
                    [ Xf[0]+1 , Xf[1]  , Xf[2] ],
                    [ Xf[0]   , Xf[1]+1, Xf[2] ],
                    [ Xf[0]   , Xf[1]  , Xf[2]+1 ],
                    [ Xf[0]+1 , Xf[1]+1, Xf[2] ],
                    [ Xf[0]   , Xf[1]+1, Xf[2]+1 ],
                    [ Xf[0]+1 , Xf[1]  , Xf[2]+1 ],
                    [ Xf[0]+1 , Xf[1]+1, Xf[2]+1 ]])

        return W,IN.astype(np.int)

    def nearest_direction(self,dx,qa,ind,odf_vertices,qa_thr=0.0245,ang_thr=60.):
        ''' Give the nearest direction to a point

        Parameters
        ----------        
        dx: array, shape(3,), as float, moving direction of the current
        tracking

        qa: array, shape(Np,), float, quantitative anisotropy matrix,
        where Np the number of peaks, found using self.Np

        ind: array, shape(Np,), float, index of the track orientation

        odf_vertices: array, shape(N,3), float, odf sampling directions

        qa_thr: float, threshold for QA, we want everything higher than
        this threshold 

        ang_thr: float, theshold, we only select fiber orientation with
        this range 

        Returns
        --------
        delta: bool, delta funtion, if 1 we give it weighting if it is 0
        we don't give any weighting

        direction: array, shape(3,), the fiber orientation to be
        consider in the interpolation

        '''

        max_dot=0
        max_doti=0
        angl = np.cos((np.pi*ang_thr)/180.) 
        if qa[0] <= qa_thr:
            return False, np.array([0,0,0])
        
        for i in range(self.Np):
            if qa[i]<= qa_thr:
                break
            curr_dot = np.abs(np.dot(dx, odf_vertices[ind[i]]))
            if curr_dot > max_dot:
                max_dot = curr_dot
                max_doti = i
                
        if max_dot < angl :
            return False, np.array([0,0,0])

        if np.dot(dx,odf_vertices[ind[max_doti]]) < 0:
            return True, - odf_vertices[ind[max_doti]]
        else:
            return True,   odf_vertices[ind[max_doti]]


        
    def propagation_direction(self,point,dx,qa,ind,odf_vertices,qa_thr,ang_thr):
        ''' Find where you are moving next
        '''
        total_w = 0 # total weighting
        new_direction = np.array([0,0,0])
        w,index=self.trilinear_interpolation(point)

        #print w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]

        #print index

        #check if you are outside of the volume
        for i in range(3):
            if index[7][i] >= qa.shape[i] or index[0][i] < 0:
                return False, np.array([0,0,0])

        #calculate qa & ind of each of the 8 corners
        for m in range(8):            
            x,y,z = index[m]
            qa_tmp = qa[x,y,z]
            ind_tmp = ind[x,y,z]
            #print qa_tmp[0]#,qa_tmp[1],qa_tmp[2],qa_tmp[3],qa_tmp[4]
            delta,direction = self.nearest_direction(dx,qa_tmp,ind_tmp,odf_vertices,qa_thr,ang_thr)
            #print delta, direction
            if not delta:
                continue
            total_w += w[m]
            new_direction = new_direction +  w[m][0]*direction

        if total_w < .5: # termination criteria
            return False, np.array([0,0,0])

        return True, new_direction/np.sqrt(np.sum(new_direction**2))
    
    def initial_direction(self,seed,qa,ind,odf_vertices,qa_thr):
        ''' First direction that we get from a seeding point

        '''
        #very tricky/cool addition/flooring that helps create a valid
        #neighborhood (grid) for the trilinear interpolation to run smoothly
        #seed+=0.5
        point=np.floor(seed+.5)
        x,y,z = point
        qa_tmp=qa[x,y,z,0]#maximum qa
        ind_tmp=ind[x,y,z,0]#corresponing orientation indices for max qa

        if qa_tmp < qa_thr:
            return False, np.array([0,0,0])
        else:
            return True, odf_vertices[ind_tmp]


    def propagation(self,seed,qa,ind,odf_vertices,qa_thr,ang_thr,step_sz):
        '''
        Parameters
        ----------
        seed: array, shape(3,), point where the tracking starts        
        qa: array, shape(Np,), float, quantitative anisotropy matrix,
        where Np the number of peaks, found using self.Np
        ind: array, shape(Np,), float, index of the track orientation        
                
        Returns
        -------
        d: bool, delta function result        
        idirection: array, shape(3,), index of the direction of the propagation

        '''
        point_bak=seed.copy()
        point=seed.copy()
        #d is the delta function 
        d,idirection=self.initial_direction(seed,qa,ind,odf_vertices,qa_thr)

        #print('FD',idirection[0],idirection[1],idirection[2])

        #print d
        if not d:
            return None
        
        dx = idirection
        #point = seed-0.5
        track = []
        track.append(point.copy())
        #track towards one direction 
        while d:
            d,dx = self.propagation_direction(point,dx,qa,ind,\
                                                  odf_vertices,qa_thr,ang_thr)
            if not d:
                break
            point = point + step_sz*dx
            track.append(point)

        d = True
        dx = - idirection
        point=point_bak.copy()
        #point = seed
        #track towards the opposite direction
        while d:
            d,dx = self.propagation_direction(point,dx,qa,ind,\
                                         odf_vertices,qa_thr,ang_thr)
            if not d:
                break
            point = point + step_sz*dx
            track.insert(0,point.copy())

        return np.array(track)













