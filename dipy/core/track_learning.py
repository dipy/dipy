''' Learning algorithms for tractography'''

import numpy as np
from . import track_metrics as tm
import dipy.core.performance as pf
from scipy import ndimage as nd
import itertools
import time


def detect_corresponding_tracks(indices,tracks1,tracks2):
    ''' Detect corresponding tracks from 1 to 2
    
    Parameters:
    ----------------
    indices: sequence
            of indices of tracks1 that are to be detected in tracks2
    
    tracks1: sequence 
            of tracks as arrays, shape (N1,3) .. (Nm,3)
                
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

        track2track[cnt-1]=np.array([cnt,i,rt.argmin()])        
        cnt+=1
        
    return track2track.astype(int)


            

def rm_far_tracks(ref,tracks,dist=25,down=False):
    ''' Remove tracks which are far away using as a distance metric the average euclidean distance of the 
    following three points start point, midpoint and end point.

    Parameters:
    ----------------
    ref:  array, shape (N,3)
       xyz points of the reference track
    
    tracks: sequence 
            of tracks as arrays, shape (N1,3) .. (Nm,3)
    
    dist: float
            average distance threshold
    
    down: bool {True, False}
            if down = True then ref and tracks are already downsampled
            if down = False then downsample them
    
    Returns:
    -----------    
    tracksr: sequence
            reduced tracks
    
    indices: sequence
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
                    if tm.intersect_sphere(tracks[tri],p,ball_radius): cnt_intersected_balls+=1
                
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

def emi_atlas():
    ''' Eleftherios-Matthew-Ian Atlas version 0.9
    Our atlas is based on Brain1 Scan1 from the PBC competition
    '''
    atlas={    
    
    0:{'bundle_name':'None','apr_ref':[],'selected_ref':[],'init_ref':[]},
    
    1:{'bundle_name':'Arcuate L','apr_ref':[79032],
    'selected_ref':[13355,203241,8239],
    'init_ref':[197816]},
    
    2:{'bundle_name':'Cingulum L','apr_ref':[115651],
    'selected_ref':[132955,209255],
    'init_ref':[15009]},    
    
    3:{'bundle_name':'Corticospinal R','apr_ref':[76983],
    'selected_ref':[249518,234534,174737,225536],
    'init_ref':[157189]},    
    
    4:{'bundle_name':'Forceps Major',
    'apr_ref':[17556],
    'selected_ref':[126619,109247],
    'init_ref':[64423]},            
    
    5:{'bundle_name':'Fornix','apr_ref':[206781],
    'selected_ref':[215713,48512,184169],
    'init_ref':[118191]},        
    
    6:{'bundle_name':'Inferior Occipitofrontal Fasciculus L','apr_ref':[168055],
    'selected_ref':[147881,126361,32004],
    'init_ref':[168055]},        
    
    7:{'bundle_name':'Superior Longitudinal Fasciculus L','apr_ref':[59215],
    'selected_ref':[6104,6777,224291,198813],
    'init_ref':[123041]},        
    
    8:{'bundle_name':'Uncinate R','apr_ref':[88647],
    'selected_ref':[249267,216811],
    'init_ref':[88647]}
    
    }
    
    
    return atlas
    

def detect_corresponding_bundles(bundle,tracks,zipit=1,n=10,d=3):
    ''' Not ready yet
    #downsample bundle
    bundlez=[tm.downsample(t,n) for t in bundle]
    
    #find reference track in bundlez
    ind_ref_bundlez,ref_bundlez=pf.most_similar_track_zhang(bundlez,'avg')
    
    #downsample tracks
    if zipit :
        tracksz=[tm.downsample(t,n) for t in tracks]
    else :
        tracksz=tracks
        
    #detect the most similar track in tracks with the bundlez reference track
    ref_tracksz=[pf.zhang_distances(ref_bundlez,t,'avg') for t in tracksz]
    
    #find the tracks that are close to ref_tracksz
    close_ref_group=[t for t in tracks if pf.minimum_closest_distance(ref_tracksz,t) <= d]

    #connect every track in bundlez with every track in close_ref_group
    
    '''
    pass

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

    
    