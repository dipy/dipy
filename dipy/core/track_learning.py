''' Learning algorithms for tractography'''

import numpy as np
from . import track_metrics as tm
import dipy.core.performance as pf
from scipy import ndimage as nd


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


            

def rm_far_tracks(ref,tracks,dist=20):
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
    
    Returns:
    -----------    
    tracksr: sequence
            reduced tracks
    
    indices: sequence
            indices of tracks
    '''
    
    tracksd=[tm.downsample(t,3) for t in tracks]
    refd=tm.downsample(ref,3) 
    
    indices=[i for (i,t) in enumerate(tracksd) if np.mean(np.sqrt(np.sum((t-refd)**2,axis=1))) <= dist]
    
    tracksr=[tracks[i] for i in indices]
    return tracksr, indices

  

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

def generate_representative_tracks(tracks,ball_radius=5,neighb_no=50):
    ''' Filter out unnescessary tracks and keep only a few good ones.    
    '''
    trackno=len(tracks)
    #select 1000 random tracks
    random_indices=(trackno*np.random.rand(1000)).astype(int)
    
    #store representative tracks
    representative=[]       
    
    import time
    t1=time.clock()
    
    # for every index of the possible representative tracks
    for (i,t) in enumerate(random_indices):        
        
        print i,t
        
        #rm far tracks
        tracksr,indices=rm_far_tracks(tracks[t],tracks,dist=20)
                
        cnt_neighb=0
        
        #for every possible neighbour track tr
        for tr in tracksr:                   
            
            cnt_intersected_balls=0
            
            #for every point of the possible representative track 
            for p in tracks[t]:
                
                #if you intersect the sphere surrounding the point of the random track increase a counter
                if tm.intersect_sphere(tr,p,ball_radius): cnt_intersected_balls+=1
            
            #if all spheres are covered then accept this track as your neighbour
            if cnt_intersected_balls ==len(tracks[t]): cnt_neighb+=1
        
        #if the number of possible neighbours is above threshold then accept track[t] as a representative fiber
        if cnt_neighb>=neighb_no: representative.append(t)
    
    print 'Time:',time.now()-t1
    
    return representative

def detect_references_in_atlas(atlas):
    ''' Not ready yet
    Detect curves representing the atlas's labeled regions    
    '''
    
    atlas=np.zeros((10,10,10))    
    atlas[2:-2,3:-3,2:-2]=1    
    A=A.astype('uint8')
    
    ind=np.where(atlas==1)
    points=np.vstack((ind[0],ind[1],ind[2])).T
    
    #tm.spline(
    
    pass
    
    

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
