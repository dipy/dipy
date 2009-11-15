''' Learning algorithms for tractography'''

import numpy as np
from . import track_metrics as tm
import dipy.performance as pf

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
    ''' Remove tracks which are far away using as a distance metric the average euclidean distance of the three points.    

    Parameters:
    ----------------
    ref:  array, shape (N,3)
       xyz points of the reference track
    
    tracks: sequence 
            of tracks as arrays, shape (N1,3) .. (Nm,3)
    
    dist: 
    
    Returns:
    -----------    
    tracksfd:
    
    '''
    
    tracksd=[tm.downsample(t,3) for t in tracks]
    refd=tm.downsample(ref,3) 
    
    tracksfd=[t for t in tracksd if np.mean(np.sqrt(np.sum((t-refd)**2,axis=1))) <= dist]
    
    return tracksfd

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