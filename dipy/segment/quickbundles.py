from dipy.tracking.metrics import downsample
from dipy.tracking.distances import local_skeleton_clustering


class QuickBundles(object):
    
    def __init__(self,tracks,dist_thr=4.,pts=12):       
        if pts!=None:                        
           tracksd=downsample(tracks,pts)
        else:
            tracksd=tracks        
        tracksqb=local_skeleton_clustering(tracksd,dist_thr)        
        return tracksqb
        
    
class pQuickBundles(object):
    
    def __init__(self):
        pass
            