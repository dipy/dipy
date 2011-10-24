import numpy as np
from dipy.tracking.metrics import downsample
from dipy.tracking.distances import local_skeleton_clustering
from dipy.tracking.distances import bundles_distances_mdf


class QuickBundles(object):
    
    def __init__(self,tracks,dist_thr=4.,pts=12):       
        if pts!=None:                        
            self.tracksd=[downsample(track,pts) for track in tracks]
        else:
            self.tracksd=tracks                    
        self.clustering=local_skeleton_clustering(self.tracksd,dist_thr)
        self.virts=None
        self.exemps=None                
    
    def virtuals(self):
        if self.virts==None:
            self.virts=[self.clustering[c]['hidden']/np.float(self.clustering[c]['N']) for c in self.clustering]
        return self.virts       
    
    def exemplars(self,tracks=None):
        if self.exemps==None:            
            self.exemps=[]
            self.exempsi=[]
            C=self.clustering
            if tracks==None:
                tracks=self.tracksd            
            for c in C:
                cluster=[tracks[i] for i in C[c]['indices']]                
                D=bundles_distances_mdf([C[c]['hidden']/float(C[c]['N'])],cluster)
                D=D.ravel()
                si=np.argmin(D)
                self.exempsi.append(si)
                self.exemps.append(cluster[si])                               
        return self.exemps, self.exempsi
    
    def partitions(self):
        return self.clustering
    
    def clusters(self):
        return self.clustering
    
    def label2cluster(self,id):
        return self.clustering[id]
    
    def label2tracksids(self,id):
        return [i for i in self.clustering[id]]        
    
    def label2tracks(self,tracks,id):
        return [tracks[i] for i in self.clustering[id]]
       
    def total_clusters(self):
        return len(self.clustering)
        
    def downsampled_tracks(self):
        return self.tracksd
        
        
        
    
class pQuickBundles():
    
    def __init__(self):
        pass
            