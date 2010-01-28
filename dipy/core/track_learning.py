''' Learning algorithms for tractography'''

import numpy as np
from . import track_metrics as tm
import dipy.core.performance as pf
from scipy import ndimage as nd
import itertools
import time
import numpy.linalg as npla


def near_clusters(c,C1,C2,n=-1):
    ''' Return 'n' closest clusters in C2 from cluster C1[c] using the hidden track

    Parameters:
    -----------
    c: int

    C1: dict, the structure of the dictionary is of the form 
    
    C2={0:{'hidden':c},1:{'hidden':d},2:{'hidden':e}} where c,d,e 3x3 numpy arrays

    C2: dict

    n= int, 
      default is -1 which returns all near clusters sorted from the nearest to the more distant.
            
    
    Example:
    --------

    >>> import dipy.core.track_learning as tl
    
    >>> a=np.array([[0,0,0],[1,0,0,],[2,0,0]])
    >>> b=np.array([[0,0,0],[1,0,0,],[2,1,0]])

    >>> c=np.array([[0,0,0],[1,0,0,],[2,0,0]])
    >>> d=np.array([[0,0,0],[1,0,0,],[3,1,0]])
    >>> e=np.array([[0,0,0],[1,0,0,],[4,1,0]])

    >>> C1={0:{'hidden':a},1:{'hidden':b}}
    >>> C2={0:{'hidden':c},1:{'hidden':d},2:{'hidden':e}}
  
    >>> tl.near_clusters(0,C1,C2)

    '''

    d= [pf.track_dist_3pts(C1[c]['hidden'],C2[c2]['hidden']) for c2 in C2]
        
    d=np.array(d)

    near=list(d[::-1].argsort())
    
    return near[:n]

    



    


def local_skeleton_clustering(tracks, d_thr=10):
    ''' deprecated use same funciton in performance
    
    Example:
    -----------
    from dipy.viz import fos
        
    tracks=[np.array([[0,0,0],[1,0,0,],[2,0,0]]),            
                np.array([[3,0,0],[3.5,1,0],[4,2,0]]),
                np.array([[3.2,0,0],[3.7,1,0],[4.4,2,0]]),
                np.array([[3.4,0,0],[3.9,1,0],[4.6,2,0]]),
                np.array([[0,0.2,0],[1,0.2,0],[2,0.2,0]]),
                np.array([[2,0.2,0],[1,0.2,0],[0,0.2,0]]),
                np.array([[0,0,0],[0,1,0],[0,2,0]])]
                                    
    C=local_skeleton_clustering(tracks,d_thr=0.5)    
    
    r=fos.ren()

    for c in C:
        color=np.random.rand(3)
        for i in C[c]['indices']:
            fos.add(r,fos.line(T[i],color))

    '''

    #Network C
    C={0:{'indices':[0],'hidden':tracks[0].copy(),'N':1}}
    ts=np.zeros((3,3),dtype=np.float32)
    
    for (it,t) in enumerate(tracks[1:]):
        
            
        lenC=len(C.keys())
        
        if it%1000==0:
            print it,lenC
        
        alld=np.zeros(lenC)
        flip=np.zeros(lenC)
        

        for k in xrange(lenC):
        
            h=C[k]['hidden']/C[k]['N']
            #print it+1
            #print t
            #print h
            d=np.sum(np.sqrt(np.sum((t-h)**2,axis=1)))/3.0
            ts[0]=t[-1];ts[1]=t[1];ts[-1]=t[0]
            ds=np.sum(np.sqrt(np.sum((ts-h)**2,axis=1)))/3.0
            
            #print d,ds
            
            if ds<d:                
                d=ds;
                flip[k]=1
                
            alld[k]=d

        m_k=np.min(alld)
        i_k=np.argmin(alld)
        
        if m_k<d_thr:            
            
            if flip[i_k]==1:                
                ts[0]=t[-1];ts[1]=t[1];ts[-1]=t[0]
                C[i_k]['hidden']+=ts
            else:                
                C[i_k]['hidden']+=t
                
            C[i_k]['N']+=1
            C[i_k]['indices'].append(it+1)
            
        else:
            C[lenC]={}
            C[lenC]['hidden']=t.copy()
            C[lenC]['N']=1
            C[lenC]['indices']=[it+1]
    
    '''   
    fos.clear(r)

    color=[fos.red,fos.green,fos.blue,fos.yellow]
    for c in C:
        for i in C[c]['indices']:
            fos.add(r,fos.line(tracks[i],color[c]))
                
    fos.show(r)
    '''
    
    return C


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
    Parameters:
    ----------------
    ref:  array, shape (N,3)
       xyz points of the reference track
    
    tracks: sequence 
            of tracks as arrays, shape (N1,3) .. (Nm,3)
    
    dist: float
            endpoint distance threshold
        
    Returns:
    -----------    
    tracksr: sequence
            reduced tracks
    
    indices: sequence
            indices of tracks
    '''
    
    indices=[i for (i,t) in enumerate(tracks) if tm.max_end_distance(t,ref) <= dist]
    
    tracksr=[tracks[i] for i in indices]
    
    return tracksr,indices
 

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
    ''' Eleftherios-Matthew-Ian Atlas version 0.00001
    Our atlas is based on Brain1 Scan1 from the PBC competition and the ICBM DTI-81 Atlas where
    ref are indices of tracks selected or given by distance metrics and value is the corresponding value for the bundle that this track
    belongs in the ICBM atlas.
    '''
        
    #Some common colors
    red=(1,0,0);    green=(0,1,0);    blue=(0,0,1);    yellow=(1,1,0);    cyan=(0,1,1);    azure=(0,0.49,1);    golden=(1,0.84,0);    white=(1,1,1)
    black=(0,0,0);    aquamarine=(0.498,1.,0.83);    indigo=(0.294,  0.,  0.5098);    lime=( 0.749,  1.,  0.);    hot_pink=( 0.988,  0.0588,  0.7529)
    gray=(0.5,0.5,0.5);    dark_red=(0.5,0,0);    dark_green=(0,0.5,0);    dark_blue=(0,0,0.5);    tan=( 0.8235,  0.7058,  0.549);    
    chartreuse=( 0.498, 1. , 0. );    coral=( 1. , 0.498, 0.3137)

    combined_atlas={    
    
    0:{'bundle_name':['None'],'apr_ref':[],'selected_ref':[],'init_ref':[]},    
    1:{'bundle_name':['Arcuate L'],'apr_ref':[79032],'selected_ref':[13355,203241,8239],'init_ref':[197816],'value':[41],'color':red},    
    2:{'bundle_name':['Cingulum L'],'apr_ref':[115651],'selected_ref':[132955,209255], 'init_ref':[15009],'value':[35],'color':blue},     
    3:{'bundle_name':['Corticospinal R','Cerebral peduncle R'],'apr_ref':[76983],    'selected_ref':[249518,234534,174737,225536],    'init_ref':[157189],'value':[9,17],'color':yellow},        
    4:{'bundle_name':['Forceps Major'],    'apr_ref':[17556],    'selected_ref':[126619,109247],    'init_ref':[64423],'value':[5],'color':green},                
    5:{'bundle_name':['Fornix'],'apr_ref':[206781],    'selected_ref':[215713,48512,184169],    'init_ref':[118191],'value':[6],'color':indigo},            
    6:{'bundle_name':['Inferior Occipitofrontal Fasciculus (Sagittal stratum) L','Inferior Occipitofrontal Fasciculus L'],'apr_ref':[168055],    'selected_ref':[147881,126361,32004],    'init_ref':[168055],'value':[31,45],'color':lime},            
    7:{'bundle_name':['Superior Longitudinal Fasciculus L'],'apr_ref':[59215],    'selected_ref':[6104,6777,224291,198813],    'init_ref':[123041],'value':[41],'color':gray},            
    8:{'bundle_name':['Uncinate R'],'apr_ref':[88647],    'selected_ref':[249267,216811],    'init_ref':[88647],'value':[48],'color':cyan},
    9:{'bundle_name':['Cingulum R'],'apr_ref':[],'selected_ref':[109309],'init_ref':[],'value':[36],'color':blue},    
    10:{'bundle_name':['Corticospinal L','Cerebral peduncle L'],'apr_ref':[],'selected_ref':[144023,235741,69257,219905],'init_ref':[],'value':[8,16],'color':yellow},    
    11:{'bundle_name':['Forceps Minor'],'apr_ref':[],'selected_ref':[56022],'init_ref':[],'value':[3],'color':green},    
    12:{'bundle_name':['Corpus Callosum Body'],'apr_ref':[],'selected_ref':[154373,25104,31959],'init_ref':[],'value':[4],'color':dark_red},    
    13:{'bundle_name':['Inferior Occipitofrontal Fasciculus (Sagittal stratum) R','Inferior Occipitofrontal Fasciculus R'],'apr_ref':[],'selected_ref':[197404],'init_ref':[],'value':[32,46],'color':lime},    
    14:{'bundle_name':['Superior Longitudinal Fasciculus R'],'apr_ref':[],'selected_ref':[64917,66270],'init_ref':[],'value':[42],'color':gray},    
    15:{'bundle_name':['Uncinate L'],'apr_ref':[],'selected_ref':[33418,115381,230360],'init_ref':[],'value':[47],'color':cyan},    
    16:{'bundle_name':['Middle cerebellar peduncle'],'apr_ref':[],'selected_ref':[177742],'init_ref':[],'value':[1],'color':hot_pink},    
    17:{'bundle_name':['Medial lemniscus R'],'apr_ref':[],'selected_ref':[45579,179716,196524],'init_ref':[],'value':[11],'color':aquamarine},    
    18:{'bundle_name':['Medial lemniscus L'],'apr_ref':[],'selected_ref':[88291,89525],'init_ref':[],'value':[10],'color':aquamarine},    
    19:{'bundle_name':['Tapatum R'],'apr_ref':[],'selected_ref':[114248],'init_ref':[],'value':[50],'color':azure},    
    20:{'bundle_name':['Tapatum L'],'apr_ref':[],'selected_ref':[202092],'init_ref':[],'value':[49],'color':azure}        

    #21:{'name':['Optic Radiation R'],'value':[30],'color':coral},
    #22:{'name':['Optic Radiation L'],'value':[29],'color':coral}

    }
    
    
    return combined_atlas



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


def relabel_by_atlas_value_and_zhang(atlas_tracks,atlas,tes,tracks,tracksd,zhang_thr):
    
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
