from scipy.io import loadmat
import numpy as np

#dsi_steam_b4000=loadmat('/home/eg309/Desktop/20100511_M030Y_CBU100624_steam.src.gz.odf6.gqi1.3.fib',struct_as_record=True)

dsi_steam_b4000=loadmat('/home/eg01/Data_Backup/Data/Frank_Eleftherios/frank/20100511_M030Y_CBU100624_steam.src.gz.odf6.gqi1.3.fib',struct_as_record=True)

x,y,z=dsi_steam_b4000['dimension'][0]

qa0=dsi_steam_b4000['fa0'].reshape(x,y,z,order='F')

qa1=dsi_steam_b4000['fa1'].reshape(x,y,z,order='F')

qa2=dsi_steam_b4000['fa2'].reshape(x,y,z,order='F')

qa3=dsi_steam_b4000['fa3'].reshape(x,y,z,order='F')

qa4=dsi_steam_b4000['fa4'].reshape(x,y,z,order='F')


qa=np.array([qa0,qa1,qa2,qa3,qa4])

qa=np.rollaxis(qa,0,3)

qa=np.rollaxis(qa,3,2)


ind0=dsi_steam_b4000['index0'].reshape(x,y,z,order='F')

ind1=dsi_steam_b4000['index1'].reshape(x,y,z,order='F')

ind2=dsi_steam_b4000['index2'].reshape(x,y,z,order='F')

ind3=dsi_steam_b4000['index3'].reshape(x,y,z,order='F')

ind4=dsi_steam_b4000['index4'].reshape(x,y,z,order='F')


ind=np.array([ind0,ind1,ind2,ind3,ind4])

ind=np.rollaxis(ind,0,3)

ind=np.rollaxis(ind,3,2)


del qa0,qa1,qa2,qa3,qa4

del ind0,ind1,ind2,ind4


def trilinear_interpolation(X):

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
                
                [  d[0] *  d[1] *  d[2] ]] )

    

    IN=np.array([[ Xf[0]  , Xf[1]  , Xf[2] ],

                [ Xf[0]+1 , Xf[1]  , Xf[2] ],

                [ Xf[0]   , Xf[1]+1, Xf[2] ],

                [ Xf[0]   , Xf[1]  , Xf[2]+1 ],

                [ Xf[0]+1 , Xf[1]+1, Xf[2] ],

                [ Xf[0]   , Xf[1]+1, Xf[2]+1 ],

                [ Xf[0]+1 , Xf[1]  , Xf[2]+1 ],
                
                [ Xf[0]+1 , Xf[1]+1, Xf[2]+1 ]])


    return W,IN.astype(np.int)


#test with one point

point = np.array([34.0928, 58.2152, 30.3443])

w,index = trilinear_interpolation(point)


#nearest direection to a point
def nearest_direction(dx,qa,ind,odf_vertices,qa_thr=0.0245,ang_thr=60.):

    ''' Give the nearest direction to a point

    Parameters
    -----------
    
    dx: array, shape(3,), as float,
    moving direction of the current tracking

    qa: array, shape(5,), float,
    quantitative anisotropy matrix for the whole volume

    ind: array, shape(5,), float,
    index of the track orientation

    odf_vertices: array, shape(N,3), float, odf sampling directions

    qa_thr: float, threshold for QA,
    we want everything higher than this threshold 

    ang_thr: float, theshold,
    we only select fiber orientation with this
    range 


    Returns:
    --------

    delta: bool, delta funtion,
    if 1 we give it weighting
    if it is 0 we don't give any weighting

    direction: array, shape(3,),
    the fiber orientation to be consider in the interpolation
 
    

    '''

    max_dot=0

    max_doti=0

    angl = np.cos((np.pi*ang_thr)/180.) 

    if qa[0] <= qa_thr:

        return False, np.array([0,0,0])
    

    for i in range(5):

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
    
#test nearest_direction

dx=np.array([0.94184637, -0.2702789, -0.19968657])
    
qa_thr=0.0239

odf_vertices=dsi_steam_b4000['odf_vertices'].T

for m in range(8):

    i,j,k = index[m]

    qa_tmp = qa[i,j,k]

    ind_tmp = ind[i,j,k]

    delta,direction = nearest_direction(dx,qa_tmp,ind_tmp,odf_vertices,qa_thr,60.)

    #print delta, direction



def propagation_direction(point,dx,qa,ind,odf_vertices,qa_thr,ang_thr):

    total_w = 0 # total weighting

    new_direction = np.array([0,0,0])
    
    w,index=trilinear_interpolation(point)

    for i in range(3):

        if index[7][i] >= qa.shape[i] or index[0][i] < 0:

            return False, np.array([0,0,0])


    for m in range(8):


        x,y,z = index[m]

        qa_tmp = qa[x,y,z]

        ind_tmp = ind[x,y,z]

        delta,direction = nearest_direction(dx,qa_tmp,ind_tmp,odf_vertices,qa_thr,ang_thr)

        #print delta, direction

        if not delta:

            continue

        total_w += w[m]

        #print 'yo',direction,w[m][0], w[m][0]*direction

        new_direction = new_direction +  w[m][0]*direction

        #print 'new', new_direction

        

    if total_w < .5: # termination criteria

        return False, np.array([0,0,0])

    #print total_w, new_direction


    return True, new_direction/np.sqrt(np.sum(new_direction**2))
            

#testing propagation direction 

delta,pdirection = propagation_direction(point,dx,qa,ind,odf_vertices,qa_thr,60.)


point = np.array([61.092846,57.215191,33.344265])

dx = np.array([.58350962,-.47178298, -.66101241])

delta,pdirection = propagation_direction(point,dx,qa,ind,odf_vertices,qa_thr,60.)

dx = np.array([-.37201062,.61045986, -.69924736])

delta,pdirection = propagation_direction(point,dx,qa,ind,odf_vertices,qa_thr,60.)




def initial_direction(seed,qa,ind,odf_vertices,qa_thr):
    ''' First direction that we get from a seeding point

    '''

    seed+=0.5

    point=np.floor(seed)

    x,y,z = point
    
    qa_tmp=qa[x,y,z,0]

    ind_tmp=ind[x,y,z,0]

    if qa_tmp < qa_thr:

        return False, np.array([0,0,0])

    else:

        return True, odf_vertices[ind_tmp]

  


#testing initial direction

seed = np.array([62.092846, 56.215191, 34.344265])

delta,idirection=initial_direction(seed,qa,ind,odf_vertices,qa_thr)


def propagation(seed,qa,ind,odf_vertices,qa_thr,ang_thr,step_sz):

    
    d,idirection=initial_direction(seed,qa,ind,odf_vertices,qa_thr)

    print d
    
    if not d:

        return None

    dx = idirection

    point = seed

    track = []

    track.append(point)

    while d:

        d,dx = propagation_direction(point,dx,qa,ind, odf_vertices,qa_thr,ang_thr)

        
        if not d:

            break

        point = point + step_sz*dx

        track.append(point)

        
    d = True
    
    dx = - idirection

    point = seed

    while d:

        d,dx = propagation_direction(point,dx,qa,ind,\

                                     odf_vertices,qa_thr,ang_thr)

        if not d:

            break
        
        point = point + step_sz*dx

        track.insert(0,point)



    return np.array(track)
    

#testing propagation
step_sz=0.5

ang_thr = 60.

#seed = seed - 1

track=propagation(seed,qa,ind,odf_vertices,qa_thr,ang_thr,step_sz)


#testing random seeds

x,y,z,g=qa.shape

tlist=[]

for i in range(10000):
    
    rx=(x-1)*np.random.rand()
    ry=(y-1)*np.random.rand()
    rz=(z-1)*np.random.rand()

    #print 'yo', rx,ry,rz

    seed=np.array([rx,ry,rz])

    track=propagation(seed,qa,ind,odf_vertices,qa_thr,ang_thr,step_sz)

    if track== None:

        pass
    
    else:

        tlist.append(track)

