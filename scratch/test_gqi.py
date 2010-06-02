from scipy.io import loadmat
import numpy as np
#?loadmat

#phantom=loadmat('/home/eg309/Desktop/phantom_test_data.mat',struct_as_record=True)
phantom=loadmat('/home/ian/Data/Frank_Eleftherios/phantom_test_data.mat',struct_as_record=True)

all=phantom['all']
b_table=phantom['b_table']
odf_vertices=phantom['odf_vertices']
odf_faces=phantom['odf_faces']

#s = all[14,14,1]

#Yeh et.al, IEEE TMI, 2010
#calculate the odf using GQI

scaling=np.sqrt(b_table[0]*0.01506) # 0.01506 = 6*D where D is the free
# water diffusion coefficient 
# l_values sqrt(6 D tau) D free water
# diffusio coefficiet and tau included in the b-value
# sqrt(6Db(q))

tmp=np.tile(scaling,(3,1))

b_vector=b_table[1:4,:]*tmp # sqrt(6Db(q)).(q/|q|)

Lambda = 1.2 # smoothing parameter - diffusion sampling length

#np.dot(b_vector.T, odf_vertices) is  before.u where before is sqrt(6Db(q)).(q/|q|)
q2odf_params=np.sinc(np.dot(b_vector.T, odf_vertices) * Lambda/np.pi) # implements equation no. 9 from Yeh et.al.


def Q2odf(s,q2odf_params):


    odf=np.dot(s,q2odf_params)

    return odf

def peak_finding(odf,odf_faces,odf_vertices):

    #proton density already include from the scaling b_table[0][0] and s[0]


    #find local maxima

    peak=odf.copy()

    # where the smallest odf values in the vertices of a face remove the
    # two smallest vertices 

    for face in odf_faces.T:

        i, j, k = face

        check=np.array([odf[i],odf[j],odf[k]])

        zeroing=check.argsort()

        peak[face[zeroing[0]]]=0

        peak[face[zeroing[1]]]=0

    #for later testing expecting peak.max 794595.94774980657 and
    #np.where(peak>0) (array([166, 347]),)


    #we just need the first half of peak

    peak=peak[0:len(peak)/2]

    #find local maxima and give fiber orientation (inds) and magnitude
    #peaks in a descending order

    inds=np.where(peak>0)[0]

    pinds=np.argsort(peak[inds])
    
    peaks=peak[inds[pinds]][::-1]


    return peaks, inds[pinds][::-1]




'''

#s = all[14,14,1]

s = all[0,0,0]

#s = all[]
        
odf = Q2odf(s,q2odf_params)

peaks,inds=peak_finding(odf,odf_faces,odf_vertices)

'''



S=all

x,y,z,g=S.shape

S=S.reshape(x*y*z,g)

QA = np.zeros((x*y*z,5))

IN = np.zeros((x*y*z,5))

fwd = 0

#Calculate Quantitative Anisotropy and find the peaks and the indices
#for every voxel

for (i,s) in enumerate(S):

    odf = Q2odf(s,q2odf_params)

    peaks,inds=peak_finding(odf,odf_faces,odf_vertices)

    fwd=max(np.max(odf),fwd) #finding the maximum value for the whole brain

    peaks = peaks - np.min(odf) # removes the isotropic part

    l=min(len(peaks),5)

    QA[i][:l] = peaks[:l]

    IN[i][:l] = inds[:l]

    
QA/=fwd

#If you just want to generate a volume then is enough to use the first QA[0]


#Generate streamlines














