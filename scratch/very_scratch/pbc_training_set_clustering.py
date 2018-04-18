import time
import numpy as np
import dipy.core.track_performance as pf
import dipy.io.pickles as pkl
import dipy.core.track_metrics as tm
from dipy.viz import fos


#fname='/home/eg01/Data/tmp/pbc_training_set.pkl'
#fname='/tmp/pbc_training_set.pkl'
fname='/home/eg309/Data/PBC/pbc2009icdm/fornix.pkl'



def show_rep3(C,r=None,color=fos.white):

    if r==None: r=fos.ren()

    for c in C:
        fos.add(r,fos.line(C[c]['rep3']/C[c]['N'],color))

    fos.show(r)

    return r


def merge(C,thr):


    k=C.keys()

    #print 'k', k

    to_be_deleted=np.zeros(len(k))

    if len(k)<=1: return C

    for i in range(1,len(k)-1):

        c=k[i]

        for j in range(i+1,len(k)):

    
            h=k[j]

            #print i,j
            
            t1=C[c]['rep3']/C[c]['N']
            t2=C[h]['rep3']/C[h]['N']

            #print 'yo',tm.zhang_distances(t1,t2,'avg')

            if tm.zhang_distances(t1,t2,'avg') < thr:

                C[h]['indices']+=C[c]['indices']
                C[h]['N']+=C[c]['N']
                C[h]['rep3']+=C[c]['rep3']

                to_be_deleted[i]=1
                

    for i in np.where(to_be_deleted>0)[0]: del C[k[i]] 
        
    return C


def most(C):

    for _ in C:
        pass # pf.most_similar_track_mam()


T=pkl.load_pickle(fname)

print 'Reducing the number of points...'
T=[pf.approx_polygon_track(t) for t in T]

print 'Reducing further to tracks with 3 pts...'
T2=[tm.downsample(t,3) for t in T]

print 'LARCH ...'
print 'Splitting ...'
t=time.clock()
C=pf.larch_3split(T2,None,5.)
print time.clock()-t, len(C)

for c in C: print c, C[c]['rep3']/C[c]['N']

r=show_rep3(C)


print 'Merging ...'
t=time.clock()
C=merge(C,5.)
print time.clock()-t, len(C)

for c in C: print c, C[c]['rep3']/C[c]['N']

show_rep3(C,r,fos.red)




'''

#print 'Showing initial dataset.'
r=fos.ren()
#fos.add(r,fos.line(T,fos.white,opacity=1))
#fos.show(r)

print 'Showing dataset after clustering.'
#fos.clear(r)

colors=np.zeros((len(T),3))
for c in C:
    color=np.random.rand(1,3)
    for i in C[c]['indices']:
        colors[i]=color
fos.add(r,fos.line(T,colors,opacity=1))
fos.show(r)

print 'Some statistics about the clusters'
print 'Number of clusters',len(C.keys())
lens=[len(C[c]['indices']) for c in C]
print 'max ',max(lens), 'min ',min(lens)
    
print 'singletons ',lens.count(1)
print 'doubletons ',lens.count(2)
print 'tripletons ',lens.count(3)


print 'Showing dataset after merging.'
fos.clear(r)

T=[t + np.array([120,0,0]) for t in T]

colors=np.zeros((len(T),3))
for c in C2:
    color=np.random.rand(1,3)
    for i in C2[c]['indices']:
        colors[i]=color
fos.add(r,fos.line(T,colors,opacity=1))
fos.show(r)

print 'Some statistics about the clusters'
print 'Number of clusters',len(C.keys())
lens=[len(C2[c]['indices']) for c in C]
print 'max ',max(lens), 'min ',min(lens)
    
print 'singletons ',lens.count(1)
print 'doubletons ',lens.count(2)
print 'tripletons ',lens.count(3)

'''
