import numpy as np


def load(fname,maxsize=3000000):
    fp = np.memmap(fname, dtype=np.object, mode='r', shape=(maxsize,))
    return fp

def save(fname,maxsize=3000000):    
    fp = np.memmap(fname, dtype=np.object, mode='w+', shape=(maxsize,))
    return fp


if __name__ == '__main__':
    
    fname='dummy.dpy'
    
    T=[np.ones((4,3)),2*np.ones((5,3)),3*np.ones((3,3))]
    Tn=np.array(T,dtype=np.object)

    fp = save(fname)
    for i in range(len(Tn)):
        fp[i]=Tn[i]
    print(fp)
    del fp

    fp2 = load(fname)
    print 'read'
    a=fp2[0]
    i=0
    while a != None:    
        print(fp2[i])
        a=fp2[i]
        i+=1    
    print i




                            