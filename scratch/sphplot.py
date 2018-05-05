import numpy as np
from dipy.viz import fos
import dipy.core.geometry as geometry
import matplotlib.pyplot as mplp

def plot_sphere(v,key):
    r = fos.ren()
    fos.add(r,fos.point(v,fos.green, point_radius= 0.01))
    fos.show(r, title=key, size=(1000,1000))

def plot_lambert(v,key,centre=np.array([0,0])):
    lamb = geometry.lambert_equal_area_projection_cart(*v.T).T
    (y1,y2) = lamb
    radius = np.sum(lamb**2,axis=0) < 1
    fig = mplp.figure(facecolor='w')
    current = fig.add_subplot(111)
    current.patch.set_color('k')
    current.plot(y1[radius],y2[radius],'.g')
    current.plot(y1[-radius],y2[-radius],'.r')
    current.plot([0.],[0.],'ob')
    #current.patches.Circle(*centre, radius=50, color='w', fill=True, alpha=0.7)
    current.axes.set_aspect(aspect = 'equal', adjustable = 'box')
    current.title.set_text(key)
    fig.show()
    fig.waitforbuttonpress()
    mplp.close()
    
