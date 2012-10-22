import numpy as np
import nibabel as nib
from dipy.segment.quickbundles import QuickBundles
from dipy.viz.fos.streamshow import StreamlineLabeler
from dipy.viz.fos.streamwindow import Window
from dipy.viz.fos.guillotine import Guillotine
from dipy.io.dpy import Dpy
from dipy.tracking.metrics import downsample
from fos import Scene

def rotation_matrix(axis, theta_degree):
    theta = 1. * theta_degree * np.pi / 180.
    axis = 1. * axis / np.sqrt(np.dot(axis,axis))
    a = np.cos(theta / 2)
    b, c, d = - axis * np.sin(theta / 2)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])


if __name__ == '__main__':
    
    #load T1 volume registered in MNI space 
    dname='/home/eg309/Devel/fos_legacy/applications/'
    img = nib.load(dname+'data/subj_05/MPRAGE_32/T1_flirt_out.nii.gz')
    data = img.get_data()
    affine = img.get_affine()

    #load the tracks registered in MNI space 
    fdpyw = dname+'data/subj_05/101_32/DTI/tracks_gqi_1M_linear.dpy'    
    dpr = Dpy(fdpyw, 'r')
    T = dpr.read_tracks()
    dpr.close() 
    
    T = T[:2000]

    T = [downsample(t, 12) - np.array(data.shape[:3]) / 2. for t in T]
    axis = np.array([1, 0, 0])
    theta = - 90. 
    T = np.dot(T,rotation_matrix(axis, theta))
    axis = np.array([0, 1, 0])
    theta = 180. 
    T = np.dot(T, rotation_matrix(axis, theta))
    
    #load initial QuickBundles with threshold 30mm
    #fpkl = dname+'data/subj_05/101_32/DTI/qb_gqi_1M_linear_30.pkl'
    qb=QuickBundles(T, 30., 12)
    #save_pickle(fpkl,qb)
    #qb=load_pickle(fpkl)

    #create the interaction system for tracks 
    tl = StreamlineLabeler('Bundle Picker', 
                        qb,qb.downsampled_tracks(), 
                        vol_shape=None, 
                        tracks_alpha=1)   

    title = 'Streamline Interaction and Segmentation'
    w = Window(caption = title, 
                width = 1200, 
                height = 800, 
                bgcolor = (.5, .5, 0.9) )

    scene = Scene(scenename = 'Main Scene', activate_aabb = False)

    data = np.interp(data, [data.min(), data.max()], [0, 255])    
    guil = Guillotine('Volume Slicer', data)

    scene.add_actor(guil)
    scene.add_actor(tl)

    w.add_scene(scene)
    w.refocus_camera()

