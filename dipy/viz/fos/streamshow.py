import numpy as np
import nibabel as nib
#fos modules
from fos import Actor, Scene
from fos.actor import Axes, Text3D
from fos.modelmat import screen_to_model
import fos.interact.collision as cll
from pyglet.gl import *
from pyglet.lib import load_library
#dipy modules
from dipy.segment.quickbundles import QuickBundles
from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle
from dipy.viz.colormap import orient2rgb
from dipy.viz.fos.guillotine import Guillotine
from dipy.tracking.metrics import downsample
from dipy.tracking.vox2track import track_counts
#other
import copy 
import cPickle as pickle
#trick for the bug of pyglet multiarrays
glib=load_library('GL')
#Tk dialogs
import Tkinter, tkFileDialog
#Pyside for windowing
from PySide.QtCore import Qt


question_message="""
>>>>Track Labeler
P : select/unselect the representative track.
E : expand/collapse the selected tracks 
F : keep selected tracks rerun QuickBundles and hide everything else.
A : select all representative tracks which are currently visible.
I : invert selected tracks to unselected
H : hide/show all representative tracks.
>>>Mouse
Left Button: keep pressed with dragging - rotation
Scrolling :  zoom
Shift + Scrolling : fast zoom
Right Button : panning - translation
Shift + Right Button : fast panning - translation
>>>General
F1 : Fullscreen.
F2 : Next time frame.
F3 : Previous time frame.
F4 : Automatic rotation.
F12 : Reset camera.
ESC: Exit.
? : Print this help information.
"""


def track2rgb(track):
    """Compute orientation of a track and retrieve and appropriate RGB
    color to represent it.
    """
    # simplest implementation:
    return orient2rgb(track[0] - track[-1])


class StreamlineLabeler(Actor):   
    
    def __init__(self, name,qb, tracks, reps='exemplars',colors=None, vol_shape=None, virtuals_line_width=5.0, tracks_line_width=2.0, virtuals_alpha=1.0, tracks_alpha=0.6, affine=None, verbose=False):
        """TrackLabeler is meant to explore and select subsets of the
        tracks. The exploration occurs through QuickBundles (qb) in
        order to simplify the scene.
        """
        super(StreamlineLabeler, self).__init__(name)

        if affine is None: self.affine = np.eye(4, dtype = np.float32)
        else: self.affine = affine      
         
        self.mouse_x=None
        self.mouse_y=None
        self.cache = {}
        self.qb = qb
        self.reps = reps
        #virtual tracks
        if self.reps=='virtuals':
            self.virtuals=qb.virtuals()
        if self.reps=='exemplars':
            self.virtuals,self.ex_ids = qb.exemplars()
        self.virtuals_alpha = virtuals_alpha
        self.virtuals_buffer, self.virtuals_colors, self.virtuals_first, self.virtuals_count = self.compute_buffers(self.virtuals, self.virtuals_alpha)
        #full tractography (downsampled at 12 pts per track)
        self.tracks = tracks
        self.tracks_alpha = tracks_alpha
        self.tracks_ids = np.arange(len(self.tracks), dtype=np.int)
        self.tracks_buffer, self.tracks_colors, self.tracks_first, self.tracks_count = self.compute_buffers(self.tracks, self.tracks_alpha)
        #calculate boundary box for entire tractography
        self.min = np.min(self.tracks_buffer,axis=0)
        self.max = np.max(self.tracks_buffer,axis=0)      
        self.vertices=self.tracks_buffer
        #coord1 = np.array([self.tracks_buffer[:,0].min(),self.tracks_buffer[:,1].min(),self.tracks_buffer[:,2].min()], dtype = 'f4')        
        #coord2 = np.array([self.tracks_buffer[:,0].max(),self.tracks_buffer[:,1].max(),self.tracks_buffer[:,2].max()], dtype = 'f4')
        #self.make_aabb((coord1,coord2),0)
        #show size of tractography buffer
        print('MBytes %f' % (self.tracks_buffer.nbytes/2.**20,))
        self.position = (0,0,0)
        #buffer for selected virtual tracks
        self.selected = []
        self.virtuals_line_width = virtuals_line_width
        self.tracks_line_width = tracks_line_width
        self.old_color = {}
        self.hide_virtuals = False
        self.expand = False
        self.verbose = verbose
        self.tracks_visualized_first = np.array([], dtype='i4')
        self.tracks_visualized_count = np.array([], dtype='i4')
        self.history = [[self.qb, self.tracks, self.tracks_ids, self.virtuals_buffer, self.virtuals_colors, self.virtuals_first, self.virtuals_count, self.tracks_buffer, self.tracks_colors, self.tracks_first, self.tracks_count]]
        #shifting of track is necessary for dipy.tracking.vox2track.track_counts
        #we also upsample using 30 points in order to increase the accuracy of track counts
        self.vol_shape = vol_shape
        if self.vol_shape !=None:
            #self.tracks_shifted =[t+np.array(vol_shape)/2. for t in self.tracks]
            self.virtuals_shifted =[downsample(t+np.array(self.vol_shape)/2.,30) for t in self.virtuals]

        else:
            #self.tracks_shifted=None
            self.virtuals_shifted=None

    def compute_buffers(self, tracks, alpha):
        """Compute buffers for GL compilation.
        """
        tracks_buffer = np.ascontiguousarray(np.concatenate(tracks).astype('f4'))
        tracks_colors = np.ascontiguousarray(self.compute_colors(tracks, alpha))
        tracks_count = np.ascontiguousarray(np.array([len(v) for v in tracks],dtype='i4'))
        tracks_first = np.ascontiguousarray(np.r_[0,np.cumsum(tracks_count)[:-1]].astype('i4'))
        
        if isinstance(tracks_count,tuple): print '== count'
        if isinstance(tracks_first,tuple): print '== first'

        return tracks_buffer, tracks_colors, tracks_first, tracks_count

    def compute_colors(self, tracks, alpha):
        """Compute colors for a list of tracks.
        """
        assert(type(tracks)==type([]))
        tot_vertices = np.sum([len(curve) for curve in tracks])
        color = np.empty((tot_vertices,4), dtype='f4')
        counter = 0
        for curve in tracks:
            color[counter:counter+len(curve),:3] = track2rgb(curve).astype('f4')
            counter += len(curve)

        color[:,3] = alpha
        return color

    def draw(self):
        """Draw virtual and real tracks.

        This is done at every frame and therefore must be real fast.
        """
        glDisable(GL_LIGHTING)
        # virtuals
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)        
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        if not self.hide_virtuals:
            glVertexPointer(3,GL_FLOAT,0,self.virtuals_buffer.ctypes.data)
            glColorPointer(4,GL_FLOAT,0,self.virtuals_colors.ctypes.data)
            glLineWidth(self.virtuals_line_width)
            glPushMatrix()
            if isinstance(self.virtuals_first, tuple): print '>> first Tuple'
            if isinstance(self.virtuals_count, tuple): print '>> count Tuple'
            glib.glMultiDrawArrays(GL_LINE_STRIP, 
                                   self.virtuals_first.ctypes.data, 
                                   self.virtuals_count.ctypes.data, 
                                   len(self.virtuals))
            glPopMatrix()
        # reals:
        if self.expand and self.tracks_visualized_first.size > 0:
            glVertexPointer(3,GL_FLOAT,0,self.tracks_buffer.ctypes.data)
            glColorPointer(4,GL_FLOAT,0,self.tracks_colors.ctypes.data)
            glLineWidth(self.tracks_line_width)
            glPushMatrix()
            glib.glMultiDrawArrays(GL_LINE_STRIP, 
                                    self.tracks_visualized_first.ctypes.data, 
                                    self.tracks_visualized_count.ctypes.data, 
                                    len(self.tracks_visualized_count))
            glPopMatrix()
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)      
        glLineWidth(1.)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)

    def process_mouse_position(self,x,y):
        self.mouse_x=x
        self.mouse_y=y

    def process_pickray(self,near,far):
        pass

    def update(self,dt):
        pass

    def select_track(self, ids):
        """Do visual selection of given virtuals.
        """
        # WARNING: we assume that no tracks can ever have color_selected as original color
        color_selected = np.array([1.0, 1.0, 1.0, 1.0], dtype='f4')
        if ids == 'all':
            ids = range(len(self.virtuals))
        elif np.isscalar(ids):
            ids = [ids]
        for id in ids:
            if not id in self.old_color:
                self.old_color[id] = self.virtuals_colors[self.virtuals_first[id]:self.virtuals_first[id]+self.virtuals_count[id],:].copy()
                new_color = np.ones(self.old_color[id].shape, dtype='f4') * color_selected
                if self.verbose: print("Storing old color: %s" % self.old_color[id][0])
                self.virtuals_colors[self.virtuals_first[id]:self.virtuals_first[id]+self.virtuals_count[id],:] = new_color
                self.selected.append(id)

    def unselect_track(self, ids):
        """Do visual un-selection of given virtuals.
        """
        if ids == 'all':
            ids = range(len(self.virtuals))
        elif np.isscalar(ids):
            ids = [ids]
        for id in ids:
            if id in self.old_color:
                self.virtuals_colors[self.virtuals_first[id]:self.virtuals_first[id]+self.virtuals_count[id],:] = self.old_color[id]
                if self.verbose: print("Setting old color: %s" % self.old_color[id][0])
                self.old_color.pop(id)
                if id in self.selected:
                    self.selected.remove(id)
                else:
                    print('WARNING: unselecting id %s but not in %s' % (id, self.selected))
                    
    def invert_tracks(self):
        """ invert selected tracks to unselected
        """        
        tmp_selected=list(set(range(len(self.virtuals))).difference(set(self.selected)))
        self.unselect_track('all')
        #print tmp_selected
        self.selected=[]
        self.select_track(tmp_selected)

    def process_messages(self,messages):
        msg=messages['key_pressed']
        #print 'Processing messages in actor', self.name, 
        #' key_press message ', msg
        if msg!=None:
            self.process_keys(msg,None)
        msg=messages['mouse_position']            
        #print 'Processing messages in actor', self.name, 
        #' mouse_pos message ', msg
        if msg!=None:
            self.process_mouse_position(*msg)

    def process_keys(self,symbol,modifiers):
        """Bind actions to key press.
        """
        prev_selected = copy.copy(self.selected)
        if symbol == Qt.Key_P:     
            print 'P'
            id = self.picking_virtuals(symbol, modifiers)
            print('Track id %d' % id)
            if prev_selected.count(id) == 0:
                self.select_track(id)
            else:
                self.unselect_track(id)
            if self.verbose: 
                print 'Selected:'
                print self.selected

        if symbol==Qt.Key_E:
            print 'E'
            if self.verbose: print("Expand/collapse selected clusters.")
            if not self.expand and len(self.selected)>0:
                tracks_selected = []
                for tid in self.selected: tracks_selected += self.qb.label2tracksids(tid)
                self.tracks_visualized_first = np.ascontiguousarray(self.tracks_first[tracks_selected, :])
                self.tracks_visualized_count = np.ascontiguousarray(self.tracks_count[tracks_selected, :])
                self.expand = True
            else:
                self.expand = False
        
        # Freeze and restart:
        elif symbol == Qt.Key_F and len(self.selected) > 0:
            print 'F'
            self.freeze()

        elif symbol == Qt.Key_A:
            print 'A'        
            print('Select/unselect all virtuals')
            if len(self.selected) < len(self.virtuals):
                self.select_track('all')
            else:
                self.unselect_track('all')
        
        elif symbol == Qt.Key_I:
            print 'I'
            print('Invert selection')
            print self.selected
            self.invert_tracks()
            
        elif symbol == Qt.Key_H:
            print 'H'
            print('Hide/show virtuals.')
            self.hide_virtuals = not self.hide_virtuals       
            
        elif symbol == Qt.Key_S:
            print 'S'
            print('Save selected tracks_ids as pickle file.')
            self.tracks_ids_to_be_saved = self.tracks_ids
            if len(self.selected)>0:
                self.tracks_ids_to_be_saved = self.tracks_ids[np.concatenate([self.qb.label2tracksids(tid) for tid in self.selected])]
            print("Saving %s tracks." % len(self.tracks_ids_to_be_saved))
            root = Tkinter.Tk()
            root.withdraw()
            pickle.dump(self.tracks_ids_to_be_saved, 
                    tkFileDialog.asksaveasfile(), 
                    protocol=pickle.HIGHEST_PROTOCOL)

        elif symbol == Qt.Key_Question:
            print question_message
        elif symbol == Qt.Key_B:
            print 'B'
            print('Go back in the freezing history.')
            if len(self.history) > 1:
                self.history.pop()
                self.qb, self.tracks, self.tracks_ids, self.virtuals_buffer, self.virtuals_colors, self.virtuals_first, self.virtuals_count, self.tracks_buffer, self.tracks_colors, self.tracks_first, self.tracks_count = self.history[-1]
                if self.reps=='virtuals':
                    self.virtuals=qb.virtuals()
                if self.reps=='exemplars':
                    self.virtuals, self.ex_ids = self.qb.exemplars()#virtuals()
                print len(self.virtuals), 'virtuals'
                self.selected = []
                self.old_color = {}
                self.expand = False
                self.hide_virtuals = False

        elif symbol == Qt.Key_G:
            print 'G'
            print('Get tracks from mask.')
            ids = self.maskout_tracks()
            self.select_track(ids)

    def freeze(self):
        print("Freezing current expanded real tracks, then doing QB on them, then restarting.")
        print("Selected virtuals: %s" % self.selected)
        tracks_frozen = []
        tracks_frozen_ids = []
        for tid in self.selected:
            print tid
            part_tracks = self.qb.label2tracks(self.tracks, tid)
            part_tracks_ids = self.qb.label2tracksids(tid)
            print("virtual %s represents %s tracks." % (tid, len(part_tracks)))
            tracks_frozen += part_tracks
            tracks_frozen_ids += part_tracks_ids
        print "frozen tracks size:", len(tracks_frozen)
        print "Computing quick bundles...",
        self.unselect_track('all')
        self.tracks = tracks_frozen
        self.tracks_ids = self.tracks_ids[tracks_frozen_ids] 
        
        root = Tkinter.Tk()
        root.wm_title('QuickBundles threshold')
        ts = ThresholdSelector(root, default_value=self.qb.dist_thr/2.0)
        root.wait_window()
        
        self.qb = QuickBundles(self.tracks, dist_thr=ts.value, pts=self.qb.pts)
        #self.qb.dist_thr = qb.dist_thr/2.
        self.qb.dist_thr = ts.value
        if self.reps=='virtuals':
            self.virtuals=qb.virtuals()
        if self.reps=='exemplars':
            self.virtuals,self.ex_ids = self.qb.exemplars()
        print len(self.virtuals), 'virtuals'
        self.virtuals_buffer, self.virtuals_colors, self.virtuals_first, self.virtuals_count = self.compute_buffers(self.virtuals, self.virtuals_alpha)
        #compute buffers
        self.tracks_buffer, self.tracks_colors, self.tracks_first, self.tracks_count = self.compute_buffers(self.tracks, self.tracks_alpha)
        # self.unselect_track('all')
        self.selected = []
        self.old_color = {}
        self.expand = False
        self.history.append([self.qb, 
                            self.tracks, 
                            self.tracks_ids, 
                            self.virtuals_buffer, 
                            self.virtuals_colors, 
                            self.virtuals_first, 
                            self.virtuals_count, 
                            self.tracks_buffer, 
                            self.tracks_colors, 
                            self.tracks_first, 
                            self.tracks_count])
        if self.vol_shape is not None:
            print("Shifting!")
            self.virtuals_shifted = [downsample(t + np.array(self.vol_shape) / 2., 30) for t in self.virtuals]
        else:
            self.virtuals_shifted = None

    def picking_virtuals(self, symbol,modifiers, min_dist=1e-3):
        """Compute the id of the closest track to the mouse pointer.
        """
        x, y = self.mouse_x, self.mouse_y
        # Define two points in model space from mouse+screen(=0) position and mouse+horizon(=1) position
        near = screen_to_model(x, y, 0)
        far = screen_to_model(x, y, 1)

        #print 'peak virtuals ', near, far, x, y
        # Compute distance of virtuals from screen and from the line defined by the two points above
        tmp = np.array([cll.mindistance_segment2track_info(near, far, xyz) \
                        for xyz in self.virtuals])
        line_distance, screen_distance = tmp[:,0], tmp[:,1]
        if False: # basic algoritm:
            # Among the virtuals within a range to the line (i.e. < min_dist) return the closest to the screen:
            closest_to_line_idx = np.argsort(line_distance)
            closest_to_line_thresholded_bool = line_distance[closest_to_line_idx] < min_dist
            if (closest_to_line_thresholded_bool).any():
                return closest_to_line_idx[np.argmin(screen_distance[closest_to_line_thresholded_bool])]
            else:
                return closest_to_line_idx[0]
        else: # simpler and apparently more effective algorithm:
            return np.argmin(line_distance + screen_distance)

    def maskout_tracks(self):
        """ retrieve ids of virtuals which go through the mask
        """
        mask = self.slicer.mask        
        #tracks = self.tracks_shifted
        tracks = self.virtuals_shifted
        #tcs,self.tes = track_counts(tracks,mask.shape,(1,1,1),True)
        tcs,tes = track_counts(tracks,mask.shape,(1,1,1),True)
        # print 'tcs:',tcs
        # print 'tes:',len(self.tes.keys())
        #find volume indices of mask's voxels
        roiinds=np.where(mask==1)
        #make it a nice 2d numpy array (Nx3)
        roiinds=np.array(roiinds).T
        #get tracks going through the roi
        # print "roiinds:", len(roiinds)
        # mask_tracks,mask_tracks_inds=bring_roi_tracks(tracks,roiinds,self.tes)
        mask_tracks_inds = []
        for voxel in roiinds:
            try:
                #mask_tracks_inds+=self.tes[tuple(voxel)]
                mask_tracks_inds+=tes[tuple(voxel)]
            except KeyError:
                pass
        mask_tracks_inds = list(set(mask_tracks_inds))
        print("Masked tracks %d" % len(mask_tracks_inds))
        print("mask_tracks_inds: %s" % mask_tracks_inds)
        return mask_tracks_inds


class ThresholdSelector(object):
    def __init__(self, parent, default_value):
        self.parent = parent
        self.s = Tkinter.Scale(self.parent, from_=1, to=30, width=25, length=300, orient=Tkinter.HORIZONTAL)
        self.s.set(default_value)
        self.s.pack()
        self.b = Tkinter.Button(self.parent, text='OK', command=self.ok)
        self.b.pack(side=Tkinter.BOTTOM)
    def ok(self):
        self.value = self.s.get()
        self.parent.destroy()



