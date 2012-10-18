import numpy as np
from fos import Window, Scene
from fos.actor.slicer import Slicer
from pyglet.gl import *


class Guillotine(Slicer):
    """ Head slicer actor
    
    """

    def draw(self):
        
        #i slice
        if self.show_i: 
            glPushMatrix()
            glRotatef(90, 0., 1., 0)
            glRotatef(90, 0., 0., 1.)
            self.tex.update_quad(self.texcoords_i, self.vertcoords_i)
            self.tex.set_state()
            self.tex.draw()
            self.tex.unset_state()
            glPopMatrix()
        
        #j slice
        if self.show_j:
            glPushMatrix()
            glRotatef(180, 0., 1., 0) # added for fsl convention
            glRotatef(90, 0., 0., 1.)
            self.tex.update_quad(self.texcoords_j, self.vertcoords_j)
            self.tex.set_state()
            self.tex.draw()
            self.tex.unset_state()
            glPopMatrix()

        #k slice
        if self.show_k:
            glPushMatrix()
            glRotatef(90, 1., 0, 0.)
            glRotatef(90, 0., 0., 1)
            glRotatef(180, 1., 0., 0.) # added for fsl
            self.tex.update_quad(self.texcoords_k, self.vertcoords_k)
            self.tex.set_state()
            self.tex.draw()
            self.tex.unset_state()
            glPopMatrix()

    def right2left(self, step):
        if self.i + step < self.I:
            self.slice_i(self.i + step)
        else:
            self.slice_i(self.I - 1)

    def left2right(self, step):
        if self.i - step >= 0:
            self.slice_i(self.i - step)
        else:
            self.slice_i(0)

    def inferior2superior(self, step):
        if self.k + step < self.K:
            self.slice_k(self.k + step)
        else:
            self.slice_k(self.K - 1)

    def superior2inferior(self, step):
        if self.k - step >= 0:
            self.slice_k(self.k - step)
        else:
            self.slice_k(0)

    def anterior2posterior(self, step):
        if self.j + step < self.J:
            self.slice_j(self.j + step)
        else:
            self.slice_j(self.J - 1)

    def posterior2anterior(self, step):
        if self.j - step >= 0:
            self.slice_j(self.j - step)
        else:
            self.slice_j(0)

    def reset_slices(self):
        self.slice_i(self.I / 2)
        self.slice_j(self.J / 2)
        self.slice_k(self.K / 2)

    def slices_ijk(self, i, j, k):
        self.slice_i(i)
        self.slice_j(j)
        self.slice_k(k)
        
    def show_coronal(self, bool=True):
        self.show_k = bool

    def show_axial(self, bool=True):
        self.show_i = bool

    def show_saggital(self, bool=True):
        self.show_j = bool

    def show_all(self, bool=True):
        self.show_i = bool
        self.show_j = bool
        self.show_k = bool


if __name__ == '__main__':

    import nibabel as nib    
    
    dname = '/usr/share/fsl/data/standard/'
    fname = dname + 'FMRIB58_FA_1mm.nii.gz'
    img=nib.load(fname)
    data = img.get_data()
    data = np.interp(data, [data.min(), data.max()], [0, 255])

    window = Window(caption="[F]OS", bgcolor=(0.4, 0.4, 0.9))
    scene = Scene(activate_aabb=False)
    guil = Guillotine('VolumeSlicer', data)
    scene.add_actor(guil)
    window.add_scene(scene)
    window.refocus_camera()

