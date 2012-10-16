import numpy as np
from fos import Window, Region
from fos.actor.slicer import Slicer


class Guillotine(Slicer):
    def left2right(self, step):
        self.slice_i(self.i + step)


if __name__ == '__main__':

    import nibabel as nib    
    
    dname = '/usr/share/fsl/data/standard/'
    fname = dname + 'FMRIB58_FA_1mm.nii.gz'
    img=nib.load(fname)
    data = img.get_data()
    data = np.interp(data, [data.min(), data.max()], [0, 255])

    window = Window(caption="Interactive Slicer", 
                        bgcolor=(0.4, 0.4, 0.9))
    region = Region(activate_aabb=False)
    guil = Guillotine('VolumeSlicer', data)
    region.add_actor(guil)
    window.add_region(region)
    window.refocus_camera()

