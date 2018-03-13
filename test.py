from dipy.viz import fvtk
r=fvtk.ren()
l=fvtk.label(r,scale=(1,1,1),text='adsadsadsadsadsadasdsadsad')
fvtk.add(r,l)
fvtk.show(r)