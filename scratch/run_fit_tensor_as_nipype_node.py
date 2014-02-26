import os
from dipy.data import fetch_stanford_hardi
from dipy.workflows.interafes import DipyFitTensor

# Download example data set
fetch_stanford_hardi()
data_dir = os.path.join(os.path.expanduser('~'), '.dipy/stanford_hardi/')

node = DipyFitTensor()
node.inputs.dwi_data = data_dir + "HARDI150.nii.gz"
node.inputs.bvec = data_dir + "HARDI150.bvec"
# Save result in current directory
node.inputs.root = "./HARDI"

node.run()
