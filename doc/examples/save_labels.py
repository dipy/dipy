"""

===========================
Save cluster labels in TRX files
===========================

Overview
========

Other than reading and writing file formats, DIPY_ can also save the cluster 
labels in using TRX files. In this example we give a short introduction on how to 
use it for saving the cluster centroids as well as streamlines with the labels.If you are not
familiar with the tractography clustering framework, read the
:ref:`clustering-framework` first.

.. contents:: Available Features
    :local:
    :depth: 1

Firstly import the necessary modules.
"""
from dipy.io.streamline import StatefulTractogram, save_tractogram, load_tractogram, Space
from dipy.segment.clustering import QuickBundles
from dipy.data import get_fnames
from dipy.tracking.streamline import Streamlines
import nibabel as nib

"""
Secondly. Get some streamlines

We use get_streamlines function defined below, for the same.
Fornix is a small streamline bundle known from neuroanatomy. This should be ideal for this example

Note that below,  "fornix" will be our reference anatomy, and fonixstreamlines will have the streamlines
"""
fname = get_fnames('fornix')
fornix = load_tractogram(fname, 'same',
                             bbox_valid_check=False)
fornix_Streamlines=fornix.streamlines
"""
We'll first look at saving streamline files

.. _Saving-Labels-Streamlines:

Saving Labels for Streamlines
=============================

Step 1. Creating labels for each streamline
"""
labels = {}
for index in enumerate(fornix_Streamlines):
    labels[index[0]]="label_"+str(index[0])

'''
The number of labels can be given by: 
'''
print("Total labels:", len(labels))

"""
::

    Total labels: 300
"""

"""
step 2. Save the TRX File

We use data_per_streamline to save the labels. the type must be a dictionary, 
with size of values same as number of streamlines. We then call the save_tractogram function 
to save the file.

"""

sft2 = StatefulTractogram(fornix_Streamlines, reference=fornix, space=Space.VOX, data_per_streamline={'Labels': labels})
save_tractogram(sft2, 'filename.trk',bbox_valid_check=False)

"""
Now lets look at saving clusters with labels

.. _Saving-Labels-Clusters:

Saving Labels for Clusters
==========================

We'll use the centroid of each cluster as our object.

Step 1. Creating clusters
The Quickbundles Function creates the bundles of streamlines with threshold of 10 mm.
The cluster() returns a ClusterMap object which stores the metadata about the clusters
"""
qb = QuickBundles(threshold=10.)
clusters = qb.cluster(fornix_Streamlines)

"""
Step 2. Creating labels for each clusters
"""

labels = {}
for index in enumerate(clusters.centroids):
    labels[index[0]]="label_"+str(index[0])

'''
The number of labels can be given by: 
'''
print("Total labels:", len(labels))

"""
::

    Total labels: 4
"""

"""
step 3. Save the TRX File

We again use data_per_streamline to save the labels.
Also instead of 'fornix_Streamlines', we'll replace it with "cluster.centroids"

"""

sft2 = StatefulTractogram(clusters.centroids, reference=fornix, space=Space.VOX, data_per_streamline={'Labels': labels})
save_tractogram(sft2, 'filename.trk',bbox_valid_check=False)