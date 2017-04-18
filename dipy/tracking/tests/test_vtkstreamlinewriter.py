from __future__ import print_function

import unittest
import numpy as np

import os
import tempfile

from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_raises

from dipy.tracking.vtkstreamlinewriter import vtkstreamlinewriter

class TestVTKStreamlineWriter(unittest.TestCase):
    def setUp(self):
        print("Testing dipy.tracking.vtkstreamlinewriter")
        import nibabel as nib

        print("Current directory:" + os.getcwd())

        self.img = nib.load("dipy/data/S0_10slices.nii.gz")
        imgdata = self.img.get_data()[:,:,:,0]
        self.imgdata = imgdata # np.where(imgdata > 100, imgdata, 0)

        self.streamlines = get_streamlines(self.imgdata)
        self.num_streamlines = len(self.streamlines)


    def test_initialization(self):
        vtks = vtkstreamlinewriter()
        assert_true(vtks is not None)

        vtks = vtkstreamlinewriter()
        assert_true(vtks is not None)

        vtks = vtkstreamlinewriter(streamlines=np.identity(2))
        assert_true(vtks is not None)
        assert_equal(0, vtks.get_number_of_streamlines())

        vtks = vtkstreamlinewriter(streamlines=[np.identity(2), np.identity(3), np.identity(4)])
        assert_true(vtks is not None)
        assert_equal(0, vtks.get_number_of_streamlines())

        vtks = vtkstreamlinewriter(streamlines=self.streamlines)
        assert_true(vtks is not None)
        assert_equal(vtks.get_number_of_streamlines(), self.num_streamlines)

        vtks = vtkstreamlinewriter(streamlines=self.streamlines)
        assert_true(vtks is not None)
        assert_equal(vtks.get_number_of_streamlines(), self.num_streamlines)

    def test_affine(self):
        vtks = vtkstreamlinewriter()

        assert_true((vtks.get_affine() == np.eye(4)).all())
        assert_true(vtks.set_affine())
        assert_true(vtks.set_affine(None))
        assert_false(vtks.set_affine(np.eye(3)))
        assert_true(vtks.set_affine(self.img.get_affine()))


    def test_setting_streamlines(self):
        vtks = vtkstreamlinewriter()

        assert_equal(vtks.get_number_of_streamlines(), 0)

        assert_true(vtks.set_streamlines(self.streamlines))
        assert_equal(vtks.get_number_of_streamlines(), self.num_streamlines)

        assert_true(vtks.set_streamlines(self.streamlines))
        assert_equal(vtks.get_number_of_streamlines(), self.num_streamlines)

        vtks = vtkstreamlinewriter()
        assert_true(vtks is not None)
        assert_false(vtks.set_streamlines(np.identity(2)))




    def test_saving(self):
        vtks = vtkstreamlinewriter()

        assert_false(vtks.save(get_temp_file_name(suffix="MyStreamlines_withVerts.vtk"), save_verts=True))

        assert_true(vtks.set_streamlines(self.streamlines))

        assert_false(vtks.save(None))

        assert_true(vtks.save(get_temp_file_name(suffix="MyStreamlines_withVerts.vtk"), ascii=True, save_verts=True))

        assert_true(vtks.save(get_temp_file_name(suffix="MyStreamlines_withoutVerts.vtk"), save_verts=False))

        assert_true(vtks.save(get_temp_file_name(suffix="MyStreamlines_withoutAffine.vtk"), save_verts=False, ignore_affine=True))

        assert_false(vtks.save(np.zeros(3)))

        assert_false(vtks.save(None))

        assert_false(vtks.save("", save_verts=True))

        # do not know how to catch error writing file
        # assert_false(vtks.save("/paththatdoesnotexist/for/sure/MyStreamlines_withoutVerts.vtk", save_verts=False))

    def test_decorating(self):
        vtks = vtkstreamlinewriter()

        assert_equal(len(vtks.get_vertex_value_names()), 0)
        assert_equal(len(vtks.get_streamline_value_names()), 0)
        assert_false(vtks.decorate(self.imgdata, "Image_Data"))

        assert_true(vtks.set_streamlines(self.streamlines))
        assert_equal(len(vtks.get_vertex_value_names()), 0)
        assert_equal(len(vtks.get_streamline_value_names()), 0)
        assert_true(vtks.decorate(self.imgdata, "Image_Data"))
        assert_equal(len(vtks.get_vertex_value_names()), 1)
        assert_true(vtks.decorate(np.asarray(range(0, vtks.get_number_of_streamlines()), float), "Number"))
        assert_equal(len(vtks.get_streamline_value_names()), 1)

        assert_true(vtks.set_streamlines(self.streamlines))
        assert_equal(len(vtks.get_vertex_value_names()), 0)
        assert_equal(len(vtks.get_streamline_value_names()), 0)
        assert_true(vtks.decorate(self.imgdata, "Image_Data"))
        assert_equal(len(vtks.get_vertex_value_names()), 1)
        assert_equal(len(vtks.get_streamline_value_names()), 0)

        assert_true(vtks.decorate(np.asarray(range(0, vtks.get_number_of_streamlines()), float), "Number"))

        assert_true(vtks.decorate(np.asarray(range(0, vtks.get_number_of_streamlines()), np.float), "Number_np_float"))

        assert_true(vtks.decorate(np.asarray(range(0, vtks.get_number_of_streamlines()), np.float64), "Number_np_float64"))

        assert_true(vtks.decorate(np.asarray(range(0, vtks.get_number_of_streamlines()), np.float128), "Number_np_float_128"))

        assert_true(vtks.decorate(np.asarray(range(0, vtks.get_number_of_streamlines()), int), "Number_int"))

        assert_false(vtks.decorate(np.asarray(range(0, vtks.get_number_of_streamlines()-1), float), "Too_Few"))

        assert_false(vtks.decorate(np.asarray(range(0, vtks.get_number_of_streamlines()+1), float), "Too_Many"))

        assert_false(vtks.decorate((np.identity(2), float), "Wrong_Data_Dimensions"))

        assert_true(isinstance(vtks.get_streamline_value_names(), list))
        assert_true(isinstance(vtks.get_streamline_value_names()[0], basestring))
        assert_true(isinstance(vtks.get_streamline_value_names(), list))
        assert_true(isinstance(vtks.get_streamline_value_names()[0], basestring))

        assert_true(vtks.save(get_temp_file_name(suffix="MyStreamlines_withDecorations.vtk"), save_verts=False, ascii=False))


def get_streamlines(imgdata):
    dim_x = imgdata.shape[0]
    dim_y = imgdata.shape[1]
    dim_z = imgdata.shape[2]

    streamlines = list()
    # add streamline that reaches outside the image
    sl = np.empty([0, 3])
    for i in xrange(0, 101):
        j = i / 100.0
        xx = j * (dim_x + 2) - 1
        yy = j * (dim_y + 2) - 1
        zz = j * (dim_z + 2) - 1
        sl = np.append(sl, [[ xx, yy, zz ]], axis=0)
    streamlines.append(sl)

    # add parallel streamlines
    zz = dim_z/2.0
    for xx in range(0, dim_x):
        sl = np.empty([0, 3])
        for yy in range(0, dim_y):
            sl = np.append(sl, [[xx, yy, zz]], axis=0)
        streamlines.append(sl)

    # add an additional curved streamline
    sl = np.empty([0, 3])
    zz = dim_z/2. + 1
    aa = 0.04
    bb = dim_x / 2.0
    for xx in xrange(0, dim_x):
        yy = aa * (xx - bb)**2
        sl = np.append(sl, [[ xx, yy, zz ]], axis=0)
    streamlines.append(sl)

    return streamlines


def get_temp_file_name(suffix):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_name = tmp.name
    tmp.close()
    return tmp_name