"""This is a document to help dipy developers write Diffusion Models and Tests.


"""
import numpy as np
import numpy.testing as nt
from dipy.reconst import odf
from ...core.sphere import interp_rbf, Sphere, unit_icosahedron

""" A diffusion model is something that holds our assumptions about diffusion
data. At the most basic level, a diffusion model should provide a fit method
that takes a 1d array of data values and returns a some fit object. Here is an
example of a silly model class. """

class SillyModel(object):
    # This model assumes that fiber tracts lie along the N smallest recorded
    # signals
    def __init__(self, bval, bvec, N=1):
        self.bval = bval
        self.bvec = bvec
        self.N = 1

    def fit(self, data):
        return SillyFit(self, data)

""" The fit object returned by Model has data specific knowledge that the model
does not have. The fit object should have attributes and methods appropriate
for that kind of fit, ie a tensor fit may have an eigenvalues and eigenvectors
as attributes. The fit object may have a directions attribute to identify fiber
tracking directions. If the fit object has a directions attribute, it should be
an (N, 3) ndarray of N unit vectors, one for each fiber population. Here is an
example of a Silly Fit class. """

class SillyFit(object):

    def __init__(self, model, data):
        # Many fit objects will save some compact representation of the data
        # but in this case we will simply save a reference to data.
        self.data = data
        self.model = model

    @property
    def directions(self):
        # Return the directions of the smallest recorded signals
        argsort = np.argsort(self.data)
        smallest = argsort[:self.model.N]
        return bvec[smallest]

""" In addition to the general idea of a model, we also define the concept of
an odf models.  An odf model is any model class that represents a continuous
function on the sphere, or an odf. An odf model should return a fit object with
an odf method. The odf method should take as it's input an instance of Sphere
(dipy.core.sphere.Sphere).

Because many odf models identify fiber populations by finding peaks on the odf,
this direction finding functionality has been built into the abstract OdfModel
and OdfFit classes. In order to take advantage of this functionality, a
developer can subclass from the OdfModel and OdfFit and override the
OdfModel.fit and OdfFit.odf methods. For example: """

class SillyOdfModel(odf.OdfModel):
    """SillyOdfModel subclasses from OdfModel to get it's direction finding
    functionality"""

    def __init__(self, bval, bvec):
        self.bval = bval
        self.bvec = bvec

    def fit(self, data):
        """Override the fit method to subclass OdfModel"""
        return SillyOdfFit(self, data)


class SillyOdfFit(odf.OdfFit):
    """SillyOdfFit subclasses from OdfFit to get it's directions attribute"""

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def odf(self, sphere):
        """Override the odf method to subclass OdfFit"""
        bvec = self.model.bvec[self.model.bval > 0]
        data = self.data[self.model.bval > 0]
        origin_sphere = Sphere(xyz=bvec)
        discrete_odf = 1. / data
        return interp_rbf(discrete_odf, origin_sphere, sphere)


""" The following tests can be subclassed to help with model and fit testing"""

bvec = np.array([[0, 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
bval = np.array([0., 1000, 1000, 1000])
data = np.array([1., .3, .3, .3])


class TestFit(object):
    """This class tests that the directions attribute of the fit"""

    # Replace this fit in subclass to test another fit class
    fit = SillyModel(bval, bvec, 2).fit(data)

    def test_fit_medod(self):
        # We test that the fit object has a directions attribute
        nt.assert_(hasattr(self.fit, "directions"))
        # We test that directions is a (N, 3) array
        nt.assert_(self.fit.directions.ndim == 2)
        nt.assert_(self.fit.directions.shape[1] == 3)


class TestOdfFit(TestFit):
    """This class runs the test from TestFit and also test that the odf method
    of the fit"""

    fit = SillyOdfModel(bval, bvec).fit(data)

    def test_odf_method(self):
        sphere = unit_icosahedron
        # We test that the fit has an odf method that can take a sphere
        # argument
        odf = self.fit.odf(sphere)
        # We test that the odf method return odf values on the points of the
        # sphere
        nt.assert_equal(len(odf), len(sphere.phi))

