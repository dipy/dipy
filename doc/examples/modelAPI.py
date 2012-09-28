
"""

Implementing new models
=======================


This document is meant to help dipy developers writing implementation of new
diffusion models. It will explain the different components that need to be in
place when you implement a new diffusion model. To be useful in fitting models
to data and in using, the different models need to conform to the
specifications of a particular API (or application programming interface). This
document describes and demostrates this API.

We will start by importing a variety of stuff to use later on:

"""

import numpy as np
import numpy.testing as nt
from dipy.reconst import odf
from dipy.core.sphere import interp_rbf, Sphere, unit_icosahedron

"""

A diffusion Model class is something that holds our assumptions about diffusion
data, but is independent of the specifics of any particular set of data, or
signal measured in any particular data set. For example, this class might hold
the design matrix used in fitting the diffusion tensor, which is the same
regardless of the data (and constant across voxels for the same measurement
scheme). 

At the very least, a diffusion model should provide a ``fit`` method
that takes a 1d array of data values and returns a Fit object (see below). Here
is an example of a silly model class, which we will appropriately name
``SillyModel``. 

"""

class SillyModel(object):
    # This model assumes that fiber tracts lie along the N smallest recorded
    # signals
    def __init__(self, bval, bvec, N=1):
        self.bval = bval
        self.bvec = bvec
        self.N = 1

    def fit(self, data):
        return SillyFit(self, data)

"""

Note that the ``fit`` method of the ``SillyModel`` class simply returns a
``SillyFit`` class instance. In the more general case, this method can also do
other things, such as fit model parameters to the data and pass those to the
Fit object. Look at implementations of some of the models in the ``reconst``
module to get an idea of how that is implemented in some of the already
existing code. 

The Fit object returned by Model now has data specific knowledge that the model
does not have. The fit object should have attributes and methods appropriate
for that kind of fit. For example, the ``TensorFit`` class has methods to
calculate eigenvalues and eigenvectors, given a certain set of data.

In order to be useful for tracking, and to be consistent with the tracking API
implemented in the ``tracking`` module, the fit object should implement a
``directions`` attribute to identify the directions along which fiber tracking
should progress in the particular location of this data. The ``directions``
attribute, should always return an (N, 3) ndarray of N unit vectors, one for
each fiber population or direction along which to track.

Here is an example of a ``SillyFit`` class:

"""

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

"""

In addition to the general idea of a Model, we also define the concept of an
ODF Model.  An OdfModel is any model class that represents a continuous
function on the sphere, or an orientation distribution function (ODF). This
kind of model should implement a ``fit`` method as well, which returns an
OdfFit class instance. OdfFit classes in turn should implement an ``odf``
method, which takes an instance of ``Sphere`` (from
``dipy.core.sphere.Sphere``) as its input and always returns the values of the
orientation distribution function on the vertices of the provided sphere. This
implies that the OdfFit class ``odf`` method should know how to calculate the
value of the orientation distribution function at any arbitrary point on the
sphere (not only the points where measurements were obtained).   

Because many ODF models identify fiber populations by finding peaks on the odf,
this direction finding functionality has been built into the abstract OdfModel
and OdfFit classes. In order to take advantage of this functionality, a
developer can subclass from the OdfModel and OdfFit and override the
OdfModel.fit and OdfFit.odf methods.

Here are example implementation s of ``SillyOdfModel`` and ``SillyOdfFit``
classes : 


"""

class SillyOdfModel(odf.OdfModel):
    ## SillyOdfModel subclasses from OdfModel to get it's direction finding
    ## functionality

    def __init__(self, bval, bvec):
        self.bval = bval
        self.bvec = bvec

    def fit(self, data):
        ## Override the fit method to subclass OdfModel
        return SillyOdfFit(self, data)


class SillyOdfFit(odf.OdfFit):
    ## SillyOdfFit subclasses from OdfFit to get it's directions attribute

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def odf(self, sphere):
        ## Override the odf method to subclass OdfFit

        bvec = self.model.bvec[self.model.bval > 0]
        data = self.data[self.model.bval > 0]
        origin_sphere = Sphere(xyz=bvec)
        discrete_odf = 1. / data
        return interp_rbf(discrete_odf, origin_sphere, sphere)


"""

To test the implementation of the Fit and OdfFit classes, you can test them in
the following manner: 

"""

bvec = np.array([[0, 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
bval = np.array([0., 1000, 1000, 1000])
data = np.array([1., .3, .3, .3])


class TestFit(object):
    ##This class tests that the directions attribute of the fit

    # Replace this fit in subclass to test another fit class
    fit = SillyModel(bval, bvec, 2).fit(data)

    def test_fit_medod(self):
        # We test that the fit object has a directions attribute
        nt.assert_(hasattr(self.fit, "directions"))
        # We test that directions is a (N, 3) array
        nt.assert_(self.fit.directions.ndim == 2)
        nt.assert_(self.fit.directions.shape[1] == 3)


class TestOdfFit(TestFit):
    ## This class runs the test from TestFit and also test that the odf method
    ## of the fit
    
    fit = SillyOdfModel(bval, bvec).fit(data)

    def test_odf_method(self):
        sphere = unit_icosahedron
        # We test that the fit has an odf method that can take a sphere
        # argument
        odf = self.fit.odf(sphere)
        # We test that the odf method return odf values on the points of the
        # sphere
        nt.assert_equal(len(odf), len(sphere.phi))

