"""

Base-classes for ivim model and fits.

All the models in the reconst module follow the same template: a Model object
is used to represent the abstract properties of the model, that are independent
of the specifics of the data . These properties are reused whenver fitting a
particular set of data (different voxels, for example).


"""


class ReconstModel(object):
    """ Abstract class for ivim models
    """

    def __init__(self, gtab):
        """Initialization of the abstract class for ivim model

        Parameters
        ----------
        gtab : GradientTable class instance

        """
        self.gtab = gtab

    def fit(self, data, mask=None, **kwargs):
        return ReconstFit(self, data)


class ReconstFit(object):
    """ Abstract class which holds the fit result of IvimModel

    For example that could be holding img, S0, D .... 
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
