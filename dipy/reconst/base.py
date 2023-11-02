"""

Base-classes for reconstruction models and reconstruction fits.

All the models in the reconst module follow the same template: a Model object
is used to represent the abstract properties of the model, that are independent
of the specifics of the data . These properties are reused whenever fitting a
particular set of data (different voxels, for example).


"""


class ReconstModel:
    """ Abstract class for signal reconstruction models
    """

    def __init__(self, gtab):
        """Initialization of the abstract class for signal reconstruction models

        Parameters
        ----------
        gtab : GradientTable class instance

        """
        self.gtab = gtab

    def fit(self, data, mask=None, **kwargs):
        return ReconstFit(self, data)


class ReconstFit:
    """ Abstract class which holds the fit result of ReconstModel

    For example that could be holding FA or GFA etc.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
