

class ReconstModel(object):
    """ Abstract class for signal reconstruction models
    """
    def get_directions(self, sig):
        """ returns Nx3 array of unit vectors
        """
        raise NotImplementedError()

    def fit(self, data, mask=None,**kargs):
        raise NotImplementedError()

class ReconstFit(object):
    """ Abstract class which holds the fit result of ReconstModel

    For example that could be holding FA or GFA etc.
    """
    pass
