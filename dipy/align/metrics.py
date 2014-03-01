import numpy as np

class Metric(object):

    def set_static(self):
        pass

    def set_moving(self):
        pass

    def compute_forward(self):
        pass

    def compute_backward(self):
        pass

    def energy(self):
        pass


class CrossCorrelation(Metric):

    def energy(self):
        pass
