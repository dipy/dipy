import numpy as np
from scipy import optimize

def watson_parameters(evals):

    cross=np.dot(np.transpose(evals),evals)

    l,d = np.linalg.eig(cross)

    
