import sympy
import numpy as np
import scipy as sc
from numpy.random import random_sample as random

def random_uniform_in_disc():
    # returns a tuple which is uniform in the disc
    theta = 2*np.pi*random()
    r2 = random()
    r = np.sqrt(r2)
    return np.array((r*np.sin(theta),r*np.cos(theta)))

def random_uniform_in_ellipse(a=1,b=1):
    x = a*random_uniform_in_disc()[0]
    y = b*np.sqrt(1-(x/a)**2)*(1-2*random())
    return np.array((x,y))

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
sample = np.array([random_uniform_in_ellipse(a=2,b=1) for i in np.arange(10000)])
ax.scatter(*sample.T)
plt.show()



