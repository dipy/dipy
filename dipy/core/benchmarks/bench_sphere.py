""" Benchmarks for sphere

Run all benchmarks with::

    import dipy.core as dipycore
    dipycore.bench()

With Pytest, Run this benchmark with:

    pytest -svv -c bench.ini /path/to/bench_sphere.py
"""

import sys
import time

import dipy.core.sphere_stats as sphere_stats
import dipy.core.sphere as sphere

from matplotlib import pyplot as plt


mode = None
if len(sys.argv) > 1 and sys.argv[1] == '-s':
    mode = "subdivide"

class Timer(object):
    def __enter__(self):
        self.__start = time.time()

    def __exit__(self, type, value, traceback):
        # Error handling here
        self.__finish = time.time()

    def duration_in_seconds(self):
        return self.__finish - self.__start


def func_minimize_adhoc(init_hemisphere, num_iterations):
    opt = sphere.disperse_charges(init_hemisphere, num_iterations)[0]
    return opt.vertices

def func_minimize_scipy(init_pointset, num_iterations):
    return sphere.disperse_charges_alt(init_pointset, num_iterations)

num_points = [20, 40, 60]
num_subdivide = [2, 3, 4]

def bench_disperse_charges_alt():

    dpi = 72
    figsize = (1920/dpi, 1080/dpi)
    fig = plt.figure(num='Electrostatic repulsion methods benchmark',
                     figsize=figsize, dpi=dpi)
    for (idx, subplot_index) in enumerate(['131', '132', '133']):
        num_repetitions = 20
        num_trials = 3

        execution_time_adhoc = []
        execution_time_scipy = []
        minimum_adhoc = []
        minimum_scipy = []

        if mode == 'subdivide':
            init_sphere = sphere.unit_octahedron.subdivide(num_subdivide[idx])
            init_hemisphere = sphere.HemiSphere.from_sphere(init_sphere)
            init_pointset = init_hemisphere.vertices
        else:
            init_pointset = sphere_stats.random_uniform_on_sphere(
                num_points[idx])
            init_hemisphere = sphere.HemiSphere(xyz=init_pointset)
        print('num_points = {}'.format(init_pointset.shape[0]))

        for j in range(num_trials):
            print('  Iteration {}/{}'.format(j + 1, num_trials))

            for num_iterations in range(12):
                # The time of an iteration of disperse charges is much
                # faster than an iteration of fmin_slsqp.
                num_iterations_dipy = 20 * num_iterations

                # Measure execution time for dipy.core.sphere.disperse_charges
                timer = Timer()
                with timer:
                    for i in range(num_repetitions):
                        opt = func_minimize_adhoc(init_hemisphere,
                                                  num_iterations_dipy)
                execution_time_adhoc.append(timer.duration_in_seconds() /
                                            num_repetitions)
                minimum_adhoc.append(sphere._get_forces_alt(opt.ravel()))

                # Measure execution time for
                # dipy.core.sphere.disperse_charges_alt
                timer = Timer()
                with timer:
                    for i in range(num_repetitions):
                        opt = func_minimize_scipy(init_pointset, num_iterations)
                execution_time_scipy.append(timer.duration_in_seconds() /
                                            num_repetitions)
                minimum_scipy.append(sphere._get_forces_alt(opt.ravel()))

        ax = fig.add_subplot(subplot_index)
        ax.plot(execution_time_adhoc, minimum_adhoc, 'r+',
                label='DIPY original')
        ax.plot(execution_time_scipy, minimum_scipy, 'g+',
                label='SciPy-based')
        ax.set_yscale('log')

        plt.xlabel('Average execution time (s)')
        plt.ylabel('Objective function value')
        if mode == 'subdivide':
            plt.title('Num subdiv: {}'.format(num_subdivide[idx]))
        else:
            plt.title('Num points: {}'.format(num_points[idx]))
        plt.legend()

    plt.show()
