import numpy as np
import numpy.testing as npt

import AFQ.utils.parallel as para


def power_it(num, n=2):
    # We define a function of the right form for parallelization
    return num ** n


def test_parfor():
    my_array = np.arange(100).reshape(10, 10)
    i, j = np.random.randint(0, 9, 2)
    my_list = list(my_array.ravel())
    for engine in ["joblib", "dask", "serial"]:
        for backend in ["threading", "multiprocessing"]:
            npt.assert_equal(para.parfor(power_it,
                                         my_list,
                                         engine=engine,
                                         backend=backend,
                                         out_shape=my_array.shape)[i, j],
                             power_it(my_array[i, j]))

            # If it's not reshaped, the first item should be the item 0, 0:
            npt.assert_equal(para.parfor(power_it,
                                         my_list,
                                         engine=engine,
                                         backend=backend)[0],
                             power_it(my_array[0, 0]))
