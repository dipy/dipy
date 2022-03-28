import numpy as np
import numpy.testing as npt
import dipy.utils.parallel as para


def power_it(num, n=2):
    # We define a function of the right form for parallelization
    return num ** n

def test_paramap():
    engines = ["serial"]
    if para.has_dask:
        engines.append("dask")
    if para.has_joblib:
        engines.append("joblib")
    if para.has_ray:
        engines.append("ray")

    my_array = np.arange(100).reshape(10, 10)
    my_list = list(my_array.ravel())
    for engine in engines:
        for backend in ["threading", "multiprocessing"]:
            npt.assert_array_equal(para.paramap(
                power_it,
                my_list,
                engine=engine,
                backend=backend,
                out_shape=my_array.shape),
                             power_it(my_array))

            # If it's not reshaped, the first item should be the item 0, 0:
            npt.assert_equal(para.paramap(power_it,
                                          my_list,
                                          engine=engine,
                                          backend=backend)[0],
                             power_it(my_array[0, 0]))
