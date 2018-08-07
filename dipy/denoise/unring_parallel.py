import time
import numpy as np
import multiprocessing
import ctypes
from contextlib import closing
from unring import unring_2d

def unring_wrapper(vol):
   inp = np.frombuffer(shared_input)
   sh_input= inp.reshape(arr_shape)

   out = np.frombuffer(shared_output)
   sh_out= out.reshape(arr_shape)

   for k in range(arr_shape[2]): 
       slice_data = sh_input[:,:,k,vol]            
       result_slice = unring_2d(slice_data, nsh,minW,maxW)
       sh_out[:,:,k,vol]=result_slice.real        

def init(shared_input_,shared_output_,arr_shape_,params_):
    #initialization of the global shared arrays
    global shared_input, shared_output,arr_shape,nsh,minW,maxW
    shared_input = shared_input_
    shared_output = shared_output_  
    arr_shape=arr_shape_        
    nsh=params_[0]
    minW=params_[1]
    maxW=params_[2]


def unring_parallel(arr, nsh=25, minW=1, maxW=5, out_dtype=None,num_threads=None):
    r"""Gibbs ringing correction for 4D DWI datasets.

    Parameters
    ----------
    arr : 4D array
        Array of data to be corrected. The dimensions are (X, Y, Z, N), where N
        are the diffusion gradient directions.   
    nsh : int, optional
        Number of shifted images on one side. Default: 25. The total number of
        shifted images will be 2*nsh+1
    minW : int, optional
        Minimum neighborhood distance. Default:1
    maxW : int, optional
        Maximum neighborhood distance. Default:5
    out_dtype : str or dtype, optional
        The dtype for the output array. Default: output has the same dtype as
        the input.
    num_threads : int, optional
         The number of threads that the algorithm can create. Default: Use all cores.

    Returns
    -------
    corrected_arr : 4D array
        This is the corrected array of the same size as that of the input data,
        clipped to non-negative values

    References    
    ----------
    .. [Kellner2015] Kellner E., Bibek D., Valerij K. G., Reisert M.(2015)
                  Gibbs-ringing artifact removal based on local subvoxel-shifts.
                  Magnetic resonance in Medicine 76(5), p1574-1581.
                  https://doi.org/10.1002/mrm.26054
    """
    start_time = time.time()

    # We perform the computations in float64. However we output 
    # with the original data_type
    if out_dtype is None:
        out_dtype = arr.dtype

    if not arr.ndim == 4:
        print('Converting input array from 3D to 4D...')
        arr=arr.reshape([arr.shape[0],arr.shape[1],arr.shape[2],1])

    if num_threads is not None:
        threads_to_use = num_threads
    else:
        threads_to_use = multiprocessing.cpu_count()

    # Creating input and output shared arrays for multi-process processing
    #input array
    mp_arr= multiprocessing.RawArray(ctypes.c_double,arr.shape[0]*arr.shape[1]*arr.shape[2]*arr.shape[3])
    shared_arr = np.frombuffer(mp_arr)
    shared_input= shared_arr.reshape(arr.shape)
    shared_input[:] =arr[:]
    #output array
    mp_arr2= multiprocessing.RawArray(ctypes.c_double,arr.shape[0]*arr.shape[1]*arr.shape[2]*arr.shape[3])
    shared_arr2 = np.frombuffer(mp_arr2)
    shared_output= shared_arr2.reshape(arr.shape)
    #parameters
    params=[nsh,minW,maxW]

    #multi-processing
    with closing(multiprocessing.Pool(threads_to_use,initializer=init, initargs=(shared_arr,shared_arr2,arr.shape,params))) as p:
        p.map_async(unring_wrapper, [vol for vol in range(0, arr.shape[3])])
    p.join()    
    

    print("Gibbs ringing correction took --- %s seconds ---" % (time.time() - start_time))

    return shared_output.astype(out_dtype)

