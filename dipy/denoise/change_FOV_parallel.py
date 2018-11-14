import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import time
import multiprocessing
import ctypes
from contextlib import closing
from change_FOV import  changeFOV2D


def init (shared_input_, shared_output_, arr_shape_, params_ ):
    global shared_input, shared_output, arr_shape,params
    shared_input = shared_input_
    shared_output = shared_output_
    arr_shape = arr_shape_
    params = params_
    # params =[old_FOV, Origin, Size, Spacing, Direction, new_size, new_FOV]


def wrapper(vol):
    inp = np.frombuffer(shared_input)
    sh_input = inp.reshape(arr_shape)
    new_size = params[-2]
    out = np.frombuffer(shared_output)
    sh_output = out.reshape(new_size)

    #print('Running Wrapper')

    for k in range(0,new_size[2]):
        print(k)
        slice_data = sh_input[:, :, k, vol]
        result_slice = changeFOV2D(slice_data, params[-1],params[0],
                                   params[3], params[2])
        sh_output[:, :, k, vol] = result_slice

def change_FOV_parallel(file_name, new_FOV, num_threads = None,doit = False):
    start_time = time.time()
    print('Start at: {}'.format(start_time))

    if num_threads is not None:
        threads_to_use = num_threads
    else:
        threads_to_use = multiprocessing.cpu_count()
    print(threads_to_use)

    arr = (nib.load(file_name)).get_data()
    #
    size1 = arr.shape[0] * arr.shape[1] * arr.shape[2] * arr.shape[3]
    mp_arr = multiprocessing.RawArray(ctypes.c_double,size1)
    shared_arr = np.frombuffer(mp_arr)
    shared_input = shared_arr.reshape(arr.shape)
    shared_input[:] = arr[:]

    image = sitk.ReadImage(file_name)
    Origin = np.asarray(image.GetOrigin())
    Size = np.asarray(image.GetSize())
    Spacing = np.asarray(image.GetSpacing())
    Direction = np.array(image.GetDirection())
    Direction = Direction.reshape(4, 4)
    old_FOV = Size * Spacing


    print("Input Image has:"
          "FOVs: {}"
          "Our New FOVs: {}".format(old_FOV, new_FOV))


    """Checking Conditions for changing FOV"""
    for f, fov in enumerate (old_FOV):
        if (abs(new_FOV[f] - old_FOV[f]) >= 2*Spacing[f]):
            doit = True
            print("OK ! Changing FOVs")
        else:
            exit()



    new_size = np.zeros(len(old_FOV)).astype(int)
    new_org_index = np.zeros(len(old_FOV)).astype(int)
    """
    Notes: How to make this shorter ? ---- ??
    """
    for d, fov in enumerate(old_FOV):
        print(d)
        if new_FOV[d] > fov:
            total_add_3D = int(np.ceil((new_FOV[d] - fov) / Spacing[d]))
            if ((total_add_3D % 2) == 1):
                total_add_3D += 1
            new_org_index[d] = - (total_add_3D/2)
            new_size[d] = int(Size[d] + total_add_3D)


        else:
            total_add_3D = -int(np.floor((new_FOV[d] - fov) / Spacing[d]))
            if ((total_add_3D % 2) == 1):
                total_add_3D -= 1
            new_org_index[d] = total_add_3D/2
            new_size[d] = int(Size[d] - total_add_3D)
        print(new_size)
        print(total_add_3D)

    ArraySize = new_size[0]*new_size[1]*new_size[2]*new_size[3]

    New_Origin = Origin + Spacing * new_org_index
    params =[old_FOV, Origin, Size, Spacing, Direction, new_size, new_FOV]

    mp_arr2 = multiprocessing.RawArray(ctypes.c_double, int(ArraySize))
    shared_arr2 = np.frombuffer(mp_arr2)
    shared_output = shared_arr2.reshape(new_size[0],new_size[1], new_size[2],new_size[3])


    with closing(multiprocessing.Pool(threads_to_use, initializer=init,
                                      initargs=(shared_arr, shared_arr2, arr.shape, params))) as p:
        print('Running, number of Vols:',arr.shape[3])
        p.map_async(wrapper, [vol for vol in range(0, arr.shape[3])])
    p.join()
    print("data processing --- %s seconds ---" % (time.time() - start_time))


    return shared_output


"""Testing"""
if __name__ == "__main__":
    filename = 'subj3_Philips_s0_all_exps_64d_AP_A_b1000_3_1.nii'
    im4D = change_FOV_parallel(file_name=filename,
                           new_FOV= [150,150,100,84])
    print('Done')