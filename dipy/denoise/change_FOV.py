import numpy as np

import matplotlib.pyplot as plt


def changeFOV2D(image_data, new_FOV, old_FOV,
              Spacing, Size):

    for f, fov in enumerate (old_FOV):
        if (abs(fov - new_FOV[f]) >= 2*Spacing[f]):
            doit = True
    print(doit)

    total_add_3D, total_remove_3D = np.zeros(2).astype(int), np.zeros(2).astype(int)

    new_origin_index = np.zeros(2).astype(int)
    new_size = np.zeros(2).astype(int)
    start_from = np.zeros(2).astype(int)
    start_from_new, from_old_size = np.zeros(2).astype(int), np.zeros(2).astype(int)

    for d, fov in enumerate(old_FOV[:2]):
        print(d)
        if new_FOV[d] > fov:
            #print("Dimension {} process add".format(d))
            total_add_3D[d] = int(np.ceil((new_FOV[d] - fov)/Spacing[d]))
            if((total_add_3D[d] % 2) == 1):
                total_add_3D[d] += 1
            new_origin_index[d] = - (total_add_3D[d]/2)
            new_size[d] = int(Size[d] + total_add_3D[d])
            start_from[d] = 0
            start_from_new[d] = total_add_3D[d]/2
            from_old_size[d]  = Size[d]

        else:
            #print("Dimension {} process remove".format(d))

            total_remove_3D[d] = -int(np.floor((new_FOV[d] - fov)/Spacing[d]))
            if((total_remove_3D[d] % 2) == 1):
                total_remove_3D[d] -= 1
            new_origin_index[d] = (total_remove_3D[d]/2)
            new_size[d] = int(Size[d] - total_remove_3D[d])
            start_from[d] = total_remove_3D[d]/2
            start_from_new[d] = 0
            from_old_size[d]  = Size[d] - total_remove_3D[d]

    new_index, old_index = np.zeros(2).astype(int), np.zeros(2).astype(int)
    print(new_size)
    new_size = new_size.astype(int)
    image2D = np.zeros(new_size)

    for j in range(int(start_from_new[1] + from_old_size[1])):
        new_index[1] = j
        old_index[1] = j + start_from[1] - start_from_new[1]
        # print('new_index:  {}, old_index: {}'.format(new_index, old_index))
        for i in range(int(start_from_new[0] + from_old_size[0])):
            new_index[0] = i
            old_index[0] = i + start_from[0] - start_from_new[0]
            #print('new_index:  {}, old_index: {}'.format(new_index, old_index))

            image2D[new_index[0], new_index[1]] = image_data[old_index[0],old_index[1]]


    return image2D

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


"""Testing"""
#im2D = changeFOV2D(dataImage[:,:,2,2],[250,250], FOVs3[:2],
                   # Spacing3[:2],Size3[:2])
# print('Done')