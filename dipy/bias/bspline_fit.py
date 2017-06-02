import numpy as np
import nibabel as nib


class cubic_bspline(object):
    """Define a class to compute cubic bspline
    """
    def __init__(self, P, resolution, spacing="uniform"):
        if len(P.shape) == 1:
            self.knot = np.array(range(P.shape[0] + 3 + 1))
        if len(P.shape) == 2:
            self.knot_u = np.array(range(P.shape[0] + 3 + 1))
            self.knot_v = np.array(range(P.shape[1] + 3 + 1))
        if len(P.shape) == 3:
            self.knot_u = np.array(range(P.shape[0] + 3 + 1))
            self.knot_v = np.array(range(P.shape[1] + 3 + 1))
            self.knot_t = np.array(range(P.shape[2] + 3 + 1))
        self.P = P
        self.resolution = resolution
        self.fit = np.zeros(resolution)

    def interp_value_2D(self, u, knot):
        """This function interp 2D curve
        """
        value = 0
        for j in range(self.P.shape[0]):
            value += self.com_nip(u, j, 3, knot) * self.P[j]
        return value

    def interp_value_3D(self, u, v, knot_u, knot_v):
        """This function interp 3D surface
        """
        value = 0
        for i in range(self.P.shape[0]):
            for j in range(self.P.shape[1]):
                value += self.com_nip(u, i, 3, knot_u) * \
                         self.com_nip(v, j, 3, knot_v) * \
                         self.P[i, j]
        return value

    def cubicbspline_2d(self):
        """This function implement 2d bspline fitting, P should be 2D,
        degree is the polynomial degree
        """
        x_min = self.knot.min()
        x_max = self.knot.max()
        step = (x_max - x_min) / self.resolution
        print(step)
        for j in range(self.resolution):
            self.fit[j] = self.interp_value_2D(x_min + j * step, self.knot)
        return self.fit

    def cubicbspline_3d(self):
        """This function implements 3d bspline fitting, P should be 3D
        """
        u_min = self.knot_u[:].min()
        u_max = self.knot_u[:].max()
        v_min = self.knot_v[:].min()
        v_max = self.knot_v[:].max()
        step_u = (u_max - u_min) / self.resolution[0]
        step_v = (v_max - v_min) / self.resolution[1]
        print(step_u)
        print(step_v)
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                self.fit[i, j] = self.interp_value_3D(u_min + i * step_u,
                                                      v_min + j * step_v,
                                                      self.knot_u,
                                                      self.knot_v)
        return self.fit

    def com_nip(self, u, i, p, knot):
        """This function recursively compute N_i_p parameters
        """
        i = int(i)
        p = int(p)
        if p == 0:
            if u < knot[i+1] and (u > knot[i] or u == knot[i]):
                return 1
            else:
                return 0

        value = (u - knot[i]) * self.com_nip(u, i, p-1, knot) / \
                (knot[i+p] - knot[i]) + \
                (knot[i+p+1] - u) * self.com_nip(u, i+1, p-1, knot) / \
                (knot[i+p+1] - knot[i+1])

        return value


def Setfieldpoints(inputimg, xscale, yscale, zscale):
    xsize = inputimg.shape[0]
    ysize = inputimg.shape[1]
    zsize = inputimg.shape[2]
    output = np.zeros(shape=(xscale, yscale, zscale))
    xsubsize = np.floor(xsize / xscale)
    ysubsize = np.floor(ysize / yscale)
    zsubsize = np.floor(zsize / zscale)
    xsubsize = xsubsize.astype(int)
    ysubsize = ysubsize.astype(int)
    zsubsize = zsubsize.astype(int)
    for i in np.array(range(zscale)):
        for j in np.array(range(xscale)):
            for k in np.array(range(yscale)):
                output[j, k, i] = np.mean(inputimg[j*xsubsize:(j+1)*xsubsize,
                                                   k*ysubsize:(k+1)*ysubsize,
                                                   i*zsubsize:(i+1)*zsubsize])
    return output
'''
test_subject = np.zeros(shape=(100, 100, 100),)

for i in np.array(range(100)):
    for j in np.array(range(100)):
        for k in np.array(range(100)):
            test_subject[i, j, k] = 50
            if(k > 30 and k < 70 and j > 30 and j < 70 and i > 30 and i < 70):
                test_subject[i, j, k] = 100
                if(k > 40 and k < 60 and j > 40 and
                   j < 60 and i > 40 and i < 60):
                    test_subject[i, j, k] = 200
'''

#Control_points = Setfieldpoints(test_subject, 20, 20, 20)

#A = np.array([[100, 75, 50, 15, 10],
#              [5, 20, 40, 75, 95],
#              [10, 30, 50, 70, 100],
#              [15, 20, 45, 80, 120]])
#
#dname = "/Users/tiwanyan/ANTs/Images"
#t1_input = "/Raw/Q_0001_T1.nii.gz"
#
#ft1 = dname + t1_input
#
#t1 = nib.load(ft1).get_data()

#logUncorrectedImage = np.log(t1)

#sharpenedimg = sharpen_image(logUncorrectedImage)

"""
P = np.array([1, 4, 10, 20, 15, 10, 20, 30, 25])
resolution = 40
knot = np.array(range(P.shape[0]+1+3))

Bspline = cubic_bspline(knot, P, resolution, spacing="uniform")

plot(np.array(range(resolution)), Bspline.cubicbspline_1d())

x_min = knot.min()
x_max = knot.max()
step = resolution / 9

for i in range(9):
    plot(np.ceil(x_min + i * step), P[i],'ro')
"""
