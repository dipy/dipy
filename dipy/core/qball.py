#from enthought.mayavi import mlab
import numpy as np
from scipy.special import sph_harm

def real_sph_harm(m, n, theta, phi):
    m = np.atleast_1d(m)
    n = np.atleast_1d(n)
    sh = sph_harm(m, n, theta, phi)
    m_ge0,junk,junk,junk = np.broadcast_arrays(m >= 0, n, theta, phi)
    m_lt0 = np.logical_not(m_ge0)

    real_sh = np.zeros(sh.shape, 'double')
    real_sh[m_ge0] = sh[m_ge0].real
    real_sh[m_lt0] = sh[m_lt0].imag
    return real_sh

def sph_harm_ind_list(sh_order):
    
    ncoef = (sh_order + 2)*(sh_order + 1)/2

    if sh_order % 2 != 0:
        raise ValueError('sh_order must be an even integer >= 0')
    
    n_range = np.arange(0, np.int(sh_order+1), 2)
    n_list = np.repeat(n_range, n_range*2+1)

    offset = 0
    m_list = np.zeros(ncoef, 'int')
    for ii in n_range:
        m_list[offset:offset+2*ii+1] = np.arange(-ii, ii+1)
        offset = offset + 2*ii + 1

    n_list = n_list[..., np.newaxis]
    m_list = m_list[..., np.newaxis]
    return (m_list, n_list)


def read_bvec_file(filename):
    import os
    import csv
    
    base, ext = os.path.splitext(filename)
    if ext == '':
        bvec = base+'.bvec'
        bvec = base+'.bvec'
    elif ext == '.bvec':
        bvec_file=filename
        bval_file=base+'.bval'
    elif ext == '.bval':
        bvec = base+'.bvec'
        bval = filename
    else:
        raise ValueError('filename must have .bvec or .bval extension')

    bvec = open(bvec_file)
    bvec_reader = csv.reader(bvec, delimiter=' ')
    grad_table = []
    for ii in bvec_reader:
        grad_table.append(ii)
    bvec.close()

    bval = open(bval_file)
    bval_reader = csv.reader(bval, delimiter=' ')
    b_values = []
    for ii in bval_reader:
        b_values.append(ii)
    bval.close()
    
    try:
        if len(grad_table) == 3:
            grad_table = np.array(grad_table)
            grad_table = grad_table[0:3, (grad_table != '').any(0)]
            grad_table = grad_table.astype('double')
        else:
            raise ValueError('bvec file should have three rows')
    except ValueError:
        raise IOError('the file, '+bvec_file+', does not seem to have a valid gradient table')

    try:
        if len(b_values) == 1:
            b_values = np.array(b_values)
            b_values = b_values[b_values != '']
            b_values = b_values.astype('double')
        else:
            raise ValueError('bval file should have one row')
    except ValueError:
        raise IOError('the file, '+bval_file+', does not seem to have a valid b values')

    if grad_table.shape[1] != b_values.shape[0]:
        raise IOError('the gradient file and b value file should have the same number of columns')

    grad_table[:,b_values > 0] = grad_table[:,b_values > 0]/np.sqrt((grad_table[:,b_values > 0]**2).sum(0))
    return (grad_table, b_values)


class odf():
    
    def __init__(self,data, sh_order, grad_table, b_values, keep_resid=False):
        from scipy.special import lpn

        if (sh_order % 2 != 0 or sh_order < 0 ):
            raise ValueError('sh_order must be an even integer >= 0')
        self.sh_order = sh_order
        dwi = b_values > 0
        self.n_grad = dwi.sum()

        theta = np.arctan2(grad_table[1, dwi], grad_table[0, dwi])
        phi = np.arccos(grad_table[2, dwi])
        

        m_list, n_list = sph_harm_ind_list(self.sh_order)
        comp_mat = real_sph_harm(m_list, n_list, theta, phi)

        self.fit_matrix = np.dot(comp_mat.T, np.linalg.inv(np.dot(comp_mat, comp_mat.T)))
        legendre0, junk = lpn(self.sh_order, 0)
        funk_radon = legendre0[n_list]
        self.fit_matrix *= funk_radon.T

        self.dim = data.shape[:-1]
        self.b0 = data[..., np.logical_not(dwi)]
        self.coef = np.dot(data[..., dwi], self.fit_matrix)

        if keep_resid:
            unfit = comp_mat / funk_radon
            self.resid = data[..., dwi] - np.dot(self.coef, unfit)
        else:
            self.resid = None

    def evaluate_at(self, theta_e, phi_e):
        
        m_list, n_list = sph_harm_ind_list(self.sh_order)
        comp_mat = real_sph_harm(m_list, n_list, theta_e.flat[:], phi_e.flat[:])
        values = np.dot(self.coef, comp_mat)
        values.shape = self.coef.shape[:-1] + np.broadcast(theta_e,phi_e).shape
        return values

    def evaluate_boot(self, theta_e, phi_e, permute=None):
        m_list, n_list = sph_harm_ind_list(self.sh_order)
        comp_mat = real_sph_harm(m_list, n_list, theta_e.flat[:], phi_e.flat[:])
        if permute == None:
            permute = np.random.permutation(self.n_grad)
        values = np.dot(self.coef + np.dot(self.resid[..., permute], self.fit_matrix), comp_mat)
        return values


    def fit_matrix_d(self):
        from scipy.special import lpn

        m_list, n_list = sph_harm_ind_list(self.sh_order)
        comp_mat = real_sph_harm(m_list, n_list, self.theta.flat[:], self.phi.flat[:])
        fit_mat = np.dot(comp_mat.T, np.linalg.inv(np.dot(comp_mat, comp_mat.T)))
        legendre0, junk = lpn(self.sh_order, 0)
        fit_mat *= legendre0[n_list.T]

        return fit_mat

    def res_mat(self):
        m_list, n_list = sph_harm_ind_list(self.sh_order)
        comp_mat = real_sph_harm(m_list, n_list, self.theta.flat[:], self.phi.flat[:])
        res_mat = np.linalg.inv(np.dot(comp_mat, comp_mat.T))
        res_mat = np.dot(comp_mat.T, res_mat)
        res_mat = np.dot(res_mat, comp_mat)
        res_mat = np.eye(self.n_grad) - res_mat
        return res_mat



    def test(self):
        print self.sh_order

