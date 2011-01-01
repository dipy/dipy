""" Class to allow masked view of data array """
import numpy as np
from operator import mul
from copy import copy

def _makearray(a):
    new = np.asarray(a)
    wrap = getattr(a, "__array_wrap__", new.__array_wrap__)
    return new, wrap

def _filled(a, *args, **kargs):
    if hasattr(a, 'filled'):
        return a.filled(*args, **kargs)
    else:
        return a

class MaskedView(object):
    """
    An interface to allow the user to interact with a data array as if it is a
    container with the same shape as mask. The contents of data are mapped to
    the nonzero elements of mask, where mask is zero fill_value is used.

    Examples
    -----------
    >>> mask = np.array([[True, False, True],[False, True, False]])
    >>> data = np.arange(2*3*4)
    >>> data.shape = (2, 3, 4)
    >>> mv = MaskedView(mask, data[mask], fill_value=10)
    >>> mv.shape
    (2, 3, 4)
    >>> data[0, 0, :]
    array([0, 1, 2, 3])
    >>> mv[0, 1]
    array([10, 10, 10, 10])
    >>> mv[:,:,0]
    array([[ 0, 10,  8],
           [10, 16, 10]])
    
    """

    def __init__(self, mask, data, fill_value=None):
        """
        Creates a MaskedView of data.

        Parameters
        ------------
        mask : ndarray of bools
            mask indicating where the data belongs
        data : ndarray, ndim >= mask.ndim
            the first dimension of data should have size equal to the number of
            nonzero elements in mask
        fill_value : optional
            fill_value is returned when MaskedView is indexed and mask is zero,
            also fill_value is used to fill out an array when the filled method
            is called. By defult is NaN or 0 depending on the dtype of data.
        
        """

        mask = mask.astype('bool')
        if len(data) != mask.sum():
            raise ValueError('the number of data elements does not match mask') 
        self._data = data
        try:
            self.fill_value = np.array(fill_value, dtype=data.dtype)
        except TypeError:
            if fill_value is None:
                self.fill_value = np.array(0, dtype=data.dtype)
            else:
                raise
        self.base = None
        self._imask = np.empty(mask.shape, 'int')
        self._imask.fill(-1)
        self._imask[mask] = np.arange(len(data))
    
    @property
    def mask(self):
        return self._imask >= 0

    @property
    def dtype(self):
        #the data type of a masked view is the same as the _data array
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim + self._imask.ndim - 1

    def filled(self, fill_value=None):
        """
        Returns an ndarray copy of itself. Where mask is zero, fill_value is used 
        (self.fill_value defult).

        Parameters
        ------------
        fill_value :
            Value to be used in place of data where mask is 0.
        """
        if fill_value is None:
            fill_value = self.fill_value
        out_arr = np.empty(self.shape, self.dtype)
        out_arr[:] = fill_value
        out_arr[self.mask] = self.__array__()
        return out_arr

    def _get_shape_contents(self):
        return self._data.shape[1:]

    def _set_shape_contents(self, shape):
        self._data.shape = self._data.shape[0:1] + shape

    def _get_shape_mask(self):
        return self._imask.shape

    def _set_shape_mask(self, shape):
        self._imask.shape = shape

    def _get_shape(self):
        return self.shape_mask + self.shape_contents

    def _set_shape(self, shape):
        try:
            shape[1]
        except (TypeError, IndexError):
            raise ValueError("a 2d shape is required such that the size of " +
                             "mask is unchanged")

        where_missing = [ii < 0 for ii in shape]
        count_missing = sum(where_missing)
        if count_missing == 1:
            ind = where_missing.index(True)
            tot_sz = reduce(mul, self.shape)
            other_sz = reduce(mul, shape[:ind] + shape[ind+1:])
            if tot_sz % other_sz != 0:
                raise ValueError("total size of new array must be unchanged")
            missing = tot_sz / other_sz
            shape = shape[:ind] + (missing,) + shape[ind+1:]
        elif count_missing > 1:
            raise ValueError("can only specify one unknown dimension")
        elif reduce(mul, shape) != reduce(mul, self.shape):
            raise ValueError("total size of new array must be unchanged")
        
        first_n = 1
        for ind, dim_i in enumerate(shape):
            first_n = dim_i*first_n
            if first_n == self._imask.size:
                self.shape_mask = shape[:ind+1]
                self.shape_contents = shape[ind+1:]
                break
            elif first_n > self._imask.size:
                raise ValueError("total size of mask must be unchanged")

    shape_contents = property(_get_shape_contents, _set_shape_contents,
                              "Tuple of contents dimensions")
    shape_mask = property(_get_shape_mask, _set_shape_mask,
                          "Tuple of mask dimensions")
    shape = property(_get_shape, _set_shape, "Tuple of array dimensions")

    def get_size(self):
        """
        Returns the number of non-empty values in MaskedView, ie where
        mask > 0.
        """

        return self.mask.sum()

    def copy(self):
        """
        Returns a copy of the MaskedView. Copies the underlying data array.
        """
        data = self._data[self._imask[self.mask]]
        return MaskedView(self.mask, data, self.fill_value)

    def __getitem__(self, index):
        """
        Indexes the MaskedView without copying the underlying data.
        """

        if type(index) is not tuple:
            index = (index,)
        #replace first Ellipsis with slices
        for ii, slc in enumerate(index):
            if slc is Ellipsis:
                n_ellipsis = len(self.shape) - len(index) + 1
                index = index[:ii] + n_ellipsis*(slice(None),) + index[ii+1:]
                break

        ndim_mask = self._imask.ndim
        if len(index) > ndim_mask:
            index_mask = index[:ndim_mask]
            index_cont = index[ndim_mask:]
        else:
            index_mask = index
            index_cont = (slice(None),)

        imask = self._imask[index_mask]
        if isinstance(imask, int):
            if imask >= 0:
                return self._data[(imask,)+index_cont]
            else:
                result = np.empty(self.shape_contents, self.dtype)
                result[:] = self.fill_value
                return result[index_cont]
        else:
            new_mp = copy(self)
            new_mp._imask = imask
            if self.base is None:
                new_mp.base = self
            data = self._data[(slice(None),) + index_cont]
            new_mp._data = data
            if data.ndim < 2:
                return new_mp.filled()
            else:
                return new_mp
    
    def __setitem__(self, index, values):
        """
        Sets part of the maskedview

        is this useful?
        """

        imask = self._imask[index]
        if isinstance(imask, int):
            if imask >= 0:
                self._data[imask] = values
            else:
                self._imask[index] = len(self._data)
                self._data = np.r_[self._data, values[np.newaxis]]
        else:
            self._data[imask[imask >= 0]] = values
    
    def __array__(self, dtype=None):
        """
        Returns the underlying data
        
        """

        #to save time only index _data when base is not None
        if self.base is None:
            data = self._data
        else:
            data = self._data[self._imask[self.mask]]

        #only makes a copy of data when dtype does not match self.dtype
        if dtype is None or np.dtype(dtype) == self.dtype:
            return data
        else:
            return data.astype(dtype)

    def __array_wrap__(self, array, context=None):
        #fill_value is not updated
        #ie if new = old + 1 new.fill_value == old.fil_value. Fixing this might
        #be a useful feature to implement at some point for numeric fill_values
        new_container = MaskedView(self.mask, array, self.fill_value)
        return new_container

