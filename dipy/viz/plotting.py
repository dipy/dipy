"""
plotting functions
"""

import numpy as np
from warnings import warn
from dipy.utils.optpkg import optional_package
plt, have_plt, _ = optional_package("matplotlib.pyplot")


def compare_maps(fits, maps, transpose=None, fit_labels=None, map_labels=None,
                 fit_kwargs={}, map_kwargs={}, filename=None):
    """ Compare one or more scalar maps for different fits or models.

    Parameters:
    -----------
    fits : list
        List of fits to be compared.
    maps : list
        Names of attributes to be compared.
        Default: 'rtop'.
    transpose : bool, optional
        If False, different fits are placed on different rows and different maps
        on different columns. If True, the order is transposed. If None, the
        figures are placed such that there are more columns than rows.
        Default: None.
    fit_labels : list, optional
        Labels for the different fitting routines. If None the fits are labeled
        by number.
        Default: None.
    map_labels : list, optional
        Labels for the different attributes. If None the attribute names are
        used.
        Default: None.
    fit_kwargs : list or dict, optional
        A dict or list of dicts with imshow options for each fitting routine.
        The dicts are passed to imshow as keyword-argument pairs.
        Default: {}.
    map_kwargs : list or dict, optional
        A dict or list of dicts with imshow options for each MAP-MRI scalar. The
        dicts are passed to imshow as keyword-argument pairs.
        Default: {}.
    filename : string, optional
        Filename where the image will be saved.
        Default: None.
    """

    if not have_plt:
        raise ValueError('matplotlib package needed for visualization.')

    fontsize = 'large'
    xscale, yscale = 12, 10

    m = len(fits)
    n = len(maps)

    if transpose is None:
        transpose = m > n

    if fit_labels is None:
        fit_labels = ['Fit ' + str(i+1) for i in range(m)]
    if map_labels is None:
        map_labels = maps

    if type(fit_kwargs) is dict:
        fit_kwargs = [fit_kwargs]*m
    if type(map_kwargs) is dict:
        map_kwargs = [map_kwargs]*n

    if transpose:
        fig, ax = plt.subplots(n, m, figsize=(xscale, yscale/m*n),
                               squeeze=False)
        ax = ax.T
        for i in range(m):
            ax[i, 0].set_title(fit_labels[i], fontsize=fontsize)
        for j in range(n):
            ax[0, j].set_ylabel(map_labels[j], fontsize=fontsize)
    else:
        fig, ax = plt.subplots(m, n, figsize=(xscale, yscale/n*m),
                               squeeze=False)
        for i in range(m):
            ax[i, 0].set_ylabel(fit_labels[i], fontsize=fontsize)
        for j in range(n):
            ax[0, j].set_title(map_labels[j], fontsize=fontsize)

    for i in range(m):
        for j in range(n):
            try:
                attr = getattr(fits[i], maps[j])
                if hasattr(attr, '__call__'):
                    attr = attr()
            except AttributeError:
                warn('Could not recover attribute ' + maps[j] + '.')
                attr = np.zeros((2, 2))
            data = np.squeeze(np.array(attr, dtype=float)).T
            ax[i, j].imshow(data, interpolation='nearest', origin='lower',
                            cmap='gray', **fit_kwargs[i], **map_kwargs[j])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False)

    fig.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
