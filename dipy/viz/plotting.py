"""
plotting functions
"""

import numpy as np
from warnings import warn
from dipy.utils.optpkg import optional_package
plt, have_plt, _ = optional_package("matplotlib.pyplot")


def compare_maps(fits, maps, transpose=None, fit_labels=None, map_labels=None,
                 fit_kwargs=None, map_kwargs=None, filename=None):
    """Compare one or more scalar maps for different fits or models.

    Parameters
    ----------
    fits : list
        List of fits to be compared.
    maps : list
        Names of attributes to be compared.
        Default: 'rtop'.
    transpose : bool, optional
        If False, different fits are placed on different rows and different
        maps on different columns. If True, the order is transposed. If None,
        the figures are placed such that there are more columns than rows.
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
        A dict or list of dicts with imshow options for each MAP-MRI scalar.
        The dicts are passed to imshow as keyword-argument pairs.
        Default: {}.
    filename : string, optional
        Filename where the image will be saved.
        Default: None.
    """
    fit_kwargs = fit_kwargs or {}
    map_kwargs = map_kwargs or {}

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


def compare_qti_maps(gt, fit1, fit2, mask,
                     maps=("fa", "ufa"),
                     fitname=("QTI", "QTI+"),
                     xlimits=([0, 1], [0.4, 1.5]),
                     disprange=([0, 1], [0, 1]),
                     slice=13):
    """Compare one or more qti derived maps obtained with
    different fitting routines.

    Parameters
    ----------
    gt : qti fit object
        The qti fit to be considered as ground truth
    fit1 : qti fit object
        First qti fit to be compared
    fit2 : qti fit object
        Second qti fit to be compared
    mask : np.ndarray
        Boolean array indicating which voxels to retain for comparing
        the values
    maps : array-like, optional
        QTI invariants to be compared
    fitname : array-like, optional
        Names of the used QTI fitting routines
    xlimits : array-like, optional
        X-Axis limits for the histograms visualization
    disprange : array-like, optional
        Display range for maps
    slice : int, optional
        Axial brain slice to be visualized
    """
    if not have_plt:
        raise ValueError(
                    'matplotlib package needed for visualization')

    n = len(maps)
    fig, ax = plt.subplots(n, 4, figsize=(12, 9))

    background = np.zeros(gt.S0_hat.shape[0:2])
    for i in range(n):
        for j in range(3):
            ax[i, j].imshow(background, cmap='gray')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    for k in range(n):
        ax[k, 0].imshow(np.rot90(getattr(gt, maps[k])[:, :, slice]),
                        cmap='gray', vmin=disprange[k][0],
                        vmax=disprange[k][1])
        ax[k, 0].set_title('GROUND TRUTH')
        ax[k, 0].set_ylabel(maps[k], fontsize=20)

        ax[k, 1].imshow(np.rot90(getattr(fit1, maps[k])[:, :, slice]),
                        cmap='gray', vmin=disprange[k][0],
                        vmax=disprange[k][1])
        ax[k, 1].set_title(fitname[0])

        ax[k, 2].imshow(np.rot90(getattr(fit2, maps[k])[:, :, slice]),
                        cmap='gray', vmin=disprange[k][0],
                        vmax=disprange[k][1])
        ax[k, 2].set_title(fitname[1])

        ax[k, 3].hist((getattr(fit1, maps[k])[mask, slice]).flatten(),
                      density=True, bins=40, label=fitname[0])
        ax[k, 3].hist((getattr(fit2, maps[k])[mask, slice]).flatten(),
                      density=True, bins=40, label=fitname[1], alpha=0.7)
        ax[k, 3].hist((getattr(gt, maps[k])[mask, slice]).flatten(),
                      histtype='stepfilled', density=True, bins=40, label='GT',
                      ec="k", alpha=1, linewidth=1.5, fc="None")
        ax[k, 3].legend()
        ax[k, 3].set_title('VALUE DISTRIBUTION')
        ax[k, 3].set_xlim(xlimits[k])

    fig.tight_layout()
    plt.show()


def bundle_shape_profile(x, shape_profile, std):
    """Plot bundlewarp bundle shape profile.

    Parameters
    ----------
    x : np.ndarray
        Integer array containing x-axis
    shape_profile : np.ndarray
        Float array containing bundlewarp displacement magnitudes along the
        length of the bundle
    std : np.ndarray
        Float array containing standard deviations
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    std_1 = shape_profile+std
    std_2 = shape_profile-std
    ax.plot(x, shape_profile, '-', label='Mean', color='Purple', linewidth=3,
            markersize=12)
    ax.fill_between(x, std_1, std_2, alpha=0.2, label='Std', color='Purple')

    plt.xticks(x)
    plt.ylim(0, max(std_1)+2)

    plt.ylabel("Average Displacement")
    plt.xlabel("Segment Number")
    plt.title('Bundle Shape Profile')
    plt.legend(loc=2)
    plt.show()
