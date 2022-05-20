"""

plotting functions

"""

import numpy as np
from dipy.utils.optpkg import optional_package
plt, have_plt, _ = optional_package("matplotlib.pyplot")


def compare_qti_maps(gt, fit1, fit2, mask,
                     maps=["fa", "ufa"],
                     fitname=["QTI", "QTI+"],
                     xlimits=[[0, 1], [0.4, 1.5]],
                     disprange=[[0, 1], [0, 1]],
                     slice=13):
    """ Compare one or more qti derived maps obtained with
    different fitting routines.

    Parameters:
    -----------
    gt : qti fit object
        The qti fit to be considered as ground truth
    fit1 : qti fit object
        First qti fit to be compared
    fit2 : qti fit object
        Second qti fit to be compared
    mask : np.ndarray
        Boolean array indicating which voxels to retain for comparing
        the values
    maps : list, optional
        QTI invariants to be compared
    fitname : list, optional
        Names of the used QTI fitting routines
    xlimits : list, optional
        X-Axis limits for the histograms visualization
    disprange : list, optional
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
