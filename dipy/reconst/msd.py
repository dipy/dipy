from dipy.data import default_sphere

def convolution_matrix(signals, B, iso_comp, m, n, bval):

    # TODO: round bval

    # Check assumptions about the basis
    # Isotropic compartments are the beginning.
    assert np.all(B[:iso_comp + 1] == B[0, 0])
    # Only even SH degrees are present.
    assert np.all((n % 2) == 0)
    # SH degrees are in ascending order.
    assert np.all(n[m == 0] == np.arange(0, n[-1] + 1, 2))

    B_dwi = B[:, iso_comp:]
    B_ax_sym = B_dwi[:, m == 0]
    conv_multipliers = {}
    for b in np.unique(bval):
        part = bval == b
        R = np.zeros(n_coeff)
        for c in range(iso_comp):
            R[c] = signals[c][part].mean() / B[0, 0]

        dwi_signal = signals[iso_comp]
        rh = np.linalg.lstsq(B_ax_sym[:, part], dwi_signal)
        R[iso_comp:] = rh[n // 2]
        conv_multipliers[b] = R
    return conv_multipliers


def singal_as_sh(signal, B):
    return np.linalg.lstsq(B, signal)[0]

def foo(signals, B, tissue_compartments):
    n_comp = tissue_compartments.max() + 1

    r = []
    for c in range(n_comp):
        part = tissue_compartm == c
        r[c] = np.linalg.lstsq(B[part], signals[c])

    return r


class MultiShellDeconvModel(ConstrainedSphericalDeconvModel):

    def __init__(self, gtab, response, reg_sphere=default_sphere, sh_order=8,
                 tissue_classes=3, *args, **kwargs):
        """
        """
        SphHarmModel.__init__(self, gtab)
        self.B_dwi, comp, m, n = multi_tissue_basis(gtab, sh_order, tissue_classes)

