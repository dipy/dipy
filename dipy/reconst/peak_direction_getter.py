from warnings import warn

import dipy
import dipy.reconst.eudx_direction_getter


def EuDXDirectionGetter(*args, **kwargs):
    warn(DeprecationWarning(
            "class 'dipy.reconst.peak_direction_getter.EuDXDirectionGetter'"
            " is deprecated since version 1.2.0, use class"
            " 'dipy.reconst.eudx_direction_getter.EuDXDirectionGetter'"
            " instead"))
    return dipy.reconst.eudx_direction_getter.EuDXDirectionGetter(*args,
                                                                  **kwargs)
