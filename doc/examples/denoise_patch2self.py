"""
==================================================================
Patch2Self: Self-Supervised Denoising via Statistical Independence
==================================================================

Patch2Self a self-supervised learning method for denoising DWI data which uses
the entire volume to learn a full-rank locally linear denoiser for that volume.
By taking advantage of the oversampled q-space of DWI data, Patch2Self can
separate structure from noise without requiring an explicit model for either.

Classical denoising algorithms such as Local PCA, Non-local Means, Total
Variation Norm, etc. which assume certain properties on the signal structure.
Patch2Self *does not* make any such assumption on the signal but only
leverages the fact that the noise across different 3D volumes of the DWI
signal originates from random fluctuations in the acquired signal.

Since Patch2Self only relies on the randomness of the noise, it can be applied
at any step in the pre-processing pipeline. The design of Patch2Self is such
that it can work on any type of diffusion data/ any body part without
requiring a noise estimation or assumptions on the type of noise (such as its
distribution).

The Patch2Self Framework:

.. _fiber_to_bundle_coherence:
.. figure:: https://github.com/dipy/dipy_data/blob/master/Patch2Self_Framework.PNG?raw=true
   :scale: 60 %
   :align: center
"""