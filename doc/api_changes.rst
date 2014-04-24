============
API changes
============

Changes between 0.7.1 and 0.6
------------------------------

The function ``peaks_from_model`` is now available from ``dipy.reconst.peaks``
. Please replace all imports like ::

    from dipy.reconst.odf import peaks_from_model

with ::

    from dipy.reconst.peaks import peaks_from_model





