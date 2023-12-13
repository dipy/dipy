.. _home:

Diffusion Imaging In Python - Documentation
===========================================

.. container:: index-paragraph

   DIPY_ is the paragon 3D/4D+ imaging library in Python. Contains generic methods for
   spatial normalization, signal processing, machine learning, statistical analysis
   and visualization of medical images. Additionally, it contains
   specialized methods for computational anatomy including diffusion,
   perfusion and structural imaging.

   DIPY is part of the `NiPy ecosystem <https://nipy.org/>`__.


***********
Quick links
***********

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: :octicon:`rocket` Get started
        :link: user_guide/installation
        :link-type: any

        New to DIPY_? Start with our installation guide and DIPY key
        concepts.

    .. grid-item-card:: :octicon:`image` Tutorials
        :link: examples_built/index
        :link-type: any

        Browse our tutorials gallery.

    .. grid-item-card:: :octicon:`image` Recipes
        :link: recipes
        :link-type: any

        How do I do X in DIPY?  This dedicated section will provide you quick and direct answer.

    .. grid-item-card:: :octicon:`zap` Workflows
        :link: interfaces/index
        :link-type: any

        Not comfortable with coding? we have command line interfaces for you.
        An easy way to use DIPY_ via a terminal.

    .. grid-item-card:: :octicon:`rocket` Theory
        :link: theory/index
        :link-type: any

        Back to the basics. Learn the theory behind the methods implemented in DIPY_.

    .. grid-item-card:: :octicon:`tools` Developer Guide
        :link: development
        :link-type: any

        Saw a typo? Found a bug? Want to improve a function? Learn how to
        contribute to DIPY_!

    .. grid-item-card:: :octicon:`repo` API reference
        :link: reference/index
        :link-type: any

        A detailed description of DIPY public Python API.

    .. grid-item-card:: :octicon:`repo` Workflows API reference
        :link: reference_cmd/index
        :link-type: any

        A detailed description of all the DIPY workflows command line.

    .. grid-item-card:: :octicon:`history` Release notes
        :link: stateoftheart
        :link-type: any

        Upgrading from a previous version? See what's new and changed between
        each release of DIPY_.

    .. grid-item-card:: :octicon:`comment-discussion` Get help :octicon:`link-external`
        :link: https://github.com/dipy/dipy/discussions

        Need help with your processing? Ask us and a large
        neuroimaging community.


**********
Highlights
**********

**DIPY 1.8.0** is now available. New features include:

- Python 3.12.0 support.
- Cython 3.0.0 compatibility.
- Migrated to Meson build system. Setuptools is no more.
- EVAC+ novel DL-based brain extraction method added.
- Parallel Transport Tractography (PTT) 10X faster.
- Many Horizon updates. Fast overlays of many images.
- New Correlation Tensor Imaging (CTI) method added.
- Improved warnings for optional dependencies.
- Large documentation update. New theme/design integration.
- Closed 197 issues and merged 130 pull requests.

**DIPY 1.7.0** is now available. New features include:

- NF: BundleWarp - Streamline-based nonlinear registration method for bundles added.
- NF: DKI+ - Diffusion Kurtosis modeling with advanced constraints added.
- NF: Synb0 - Synthetic b0 creation added using deep learning added.
- NF: New Parallel Transport Tractography (PTT) added.
- NF: Fast Streamline Search algorithm added.
- NF: New denoising methods based on 1D CNN added.
- Handle Asymmetric Spherical Functions.
- Large update of DIPY Horizon features.
- Multiple Workflows updated
- Large codebase cleaning.
- Large documentation update. Integration of Sphinx-Gallery.
- Closed 53 issues and merged 34 pull requests.

See :ref:`Older Highlights <old_highlights>`.

*************
Announcements
*************
- :doc:`DIPY 1.8.0 <release_notes/release1.8>` released December 13, 2023.
- :doc:`DIPY 1.7.0 <release_notes/release1.7>` released April 23, 2023.
- :doc:`DIPY 1.6.0 <release_notes/release1.6>` released January 16, 2023.





See some of our :ref:`Past Announcements <old_news>`

*******
Support
*******

We acknowledge support from the following organizations:

- The department of Intelligent Systems Engineering of Indiana University.

- The National Institute of Biomedical Imaging and Bioengineering, NIH.

- The Gordon and Betty Moore Foundation and the Alfred P. Sloan Foundation, through the
  University of Washington eScience Institute Data Science Environment.

- Google supported DIPY through the Google Summer of Code Program during
  Summer 2015, 2016 and 2018.

- The International Neuroinformatics Coordination Facility.


.. This tree is helping populate the side navigation panel
.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide/index
   examples_built/index
   interfaces/index
   devel/index
   theory/index
   reference/index
   reference_cmd/index
   api_changes
   stateoftheart
   old_highlights
   old_news
   glossary
   faq
   cite

.. Main content will be displayed using the jinja template


.. include:: links_names.inc
