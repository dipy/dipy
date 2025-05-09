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
        :link-type: ref

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
        :link-type: ref

        Upgrading from a previous version? See what's new and changed between
        each release of DIPY_.

    .. grid-item-card:: :octicon:`comment-discussion` Get help :octicon:`link-external`
        :link: https://github.com/dipy/dipy/discussions

        Need help with your processing? Ask us and a large
        neuroimaging community.


**********
Highlights
**********

**DIPY 1.11.0** is now available. New features include:

- NF: Refactoring of the tracking API.
- Deprecation of Tensorflow backend in favor of PyTorch.
- Performance improvements of multiple functionalities.
- DIPY Horizon improvements and minor features added.
- Added support for Python 3.13.
- Drop support for Python 3.9.
- Multiple Workflows updated and added (15 workflows).
- Documentation update.
- Closed 73 issues and merged 47 pull requests.


See :ref:`Older Highlights <old_highlights>`.

*************
Announcements
*************
- :doc:`DIPY 1.11.0 <release_notes/release1.11>` released March 15, 2025.
- :doc:`DIPY 1.10.0 <release_notes/release1.10>` released December 12, 2024.
- :doc:`DIPY 1.9.0 <release_notes/release1.9>` released March 8, 2024.



See some of our :ref:`Past Announcements <old_news>`


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
   recipes
   api_changes
   stateoftheart
   old_highlights
   old_news
   glossary
   developers
   gimbal_lock
   faq
   cite
   subscribe

.. Main content will be displayed using the jinja template


.. include:: links_names.inc
