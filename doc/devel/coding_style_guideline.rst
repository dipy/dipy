.. _coding_style_guideline:

===========================
DIPY Coding Style Guideline
===========================

The main principles behind DIPY_ development are:

* **Robustness**: the results of a piece of code must be verified
  systematically, and hence stability and robustness of the code must be
  ensured, reducing code redundancies.
* **Readability**: the code is written and read by humans, and it is read
  much more frequently than it is written.
* **Consistency**: following these guidelines will ease reading the code,
  and will make it less error-prone.
* **Documentation**: document the code. Documentation is essential as it is
  one of the key points for the adoption of DIPY as the toolkit of choice in
  diffusion by the scientific community. Documenting helps to clarify
  certain choices, helps to avoid obscure places, and is a way to allow
  other members *decode* it with less effort.
* **Language**: the code must be written in English. Norms and spelling
  should be abided by.

------------
Coding style
------------

DIPY uses the standard Python PEP8_ style to ensure the
readability and consistency across the toolkit. Conformance to the PEP8_ syntax
is checked automatically when requesting to push to DIPY. There are
`software systems <https://pypi.python.org/pypi/pep8>`_ that will check your
code for PEP8_ compliance, and most text editors can be configured to check the
compliance of your code with PEP8_. Beyond the aspects checked, as a
contributor to DIPY, you should try to ensure that your code, including
comments, conforms to the above principles.

Imports
-------
DIPY recommends using the following package shorthands to increase consistency
and readability across the library::

  import nibabel as nib
  import numpy as np
  import numpy.testing as npt
  import scipy as sp

No alias should be used for ``h5py``::

  import h5py

-------------------
Cython coding style
-------------------
DIPY recommends the use of the standard Python
PEP8_ style when writing `Cython <https://cython.org/>`_ code.

Cython-specific syntax should follow these additional rules:

Imports
-------
The ``cimport``'s should add the ``c`` prefix to the usual Python import package
shorthand, e.g.::

  cimport numpy as cnp

Adding the ``c`` prefix to the import line makes it clear that the Cython/C
symbols are being referred to as compared to the Python symbols.

Variable declaration
--------------------
Separate ``cdef``, ``cpdef``, and ``ctypedef`` statements from the following type by
exactly one space. In turn, separate the type from the variable name by exactly
one space. Declare only one ``ctypedef`` variable per line. You may ``cdef`` or
``cpdef`` multiple variables per line as long as these are simple declarations;
note that multiple assignment, references, or pointers are not allowed on the
same line. Grouping ``cdef`` statements is allowed. For example::

  # Good
  cdef int n
  cdef char * s
  cdef double Xf[3]
  cdef double d[3]
  cpdef int i, j, k
  cdef ClusterMapCentroid clusters = ClusterMapCentroid()
  cdef:
      double *ps = <double *> cnp.PyArray_DATA(seed)
      cnp.npy_intp *pstr = <cnp.npy_intp *> qa.strides
      cnp.npy_intp d, i, j, cnt
      double direction[3]
      double tmp, ftmp
  cdef int get_direction_c(self, double* point, double* direction):
      return 1

  # Bad
  cdef  char *s
  cdef char * s, * t, * u, * v
  cdef double Xf[3], d[3]
  cdef double x=42, y=x+1, z=x*y
  cdef ClusterMapCentroid     clusters   = ClusterMapCentroid()
  cdef   int   get_direction_c(self, double* point, double* direction):
      return 0

Inside of a function, place all ``cdef`` statements at the top of the function
body::

  # Good
  cdef void estimate_kernel_size(self, verbose=True):
  cdef:
      double [:] x
      double [:] y

  # Bad
  cdef void estimate_kernel_size(self, verbose=True):
      cdef double [:] x
      self.kernelmax = self.k2(x, y, r, v)
      cdef double [:] y
      x = np.array([0., 0., 0.])

Using C libraries
-----------------
The ``cimport``'s should follow the same rules defined in PEP8 for ``import``
statements. If a module is both *imported* and *cimported*, the ``cimport``
should come before the ``import``.

An example of an imported C library::

  from libc.stdlib cimport calloc, realloc, free

Do not use ``include`` statements.

Error return values
-------------------
When declaring an error return value with the ``except`` keyword, use one space on
both sides of the ``except``. If in a function definition, there should be no
spaces between the error return value and the colon ``:``. Avoid ``except *``
unless it is needed for functions returning ``void``::

  # Good
  cdef void bar() except *
  cdef void c_extract(Feature self, Data2D datum, Data2D out) nogil except *:
  cdef int front(x) except +:
      ...

  # Bad
  cdef char * hat(x) except *:
  cdef int front(x)    except   +  :
      ...

Pointers and references
-----------------------
Pointers and references may be either zero or one space away from the type name.
If followed by a variable name, they must be one space away from the variable
name. Do not put any spaces between the reference operator ``&`` and the variable
name::

  # Good
  cdef int& i
  cdef char * s
  i = &j

  # Bad
  cdef int &i
  cdef char *s
  i = & j

Casting
-------
When casting a variable there must be no whitespace between the opening ``<`` and
the type. There must one space between the closing ``>`` and the variable::

  # Good
  <float> i
  <void *> s

  # Bad
  < float >i
  <void*>  s

Loops
-----
Use Python loop syntax::

  for i in range(nrows):
    ...

Other ``for``-loop constructs are deprecated and must be avoided.

-------------
Documentation
-------------
DIPY uses `Sphinx <http://www.sphinx-doc.org/en/stable/index.html>`_ to
generate documentation. We welcome contributions of examples, and suggestions
for changes in the documentation, but please make sure that changes that are
introduced render properly into the HTML format that is used for the DIPY
website.

DIPY follows the `numpy docstring standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
for documenting modules, classes, functions, and examples.

The documentation includes an extensive library of
`examples <http://dipy.org/examples_index.html>`_. These are Python files that
are stored in the ``doc/examples`` folder and contain code to execute the
example, interleaved with multi-line comments that contain explanations of the
blocks of code. Examples demonstrate how to perform processing (segmentation,
tracking, etc.) on diffusion files using the DIPY classes. The code is
intermixed with generous comments that describe the former, and the rationale
and aim of it. If you are contributing a new feature to DIPY, please provide
an extended example, with explanations of this feature, and references to the
relevant papers.

If the feature that you are working on integrates well into one of the
existing examples, please edit the ``.py`` file of that example. Otherwise,
create a new ``.py`` file in that directory. Please also add the name of this
file into the ``doc/examples/valid_examples.txt`` file (which controls the
rendering of these examples into the documentation).

Additionally, DIPY relies on a set of reStructuredText files (``.rst``)
located in the ``doc`` folder. They contain information about theoretical
backgrounds of DIPY, installation instructions, description of the
contribution process, etc.

Again, both sets of files use the `reStructuredText markup language
<http://www.sphinx-doc.org/en/stable/rest.html>`_ for comments. Sphinx parses
the files to produce the contents that are later rendered in the DIPY_
website.

The Python examples are compiled, output images produced and corresponding
``.rst`` files produced so that the comments can be appropriately displayed
in a web page enriched with images.

Particularly, in order to ease the contribution of examples and ``.rst``
files, and with the consistency criterion in mind, beyond the numpy
docstring standard aspects, contributors are encouraged to observe the
following guidelines:

* The acronym for the Diffusion Imaging in Python toolkit should be written
  as **DIPY**.
* The classes, objects, and any other construct referenced from the code
  should be written with inverted commas, such as in *In DIPY, we use an
  object called ``GradientTable`` which holds all the acquisition specific
  parameters, e.g. b-values, b-vectors, timings, and others.*
* Cite the relevant papers. Use the *[NameYear]* convention for
  cross-referencing them, such as in [Garyfallidis2014]_, and put them
  under the :ref:`references` section.
* Cross-reference related examples and files. Use the
  ``.. _specific_filename:`` convention to label a file at the top of it.
  Thus, other pages will be able to reference the file using the standard
  Sphinx syntax ``:ref:`specific_filename```.
* Use an all-caps scheme for acronyms, and capitalize the first letters of
  the long names, such as in *Constrained Spherical Deconvolution (CSD)*,
  except in those cases where the most common convention has been to use
  lowercase, such as in *superior longitudinal fasciculus (SLF)*.
* As customary in Python, use lowercase and separate words with underscores
  for filenames, labels for references, etc.
* When including figures, use the regular font for captions (i.e. do not use
  bold faces) unless otherwise required for a specific text part (e.g. a
  DIPY object, etc.).
* When referring to relative paths, use the backquote inline markup
  the convention, such as in ``doc/devel``. Do not add the
  greater-than/less-than signs to enclose the path.


.. _references:

References
----------

.. [Garyfallidis2014] Garyfallidis E, Brett M, Amirbekian B, Rokem A, van der
   Walt S, Descoteaux M, Nimmo-Smith I and Dipy Contributors (2014). `Dipy, a
   library for the analysis of diffusion MRI data.
   <http://journal.frontiersin.org/Journal/10.3389/fninf.2014.00008/abstract>`_
   Frontiers in Neuroinformatics, vol.8, no.8.

.. include:: ../links_names.inc
