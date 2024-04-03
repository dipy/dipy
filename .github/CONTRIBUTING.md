# Contributing to DIPY

DIPY is an open-source software project, and we have an open development
process. This means that we welcome contributions from anyone. We do ask that
you first read this document and follow the guidelines we have outlined here and
that you follow the [NIPY community code of conduct](https://nipy.org/conduct.html).

## Getting started

If you are looking for places that you could make a meaningful contribution,
please contact us! We respond to queries on the [Nipy mailing
list](https://mail.python.org/mailman/listinfo/neuroimaging), and to questions
on our [gitter channel](https://gitter.im/dipy/dipy). A good place to get an
idea for things that currently need attention is the
[issues](https://github.com/dipy/dipy/issues) page of our Github repository.
This page collects outstanding issues that you can help address. Join the
conversation about the issue, by typing into the text box in the issue page.

## The development process

Please refer to the [development section](https://docs.dipy.org/stable/devel/index.html#development)
of the documentation for the procedures we use in developing the code.

## When writing code, please pay attention to the following:

### Tests and test coverage

We use [pytest](https://docs.pytest.org) to write tests of the code,
and [Azure Pipelines](https://dev.azure.com/dipy/dipy) and [Travis-CI](https://travis-ci.com/dipy/dipy)
for continuous integration.

If you are adding code into a module that already has a 'test' file (e.g., if
you are adding code into ``dipy/tracking/streamline.py``), add additional tests
into the respective file (e.g., ``dipy/tracking/tests/test_streamline.py ``).

New contributions are required to have as close to 100% code coverage as
possible. This means that the tests written cause each and every statement in
the code to be executed, covering corner-cases, error-handling, and logical
branch points. To check how much coverage the tests have, you will need.

When running:

    coverage run -m pytest -s --doctest-modules --verbose dipy

You will get the usual output of pytest, but also a table that indicates the test
coverage in each module: the percentage of coverage and also the lines of code
that are not run in the tests. You can also see the test coverage in the Travis
run corresponding to the PR (in the log for the machine with ``COVERAGE=1``).

If your contributions are to a single module, you can see test and
coverage results for only that module without running all of the DIPY
tests. For example, if you are adding code to ``dipy/core/geometry.py``,
you can use:

    coverage run --source=dipy.core.geometry -m pytest -s --doctest-modules --verbose dipy/core/tests/test_geometry.py

You can then use ``coverage report`` to view the results, or use
``coverage html`` and open htmlcov/index.html in your browser for a
nicely formatted interactive coverage report.

Contributions to tests that extend test coverage in older modules that are not
fully covered are very welcome!

### Code style

Code contributions should be formatted according to the [DIPY Coding Style Guideline](../doc/devel/coding_style_guideline.rst).
Please, read the document to conform your code contributions to the DIPY standard.


### Documentation

DIPY uses [Sphinx](https://www.sphinx-doc.org/en/master/index.html) to generate
documentation. The
[DIPY Coding Style Guideline](../doc/devel/coding_style_guideline.rst)
contains details about documenting the contributions.
