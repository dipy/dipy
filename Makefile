# Simple makefile to quickly access handy build commands for Cython extension
# code generation.  Note that the actual code to produce the extension lives in
# the setup.py file, this Makefile is just meant as a command
# convenience/reminder while doing development.

PYTHON ?= python
PKGDIR=dipy
DOCSRC_DIR=doc
DOCDIR=${PKGDIR}/${DOCSRC_DIR}
TESTDIR=${PKGDIR}/tests

help:
	@echo "Numpy/Cython tasks.  Available tasks:"
	@echo "ext  -> build the Cython extension module."
	@echo "cython-html -> create annotated HTML from the .pyx sources"
	@echo "test -> run a simple test demo."
	@echo "all  -> Call ext, html and finally test."

all: ext cython-html test

ext: recspeed.so propspeed.so vox2track.so \
    distances.so

test: ext
	nosetests .

cython-html:  ${PKGDIR}/reconst/recspeed.html ${PKGDIR}/tracking/propspeed.html ${PKGDIR}/tracking/vox2track.html ${PKGDIR}/tracking/distances.html 

recspeed.so: ${PKGDIR}/reconst/recspeed.pyx
propspeed.so: ${PKGDIR}/tracking/propspeed.pyx
vox2track.so: ${PKGDIR}/tracking/vox2track.pyx
distances.so: ${PKGDIR}/tracking/distances.pyx

	python setup.py build_ext --inplace

# Phony targets for cleanup and similar uses

.PHONY: clean

clean:
	- find ${PKGDIR} -name "*.so" -print0 | xargs -0 rm
	- find ${PKGDIR} -name "*.pyd" -print0 | xargs -0 rm
	- find ${PKGDIR} -name "*.c" -print0 | xargs -0 rm
	- find ${PKGDIR} -name "*.html" -print0 | xargs -0 rm
	rm -rf build
	rm -rf docs/_build
	rm -rf docs/dist

distclean: clean
	rm -rf dist

# Suffix rules
%.c : %.pyx
	cython $<

%.html : %.pyx
	cython -a $<

# Print out info for possible install methods
check-version-info:
	$(PYTHON) -c 'from nisext.testers import info_from_here; info_from_here("dipy")'

# Run tests from installed code
installed-tests:
	$(PYTHON) -c 'from nisext.testers import tests_installed; tests_installed("dipy")'

# Run tests from installed code
sdist-tests:
	$(PYTHON) -c 'from nisext.testers import sdist_tests; sdist_tests("dipy")'

bdist-egg-tests:
	$(PYTHON) -c 'from nisext.testers import bdist_egg_tests; bdist_egg_tests("dipy")'

source-release: clean
	python -m compileall .
	python setup.py sdist --formats=gztar,zip

binary-release: clean
	python setup_egg.py bdist_egg

