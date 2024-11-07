# Simple makefile to quickly access handy build commands for Cython extension
# code generation.  Note that the actual code to produce the extension lives in
# the setup.py file, this Makefile is just meant as a command
# convenience/reminder while doing development.

PYTHON ?= python3
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
    distances.so streamlinespeed.so denspeed.so \
    vec_val_sum.so quick_squash.so vector_fields.so \
    crosscorr.so sumsqdiff.so expectmax.so bundlemin.so \
    cythonutils.so featurespeed.so metricspeed.so \
    clusteringspeed.so clustering_algorithms.so \
    mrf.so

test: ext
	pytest -s --verbose --doctest-modules .

cython-html:  ${PKGDIR}/reconst/recspeed.html ${PKGDIR}/tracking/propspeed.html ${PKGDIR}/tracking/vox2track.html ${PKGDIR}/tracking/distances.html ${PKGDIR}/tracking/streamlinespeed.html ${PKGDIR}/segment/cythonutils.html ${PKGDIR}/segment/featurespeed.html ${PKGDIR}/segment/metricspeed.html ${PKGDIR}/segment/clusteringspeed.html ${PKGDIR}/segment/clustering_algorithms.html

recspeed.so: ${PKGDIR}/reconst/recspeed.pyx
cythonutils.so: ${PKGDIR}/segment/cythonutils.pyx
featurespeed.so: ${PKGDIR}/segment/featurespeed.pyx
metricspeed.so: ${PKGDIR}/segment/metricspeed.pyx
mrf.so: ${PKGDIR}/segment/mrf.pyx
clusteringspeed.so: ${PKGDIR}/segment/clusteringspeed.pyx
clustering_algorithms.so: ${PKGDIR}/segment/clustering_algorithms.pyx
propspeed.so: ${PKGDIR}/tracking/propspeed.pyx
vox2track.so: ${PKGDIR}/tracking/vox2track.pyx
distances.so: ${PKGDIR}/tracking/distances.pyx
streamlinespeed.so: ${PKGDIR}/tracking/streamlinespeed.pyx
denspeed.so: ${PKGDIR}/denoise/denspeed.pyx
vec_val_sum.so: ${PKGDIR}/reconst/vec_val_sum.pyx
quick_squash.so: ${PKGDIR}/reconst/quick_squash.pyx
vector_fields.so: ${PKGDIR}/align/vector_fields.pyx
crosscorr.so: ${PKGDIR}/align/crosscorr.pyx
sumsqdiff.so: ${PKGDIR}/align/sumsqdiff.pyx
expectmax.so: ${PKGDIR}/align/expectmax.pyx
bundlemin.so: ${PKGDIR}/align/bundlemin.pyx

	$(PYTHON) setup.py build_ext --inplace

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
	rm -rf dipy/dipy.egg-info

distclean: clean
	rm -rf dist

# Suffix rules
%.c : %.pyx
	cython $<

%.html : %.pyx
	cython -a $<

source-release: clean
	$(PYTHON) -m compileall .
	$(PYTHON) -m build --sdist --wheel .

binary-release: clean
	$(PYTHON) -m build --wheel .

# Checks to see if local files pass formatting rules
format:
	$(PYTHON) -m pycodestyle dipy
