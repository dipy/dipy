# Simple makefile to quickly access handy build commands for Cython extension
# code generation.  Note that the actual code to produce the extension lives in
# the setup.py file, this Makefile is just meant as a command
# convenience/reminder while doing development.

PKGDIR=dipy
DOCDIR=${PKGDIR}/doc
TESTDIR=${PKGDIR}/tests


help:
	@echo "Numpy/Cython tasks.  Available tasks:"
	@echo "ext  -> build the Cython extension module."
	@echo "html -> create annotated HTML from the .pyx sources"
	@echo "test -> run a simple test demo."
	@echo "all  -> Call ext, html and finally test."

all: ext html test

ext: performance.so

test:   ext
	nosetests .

html:  ${PKGDIR}/performance.html

performance.so: ${PKGDIR}/performance.pyx
	python setup.py build_ext --inplace

# Phony targets for cleanup and similar uses

.PHONY: clean
clean:
	rm -rf *~ ${PKGDIR}/*.so ${PKGDIR}/*.c ${PKGDIR}/*.o \
		${PKGDIR}/*.html build \
		*.pyc ${PKGDIR}/*.pyc ${TESTDIR}/*.pyc


# Suffix rules
%.c : %.pyx
	cython $<

%.html : %.pyx
	cython -a $<

