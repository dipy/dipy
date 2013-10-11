from numpy.distutils.misc_util import get_numpy_include_dirs
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules=[Extension(
                            "ornlm", ["ornlm.pyx"],
                            include_dirs=get_numpy_include_dirs(), 
                            extra_compile_args=["-msse2 -mfpmath=sse"],
                            language="c++"
                            )]
)
#Note on the usage of -msse and -mfpmath-sse compiler options
#http://stackoverflow.com/questions/16888621/why-does-returning-a-floating-point-value-change-its-value
#       -msse2 -mfpmath=sse
