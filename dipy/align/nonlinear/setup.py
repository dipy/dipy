from numpy.distutils.misc_util import get_numpy_include_dirs
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = []
ext_modules.append(Extension("VectorFieldUtils", ["VectorFieldUtils.pyx"],include_dirs=get_numpy_include_dirs()))
ext_modules.append(Extension("CrossCorrelationFunctions", ["CrossCorrelationFunctions.pyx"],include_dirs=get_numpy_include_dirs()))
ext_modules.append(Extension("SSDFunctions", ["SSDFunctions.pyx"],include_dirs=get_numpy_include_dirs()))
setup(
  name = 'Tensor Field Utilities',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)


#from numpy.distutils.misc_util import get_numpy_include_dirs
#from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext
#setup(
#    cmdclass = {'build_ext': build_ext},
#    ext_modules = [Extension("ornlm_module", ["ornlm_module.pyx", "ornlm.cpp", "upfirdn.cpp"],include_dirs=get_numpy_include_dirs(), language="c++")]
#)

