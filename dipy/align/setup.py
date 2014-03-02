from numpy.distutils.misc_util import get_numpy_include_dirs
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = []
ext_modules.append(Extension("vector_fields", ["vector_fields.pyx"],include_dirs=get_numpy_include_dirs()))
ext_modules.append(Extension("ssd", ["ssd.pyx"],include_dirs=get_numpy_include_dirs()))
ext_modules.append(Extension("cc", ["cc.pyx"],include_dirs=get_numpy_include_dirs()))
ext_modules.append(Extension("em", ["em.pyx"],include_dirs=get_numpy_include_dirs()))
setup(
  name = 'Registration API',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
