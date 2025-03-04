project(
  'dipy',
  'c', 'cpp', 'cython',
  version: run_command(['tools/gitversion.py'], check: true).stdout().strip(),
  license: 'BSD-3',
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
    'cpp_std=c++14',
    'optimization=2',
  ],
)

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py3 = py_mod.find_installation(pure: false)
py3_dep = py3.dependency()

# filesystem Module to manage files and directories
fs = import('fs')

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
cy = meson.get_compiler('cython')
host_system = host_machine.system()
host_cpu_family = host_machine.cpu_family()

cython = find_program('cython')

# Check compiler is recent enough (see "Toolchain Roadmap" for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=8.0')
    error('DIPY requires GCC >= 8.0')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('DIPY requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
endif
if not cy.version().version_compare('>=0.29.35')
  error('DIPY requires Cython >= 0.29.35')
endif


# TODO: the below -Wno flags are all needed to silence warnings in
# f2py-generated code. This should be fixed in f2py itself.
#_global_c_args = cc.get_supported_arguments(
#  '-Wno-unused-but-set-variable',
#  '-Wno-unused-function',
#  '-Wno-conversion',
#  '-Wno-misleading-indentation',
#  '-Wno-incompatible-pointer-types',
#)
#add_project_arguments(_global_c_args, language : 'c')

# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for dipy). For C++ it isn't needed, because libstdc++/libc++ is
# guaranteed to depend on it.
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

subdir('dipy')