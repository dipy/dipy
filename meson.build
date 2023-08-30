project(
  'dipy',
  'c', 'cpp', 'cython',
  version: '1.8.0dev0',
  license: 'BSD-3',
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
    'cpp_std=c++14',
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
  error('SciPy requires Cython >= 0.29.35')
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

# Hide symbols when building on Linux with GCC. For Python extension modules,
# we only need `PyInit_*` to be public, anything else may cause problems. So we
# use a linker script to avoid exporting those symbols (this is in addition to
# Meson using `-fvisibility=hidden` for C and `-fvisibility-inlines-hidden` for
# C++ code. See gh-15996 for details.
_linker_script = meson.project_source_root() / 'dipy/_build_utils/link-version-pyinit.map'
version_link_args = ['-Wl,--version-script=' + _linker_script]
# Note that FreeBSD only accepts version scripts when -shared is passed,
# hence we need to pass that to `cc.links` explicitly (flag is already
# present for `extension_module` invocations).
if not cc.links('', name: '-Wl,--version-script', args: ['-shared', version_link_args])
  version_link_args = []
endif




subdir('dipy')