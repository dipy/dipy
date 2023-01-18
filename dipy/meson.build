# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'


# ------------------------------------------------------------------------
# Preprocessor flags
# ------------------------------------------------------------------------

numpy_nodepr_api_1_9 = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'
numpy_nodepr_api_1_7 = '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'

# ------------------------------------------------------------------------
# Compiler flags
# ------------------------------------------------------------------------

# C warning flags
Wno_maybe_uninitialized = cc.get_supported_arguments('-Wno-maybe-uninitialized')
Wno_discarded_qualifiers = cc.get_supported_arguments('-Wno-discarded-qualifiers')
Wno_empty_body = cc.get_supported_arguments('-Wno-empty-body')
Wno_implicit_function_declaration = cc.get_supported_arguments('-Wno-implicit-function-declaration')
Wno_parentheses = cc.get_supported_arguments('-Wno-parentheses')
Wno_switch = cc.get_supported_arguments('-Wno-switch')
Wno_unused_label = cc.get_supported_arguments('-Wno-unused-label')
Wno_unused_variable = cc.get_supported_arguments('-Wno-unused-variable')

# C++ warning flags
_cpp_Wno_cpp = cpp.get_supported_arguments('-Wno-cpp')
_cpp_Wno_deprecated_declarations = cpp.get_supported_arguments('-Wno-deprecated-declarations')
_cpp_Wno_class_memaccess = cpp.get_supported_arguments('-Wno-class-memaccess')
_cpp_Wno_format_truncation = cpp.get_supported_arguments('-Wno-format-truncation')
_cpp_Wno_non_virtual_dtor = cpp.get_supported_arguments('-Wno-non-virtual-dtor')
_cpp_Wno_sign_compare = cpp.get_supported_arguments('-Wno-sign-compare')
_cpp_Wno_switch = cpp.get_supported_arguments('-Wno-switch')
_cpp_Wno_terminate = cpp.get_supported_arguments('-Wno-terminate')
_cpp_Wno_unused_but_set_variable = cpp.get_supported_arguments('-Wno-unused-but-set-variable')
_cpp_Wno_unused_function = cpp.get_supported_arguments('-Wno-unused-function')
_cpp_Wno_unused_local_typedefs = cpp.get_supported_arguments('-Wno-unused-local-typedefs')
_cpp_Wno_unused_variable = cpp.get_supported_arguments('-Wno-unused-variable')
_cpp_Wno_int_in_bool_context = cpp.get_supported_arguments('-Wno-int-in-bool-context')


cython_c_args = []
if is_windows
  # For mingw-w64, link statically against the UCRT.
  gcc_link_args = ['-lucrt', '-static']
  if is_mingw
    add_project_link_arguments(gcc_link_args, language: ['c', 'cpp'])
    # Force gcc to float64 long doubles for compatibility with MSVC
    # builds, for C only.
    add_project_arguments('-mlong-double-64', language: 'c')
    # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
    add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c', 'cpp'])
    # Manual add of MS_WIN64 macro when not using MSVC.
    # https://bugs.python.org/issue28267
    if target_machine.cpu_family().to_lower().contains('64')
      add_project_arguments('-DMS_WIN64', language: ['c', 'cpp'])
    endif
    # Silence warnings emitted by PyOS_snprintf for (%zd), see
    # https://github.com/rgommers/scipy/issues/118.
    # Use as c_args for extensions containing Cython code
    cython_c_args += ['-Wno-format-extra-args', '-Wno-format']
  endif
endif

# Suppress warning for deprecated Numpy API.
# (Suppress warning messages emitted by #warning directives).
# Replace with numpy_nodepr_api after Cython 3.0 is out
cython_c_args += ['-Wno-cpp']
cython_cpp_args = cython_c_args

# ------------------------------------------------------------------------
# NumPy include directory - needed in all submodules
# ------------------------------------------------------------------------
incdir_numpy = run_command(py3,
  [
    '-c',
    'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

# ------------------------------------------------------------------------
#  include openmp
# Copy the main __init__.py and pxd files to the build dir.
# Needed to trick Cython, it won't do a relative import outside a package
# ------------------------------------------------------------------------

_cython_tree = [
  fs.copyfile('__init__.py'),
  fs.copyfile('../src/conditional_omp.h'),
  fs.copyfile('../src/cythonutils.h'),
  fs.copyfile('../src/dpy_math.h'),
  fs.copyfile('../src/safe_openmp.pxd'),
]

# include some local folder
# Todo: need more explicit name
incdir_local = meson.current_build_dir()
inc_local = include_directories('.')

# ------------------------------------------------------------------------
# Include Python Sources
# ------------------------------------------------------------------------
python_sources = [
  '__init__.py',
  'conftest.py',
  'info.py',
  'pkg_info.py'
]

py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy'
)

# ------------------------------------------------------------------------
# Custom Meson Command line tools
# ------------------------------------------------------------------------

cython_args = ['-3', '--fast-fail', '--output-file', '@OUTPUT@',
               '--include-dir', incdir_local,
               '@INPUT@']
cython_cplus_args = ['--cplus'] + cython_args

cython_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
  depends : _cython_tree)

cython_gen_cpp = generator(cython,
  arguments : cython_cplus_args,
  output : '@BASENAME@.cpp',
  depends : [_cython_tree])


# ------------------------------------------------------------------------
# Add subfolders
# ------------------------------------------------------------------------

subdir('align')
subdir('boots')
subdir('core')
subdir('data')
subdir('denoise')
subdir('direction')
subdir('io')
subdir('nn')
subdir('reconst')
subdir('segment')
subdir('sims')
subdir('stats')
subdir('testing')
subdir('tests')
subdir('tracking')
subdir('utils')
subdir('viz')
subdir('workflows')