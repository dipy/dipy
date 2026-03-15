cython_sources = [
  'dirspeed',
  'dkispeed',
  'eudx_direction_getter',
  'quick_squash',
  'recspeed',
  'vec_val_sum',
  ]

cython_headers = [
  'dkispeed.pxd',
  '_force_heap.pxd',
  '_force_search.pxd',
  'recspeed.pxd',
  ]

foreach ext: cython_sources
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, inc_local],
    dependencies: [omp],
    install: true,
    subdir: 'dipy/reconst'
  )
endforeach

# Build FORCE extensions with full optimizations (matching FORCE/setup_fast.py)
# Use compiler-appropriate flags: MSVC uses /O2 and /fp:fast; GCC/Clang use -O3/-ffast-math.
# -march=native / /arch:AVX2 enables CPU-specific SIMD (AVX2/FMA).
# These flags are applied to both compile AND link args (like setup_fast.py lines 56-61)
if cpp.get_id() == 'msvc'
  force_optimization_args = ['/O2', '/fp:fast']
  force_simd_args = cpp.get_supported_arguments(['/arch:AVX2'])
else
  force_optimization_args = ['-O3', '-ffast-math']
  force_simd_args = cpp.get_supported_arguments(['-mavx2', '-mfma'])
endif

# Build _force_heap with full optimizations (used by _force_search)
py3.extension_module('_force_heap',
  cython_gen_cpp.process('_force_heap.pyx'),
  cpp_args: cython_cpp_args + force_optimization_args + force_simd_args,
  link_args: force_optimization_args,
  include_directories: [incdir_numpy, inc_local],
  dependencies: [omp],
  install: true,
  subdir: 'dipy/reconst'
)

# Build _force_search with FAISS SIMD C++ source
# This extension requires the distances_simd.cpp for AVX2/FMA optimized operations
force_src_inc = include_directories('force_src')
py3.extension_module('_force_search',
  [
    cython_gen_cpp.process('_force_search.pyx'),
    'force_src/distances_simd.cpp',
  ],
  cpp_args: cython_cpp_args + force_optimization_args + force_simd_args,
  link_args: force_optimization_args,
  include_directories: [incdir_numpy, inc_local, force_src_inc],
  dependencies: [omp],
  install: true,
  subdir: 'dipy/reconst'
)


python_sources = ['__init__.py',
  'base.py',
  'bingham.py',
  'force.py',
  'cache.py',
  'cross_validation.py',
  'csdeconv.py',
  'cti.py',
  'dki_micro.py',
  'dki.py',
  'dsi.py',
  'dti.py',
  'forecast.py',
  'fwdti.py',
  'gqi.py',
  'ivim.py',
  'mapmri.py',
  'mcsd.py',
  'msdki.py',
  'multi_voxel.py',
  'odf.py',
  'qtdmri.py',
  'qti.py',
  'rumba.py',
  'sfm.py',
  'shm.py',
  'shore.py',
  'utils.py',
  'weights_method.py'
  ]

py3.install_sources(
  python_sources + cython_headers,
  pure: false,
  subdir: 'dipy/reconst'
)


subdir('tests')
