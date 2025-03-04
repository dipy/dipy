cython_sources = [
  'test_math',
  ]

foreach ext: cython_sources
  if fs.exists(ext + '.pxd')
    extra_args += ['--depfile', meson.current_source_dir() +'/'+ ext + '.pxd', ]
  endif
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, inc_local],
    dependencies: [omp],
    install: true,
    subdir: 'dipy/core/tests'
  )
endforeach

python_sources = ['__init__.py',
  'test_geometry.py',
  'test_gradients.py',
  'test_graph.py',
  'test_interpolation.py',
  'test_ndindex.py',
  'test_optimize.py',
  'test_rng.py',
  'test_sphere.py',
  'test_subdivide_octahedron.py',
  ]


py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/core/tests'
)


