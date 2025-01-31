cython_sources = [
  'interpolation',
  'math',
]

cython_headers = [
  'interpolation.pxd',
  'math.pxd',
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
    subdir: 'dipy/core'
  )
endforeach


python_sources = ['__init__.py',
  'geometry.py',
  'gradients.py',
  'graph.py',
  'histeq.py',
  'ndindex.py',
  'onetime.py',
  'optimize.py',
  'profile.py',
  'rng.py',
  'sphere.py',
  'sphere_stats.py',
  'subdivide_octahedron.py',
  'wavelet.py',
  ]


py3.install_sources(
  python_sources + cython_headers,
  pure: false,
  subdir: 'dipy/core'
)


subdir('tests')