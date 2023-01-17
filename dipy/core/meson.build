cython_sources = ['interpolation',]

foreach ext: cython_sources
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    cpp_args: cython_c_args,
    include_directories: [incdir_numpy, ],
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
  python_sources,
  pure: false,
  subdir: 'dipy/core'
)


subdir('benchmarks')
subdir('tests')