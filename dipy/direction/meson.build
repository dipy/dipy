cython_sources = [
  'bootstrap_direction_getter',
  'closest_peak_direction_getter',
  'pmf',
  'probabilistic_direction_getter',
  'ptt_direction_getter',
  ]

foreach ext: cython_sources
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, inc_local],
    dependencies: [omp],
    install: true,
    subdir: 'dipy/direction'
  )
endforeach


python_sources = ['__init__.py',
  'peaks.py',
  ]


py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/direction'
)


subdir('tests')