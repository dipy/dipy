cython_sources = [
  'bootstrap_direction_getter',
  'closest_peak_direction_getter',
  'pmf',
  'probabilistic_direction_getter',
  'ptt_direction_getter',
  ]

cython_headers = [
  'closest_peak_direction_getter.pxd',
  'pmf.pxd',
  'probabilistic_direction_getter.pxd',
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
    subdir: 'dipy/direction'
  )
endforeach


python_sources = ['__init__.py',
  'peaks.py',
  ]


py3.install_sources(
  python_sources + cython_headers,
  pure: false,
  subdir: 'dipy/direction'
)


subdir('tests')