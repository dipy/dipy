cython_sources = [
  'direction_getter',
  'distances',
  'fast_tracking',
  'fbcmeasures',
  'localtrack',
  'propspeed',
  'stopping_criterion',
  'streamlinespeed',
  'vox2track',
  ]

cython_headers = [
  'direction_getter.pxd',
  'fast_tracking.pxd',
  'fbcmeasures.pxd',
  'propspeed.pxd',
  'stopping_criterion.pxd',
  'streamlinespeed.pxd',
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
    subdir: 'dipy/tracking'
  )
endforeach


python_sources = ['__init__.py',
  '_utils.py',
  'learning.py',
  'life.py',
  'local_tracking.py',
  'mesh.py',
  'metrics.py',
  'streamline.py',
  'utils.py',
  ]

py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/tracking'
)


subdir('tests')