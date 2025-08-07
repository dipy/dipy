cython_sources = [
  'direction_getter',
  'distances',
  'tractogen',
  'fbcmeasures',
  'localtrack',
  'propspeed',
  'stopping_criterion',
  'streamlinespeed',
  'tracker_parameters',
  'vox2track',
  ]

cython_headers = [
  'direction_getter.pxd',
  'tractogen.pxd',
  'fbcmeasures.pxd',
  'propspeed.pxd',
  'stopping_criterion.pxd',
  'streamlinespeed.pxd',
  'tracker_parameters.pxd'
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
  'tracker.py',
  'utils.py',
  ]

py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/tracking'
)


subdir('tests')