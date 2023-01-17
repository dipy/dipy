cython_sources = [
  'direction_getter',
  'distances',
  'fbcmeasures',
  'localtrack',
  'propspeed',
  'stopping_criterion',
  'streamlinespeed',
  'vox2track',
  ]

foreach ext: cython_sources
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    cpp_args: cython_c_args,
    include_directories: [incdir_numpy, ],
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


subdir('benchmarks')
subdir('tests')