cython_sources = [
  'clustering_algorithms',
  'clusteringspeed',
  'cythonutils',
  'featurespeed',
  'metricspeed',
  'mrf',
  ]

foreach ext: cython_sources
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    cpp_args: cython_c_args,
    include_directories: [incdir_numpy, ],
    install: true,
    subdir: 'dipy/segment'
  )
endforeach


python_sources = ['__init__.py',
  'bundles.py',
  'clustering.py',
  'mask.py',
  'metric.py',
  'threshold.py',
  'tissue.py',
  ]

py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/segment'
)


subdir('benchmarks')
subdir('tests')