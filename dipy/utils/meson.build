cython_sources = [
  'fast_numpy',
  'omp',
  ]

foreach ext: cython_sources
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, inc_local],
    install: true,
    subdir: 'dipy/utils'
  )
endforeach


python_sources = [
  '__init__.py',
  '_importlib.py',
  'arrfuncs.py',
  'deprecator.py',
  'multiproc.py',
  'optpkg.py',
  'parallel.py',
  'tripwire.py',
  'volume.py',
  ]


py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/utils'
)


subdir('tests')