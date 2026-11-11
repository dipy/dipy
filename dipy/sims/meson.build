cython_sources = [
  '_force_core',
  '_multi_tensor_omp',
  ]

foreach ext: cython_sources
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, inc_local],
    dependencies: [omp],
    install: true,
    subdir: 'dipy/sims'
  )
endforeach


python_sources = [
  '__init__.py',
  'force.py',
  'phantom.py',
  'voxel.py',
  ]


py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/sims'
)


subdir('tests')
