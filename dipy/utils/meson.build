cython_sources = [
  'fast_numpy',
  'omp',
  ]

cython_headers = [
  'fast_numpy.pxd',
  'omp.pxd',
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
    subdir: 'dipy/utils'
  )
endforeach


python_sources = [
  '__init__.py',
  'arrfuncs.py',
  'compatibility.py',
  'convert.py',
  'deprecator.py',
  'multiproc.py',
  'optpkg.py',
  'parallel.py',
  'tractogram.py',
  'tripwire.py',
  'volume.py',
  ]


py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/utils'
)


subdir('tests')