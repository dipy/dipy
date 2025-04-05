cython_sources = [
  'test_fast_numpy',
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
    subdir: 'dipy/utils/tests'
  )
endforeach

python_sources = [
  '__init__.py',
  'test_arrfuncs.py',
  'test_compatibility.py',
  'test_convert.py',
  'test_deprecator.py',
  'test_multiproc.py',
  'test_omp.py',
  'test_optpkg.py',
  'test_parallel.py',
  'test_tractogram.py',
  'test_tripwire.py',
  'test_volume.py',
  ]


py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/utils/tests'
)


