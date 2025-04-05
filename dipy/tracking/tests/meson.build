cython_sources = [
  'test_tractogen',
  'test_propspeed',
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
    subdir: 'dipy/tracking/tests'
  )
endforeach

python_sources = [
  '__init__.py',
  'test_distances.py',
  'test_fbc.py',
  'test_learning.py',
  'test_life.py',
  'test_mesh.py',
  'test_metrics.py',
  'test_stopping_criterion.py',
  'test_streamline.py',
  'test_tracker.py',
  'test_track_volumes.py',
  'test_tracking.py',
  'test_utils.py',
  ]


py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/tracking/tests'
)


