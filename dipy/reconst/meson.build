cython_sources = [
  'dirspeed',
  'eudx_direction_getter',
  'quick_squash',
  'recspeed',
  'vec_val_sum',
  ]

cython_headers = [
  'recspeed.pxd',
  ]

foreach ext: cython_sources
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, inc_local],
    dependencies: [omp],
    install: true,
    subdir: 'dipy/reconst'
  )
endforeach


python_sources = ['__init__.py',
  'base.py',
  'bingham.py',
  'cache.py',
  'cross_validation.py',
  'csdeconv.py',
  'cti.py',
  'dki_micro.py',
  'dki.py',
  'dsi.py',
  'dti.py',
  'forecast.py',
  'fwdti.py',
  'gqi.py',
  'ivim.py',
  'mapmri.py',
  'mcsd.py',
  'msdki.py',
  'multi_voxel.py',
  'odf.py',
  'qtdmri.py',
  'qti.py',
  'rumba.py',
  'sfm.py',
  'shm.py',
  'shore.py',
  'utils.py',
  'weights_method.py'
  ]

py3.install_sources(
  python_sources + cython_headers,
  pure: false,
  subdir: 'dipy/reconst'
)


subdir('tests')
