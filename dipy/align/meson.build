cython_sources = ['bundlemin',
'crosscorr',
'expectmax',
'parzenhist',
'sumsqdiff',
'transforms',
'vector_fields',]

foreach ext: cython_sources
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, inc_local],
    install: true,
    subdir: 'dipy/align'
  )
endforeach


python_sources = ['__init__.py',
  '_public.py',
  'imaffine.py',
  'imwarp.py',
  'metrics.py',
  'reslice.py',
  'scalespace.py',
  'streamlinear.py',
  ]


py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/align'
)


subdir('tests')