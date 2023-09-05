cython_sources = ['bundlemin',
'crosscorr',
'expectmax',
'parzenhist',
'sumsqdiff',
'transforms',
'vector_fields',]

cython_headers = [
  'fused_types.pxd',
  'transforms.pxd',
  'vector_fields.pxd',
]

foreach ext: cython_sources
  extra_args = []
  # Todo: check why it is failing to compile with transforms.pxd
  # C attributes cannot be added in implementation part of extension type
  # defined in a pxd
  # if fs.exists(ext + '.pxd')
  #   extra_args += ['--depfile', meson.current_source_dir() +'/'+ ext + '.pxd', ]
  # endif
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx', extra_args: extra_args),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, inc_local],
    dependencies: [omp],
    install: true,
    subdir: 'dipy/align'
  )
endforeach


python_sources = ['__init__.py',
  '_public.py',
  'cpd.py',
  'imaffine.py',
  'imwarp.py',
  'metrics.py',
  'reslice.py',
  'scalespace.py',
  'streamlinear.py',
  'streamwarp.py',
  ]


py3.install_sources(
  python_sources + cython_headers,
  pure: false,
  subdir: 'dipy/align'
)


subdir('tests')