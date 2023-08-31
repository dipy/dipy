cython_sources = [
  'denspeed',
  'enhancement_kernel',
  'nlmeans_block',
  'pca_noise_estimate',
  'shift_twist_convolution',
  ]

foreach ext: cython_sources
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, inc_local],
    dependencies: [omp],
    install: true,
    subdir: 'dipy/denoise'
  )
endforeach


python_sources = ['__init__.py',
  'adaptive_soft_matching.py',
  'gibbs.py',
  'localpca.py',
  'nlmeans.py',
  'noise_estimate.py',
  'non_local_means.py',
  'patch2self.py',
  ]


py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/denoise'
)


subdir('tests')