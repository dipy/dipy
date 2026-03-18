cython_sources = [
  'denspeed',
  'enhancement_kernel',
  'pca_noise_estimate',
  'shift_twist_convolution',
  '_bias_correction',
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
  'bias_correction.py',
  'gibbs.py',
  'localpca.py',
  'nlmeans.py',
  'noise_estimate.py',
  'patch2self.py',
  '_gpu_mppca.py',
  ]


py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/denoise'
)

gpu_shader_sources = [
  '_gpu_shaders/__init__.py',
  '_gpu_shaders/extract_covariance.wgsl',
  '_gpu_shaders/jacobi_eigh.wgsl',
  '_gpu_shaders/jacobi_eigh_global.wgsl',
  '_gpu_shaders/classify_reconstruct.wgsl',
  '_gpu_shaders/classify_reconstruct_global.wgsl',
  '_gpu_shaders/accumulate.wgsl',
]

py3.install_sources(
  gpu_shader_sources,
  pure: false,
  subdir: 'dipy/denoise/_gpu_shaders'
)


subdir('tests')