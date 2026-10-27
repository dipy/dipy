# Plain C kernel, built at -O3 so the convolution loops get vectorized.
meshnet_cpu_lib = static_library('meshnet_cpu',
  'meshnet_cpu.c',
  c_args: cc.get_supported_arguments('-ffp-contract=fast'),
  dependencies: [omp],
  override_options: ['optimization=3'],
)

py3.extension_module('meshnet',
  cython_gen.process('meshnet.pyx'),
  c_args: cython_c_args,
  include_directories: [incdir_numpy, inc_local],
  link_with: meshnet_cpu_lib,
  dependencies: [omp],
  install: true,
  subdir: 'dipy/nn'
)


python_sources = [
  '__init__.py',
  'meshnet_webgpu.py',
  'mindgrab.py',
  'utils.py',
  ]


py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'dipy/nn'
)

subdir('tests')
subdir('torch')
subdir('tf')
