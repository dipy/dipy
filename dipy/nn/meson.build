# Plain C kernel, built at -O3 so the convolution loops get vectorized.
meshnet_cpu_args = cc.get_supported_arguments('-ffp-contract=fast')
meshnet_cpu_libs = []
# MSVC cannot target AVX2 per function: compile the kernels a second time
# with /arch:AVX2, selected at run time by meshnet_cpu.c.
if cc.get_argument_syntax() == 'msvc' and host_machine.cpu_family() in ['x86', 'x86_64']
  meshnet_cpu_libs += static_library('meshnet_cpu_avx2',
    'meshnet_cpu.c',
    c_args: ['/arch:AVX2', '-DMESHNET_AVX2_UNIT']
      + cc.get_supported_arguments('/fp:contract'),
    override_options: ['optimization=3'],
  )
  meshnet_cpu_args += '-DMESHNET_MSVC_AVX2'
endif
meshnet_cpu_libs += static_library('meshnet_cpu',
  'meshnet_cpu.c',
  c_args: meshnet_cpu_args,
  dependencies: [omp],
  override_options: ['optimization=3'],
)

py3.extension_module('meshnet',
  cython_gen.process('meshnet.pyx'),
  c_args: cython_c_args,
  include_directories: [incdir_numpy, inc_local],
  link_with: meshnet_cpu_libs,
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
