python_sources = [
  '__init__.py',
  'cuda.py',
  'metal.py',
  'tracker.py',
  'webgpu.py',
  ]

# GPU kernel sources, compiled at runtime
kernel_sources = [
  'kernel.cu',
  'kernel.metal',
  'kernel.wgsl',
  ]

py3.install_sources(
  python_sources + kernel_sources,
  pure: false,
  subdir: 'dipy/tracking/simplet'
)
