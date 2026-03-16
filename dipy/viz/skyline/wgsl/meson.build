python_sources = [
  '__init__.py',
]

wgsl_sources = [
  'sh_billboard.wgsl',
  'sh_cube_hermite_lut_compute.wgsl',
]

py3.install_sources(
  python_sources + wgsl_sources,
  pure: false,
  subdir: 'dipy/viz/skyline/wgsl'
)
