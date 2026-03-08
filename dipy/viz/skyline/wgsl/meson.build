python_sources = [
  '__init__.py',
  ]

wgsl_sources = [
  'billboard_sphere_render.wgsl',
  'sh_analytical_lut_populate.wgsl',
  'sh_billboard.wgsl',
  'utils.wgsl',
  ]



py3.install_sources(
  python_sources + wgsl_sources,
  pure: false,
  subdir: 'dipy/viz/skyline/wgsl'
)
