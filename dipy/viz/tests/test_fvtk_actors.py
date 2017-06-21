import os
import numpy as np

from dipy.viz import actor, window

import numpy.testing as npt
from nibabel.tmpdirs import TemporaryDirectory
from dipy.tracking.streamline import center_streamlines, transform_streamlines
from dipy.align.tests.test_streamlinear import fornix_streamlines
from dipy.testing.decorators import xvfb_it
from dipy.data import get_sphere
from tempfile import mkstemp


use_xvfb = os.environ.get('TEST_WITH_XVFB', False)
if use_xvfb == 'skip':
    skip_it = True
else:
    skip_it = False

run_test = (actor.have_vtk and
            actor.have_vtk_colors and
            window.have_imread and
            not skip_it)

if actor.have_vtk:
    if actor.major_version == 5 and use_xvfb:
        skip_slicer = True
    else:
        skip_slicer = False
else:
    skip_slicer = False


@npt.dec.skipif(skip_slicer)
@npt.dec.skipif(not run_test)
@xvfb_it
def test_slicer():
    renderer = window.renderer()
    data = (255 * np.random.rand(50, 50, 50))
    affine = np.eye(4)
    slicer = actor.slicer(data, affine)
    slicer.display(None, None, 25)
    renderer.add(slicer)

    renderer.reset_camera()
    renderer.reset_clipping_range()
    # window.show(renderer)

    # copy pixels in numpy array directly
    arr = window.snapshot(renderer, 'test_slicer.png', offscreen=False)
    import scipy
    print(scipy.__version__)
    print(scipy.__file__)

    print(arr.sum())
    print(np.sum(arr == 0))
    print(np.sum(arr > 0))
    print(arr.shape)
    print(arr.dtype)

    report = window.analyze_snapshot(arr, find_objects=True)

    print(report)

    npt.assert_equal(report.objects, 1)
    # print(arr[..., 0])

    # The slicer can cut directly a smaller part of the image
    slicer.display_extent(10, 30, 10, 30, 35, 35)
    renderer.ResetCamera()

    renderer.add(slicer)

    # save pixels in png file not a numpy array
    with TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'slice.png')
        # window.show(renderer)
        arr = window.snapshot(renderer, fname, offscreen=False)
        report = window.analyze_snapshot(fname, find_objects=True)
        npt.assert_equal(report.objects, 1)

    npt.assert_raises(ValueError, actor.slicer, np.ones(10))

    renderer.clear()

    rgb = np.zeros((30, 30, 30, 3))
    rgb[..., 0] = 1.
    rgb_actor = actor.slicer(rgb)

    renderer.add(rgb_actor)

    renderer.reset_camera()
    renderer.reset_clipping_range()

    arr = window.snapshot(renderer, offscreen=False)
    report = window.analyze_snapshot(arr, colors=[(255, 0, 0)])
    npt.assert_equal(report.objects, 1)
    npt.assert_equal(report.colors_found, [True])

    lut = actor.colormap_lookup_table(scale_range=(0, 255),
                                      hue_range=(0.4, 1.),
                                      saturation_range=(1, 1.),
                                      value_range=(0., 1.))
    renderer.clear()
    slicer_lut = actor.slicer(data, lookup_colormap=lut)

    slicer_lut.display(10, None, None)
    slicer_lut.display(None, 10, None)
    slicer_lut.display(None, None, 10)

    slicer_lut2 = slicer_lut.copy()
    slicer_lut2.display(None, None, 10)
    renderer.add(slicer_lut2)

    renderer.reset_clipping_range()

    arr = window.snapshot(renderer, offscreen=False)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 1)

    renderer.clear()

    data = (255 * np.random.rand(50, 50, 50))
    affine = np.diag([1, 3, 2, 1])
    slicer = actor.slicer(data, affine, interpolation='nearest')
    slicer.display(None, None, 25)

    renderer.add(slicer)
    renderer.reset_camera()
    renderer.reset_clipping_range()

    arr = window.snapshot(renderer, offscreen=False)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 1)
    npt.assert_equal(data.shape, slicer.shape)

    renderer.clear()

    data = (255 * np.random.rand(50, 50, 50))
    affine = np.diag([1, 3, 2, 1])

    from dipy.align.reslice import reslice

    data2, affine2 = reslice(data, affine, zooms=(1, 3, 2),
                             new_zooms=(1, 1, 1))

    slicer = actor.slicer(data2, affine2, interpolation='linear')
    slicer.display(None, None, 25)

    renderer.add(slicer)
    renderer.reset_camera()
    renderer.reset_clipping_range()

    # window.show(renderer, reset_camera=False)
    arr = window.snapshot(renderer, offscreen=False)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 1)
    npt.assert_array_equal([1, 3, 2] * np.array(data.shape),
                           np.array(slicer.shape))


@npt.dec.skipif(not run_test)
@xvfb_it
def test_streamtube_and_line_actors():
    renderer = window.renderer()

    line1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.]])
    line2 = line1 + np.array([0.5, 0., 0.])

    lines = [line1, line2]
    colors = np.array([[1, 0, 0], [0, 0, 1.]])
    c = actor.line(lines, colors, linewidth=3)
    window.add(renderer, c)

    c = actor.line(lines, colors, spline_subdiv=5, linewidth=3)
    window.add(renderer, c)

    # create streamtubes of the same lines and shift them a bit
    c2 = actor.streamtube(lines, colors, linewidth=.1)
    c2.SetPosition(2, 0, 0)
    window.add(renderer, c2)

    arr = window.snapshot(renderer)

    report = window.analyze_snapshot(arr,
                                     colors=[(255, 0, 0), (0, 0, 255)],
                                     find_objects=True)

    npt.assert_equal(report.objects, 4)
    npt.assert_equal(report.colors_found, [True, True])

    # as before with splines
    c2 = actor.streamtube(lines, colors, spline_subdiv=5, linewidth=.1)
    c2.SetPosition(2, 0, 0)
    window.add(renderer, c2)

    arr = window.snapshot(renderer)

    report = window.analyze_snapshot(arr,
                                     colors=[(255, 0, 0), (0, 0, 255)],
                                     find_objects=True)

    npt.assert_equal(report.objects, 4)
    npt.assert_equal(report.colors_found, [True, True])


@npt.dec.skipif(not run_test)
@xvfb_it
def test_bundle_maps():
    renderer = window.renderer()
    bundle = fornix_streamlines()
    bundle, shift = center_streamlines(bundle)

    mat = np.array([[1, 0, 0, 100],
                    [0, 1, 0, 100],
                    [0, 0, 1, 100],
                    [0, 0, 0, 1.]])

    bundle = transform_streamlines(bundle, mat)

    # metric = np.random.rand(*(200, 200, 200))
    metric = 100 * np.ones((200, 200, 200))

    # add lower values
    metric[100, :, :] = 100 * 0.5

    # create a nice orange-red colormap
    lut = actor.colormap_lookup_table(scale_range=(0., 100.),
                                      hue_range=(0., 0.1),
                                      saturation_range=(1, 1),
                                      value_range=(1., 1))

    line = actor.line(bundle, metric, linewidth=0.1, lookup_colormap=lut)
    window.add(renderer, line)
    window.add(renderer, actor.scalar_bar(lut, ' '))

    report = window.analyze_renderer(renderer)

    npt.assert_almost_equal(report.actors, 1)
    # window.show(renderer)

    renderer.clear()

    nb_points = np.sum([len(b) for b in bundle])
    values = 100 * np.random.rand(nb_points)
    # values[:nb_points/2] = 0

    line = actor.streamtube(bundle, values, linewidth=0.1, lookup_colormap=lut)
    renderer.add(line)
    # window.show(renderer)

    report = window.analyze_renderer(renderer)
    npt.assert_equal(report.actors_classnames[0], 'vtkLODActor')

    renderer.clear()

    colors = np.random.rand(nb_points, 3)
    # values[:nb_points/2] = 0

    line = actor.line(bundle, colors, linewidth=2)
    renderer.add(line)
    # window.show(renderer)

    report = window.analyze_renderer(renderer)
    npt.assert_equal(report.actors_classnames[0], 'vtkLODActor')
    # window.show(renderer)

    arr = window.snapshot(renderer)
    report2 = window.analyze_snapshot(arr)
    npt.assert_equal(report2.objects, 1)

    # try other input options for colors
    renderer.clear()
    actor.line(bundle, (1., 0.5, 0))
    actor.line(bundle, np.arange(len(bundle)))
    actor.line(bundle)
    colors = [np.random.rand(*b.shape) for b in bundle]
    actor.line(bundle, colors=colors)


@npt.dec.skipif(not run_test)
@xvfb_it
def test_odf_slicer(interactive=False):

    sphere = get_sphere('symmetric362')

    shape = (11, 11, 11, sphere.vertices.shape[0])

    fid, fname = mkstemp(suffix='_odf_slicer.mmap')
    print(fid)
    print(fname)
    
    odfs = np.memmap(fname, dtype=np.float64, mode='w+',
                     shape=shape)
    
    odfs[:] = 1

    affine = np.eye(4)
    renderer = window.Renderer()

    mask = np.ones(odfs.shape[:3])
    mask[:4, :4, :4] = 0

    odfs[..., 0] = 1

    odf_actor = actor.odf_slicer(odfs, affine,
                                 mask=mask, sphere=sphere, scale=.25,
                                 colormap='jet')
    fa = 0. * np.zeros(odfs.shape[:3])
    fa[:, 0, :] = 1.
    fa[:, -1, :] = 1.
    fa[0, :, :] = 1.
    fa[-1, :, :] = 1.
    fa[5, 5, 5] = 1

    k = 5
    I, J, K = odfs.shape[:3]

    fa_actor = actor.slicer(fa, affine)
    fa_actor.display_extent(0, I, 0, J, k, k)
    renderer.add(odf_actor)
    renderer.reset_camera()
    renderer.reset_clipping_range()
    
    odf_actor.display_extent(0, I, 0, J, k, k)
    odf_actor.GetProperty().SetOpacity(1.0)
    if interactive:
        window.show(renderer, reset_camera=False)
    
    arr = window.snapshot(renderer)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 11 * 11)
    
    renderer.clear()
    renderer.add(fa_actor)
    renderer.reset_camera()
    renderer.reset_clipping_range()
    if interactive:
        window.show(renderer)

    mask[:] = 0
    mask[5, 5, 5] = 1
    fa[5, 5, 5] = 0
    fa_actor = actor.slicer(fa, None)
    fa_actor.display(None, None, 5)
    odf_actor = actor.odf_slicer(odfs, None, mask=mask,
                                 sphere=sphere, scale=.25,
                                 colormap='jet',
                                 norm=False, global_cm=True)
    renderer.clear()
    renderer.add(fa_actor)
    renderer.add(odf_actor)
    renderer.reset_camera()
    renderer.reset_clipping_range()
    if interactive:
        window.show(renderer)

    renderer.clear()
    renderer.add(odf_actor)
    renderer.add(fa_actor)
    odfs[:, :, :] = 1
    mask = np.ones(odfs.shape[:3])
    odf_actor = actor.odf_slicer(odfs, None, mask=mask,
                                 sphere=sphere, scale=.25,
                                 colormap='jet',
                                 norm=False, global_cm=True)

    renderer.clear()
    renderer.add(odf_actor)
    renderer.add(fa_actor)
    renderer.add(actor.axes((11, 11, 11)))
    for i in range(11):
        odf_actor.display(i, None, None)
        fa_actor.display(i, None, None)
        if interactive:
            window.show(renderer)
    for j in range(11):
        odf_actor.display(None, j, None)
        fa_actor.display(None, j, None)
        if interactive:
            window.show(renderer)
    # with mask equal to zero everything should be black
    mask = np.zeros(odfs.shape[:3])
    odf_actor = actor.odf_slicer(odfs, None, mask=mask,
                                 sphere=sphere, scale=.25,
                                 colormap='plasma',
                                 norm=False, global_cm=True)
    renderer.clear()
    renderer.add(odf_actor)
    renderer.reset_camera()
    renderer.reset_clipping_range()
    if interactive:
        window.show(renderer)

    report = window.analyze_renderer(renderer)
    npt.assert_equal(report.actors, 1)
    npt.assert_equal(report.actors_classnames[0], 'vtkLODActor')
        
    del odf_actor
    odfs._mmap.close()
    del odfs
    os.close(fid)
    
    os.remove(fname)


@npt.dec.skipif(not run_test)
@xvfb_it
def test_peak_slicer(interactive=False):

    _peak_dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='f4')
    # peak_dirs.shape = (1, 1, 1) + peak_dirs.shape

    peak_dirs = np.zeros((11, 11, 11, 3, 3))

    peak_values = np.random.rand(11, 11, 11, 3)

    peak_dirs[:, :, :] = _peak_dirs

    renderer = window.Renderer()
    peak_actor = actor.peak_slicer(peak_dirs)
    renderer.add(peak_actor)
    renderer.add(actor.axes((11, 11, 11)))
    if interactive:
        window.show(renderer)

    renderer.clear()
    renderer.add(peak_actor)
    renderer.add(actor.axes((11, 11, 11)))
    for k in range(11):
        peak_actor.display_extent(0, 10, 0, 10, k, k)

    for j in range(11):
        peak_actor.display_extent(0, 10, j, j, 0, 10)

    for i in range(11):
        peak_actor.display(i, None, None)

    renderer.rm_all()

    peak_actor = actor.peak_slicer(
        peak_dirs,
        peak_values,
        mask=None,
        affine=np.diag([3, 2, 1, 1]),
        colors=None,
        opacity=1,
        linewidth=3,
        lod=True,
        lod_points=10 ** 4,
        lod_points_size=3)

    renderer.add(peak_actor)
    renderer.add(actor.axes((11, 11, 11)))
    if interactive:
        window.show(renderer)

    report = window.analyze_renderer(renderer)
    ex = ['vtkLODActor', 'vtkOpenGLActor', 'vtkOpenGLActor', 'vtkOpenGLActor']
    npt.assert_equal(report.actors_classnames, ex)
    
    
if __name__ == "__main__":
    npt.run_module_suite()
