from itertools import product
import zipfile

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.nn import mindgrab
from dipy.nn.meshnet import (
    _set_simd_allowed,
    meshnet_cpu_kernels,
    meshnet_forward_cpu,
)
from dipy.testing.decorators import set_random_number_generator
from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")


def reference_meshnet(volume, kernels, dilations, classifier, bias):
    """Straightforward float64 MeshNet forward pass."""
    x = volume[None].astype(np.float64)
    shape = volume.shape
    for kernel, d in zip(kernels, dilations):
        padded = np.pad(x, ((0, 0),) + ((d, d),) * 3)
        out = np.zeros((kernel.shape[0],) + shape)
        for a, b, c in product(range(3), repeat=3):
            patch = padded[
                :,
                a * d : a * d + shape[0],
                b * d : b * d + shape[1],
                c * d : c * d + shape[2],
            ]
            out += np.einsum("oi,izyx->ozyx", kernel[:, :, a, b, c], patch)
        mean = out.mean(axis=(1, 2, 3), keepdims=True)
        var = out.var(axis=(1, 2, 3), keepdims=True)
        x = (out - mean) / np.sqrt(var + 1e-5)
        x = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    return np.einsum("ci,izyx->czyx", classifier, x) + bias[:, None, None, None]


def random_meshnet(rng, *, channels, n_layers, n_classes):
    kernels = [rng.standard_normal((channels, 1, 3, 3, 3))] + [
        0.2 * rng.standard_normal((channels, channels, 3, 3, 3))
        for _ in range(n_layers - 1)
    ]
    kernels = [kernel.astype(np.float32) for kernel in kernels]
    classifier = rng.standard_normal((n_classes, channels)).astype(np.float32)
    bias = rng.standard_normal(n_classes).astype(np.float32)
    return kernels, classifier, bias


def test_orientation_round_trip():
    volume = np.arange(4 * 5 * 6).reshape((4, 5, 6))
    lia = mindgrab.ras_to_lia(volume)

    npt.assert_equal(lia.shape, (4, 6, 5))
    # LIA axes point to the left, inferior and anterior, so its first voxel
    # is the most right, superior and posterior one.
    npt.assert_equal(lia[0, 0, 0], volume[-1, 0, -1])
    npt.assert_array_equal(mindgrab.lia_to_ras(lia), volume)


def test_quantile_normalize():
    volume = np.linspace(-10, 90, 1000).reshape((10, 10, 10))
    normalized = mindgrab.quantile_normalize(volume)

    npt.assert_equal(normalized.dtype, np.float32)
    npt.assert_almost_equal(normalized.min(), 0)
    npt.assert_almost_equal(normalized.max(), 1)
    npt.assert_equal(np.mean(normalized == 0) > 0.01, True)
    npt.assert_equal(np.mean(normalized == 1) > 0.01, True)


@set_random_number_generator()
def test_meshnet_forward_cpu(rng=None):
    # Volumes narrower and wider than twice the dilations, so that both the
    # bounds-checked and the segmented convolution loops are exercised, with
    # MindGrab's 15 channels and a different count.
    cases = [
        ((7, 9, 11), (4, 2, 1), 15, 2),
        ((12, 10, 21), (1, 3, 2, 1), 15, 3),
        ((5, 6, 3), (1,), 15, 2),
        ((9, 8, 30), (2, 5), 7, 4),
    ]
    for shape, dilations, channels, n_classes in cases:
        kernels, classifier, bias = random_meshnet(
            rng, channels=channels, n_layers=len(dilations), n_classes=n_classes
        )
        volume = rng.random(shape).astype(np.float32)

        logits = meshnet_forward_cpu(
            volume, kernels, dilations, classifier, bias, num_threads=1
        )
        expected = reference_meshnet(volume, kernels, dilations, classifier, bias)
        npt.assert_equal(logits.dtype, np.float32)
        npt.assert_equal(logits.shape, (n_classes,) + shape)
        npt.assert_allclose(logits, expected, atol=1e-4)

        # Moments are folded serially, so results do not depend on threads.
        logits_threads = meshnet_forward_cpu(
            volume, kernels, dilations, classifier, bias, num_threads=3
        )
        npt.assert_array_equal(logits_threads, logits)


@set_random_number_generator()
def test_meshnet_cpu_kernels(rng=None):
    kernels = meshnet_cpu_kernels()
    npt.assert_equal(kernels in ("avx2", "baseline"), True)
    if kernels == "baseline":
        pytest.skip("No AVX2 kernels on this CPU")

    dilations = (4, 2, 1)
    weights = random_meshnet(rng, channels=15, n_layers=3, n_classes=2)
    volume = rng.random((10, 12, 14)).astype(np.float32)
    logits = meshnet_forward_cpu(volume, weights[0], dilations, *weights[1:])
    try:
        _set_simd_allowed(allow=False)
        npt.assert_equal(meshnet_cpu_kernels(), "baseline")
        logits_baseline = meshnet_forward_cpu(
            volume, weights[0], dilations, *weights[1:]
        )
    finally:
        _set_simd_allowed(allow=True)
    npt.assert_allclose(logits, logits_baseline, atol=1e-4)


@set_random_number_generator()
def test_meshnet_forward_cpu_wrong_inputs(rng=None):
    kernels, classifier, bias = random_meshnet(
        rng, channels=15, n_layers=2, n_classes=2
    )
    volume = rng.random((6, 6, 6)).astype(np.float32)
    wide = [np.zeros((17, 1, 3, 3, 3)), np.zeros((17, 17, 3, 3, 3))]
    wrong_args = [
        (volume[0], kernels, (2, 1), classifier, bias),
        (volume, kernels, (1,), classifier, bias),
        (volume, kernels, (0, 1), classifier, bias),
        (volume, kernels[::-1], (2, 1), classifier, bias),
        (volume, kernels[:1] * 2, (2, 1), classifier, bias),
        (volume, wide, (2, 1), classifier, bias),
        (volume, kernels, (2, 1), classifier[:, :5], bias),
        (volume, kernels, (2, 1), classifier, bias[:1]),
    ]
    for args in wrong_args:
        npt.assert_raises(ValueError, meshnet_forward_cpu, *args)
    npt.assert_raises(
        ValueError,
        meshnet_forward_cpu,
        volume,
        kernels,
        (2, 1),
        classifier,
        bias,
        num_threads=0,
    )


def test_read_mindgrab_weights(tmp_path):
    kernels, classifier, bias = mindgrab.read_mindgrab_weights(
        get_fnames(name="mindgrab_weights")
    )
    n_params = sum(k.size for k in kernels) + classifier.size + bias.size
    npt.assert_equal(n_params, 146237)
    npt.assert_equal(len(kernels), len(mindgrab.DILATIONS))
    npt.assert_equal(kernels[0].shape, (15, 1, 3, 3, 3))
    npt.assert_equal(classifier.shape, (2, 15))
    npt.assert_equal(np.all(bias == 0), False)

    wrong_weights = tmp_path / "wrong.pth"
    with zipfile.ZipFile(wrong_weights, "w") as archive:
        archive.writestr("archive/version", "3")
    npt.assert_raises(ValueError, mindgrab.read_mindgrab_weights, wrong_weights)


def test_wrong_inputs():
    model = mindgrab.MindGrab()
    npt.assert_raises(ValueError, model.predict, np.zeros((5, 5)), np.eye(4))
    npt.assert_raises(ValueError, model.predict, np.zeros((5, 5, 5, 5)), np.eye(4))


@pytest.mark.skipif(not have_torch, reason="Requires Torch")
def test_meshnet():
    model = mindgrab.MeshNet()
    n_params = sum(p.numel() for p in model.parameters())
    npt.assert_equal(n_params, 146237)

    with torch.no_grad():
        out = model(torch.zeros((1, 1, 8, 9, 10)))
    npt.assert_equal(tuple(out.shape), (1, 2, 8, 9, 10))


@pytest.mark.skipif(not have_torch, reason="Requires Torch")
@set_random_number_generator()
def test_torch_matches_cpu(rng=None):
    model = mindgrab.MindGrab()
    volume = rng.random((40, 36, 33)).astype(np.float32)
    prob_cpu = model._predict_volume(volume)

    model.device = torch.device("cpu")
    model.load_model_weights(get_fnames(name="mindgrab_weights"))
    prob_torch = model._predict_volume(volume)

    npt.assert_equal(prob_cpu.shape, volume.shape)
    npt.assert_allclose(prob_cpu, prob_torch, atol=1e-4)
