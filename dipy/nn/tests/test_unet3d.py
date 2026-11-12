from itertools import product

import numpy as np
import numpy.testing as npt
import pytest

from dipy.nn.synb0 import UNet3D
from dipy.nn.unet3d import pack_unet3d_weights, unet3d_forward_cpu
from dipy.testing.decorators import set_random_number_generator
from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")

ENCODER = ((32, 2), (64, 32), (64, 64), (128, 64), (128, 128), (256, 128))
ENCODER += ((256, 256), (512, 256))


def random_state(rng):
    """Random weights with the shapes of the Synb0 UNet3D state dict."""

    def weight(*shape):
        fan_in = np.prod(shape[1:])
        return (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)

    state = {
        f"ec{i}.0.weight": weight(o, c, 3, 3, 3) for i, (o, c) in enumerate(ENCODER)
    }
    state["el.weight"] = weight(512, 512, 1, 1, 1)
    state["el.bias"] = weight(512, 1)[:, 0]
    for up, conv, width in ((9, 8, 256), (6, 5, 128), (3, 2, 64)):
        state[f"dc{up}.0.weight"] = weight(2 * width, 2 * width, 2, 2, 2)
        state[f"dc{conv}.0.weight"] = weight(3 * width, width, 3, 3, 3)
        state[f"dc{conv - 1}.0.weight"] = weight(width, width, 3, 3, 3)
    state["dc0.0.weight"] = weight(64, 1, 1, 1, 1)
    state["dl.weight"] = weight(1, 1, 1, 1, 1)
    state["dl.bias"] = weight(1, 1)[:, 0]
    return state


def reference_unet3d(volume, state):
    """Straightforward float64 forward pass, with PyTorch semantics."""

    def norm_leaky_relu(x):
        mean = x.mean(axis=(1, 2, 3), keepdims=True)
        x = (x - mean) / np.sqrt(x.var(axis=(1, 2, 3), keepdims=True) + 1e-5)
        return np.where(x > 0, x, 0.01 * x)

    def conv(x, name):
        w = state[f"{name}.0.weight"].astype(np.float64)
        padded = np.pad(x, ((0, 0),) + ((1, 1),) * 3)
        out = 0
        for a, b, c in product(range(3), repeat=3):
            patch = padded[
                :, a : a + x.shape[1], b : b + x.shape[2], c : c + x.shape[3]
            ]
            out = out + np.einsum("oi,izyx->ozyx", w[:, :, a, b, c], patch)
        return norm_leaky_relu(out)

    def conv_transpose(x, name):
        # out[k + j - 1] += x[j] * w[k], cropped to the input size.
        w = state[f"{name}.0.weight"].astype(np.float64)
        out = np.zeros((w.shape[1],) + tuple(s + 2 for s in x.shape[1:]))
        for a, b, c in product(range(3), repeat=3):
            out[:, a : a + x.shape[1], b : b + x.shape[2], c : c + x.shape[3]] += (
                np.einsum("io,izyx->ozyx", w[:, :, a, b, c], x)
            )
        return norm_leaky_relu(out[:, 1:-1, 1:-1, 1:-1])

    def upsample(x, name):
        w = state[f"{name}.0.weight"].astype(np.float64)
        out = np.zeros((w.shape[1],) + tuple(2 * s for s in x.shape[1:]))
        for a, b, c in product(range(2), repeat=3):
            out[:, a::2, b::2, c::2] = np.einsum("io,izyx->ozyx", w[:, :, a, b, c], x)
        return norm_leaky_relu(out)

    def pool(x):
        c, z, y, w = x.shape
        return x.reshape(c, z // 2, 2, y // 2, 2, w // 2, 2).max(axis=(2, 4, 6))

    x = volume.astype(np.float64)
    skips = []
    for level in range(3):
        x = conv(conv(x, f"ec{2 * level}"), f"ec{2 * level + 1}")
        skips.append(x)
        x = pool(x)
    x = conv(conv(x, "ec6"), "ec7")
    x = np.einsum("oi,izyx->ozyx", state["el.weight"][:, :, 0, 0, 0], x)
    x = x + state["el.bias"][:, None, None, None]
    for up, conv_name in ((9, 8), (6, 5), (3, 2)):
        x = np.concatenate((upsample(x, f"dc{up}"), skips.pop()))
        x = conv_transpose(conv_transpose(x, f"dc{conv_name}"), f"dc{conv_name - 1}")
    x = norm_leaky_relu(
        np.einsum("io,izyx->ozyx", state["dc0.0.weight"][:, :, 0, 0, 0], x)
    )
    x = np.einsum("io,izyx->ozyx", state["dl.weight"][:, :, 0, 0, 0], x)
    return x + state["dl.bias"][:, None, None, None]


@set_random_number_generator()
def test_unet3d_forward_cpu(rng=None):
    state = random_state(rng)
    volume = rng.standard_normal((2, 16, 24, 16)).astype(np.float32)

    expected = reference_unet3d(volume, state)
    result = unet3d_forward_cpu(volume, state, num_threads=1)
    npt.assert_equal(result.dtype, np.float32)
    npt.assert_equal(result.shape, (1, 16, 24, 16))
    npt.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    packed = pack_unet3d_weights(state)
    for num_threads in (2, -1):
        npt.assert_array_equal(
            unet3d_forward_cpu(volume, packed, num_threads=num_threads), result
        )


@set_random_number_generator()
def test_unet3d_forward_cpu_wrong_inputs(rng=None):
    state = random_state(rng)
    volume = rng.standard_normal((2, 16, 16, 16)).astype(np.float32)

    npt.assert_raises(ValueError, unet3d_forward_cpu, volume[0], state)
    npt.assert_raises(ValueError, unet3d_forward_cpu, volume[:, :12], state)
    npt.assert_raises(ValueError, unet3d_forward_cpu, volume[:1], state)
    npt.assert_raises(ValueError, unet3d_forward_cpu, volume, state, num_threads=0)

    missing = dict(state)
    del missing["dc5.0.weight"]
    npt.assert_raises(ValueError, pack_unet3d_weights, missing)
    extra = dict(state, **{"ec0.1.running_mean": np.zeros(32, dtype=np.float32)})
    npt.assert_raises(ValueError, pack_unet3d_weights, extra)
    wrong = dict(state, **{"dc8.0.weight": state["dc8.0.weight"][:, :128]})
    npt.assert_raises(ValueError, pack_unet3d_weights, wrong)


@pytest.mark.skipif(not have_torch, reason="Requires Torch")
@set_random_number_generator()
def test_torch_matches_cpu(rng=None):
    torch.manual_seed(int(rng.integers(2**31)))
    model = UNet3D(n_in=2, n_out=1).eval()
    state = {k: v.numpy() for k, v in model.state_dict().items()}
    volume = rng.standard_normal((2, 16, 24, 16)).astype(np.float32)

    with torch.no_grad():
        expected = model(torch.from_numpy(volume[None]))[0].numpy()
    npt.assert_allclose(
        unet3d_forward_cpu(volume, state), expected, rtol=1e-4, atol=1e-4
    )
