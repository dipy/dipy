/* CPU inference of MeshNet, the dilated convolutional network behind MindGrab.
 *
 * Adapted from src/meshnet_cpu.c of brainchopC, distributed under the MIT
 * license reproduced in meshnet_cpu.c.
 */
#ifndef DIPY_MESHNET_CPU_H
#define DIPY_MESHNET_CPU_H

/* Activations are stored channel-last, padded to this many lanes. */
#define MESHNET_STRIDE 16
#define MESHNET_MAX_CLASSES 64

/*
 * Run a MeshNet forward pass over a whole volume.
 *
 * volume          nz * ny * nx floats, x fastest.
 * logits          n_classes * nz * ny * nx floats, written class by class.
 * first_kernel    27 * MESHNET_STRIDE floats, taps ordered (dz, dy, dx),
 *                 zero beyond `channels`.
 * hidden_kernels  (n_layers - 1) * 27 * MESHNET_STRIDE * MESHNET_STRIDE floats,
 *                 indexed (layer, tap, input channel, output channel), zero
 *                 beyond `channels` on both channel axes.
 * dilations       n_layers dilations, the first one for first_kernel.
 * classifier      channels * n_classes floats.
 * bias            n_classes floats.
 * n_threads       number of OpenMP threads; ignored without OpenMP.
 *
 * Every convolution zero-pads its input and is followed by a parameter-free
 * group normalization (one group per channel) and a GELU activation.
 *
 * Returns 0 on success, 1 on invalid arguments, 2 when out of memory.
 */
int meshnet_forward(const float *volume, float *logits, int nz, int ny, int nx,
                    const float *first_kernel, const float *hidden_kernels,
                    const int *dilations, int n_layers, int channels,
                    const float *classifier, const float *bias, int n_classes,
                    int n_threads);

#endif
