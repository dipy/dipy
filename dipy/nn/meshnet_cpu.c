/* CPU inference of MeshNet, the dilated convolutional network behind MindGrab.
 *
 * Adapted from src/meshnet_cpu.c of brainchopC, under the following license:
 *
 * MIT License
 *
 * Copyright (c) 2024-2026 spikedoanz, Sergey Plis
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include "meshnet_cpu.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#define MN_CS MESHNET_STRIDE
#define MN_BLOCK 4
#define MN_EPS 1e-5

#if defined(_MSC_VER)
#define MN_RESTRICT __restrict
#define MN_INLINE static __forceinline
#elif defined(__GNUC__)
#define MN_RESTRICT __restrict__
#define MN_INLINE static inline __attribute__((always_inline))
#else
#define MN_RESTRICT
#define MN_INLINE static
#endif

typedef struct {
    int nz, ny, nx;
} mn_shape;

/*
 * One z slice of a zero-padded dilated 3x3x3 convolution.
 *
 * Always inlined so that each caller gets a copy specialized on its literal
 * source stride: the MN_CS accumulators of a block of MN_BLOCK voxels then
 * stay in vector registers, and each weight vector is shared by the block.
 * The x range is split in three segments so that the accumulation loops need
 * no bounds test.
 */
MN_INLINE void mn_conv_slice(const float *MN_RESTRICT source,
                             float *MN_RESTRICT destination,
                             const float *MN_RESTRICT weights, int dilation,
                             int z, mn_shape shape, int source_stride,
                             int input_channels)
{
    const size_t weight_layer = (size_t)source_stride * MN_CS;
    const int nx = shape.nx;
    for (int y = 0; y < shape.ny; y++) {
        ptrdiff_t row_offset[9];
        int row_tap[9];
        int row_count = 0;
        for (int tz = 0; tz < 3; tz++) {
            int zz = z + (tz - 1) * dilation;
            if (zz < 0 || zz >= shape.nz) continue;
            for (int ty = 0; ty < 3; ty++) {
                int yy = y + (ty - 1) * dilation;
                if (yy < 0 || yy >= shape.ny) continue;
                row_offset[row_count] = ((ptrdiff_t)zz * shape.ny + yy) * nx;
                row_tap[row_count] = (tz * 3 + ty) * 3;
                row_count++;
            }
        }
        float *row_output = destination + ((size_t)z * shape.ny + y) * nx * MN_CS;

        if (nx <= 2 * dilation) {
            for (int x = 0; x < nx; x++) {
                float accumulator[MN_CS];
                for (int channel = 0; channel < MN_CS; channel++)
                    accumulator[channel] = 0.0f;
                for (int row = 0; row < row_count; row++)
                    for (int tx = 0; tx < 3; tx++) {
                        int xx = x + (tx - 1) * dilation;
                        if (xx < 0 || xx >= nx) continue;
                        const float *input =
                            source + (row_offset[row] + xx) * source_stride;
                        const float *tap_weights =
                            weights + (size_t)(row_tap[row] + tx) * weight_layer;
                        for (int ic = 0; ic < input_channels; ic++) {
                            float value = input[ic];
                            for (int channel = 0; channel < MN_CS; channel++)
                                accumulator[channel] +=
                                    value * tap_weights[(size_t)ic * MN_CS + channel];
                        }
                    }
                for (int channel = 0; channel < MN_CS; channel++)
                    row_output[(size_t)x * MN_CS + channel] = accumulator[channel];
            }
            continue;
        }

        const int segment_x[4] = {0, dilation, nx - dilation, nx};
        for (int segment = 0; segment < 3; segment++) {
            const int tap_x_low = segment == 0 ? 1 : 0;
            const int tap_x_high = segment == 2 ? 1 : 2;
            ptrdiff_t tap_voxel[27];
            const float *tap_weight[27];
            int tap_count = 0;
            for (int row = 0; row < row_count; row++)
                for (int tx = tap_x_low; tx <= tap_x_high; tx++) {
                    tap_voxel[tap_count] = row_offset[row] + (tx - 1) * dilation;
                    tap_weight[tap_count] =
                        weights + (size_t)(row_tap[row] + tx) * weight_layer;
                    tap_count++;
                }
            int x = segment_x[segment];
            const int x_end = segment_x[segment + 1];
            for (; x + MN_BLOCK <= x_end; x += MN_BLOCK) {
                float accumulator[MN_BLOCK][MN_CS];
                for (int block = 0; block < MN_BLOCK; block++)
                    for (int channel = 0; channel < MN_CS; channel++)
                        accumulator[block][channel] = 0.0f;
                for (int tap = 0; tap < tap_count; tap++) {
                    const float *input = source + (tap_voxel[tap] + x) * source_stride;
                    const float *tap_weights = tap_weight[tap];
                    for (int ic = 0; ic < input_channels; ic++) {
                        const float *weight_vector = tap_weights + (size_t)ic * MN_CS;
                        float input_value[MN_BLOCK];
                        for (int block = 0; block < MN_BLOCK; block++)
                            input_value[block] = input[block * source_stride + ic];
                        for (int channel = 0; channel < MN_CS; channel++) {
                            float weight = weight_vector[channel];
                            for (int block = 0; block < MN_BLOCK; block++)
                                accumulator[block][channel] += input_value[block] * weight;
                        }
                    }
                }
                for (int block = 0; block < MN_BLOCK; block++)
                    for (int channel = 0; channel < MN_CS; channel++)
                        row_output[(size_t)(x + block) * MN_CS + channel] =
                            accumulator[block][channel];
            }
            for (; x < x_end; x++) {
                float accumulator[MN_CS];
                for (int channel = 0; channel < MN_CS; channel++)
                    accumulator[channel] = 0.0f;
                for (int tap = 0; tap < tap_count; tap++) {
                    const float *input = source + (tap_voxel[tap] + x) * source_stride;
                    const float *tap_weights = tap_weight[tap];
                    for (int ic = 0; ic < input_channels; ic++) {
                        float value = input[ic];
                        for (int channel = 0; channel < MN_CS; channel++)
                            accumulator[channel] +=
                                value * tap_weights[(size_t)ic * MN_CS + channel];
                    }
                }
                for (int channel = 0; channel < MN_CS; channel++)
                    row_output[(size_t)x * MN_CS + channel] = accumulator[channel];
            }
        }
    }
}

static void mn_conv_first(const float *source, float *destination,
                          const float *weights, int dilation, mn_shape shape,
                          int n_threads)
{
    int z;
    (void)n_threads;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4) num_threads(n_threads)
#endif
    for (z = 0; z < shape.nz; z++)
        mn_conv_slice(source, destination, weights, dilation, z, shape, 1, 1);
}

static void mn_conv_hidden(const float *source, float *destination,
                           const float *weights, int dilation, int channels,
                           mn_shape shape, int n_threads)
{
    int z;
    (void)n_threads;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4) num_threads(n_threads)
#endif
    for (z = 0; z < shape.nz; z++) {
        /* A literal input channel count lets the compiler fully unroll the
           inner loop; skipping MindGrab's zero padding lane saves ~15%. */
        if (channels == 15)
            mn_conv_slice(source, destination, weights, dilation, z, shape,
                          MN_CS, 15);
        else
            mn_conv_slice(source, destination, weights, dilation, z, shape,
                          MN_CS, MN_CS);
    }
}

/* tanh-form GELU, with the rational tanh approximation of brainchopC, which
   unlike tanhf lets the compiler vectorize the activation loop. */
MN_INLINE float mn_tanh(float x)
{
    if (x > 9.0f) x = 9.0f;
    if (x < -9.0f) x = -9.0f;
    float u = x * x;
    float p = -8.29118133e-14f;
    p = p * u + 5.19263868e-11f;
    p = p * u - 2.00294448e-08f;
    p = p * u + 1.11017944e-05f;
    p = p * u + 0.00309865153f;
    p = p * u + 0.130791619f;
    p = p * u + 0.99999994f;
    float q = 0.000253859733f;
    q = q * u + 0.024473751f;
    q = q * u + 0.464124829f;
    q = q * u + 1.0f;
    return x * p / q;
}

MN_INLINE float mn_gelu(float x)
{
    float u = 0.797884583f * (x + 0.044715f * (x * x * x));
    return (0.5f * x) * (1.0f + mn_tanh(u));
}

/*
 * Group normalization with one group per channel, followed by GELU, in place.
 *
 * The per-slice moments are folded serially, so the result does not depend on
 * the number of threads. Padding lanes hold zeros, which stay zeros.
 */
static void mn_norm_gelu(float *buffer, double *partial, mn_shape shape,
                         int n_threads)
{
    const size_t plane = (size_t)shape.ny * shape.nx;
    const double count = (double)plane * shape.nz;
    float mean[MN_CS], invstd[MN_CS];
    int z;
    (void)n_threads;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4) num_threads(n_threads)
#endif
    for (z = 0; z < shape.nz; z++) {
        double sum[MN_CS], square[MN_CS];
        for (int channel = 0; channel < MN_CS; channel++)
            sum[channel] = square[channel] = 0.0;
        const float *slice = buffer + (size_t)z * plane * MN_CS;
        for (size_t i = 0; i < plane; i++)
            for (int channel = 0; channel < MN_CS; channel++) {
                double value = slice[i * MN_CS + channel];
                sum[channel] += value;
                square[channel] += value * value;
            }
        for (int channel = 0; channel < MN_CS; channel++) {
            partial[(size_t)z * 2 * MN_CS + channel] = sum[channel];
            partial[(size_t)z * 2 * MN_CS + MN_CS + channel] = square[channel];
        }
    }

    for (int channel = 0; channel < MN_CS; channel++) {
        double sum = 0.0, square = 0.0;
        for (int zz = 0; zz < shape.nz; zz++) {
            sum += partial[(size_t)zz * 2 * MN_CS + channel];
            square += partial[(size_t)zz * 2 * MN_CS + MN_CS + channel];
        }
        double channel_mean = sum / count;
        double variance = square / count - channel_mean * channel_mean;
        if (!(variance > 0.0)) variance = 0.0;
        mean[channel] = (float)channel_mean;
        invstd[channel] = (float)(1.0 / sqrt(variance + MN_EPS));
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4) num_threads(n_threads)
#endif
    for (z = 0; z < shape.nz; z++) {
        float *slice = buffer + (size_t)z * plane * MN_CS;
        for (size_t i = 0; i < plane; i++) {
            float *value = slice + i * MN_CS;
            for (int channel = 0; channel < MN_CS; channel++)
                value[channel] =
                    mn_gelu((value[channel] - mean[channel]) * invstd[channel]);
        }
    }
}

static void mn_classify(const float *source, float *logits,
                        const float *classifier, const float *bias,
                        int channels, int n_classes, mn_shape shape,
                        int n_threads)
{
    const size_t plane = (size_t)shape.ny * shape.nx;
    const size_t n_voxels = plane * shape.nz;
    int z;
    (void)n_threads;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4) num_threads(n_threads)
#endif
    for (z = 0; z < shape.nz; z++) {
        float logit[MESHNET_MAX_CLASSES];
        for (size_t i = 0; i < plane; i++) {
            size_t voxel = (size_t)z * plane + i;
            const float *value = source + voxel * MN_CS;
            for (int k = 0; k < n_classes; k++)
                logit[k] = bias[k];
            for (int channel = 0; channel < channels; channel++)
                for (int k = 0; k < n_classes; k++)
                    logit[k] += value[channel] * classifier[channel * n_classes + k];
            for (int k = 0; k < n_classes; k++)
                logits[(size_t)k * n_voxels + voxel] = logit[k];
        }
    }
}

int meshnet_forward(const float *volume, float *logits, int nz, int ny, int nx,
                    const float *first_kernel, const float *hidden_kernels,
                    const int *dilations, int n_layers, int channels,
                    const float *classifier, const float *bias, int n_classes,
                    int n_threads)
{
    if (nz < 1 || ny < 1 || nx < 1 || n_layers < 1 || channels < 1 ||
            channels > MN_CS || n_classes < 1 ||
            n_classes > MESHNET_MAX_CLASSES || n_threads < 1)
        return 1;
    for (int layer = 0; layer < n_layers; layer++)
        if (dilations[layer] < 1) return 1;

    const mn_shape shape = {nz, ny, nx};
    const size_t n_values = (size_t)nz * ny * nx * MN_CS;
    float *first = (float *)malloc(n_values * sizeof(float));
    float *second = n_layers > 1 ? (float *)malloc(n_values * sizeof(float)) : NULL;
    double *partial = (double *)malloc((size_t)nz * 2 * MN_CS * sizeof(double));
    if (!first || (n_layers > 1 && !second) || !partial) {
        free(first);
        free(second);
        free(partial);
        return 2;
    }

    mn_conv_first(volume, first, first_kernel, dilations[0], shape, n_threads);
    mn_norm_gelu(first, partial, shape, n_threads);

    const size_t hidden_size = (size_t)27 * MN_CS * MN_CS;
    float *source = first, *destination = second;
    for (int layer = 1; layer < n_layers; layer++) {
        mn_conv_hidden(source, destination,
                       hidden_kernels + (size_t)(layer - 1) * hidden_size,
                       dilations[layer], channels, shape, n_threads);
        mn_norm_gelu(destination, partial, shape, n_threads);
        float *swap = source;
        source = destination;
        destination = swap;
    }

    mn_classify(source, logits, classifier, bias, channels, n_classes, shape,
                n_threads);
    free(first);
    free(second);
    free(partial);
    return 0;
}
