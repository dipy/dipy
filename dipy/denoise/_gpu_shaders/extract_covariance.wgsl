override workgroup_size: u32 = 32;

struct Params {
    D:            u32,
    Nx:           u32,
    Ny:           u32,
    Nz:           u32,
    num_samples:  u32,
    tile_offset:  u32,
    tile_size:    u32,
}

@group(0) @binding(0) var<uniform>          params:         Params;
@group(0) @binding(1) var<storage, read>    input_volume:   array<f32>;
@group(0) @binding(2) var<storage, read>    patch_indices:  array<u32>;
@group(0) @binding(3) var<storage, read_write> cov_matrices: array<f32>;
@group(0) @binding(4) var<storage, read_write> means:        array<f32>;

@compute @workgroup_size(workgroup_size)
fn main(
    @builtin(workgroup_id)    wg_id:    vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let patch_idx = wg_id.x;
    if (patch_idx >= params.tile_size) {
        return;
    }

    let D  = params.D;
    let Ny = params.Ny;
    let Nz = params.Nz;

    // Load bounding box for this pid from patch_indices.
    let base = (params.tile_offset + patch_idx) * 6u;
    let ix1 = patch_indices[base + 0u];
    let ix2 = patch_indices[base + 1u];
    let jx1 = patch_indices[base + 2u];
    let jx2 = patch_indices[base + 3u];
    let kx1 = patch_indices[base + 4u];
    let kx2 = patch_indices[base + 5u];

    let num_samples = (ix2 - ix1) * (jx2 - jx1) * (kx2 - kx1);
    let inv_n = 1.0 / f32(num_samples);

    let tid = lid.x;

    for (var d = tid; d < D; d += workgroup_size) {
        var sum = 0.0;
        for (var i = ix1; i < ix2; i++) {
            for (var j = jx1; j < jx2; j++) {
                for (var k = kx1; k < kx2; k++) {
                    let voxel = i * Ny * Nz + j * Nz + k;
                    sum += input_volume[voxel * D + d];
                }
            }
        }
        means[patch_idx * D + d] = sum * inv_n;
    }

    workgroupBarrier();

    // Only compute upper triangle and mirror; the covariance matrix is symmetric.
    let cov_base = patch_idx * D * D;
    for (var d = tid; d < D; d += workgroup_size) {
        let mean_d = means[patch_idx * D + d];

        for (var d2 = d; d2 < D; d2++) {
            let mean_d2 = means[patch_idx * D + d2];
            var acc = 0.0;

            for (var i = ix1; i < ix2; i++) {
                for (var j = jx1; j < jx2; j++) {
                    for (var k = kx1; k < kx2; k++) {
                        let voxel = i * Ny * Nz + j * Nz + k;
                        let a = input_volume[voxel * D + d]  - mean_d;
                        let b = input_volume[voxel * D + d2] - mean_d2;
                        acc += a * b;
                    }
                }
            }

            let cov_val = acc * inv_n;
            cov_matrices[cov_base + d * D + d2] = cov_val;
            if (d2 != d) {
                cov_matrices[cov_base + d2 * D + d] = cov_val;
            }
        }
    }
}
