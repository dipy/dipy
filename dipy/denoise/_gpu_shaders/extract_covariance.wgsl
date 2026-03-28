override workgroup_size: u32 = 32;
override b_max: u32 = 192;

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

var<workgroup> shared_means: array<f32, 512>;
var<workgroup> shared_patch: array<f32, 123072>;

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

    let patch_nz = kx2 - kx1;
    let patch_ny = jx2 - jx1;

    let B = min(b_max, num_samples);

    for (var d = tid; d < D; d += workgroup_size) {
        shared_means[d] = 0.0;
    }
    workgroupBarrier();

    var tile_start = 0u;
    while (tile_start < num_samples) {
        let tile_end = min(tile_start + B, num_samples);
        let tile_len = tile_end - tile_start;

        for (var load_idx = tid; load_idx < tile_len * D; load_idx += workgroup_size) {
            let s = load_idx / D;
            let d = load_idx % D;
            let global_s = tile_start + s;
            let off_i = global_s / (patch_ny * patch_nz);
            let rem2 = global_s % (patch_ny * patch_nz);
            let off_j = rem2 / patch_nz;
            let off_k = rem2 % patch_nz;
            let voxel = (ix1 + off_i) * Ny * Nz + (jx1 + off_j) * Nz + (kx1 + off_k);
            shared_patch[s * D + d] = input_volume[voxel * D + d];
        }
        workgroupBarrier();

        for (var d = tid; d < D; d += workgroup_size) {
            var sum = 0.0;
            for (var s = 0u; s < tile_len; s++) {
                sum += shared_patch[s * D + d];
            }
            shared_means[d] += sum;
        }
        workgroupBarrier();

        tile_start = tile_end;
    }

    for (var d = tid; d < D; d += workgroup_size) {
        shared_means[d] *= inv_n;
        means[patch_idx * D + d] = shared_means[d];
    }
    workgroupBarrier();

    let cov_base = patch_idx * D * D;
    for (var idx = tid; idx < D * D; idx += workgroup_size) {
        cov_matrices[cov_base + idx] = 0.0;
    }
    workgroupBarrier();

    tile_start = 0u;
    while (tile_start < num_samples) {
        let tile_end = min(tile_start + B, num_samples);
        let tile_len = tile_end - tile_start;

        for (var load_idx = tid; load_idx < tile_len * D; load_idx += workgroup_size) {
            let s = load_idx / D;
            let d = load_idx % D;
            let global_s = tile_start + s;
            let off_i = global_s / (patch_ny * patch_nz);
            let rem2 = global_s % (patch_ny * patch_nz);
            let off_j = rem2 / patch_nz;
            let off_k = rem2 % patch_nz;
            let voxel = (ix1 + off_i) * Ny * Nz + (jx1 + off_j) * Nz + (kx1 + off_k);
            shared_patch[s * D + d] = input_volume[voxel * D + d];
        }
        workgroupBarrier();

        for (var d = tid; d < D; d += workgroup_size) {
            let mean_d = shared_means[d];

            for (var d2 = d; d2 < D; d2++) {
                let mean_d2 = shared_means[d2];
                var acc = 0.0;

                for (var s = 0u; s < tile_len; s++) {
                    let a = shared_patch[s * D + d]  - mean_d;
                    let b = shared_patch[s * D + d2] - mean_d2;
                    acc += a * b;
                }

                cov_matrices[cov_base + d * D + d2] += acc * inv_n;
            }
        }
        workgroupBarrier();

        tile_start = tile_end;
    }

    for (var d = tid; d < D; d += workgroup_size) {
        for (var d2 = d + 1u; d2 < D; d2++) {
            cov_matrices[cov_base + d2 * D + d] = cov_matrices[cov_base + d * D + d2];
        }
    }
}
