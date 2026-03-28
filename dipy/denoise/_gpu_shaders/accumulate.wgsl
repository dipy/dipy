{{#native}}
enable shader_f32_atomic;
{{/native}}

override workgroup_size: u32 = 64;

struct Params {
    D:            u32,
    Nx:           u32,
    Ny:           u32,
    Nz:           u32,
    num_samples:  u32,
    tile_offset:  u32,
    tile_size:    u32,
}

@group(0) @binding(0) var<uniform>             params:         Params;
@group(0) @binding(1) var<storage, read>       patch_indices:  array<u32>;
@group(0) @binding(2) var<storage, read>       reconstructed:  array<f32>;
@group(0) @binding(3) var<storage, read>       ncomps:         array<u32>;
@group(0) @binding(4) var<storage, read>       sigma_est:      array<f32>;
{{#cas}}
@group(0) @binding(5) var<storage, read_write> output_volume:  array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> theta:          array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> sigma_out:      array<atomic<u32>>;
{{/cas}}
{{#native}}
@group(0) @binding(5) var<storage, read_write> output_volume:  array<atomic<f32>>;
@group(0) @binding(6) var<storage, read_write> theta:          array<atomic<f32>>;
@group(0) @binding(7) var<storage, read_write> sigma_out:      array<atomic<f32>>;
{{/native}}

@compute @workgroup_size(workgroup_size)
fn main(
    @builtin(workgroup_id)        wg_id: vec3<u32>,
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

    let num_samples = params.num_samples;

    let weight = 1.0 / (1.0 + f32(D) - f32(ncomps[patch_idx]));
    let sigma_w = sigma_est[patch_idx] * weight;

    let tid = lid.x;

    let patch_nx = ix2 - ix1;
    let patch_ny = jx2 - jx1;
    let patch_nz = kx2 - kx1;
    let total_voxels = patch_nx * patch_ny * patch_nz;

    for (var v = tid; v < total_voxels; v += workgroup_size) {
        let off_i = v / (patch_ny * patch_nz);
        let rem   = v % (patch_ny * patch_nz);
        let off_j = rem / patch_nz;
        let off_k = rem % patch_nz;

        let i = ix1 + off_i;
        let j = jx1 + off_j;
        let k = kx1 + off_k;

        let voxel_linear = i * Ny * Nz + j * Nz + k;
        let sample_idx = v;

{{#cas}}
        {
            var old_val = atomicLoad(&theta[voxel_linear]);
            loop {
                let new_val = bitcast<u32>(bitcast<f32>(old_val) + weight);
                let result = atomicCompareExchangeWeak(&theta[voxel_linear], old_val, new_val);
                if result.exchanged { break; }
                old_val = result.old_value;
            }
        }

        {
            var old_val = atomicLoad(&sigma_out[voxel_linear]);
            loop {
                let new_val = bitcast<u32>(bitcast<f32>(old_val) + sigma_w);
                let result = atomicCompareExchangeWeak(&sigma_out[voxel_linear], old_val, new_val);
                if result.exchanged { break; }
                old_val = result.old_value;
            }
        }

        let recon_base = patch_idx * num_samples * D + sample_idx * D;
        for (var d = 0u; d < D; d++) {
            let val = reconstructed[recon_base + d] * weight;
            let out_idx = voxel_linear * D + d;
            var old_val = atomicLoad(&output_volume[out_idx]);
            loop {
                let new_val = bitcast<u32>(bitcast<f32>(old_val) + val);
                let result = atomicCompareExchangeWeak(&output_volume[out_idx], old_val, new_val);
                if result.exchanged { break; }
                old_val = result.old_value;
            }
        }
{{/cas}}
{{#native}}
        atomicAdd(&theta[voxel_linear], weight);
        atomicAdd(&sigma_out[voxel_linear], sigma_w);

        let recon_base = patch_idx * num_samples * D + sample_idx * D;
        for (var d = 0u; d < D; d++) {
            let val = reconstructed[recon_base + d] * weight;
            let out_idx = voxel_linear * D + d;
            atomicAdd(&output_volume[out_idx], val);
        }
{{/native}}
    }
}
