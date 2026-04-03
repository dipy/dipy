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

@group(0) @binding(0) var<uniform>             params:         Params;
@group(0) @binding(1) var<storage, read>       input_volume:   array<f32>;
@group(0) @binding(2) var<storage, read>       patch_indices:  array<u32>;
@group(0) @binding(3) var<storage, read>       eigenvalues:    array<f32>;
@group(0) @binding(4) var<storage, read_write> eigenvectors:   array<f32>;
@group(0) @binding(5) var<storage, read>       means:          array<f32>;
@group(0) @binding(6) var<storage, read_write> reconstructed:  array<f32>;
@group(0) @binding(7) var<storage, read_write> ncomps_out:     array<u32>;
@group(0) @binding(8) var<storage, read_write> sigma_est_out:  array<f32>;

{{#shared}}
var<workgroup> shared_mat: array<f32, 4096>;  // D * (D + 1), patched at load time

fn project_voxel(
    d: u32, voxel: u32, D: u32, nc: u32,
    eig_offset: u32, mean_offset: u32,
) -> f32 {
    var val = 0.0;
    for (var d2 = 0u; d2 < D; d2++) {
        let x = input_volume[voxel * D + d2] - means[mean_offset + d2];
        val += shared_mat[d * (D + 1u) + d2] * x;
    }
    return val + means[mean_offset + d];
}
{{/shared}}
{{#global}}
// Avoids materializing the full D x D projection matrix.
fn project_voxel(
    d: u32, voxel: u32, D: u32, nc: u32,
    eig_offset: u32, mean_offset: u32,
) -> f32 {
    var val = 0.0;
    for (var col = nc; col < D; col++) {
        let w = eigenvectors[eig_offset + d * D + col];
        var inner = 0.0;
        for (var d2 = 0u; d2 < D; d2++) {
            let x = input_volume[voxel * D + d2] - means[mean_offset + d2];
            inner += eigenvectors[eig_offset + d2 * D + col] * x;
        }
        val += w * inner;
    }
    return val + means[mean_offset + d];
}
{{/global}}

var<workgroup> ncomps_shared: u32;
var<workgroup> sigma_shared: f32;

@compute @workgroup_size(workgroup_size)
fn main(
    @builtin(workgroup_id)         wg_id: vec3<u32>,
    @builtin(local_invocation_id)  lid:   vec3<u32>,
) {
    let patch_idx = wg_id.x;
    if (patch_idx >= params.tile_size) { return; }

    let D  = params.D;
    let Ny = params.Ny;
    let Nz = params.Nz;
    let tid = lid.x;

    let base = (params.tile_offset + patch_idx) * 6u;
    let ix1 = patch_indices[base + 0u];
    let ix2 = patch_indices[base + 1u];
    let jx1 = patch_indices[base + 2u];
    let jx2 = patch_indices[base + 3u];
    let kx1 = patch_indices[base + 4u];
    let kx2 = patch_indices[base + 5u];

    let num_samples = (ix2 - ix1) * (jx2 - jx1) * (kx2 - kx1);
    let eig_offset = patch_idx * D * D;
    let eval_offset = patch_idx * D;

    // Marcenko-Pastur classification + tau threshold (single-threaded).
    // Matches _pca_classifier() + tau_factor logic in genpca() exactly.
    if (tid == 0u) {
        var eff_D = D;
        var eval_start = 0u;
        if (D > num_samples - 1u) {
            eff_D = num_samples - 1u;
            eval_start = D - eff_D;
        }

        var sum_all = 0.0;
        for (var ii = eval_start; ii < D; ii++) {
            sum_all += eigenvalues[eval_offset + ii];
        }
        var variance = sum_all / f32(eff_D);

        var c = i32(eff_D) - 1;
        let d_first = eigenvalues[eval_offset + eval_start];
        var d_c = eigenvalues[eval_offset + eval_start + u32(c)];
        var r = d_c - d_first - 4.0 * sqrt(f32(c + 1) / f32(num_samples)) * variance;

        while (r > 0.0) {
            if (c <= 0) { break; }
            var partial_sum = 0.0;
            for (var ii = eval_start; ii < eval_start + u32(c); ii++) {
                partial_sum += eigenvalues[eval_offset + ii];
            }
            variance = partial_sum / f32(u32(c));
            c = c - 1;
            d_c = eigenvalues[eval_offset + eval_start + u32(c)];
            r = d_c - d_first - 4.0 * sqrt(f32(c + 1) / f32(num_samples)) * variance;
        }

        let tau_factor = 1.0 + sqrt(f32(D) / f32(num_samples));
        let tau = tau_factor * tau_factor * variance;

        var nc = 0u;
        for (var ii = 0u; ii < D; ii++) {
            if (eigenvalues[eval_offset + ii] < tau) { nc++; }
        }

        ncomps_shared = nc;
        // Store the variance (not sigma) for correct weighted accumulation.
        // The CPU accumulates this_var * theta, then at the end computes
        // sqrt(weighted_avg_var). So we store variance here.
        sigma_shared = max(variance, 0.0);
        ncomps_out[patch_idx] = nc;
        sigma_est_out[patch_idx] = sigma_shared;
    }

    workgroupBarrier();
    let nc = ncomps_shared;

{{#shared}}
    for (var ii = tid; ii < D * D; ii += workgroup_size) {
        let r1 = ii / D;
        let r2 = ii % D;
        var dot_val = 0.0;
        for (var col = nc; col < D; col++) {
            dot_val += eigenvectors[eig_offset + r1 * D + col]
                     * eigenvectors[eig_offset + r2 * D + col];
        }
        shared_mat[r1 * (D + 1u) + r2] = dot_val;
    }
    workgroupBarrier();
{{/shared}}

    let recon_offset = patch_idx * params.num_samples * D;
    let mean_offset = patch_idx * D;

    var s = 0u;
    for (var i = ix1; i < ix2; i++) {
        for (var j = jx1; j < jx2; j++) {
            for (var k = kx1; k < kx2; k++) {
                let voxel = i * Ny * Nz + j * Nz + k;
                for (var d = tid; d < D; d += workgroup_size) {
                    reconstructed[recon_offset + s * D + d] =
                        project_voxel(d, voxel, D, nc, eig_offset, mean_offset);
                }
                s++;
            }
        }
    }
}
