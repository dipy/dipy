override workgroup_size: u32 = 32;

struct Params {
    D:         u32,
    tile_size: u32,
    sweep_idx: u32,
    epsilon:   f32,
}

@group(0) @binding(0) var<uniform>             params:         Params;
@group(0) @binding(1) var<storage, read_write> cov_matrices:   array<f32>;
@group(0) @binding(2) var<storage, read_write> eigenvectors:   array<f32>;
@group(0) @binding(3) var<storage, read_write> eigenvalues:    array<f32>;
@group(0) @binding(4) var<storage, read_write> converged:      array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> off_diag_norms: array<f32>;

var<workgroup> reduce_buf: array<f32, 256>;
var<workgroup> did_converge: u32;

fn idx(base: u32, row: u32, col: u32, stride: u32) -> u32 {
    return base + row * stride + col;
}

fn workgroup_reduce_sum(tid: u32, val: f32) -> f32 {
    reduce_buf[tid] = val;
    workgroupBarrier();

    var stride = workgroup_size >> 1u;
    while (stride > 0u) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let result = reduce_buf[0];
    workgroupBarrier();
    return result;
}

fn off_diag_norm_sq(tid: u32, D: u32, cov_base: u32) -> f32 {
    var local_sum = 0.0;
    for (var r = tid; r < D; r += workgroup_size) {
        for (var c = 0u; c < D; c++) {
            if (c != r) {
                let v = cov_matrices[idx(cov_base, r, c, D)];
                local_sum += v * v;
            }
        }
    }
    return workgroup_reduce_sum(tid, local_sum);
}

@compute @workgroup_size(workgroup_size)
fn main(
    @builtin(workgroup_id)        wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let pid = wg_id.x;
    if (pid >= params.tile_size) {
        return;
    }

    let D   = params.D;
    let tid = lid.x;

    let cov_base = pid * D * D;
    let ev_base  = pid * D * D;
    let total    = D * D;

    if (params.sweep_idx == 0u) {
        for (var i = tid; i < total; i += workgroup_size) {
            let r = i / D;
            let c = i % D;
            eigenvectors[ev_base + i] = select(0.0, 1.0, r == c);
        }
        storageBarrier();
        workgroupBarrier();

        let init_norm_sq = off_diag_norm_sq(tid, D, cov_base);
        if (tid == 0u) {
            off_diag_norms[pid * 2u] = sqrt(init_norm_sq);
            atomicStore(&converged[pid], 0u);
        }
        storageBarrier();
        workgroupBarrier();
    }

    if (atomicLoad(&converged[pid]) == 1u) {
        return;
    }

    // Cyclic Jacobi sweep: Givens rotations for all (p, q) pairs, p < q.
    for (var p = 0u; p < D; p++) {
        for (var q = p + 1u; q < D; q++) {
            let c_pq = cov_matrices[idx(cov_base, p, q, D)];

            if (abs(c_pq) < 1e-20) {
                continue;
            }

            let c_pp = cov_matrices[idx(cov_base, p, p, D)];
            let c_qq = cov_matrices[idx(cov_base, q, q, D)];
            let tau_val = (c_qq - c_pp) / (2.0 * c_pq);
            let t = sign(tau_val) / (abs(tau_val) + sqrt(1.0 + tau_val * tau_val));
            let cos_t = 1.0 / sqrt(1.0 + t * t);
            let sin_t = t * cos_t;

            for (var c = tid; c < D; c += workgroup_size) {
                if (c != p && c != q) {
                    let rp = cov_matrices[idx(cov_base, p, c, D)];
                    let rq = cov_matrices[idx(cov_base, q, c, D)];
                    let new_pc = cos_t * rp - sin_t * rq;
                    let new_qc = sin_t * rp + cos_t * rq;
                    cov_matrices[idx(cov_base, p, c, D)] = new_pc;
                    cov_matrices[idx(cov_base, q, c, D)] = new_qc;
                    cov_matrices[idx(cov_base, c, p, D)] = new_pc;
                    cov_matrices[idx(cov_base, c, q, D)] = new_qc;
                }
            }
            storageBarrier();
            workgroupBarrier();

            if (tid == 0u) {
                cov_matrices[idx(cov_base, p, p, D)] = cos_t * cos_t * c_pp - 2.0 * cos_t * sin_t * c_pq + sin_t * sin_t * c_qq;
                cov_matrices[idx(cov_base, q, q, D)] = sin_t * sin_t * c_pp + 2.0 * cos_t * sin_t * c_pq + cos_t * cos_t * c_qq;
                cov_matrices[idx(cov_base, p, q, D)] = 0.0;
                cov_matrices[idx(cov_base, q, p, D)] = 0.0;
            }
            storageBarrier();
            workgroupBarrier();

            for (var r = tid; r < D; r += workgroup_size) {
                let vp = eigenvectors[idx(ev_base, r, p, D)];
                let vq = eigenvectors[idx(ev_base, r, q, D)];
                eigenvectors[idx(ev_base, r, p, D)] = cos_t * vp - sin_t * vq;
                eigenvectors[idx(ev_base, r, q, D)] = sin_t * vp + cos_t * vq;
            }
            storageBarrier();
            workgroupBarrier();
        }
    }

    let cur_norm_sq = off_diag_norm_sq(tid, D, cov_base);

    if (tid == 0u) {
        let cur_norm = sqrt(cur_norm_sq);
        off_diag_norms[pid * 2u + 1u] = cur_norm;

        let init_norm = off_diag_norms[pid * 2u];
        if (init_norm <= 0.0 || cur_norm / init_norm < params.epsilon) {
            atomicStore(&converged[pid], 1u);
            did_converge = 1u;
        } else {
            did_converge = 0u;
        }
    }
    workgroupBarrier();

    // Sort eigenvalues ascending and permute eigenvector columns on convergence.
    if (did_converge == 1u) {
        if (tid == 0u) {
            for (var i = 0u; i < D; i++) {
                reduce_buf[i] = cov_matrices[idx(cov_base, i, i, D)];
            }

            for (var i = 1u; i < D; i++) {
                let key = reduce_buf[i];
                var j = i32(i) - 1;
                while (j >= 0 && reduce_buf[u32(j)] > key) {
                    reduce_buf[u32(j) + 1u] = reduce_buf[u32(j)];
                    for (var r = 0u; r < D; r++) {
                        let tmp = eigenvectors[idx(ev_base, r, u32(j) + 1u, D)];
                        eigenvectors[idx(ev_base, r, u32(j) + 1u, D)] = eigenvectors[idx(ev_base, r, u32(j), D)];
                        eigenvectors[idx(ev_base, r, u32(j), D)] = tmp;
                    }
                    j = j - 1;
                }
                reduce_buf[u32(j + 1)] = key;
            }

            for (var i = 0u; i < D; i++) {
                eigenvalues[pid * D + i] = reduce_buf[i];
            }
        }
    } else {
        for (var d = tid; d < D; d += workgroup_size) {
            eigenvalues[pid * D + d] = cov_matrices[idx(cov_base, d, d, D)];
        }
    }
}
