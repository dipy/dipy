override workgroup_size: u32 = 32;
override d_max: u32 = 64;

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

var<workgroup> shared_C: array<f32, 4096>;  // d_max * d_max
var<workgroup> shared_V: array<f32, 4096>;  // d_max * d_max

var<workgroup> reduce_buf: array<f32, 256>;
var<workgroup> did_converge: u32;

fn idx(row: u32, col: u32, stride: u32) -> u32 {
    return row * stride + col;
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

fn off_diag_norm_sq(tid: u32, D: u32) -> f32 {
    var local_sum = 0.0;
    for (var r = tid; r < D; r += workgroup_size) {
        for (var c = 0u; c < D; c++) {
            if (c != r) {
                let v = shared_C[idx(r, c, D)];
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

    let global_base = pid * D * D;

    let total = D * D;
    for (var i = tid; i < total; i += workgroup_size) {
        shared_C[i] = cov_matrices[global_base + i];
    }

    if (params.sweep_idx == 0u) {
        for (var i = tid; i < total; i += workgroup_size) {
            let r = i / D;
            let c = i % D;
            shared_V[i] = select(0.0, 1.0, r == c);
        }
        workgroupBarrier();

        let init_norm_sq = off_diag_norm_sq(tid, D);
        if (tid == 0u) {
            off_diag_norms[pid * 2u] = sqrt(init_norm_sq);
            atomicStore(&converged[pid], 0u);
        }
        workgroupBarrier();
    } else {
        let ev_base = pid * D * D;
        for (var i = tid; i < total; i += workgroup_size) {
            shared_V[i] = eigenvectors[ev_base + i];
        }
        workgroupBarrier();
    }

    if (atomicLoad(&converged[pid]) == 1u) {
        return;
    }

    // Cyclic Jacobi sweep: Givens rotations for all (p, q) pairs, p < q.
    // The 2x2 submatrix at (p,q) is set analytically after updating the
    // remaining rows/columns to avoid read-after-write hazards.
    for (var p = 0u; p < D; p++) {
        for (var q = p + 1u; q < D; q++) {
            let c_pq = shared_C[idx(p, q, D)];

            if (abs(c_pq) < 1e-20) {
                continue;
            }

            let c_pp = shared_C[idx(p, p, D)];
            let c_qq = shared_C[idx(q, q, D)];
            let tau_val = (c_qq - c_pp) / (2.0 * c_pq);
            let t = sign(tau_val) / (abs(tau_val) + sqrt(1.0 + tau_val * tau_val));
            let cos_t = 1.0 / sqrt(1.0 + t * t);
            let sin_t = t * cos_t;

            for (var c = tid; c < D; c += workgroup_size) {
                if (c != p && c != q) {
                    let rp = shared_C[idx(p, c, D)];
                    let rq = shared_C[idx(q, c, D)];
                    let new_pc = cos_t * rp - sin_t * rq;
                    let new_qc = sin_t * rp + cos_t * rq;
                    shared_C[idx(p, c, D)] = new_pc;
                    shared_C[idx(q, c, D)] = new_qc;
                    shared_C[idx(c, p, D)] = new_pc;
                    shared_C[idx(c, q, D)] = new_qc;
                }
            }
            workgroupBarrier();

            if (tid == 0u) {
                shared_C[idx(p, p, D)] = cos_t * cos_t * c_pp - 2.0 * cos_t * sin_t * c_pq + sin_t * sin_t * c_qq;
                shared_C[idx(q, q, D)] = sin_t * sin_t * c_pp + 2.0 * cos_t * sin_t * c_pq + cos_t * cos_t * c_qq;
                shared_C[idx(p, q, D)] = 0.0;
                shared_C[idx(q, p, D)] = 0.0;
            }
            workgroupBarrier();

            for (var r = tid; r < D; r += workgroup_size) {
                let vp = shared_V[idx(r, p, D)];
                let vq = shared_V[idx(r, q, D)];
                shared_V[idx(r, p, D)] = cos_t * vp - sin_t * vq;
                shared_V[idx(r, q, D)] = sin_t * vp + cos_t * vq;
            }
            workgroupBarrier();
        }
    }

    let cur_norm_sq = off_diag_norm_sq(tid, D);

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
                reduce_buf[i] = shared_C[idx(i, i, D)];
            }

            for (var i = 1u; i < D; i++) {
                let key = reduce_buf[i];
                var j = i32(i) - 1;
                while (j >= 0 && reduce_buf[u32(j)] > key) {
                    reduce_buf[u32(j) + 1u] = reduce_buf[u32(j)];
                    for (var r = 0u; r < D; r++) {
                        let tmp = shared_V[idx(r, u32(j) + 1u, D)];
                        shared_V[idx(r, u32(j) + 1u, D)] = shared_V[idx(r, u32(j), D)];
                        shared_V[idx(r, u32(j), D)] = tmp;
                    }
                    j = j - 1;
                }
                reduce_buf[u32(j + 1)] = key;
            }

            for (var i = 0u; i < D; i++) {
                eigenvalues[pid * D + i] = reduce_buf[i];
            }
        }
        workgroupBarrier();

        for (var i = tid; i < total; i += workgroup_size) {
            eigenvectors[pid * D * D + i] = shared_V[i];
        }
    } else {
        for (var d = tid; d < D; d += workgroup_size) {
            eigenvalues[pid * D + d] = shared_C[idx(d, d, D)];
        }

        for (var i = tid; i < total; i += workgroup_size) {
            cov_matrices[global_base + i] = shared_C[i];
            eigenvectors[pid * D * D + i] = shared_V[i];
        }
    }
}
