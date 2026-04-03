override workgroup_size: u32 = 32;

struct Params {
    D:         u32,
    tile_size: u32,
    epsilon:   f32,
    _pad:      u32,
}

@group(0) @binding(0) var<uniform>             params:         Params;
@group(0) @binding(1) var<storage, read_write> cov_matrices:   array<f32>;
@group(0) @binding(2) var<storage, read_write> eigenvectors:   array<f32>;
@group(0) @binding(3) var<storage, read_write> eigenvalues:    array<f32>;
@group(0) @binding(4) var<storage, read_write> converged:      array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> off_diag_norms: array<f32>;

var<workgroup> reduce_buf: array<f32, 256>;
var<workgroup> did_converge: u32;

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

{{#shared}}
var<workgroup> shared_C: array<f32, 4096>;  // D * (D + 1), patched at load time
var<workgroup> shared_V: array<f32, 4096>;  // D * (D + 1), patched at load time

fn c_get(pid: u32, r: u32, c: u32, D: u32) -> f32 { return shared_C[r * (D + 1u) + c]; }
fn c_set(pid: u32, r: u32, c: u32, D: u32, v: f32) { shared_C[r * (D + 1u) + c] = v; }
fn v_get(pid: u32, r: u32, c: u32, D: u32) -> f32 { return shared_V[r * (D + 1u) + c]; }
fn v_set(pid: u32, r: u32, c: u32, D: u32, v: f32) { shared_V[r * (D + 1u) + c] = v; }
fn sync() { workgroupBarrier(); }
{{/shared}}
{{#global}}
fn c_get(pid: u32, r: u32, c: u32, D: u32) -> f32 { return cov_matrices[pid * D * D + r * D + c]; }
fn c_set(pid: u32, r: u32, c: u32, D: u32, v: f32) { cov_matrices[pid * D * D + r * D + c] = v; }
fn v_get(pid: u32, r: u32, c: u32, D: u32) -> f32 { return eigenvectors[pid * D * D + r * D + c]; }
fn v_set(pid: u32, r: u32, c: u32, D: u32, v: f32) { eigenvectors[pid * D * D + r * D + c] = v; }
fn sync() { storageBarrier(); workgroupBarrier(); }
{{/global}}

fn off_diag_norm_sq(pid: u32, tid: u32, D: u32) -> f32 {
    var local_sum = 0.0;
    for (var r = tid; r < D; r += workgroup_size) {
        for (var c = 0u; c < D; c++) {
            if (c != r) {
                let v = c_get(pid, r, c, D);
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
    if (pid >= params.tile_size) { return; }

    let D   = params.D;
    let tid = lid.x;

    // Init: runs once per tile (off_diag buffer zeroed by CPU).
    // Uses max(norm, 1e-38) so a zero-norm matrix doesn't re-trigger.
    if (off_diag_norms[pid * 2u] == 0.0) {
        let total_init = D * D;
        for (var i = tid; i < total_init; i += workgroup_size) {
            eigenvectors[pid * D * D + i] = select(0.0, 1.0, (i / D) == (i % D));
        }
        storageBarrier();
        workgroupBarrier();

        var local_sum = 0.0;
        for (var r = tid; r < D; r += workgroup_size) {
            for (var c = 0u; c < D; c++) {
                if (c != r) {
                    let v = cov_matrices[pid * D * D + r * D + c];
                    local_sum += v * v;
                }
            }
        }
        let norm_sq = workgroup_reduce_sum(tid, local_sum);
        if (tid == 0u) {
            off_diag_norms[pid * 2u] = max(sqrt(norm_sq), 1e-38);
        }
        workgroupBarrier();
    }

    if (atomicLoad(&converged[pid]) == 1u) { return; }

    let total = D * D;

{{#shared}}
    let global_base = pid * D * D;
    for (var i = tid; i < total; i += workgroup_size) {
        let r = i / D;
        let c = i % D;
        shared_C[r * (D + 1u) + c] = cov_matrices[global_base + i];
        shared_V[r * (D + 1u) + c] = eigenvectors[global_base + i];
    }
    workgroupBarrier();
{{/shared}}

    // Cyclic Jacobi sweep — the 2x2 submatrix at (p,q) is set analytically
    // after the remaining rows/columns to avoid read-after-write hazards.
    for (var p = 0u; p < D; p++) {
        for (var q = p + 1u; q < D; q++) {
            let c_pq = c_get(pid, p, q, D);
            if (abs(c_pq) < 1e-20) { continue; }

            let c_pp = c_get(pid, p, p, D);
            let c_qq = c_get(pid, q, q, D);
            let tau_val = (c_qq - c_pp) / (2.0 * c_pq);
            var t: f32;
            if (abs(tau_val) < 1e-30) {
                t = 1.0;
            } else {
                t = sign(tau_val) / (abs(tau_val) + sqrt(1.0 + tau_val * tau_val));
            }
            let cos_t = 1.0 / sqrt(1.0 + t * t);
            let sin_t = t * cos_t;

            for (var c = tid; c < D; c += workgroup_size) {
                if (c != p && c != q) {
                    let rp = c_get(pid, p, c, D);
                    let rq = c_get(pid, q, c, D);
                    let new_pc = cos_t * rp - sin_t * rq;
                    let new_qc = sin_t * rp + cos_t * rq;
                    c_set(pid, p, c, D, new_pc);
                    c_set(pid, q, c, D, new_qc);
                    c_set(pid, c, p, D, new_pc);
                    c_set(pid, c, q, D, new_qc);
                }
            }
            sync();

            if (tid == 0u) {
                c_set(pid, p, p, D, cos_t * cos_t * c_pp - 2.0 * cos_t * sin_t * c_pq + sin_t * sin_t * c_qq);
                c_set(pid, q, q, D, sin_t * sin_t * c_pp + 2.0 * cos_t * sin_t * c_pq + cos_t * cos_t * c_qq);
                c_set(pid, p, q, D, 0.0);
                c_set(pid, q, p, D, 0.0);
            }
            sync();

            for (var r = tid; r < D; r += workgroup_size) {
                let vp = v_get(pid, r, p, D);
                let vq = v_get(pid, r, q, D);
                v_set(pid, r, p, D, cos_t * vp - sin_t * vq);
                v_set(pid, r, q, D, sin_t * vp + cos_t * vq);
            }
            sync();
        }
    }

    let cur_norm_sq = off_diag_norm_sq(pid, tid, D);

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

    if (did_converge == 1u) {
        if (tid == 0u) {
            for (var i = 0u; i < D; i++) {
                reduce_buf[i] = c_get(pid, i, i, D);
            }
            for (var i = 1u; i < D; i++) {
                let key = reduce_buf[i];
                var j = i32(i) - 1;
                while (j >= 0 && reduce_buf[u32(j)] > key) {
                    reduce_buf[u32(j) + 1u] = reduce_buf[u32(j)];
                    for (var r = 0u; r < D; r++) {
                        let tmp = v_get(pid, r, u32(j) + 1u, D);
                        v_set(pid, r, u32(j) + 1u, D, v_get(pid, r, u32(j), D));
                        v_set(pid, r, u32(j), D, tmp);
                    }
                    j = j - 1;
                }
                reduce_buf[u32(j + 1)] = key;
            }
            for (var i = 0u; i < D; i++) {
                eigenvalues[pid * D + i] = reduce_buf[i];
            }
        }
{{#shared}}
        workgroupBarrier();
        for (var i = tid; i < total; i += workgroup_size) {
            let r = i / D;
            let c = i % D;
            eigenvectors[pid * D * D + i] = shared_V[r * (D + 1u) + c];
        }
{{/shared}}
    } else {
        for (var d = tid; d < D; d += workgroup_size) {
            eigenvalues[pid * D + d] = c_get(pid, d, d, D);
        }
{{#shared}}
        for (var i = tid; i < total; i += workgroup_size) {
            let r = i / D;
            let c = i % D;
            cov_matrices[global_base + i] = shared_C[r * (D + 1u) + c];
            eigenvectors[pid * D * D + i] = shared_V[r * (D + 1u) + c];
        }
{{/shared}}
    }
}
