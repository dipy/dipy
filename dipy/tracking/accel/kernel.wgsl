// WebGPU (WGSL) backend for dipy.tracking.accel
//
// Constants prepended by dipy/tracking/accel/webgpu.py:
//   const SPHERE_SYMM: u32   1 if sphere vertices are antipodally symmetric
//   const N32DIMT: u32       dimt rounded up to a multiple of 32
//
// Conventions:
//   - vec3 data is stored as 3 consecutive f32 in flat storage buffers
//   - one subgroup (32 threads) tracks one streamline, BLOCK_Y streamlines
//     per workgroup; subgroup operations are required
//   - WGSL only supports f32
//   - PhiloxState is passed by value and returned in result structs (WGSL
//     has no mutable references across function boundaries)
//   - the tracking loop, random number generator and output layout match
//     kernel.cu (see the comments there)

const THR_X_SL: u32 = 32u;
const BLOCK_Y: u32 = 2u;

const REAL_MIN: f32 = -3.4028235e+38;

const OUTSIDEIMAGE: i32 = 0;
const INVALIDPOINT: i32 = 1;
const TRACKPOINT: i32 = 2;
const ENDPOINT: i32 = 3;

struct ProbTrackingParams {
    max_angle: f32,
    tc_threshold: f32,
    step_size: f32,
    pmf_threshold: f32,
    rng_seed_lo: u32,
    rng_seed_hi: u32,
    sl_base_lo: u32,     // index in the tractogram of the first streamline
    sl_base_hi: u32,     // of this dispatch
    nseed: i32,
    dimx: i32,
    dimy: i32,
    dimz: i32,
    dimt: i32,
    max_sline_len: i32,
}

// ── buffer bindings ─────────────────────────────────────────────────
// Group 0: static data
@group(0) @binding(0) var<storage, read> params: ProbTrackingParams;
@group(0) @binding(1) var<storage, read> seeds: array<f32>;
@group(0) @binding(2) var<storage, read> dataf: array<f32>;
@group(0) @binding(3) var<storage, read> metric_map: array<f32>;
@group(0) @binding(4) var<storage, read> sphere_vertices: array<f32>;

// Group 1: per-batch buffers
@group(1) @binding(0) var<storage, read> slineOutOff: array<i32>;
@group(1) @binding(1) var<storage, read> shDir0: array<f32>;
@group(1) @binding(2) var<storage, read_write> slineStart: array<i32>;
@group(1) @binding(3) var<storage, read_write> slineLen: array<i32>;
@group(1) @binding(4) var<storage, read_write> sline: array<f32>;

fn load_seeds_f3(idx: u32) -> vec3<f32> {
    let base = idx * 3u;
    return vec3<f32>(seeds[base], seeds[base + 1u], seeds[base + 2u]);
}

fn load_sphere_verts_f3(idx: u32) -> vec3<f32> {
    let base = idx * 3u;
    return vec3<f32>(sphere_vertices[base], sphere_vertices[base + 1u], sphere_vertices[base + 2u]);
}

fn load_shDir0_f3(idx: u32) -> vec3<f32> {
    let base = idx * 3u;
    return vec3<f32>(shDir0[base], shDir0[base + 1u], shDir0[base + 2u]);
}

fn store_sline_f3(idx: u32, v: vec3<f32>) {
    let base = idx * 3u;
    sline[base] = v.x;
    sline[base + 1u] = v.y;
    sline[base + 2u] = v.z;
}

// ── workgroup memory ────────────────────────────────────────────────
var<workgroup> wg_sh_mem: array<f32, BLOCK_Y * N32DIMT>;
var<workgroup> wg_new_dir: array<vec3<f32>, BLOCK_Y>;

// ── Philox4x32-10 RNG (same generator as kernel.cu) ─────────────────
// Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"
//            (SC '11).  DOI 10.1145/2063384.2063405

const PHILOX_M4x32_0: u32 = 0xD2511F53u;
const PHILOX_M4x32_1: u32 = 0xCD9E8D57u;
const PHILOX_W32_0: u32   = 0x9E3779B9u;
const PHILOX_W32_1: u32   = 0xBB67AE85u;

struct PhiloxState {
    counter: vec4<u32>,
    key: vec2<u32>,
    output: vec4<u32>,
    idx: u32,
}

// upper 32 bits of a*b (WGSL has no u64)
fn mulhi32(a: u32, b: u32) -> u32 {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let lo_lo = a_lo * b_lo;
    let lo_hi = a_lo * b_hi;
    let hi_lo = a_hi * b_lo;
    let hi_hi = a_hi * b_hi;

    let mid_sum = (lo_lo >> 16u) + (lo_hi & 0xFFFFu) + (hi_lo & 0xFFFFu);
    return hi_hi + (lo_hi >> 16u) + (hi_lo >> 16u) + (mid_sum >> 16u);
}

fn philox4x32_single_round(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    let lo0 = ctr.x * PHILOX_M4x32_0;
    let hi0 = mulhi32(ctr.x, PHILOX_M4x32_0);
    let lo1 = ctr.z * PHILOX_M4x32_1;
    let hi1 = mulhi32(ctr.z, PHILOX_M4x32_1);
    return vec4<u32>(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

fn philox4x32_10(ctr_in: vec4<u32>, key_in: vec2<u32>) -> vec4<u32> {
    var ctr = ctr_in;
    var key = key_in;
    let bump = vec2<u32>(PHILOX_W32_0, PHILOX_W32_1);
    for (var r = 0; r < 9; r++) {
        ctr = philox4x32_single_round(ctr, key);
        key += bump;
    }
    return philox4x32_single_round(ctr, key);
}

// random stream of the streamline with index (sl_hi << 32) | sl_lo
fn philox_init(seed_lo: u32, seed_hi: u32, sl_lo: u32, sl_hi: u32) -> PhiloxState {
    var s: PhiloxState;
    s.key = vec2<u32>(seed_lo, seed_hi);
    s.counter = vec4<u32>(0u, sl_lo, sl_hi, 0u);
    s.output = vec4<u32>(0u);
    s.idx = 4u;
    return s;
}

struct PhiloxUniformResult {
    state: PhiloxState,
    value: f32,
}

// uniform f32 in [0, 1) from the 24 high bits: exact in f32 and f64
fn philox_uniform(s: PhiloxState) -> PhiloxUniformResult {
    var r = s;
    if (r.idx == 4u) {
        r.output = philox4x32_10(r.counter, r.key);
        r.counter.x += 1u;
        r.idx = 0u;
    }
    var bits: u32;
    switch (r.idx) {
        case 0u: { bits = r.output.x; }
        case 1u: { bits = r.output.y; }
        case 2u: { bits = r.output.z; }
        default: { bits = r.output.w; }
    }
    r.idx += 1u;
    let value = f32(bits >> 8u) * (1.0 / 16777216.0);
    return PhiloxUniformResult(r, value);
}

// ── subgroup reductions on wg_sh_mem ────────────────────────────────

fn sg_max_reduce_wg(n: i32, wg_offset: u32, min_val: f32, tidx: u32) -> f32 {
    var m = min_val;
    for (var i = i32(tidx); i < n; i += i32(THR_X_SL)) {
        m = max(m, wg_sh_mem[wg_offset + u32(i)]);
    }
    m = max(m, subgroupShuffleXor(m, 16u));
    m = max(m, subgroupShuffleXor(m, 8u));
    m = max(m, subgroupShuffleXor(m, 4u));
    m = max(m, subgroupShuffleXor(m, 2u));
    m = max(m, subgroupShuffleXor(m, 1u));
    return m;
}

fn prefix_sum_sh(wg_offset: u32, len: i32, tidx: u32) {
    for (var j = 0; j < len; j += i32(THR_X_SL)) {
        if (tidx == 0u && j != 0) {
            wg_sh_mem[wg_offset + u32(j)] += wg_sh_mem[wg_offset + u32(j - 1)];
        }
        subgroupBarrier();

        var t_pmf: f32 = 0.0;
        if (j + i32(tidx) < len) {
            t_pmf = wg_sh_mem[wg_offset + u32(j) + tidx];
        }
        for (var i = 1u; i < THR_X_SL; i *= 2u) {
            let tmp = subgroupShuffleUp(t_pmf, i);
            if (tidx >= i && j + i32(tidx) < len) {
                t_pmf += tmp;
            }
        }
        if (j + i32(tidx) < len) {
            wg_sh_mem[wg_offset + u32(j) + tidx] = t_pmf;
        }
        subgroupBarrier();
    }
}

// ── trilinear interpolation ─────────────────────────────────────────

// The 8 corners of a point: flat voxel offsets (times stride) and weights,
// kept in vectors so that they stay in registers across the channel loop.
struct Corners {
    status: i32,
    off_lo: vec4<i32>,
    off_hi: vec4<i32>,
    w_lo: vec4<f32>,
    w_hi: vec4<f32>,
}

fn trilinear_corners(dimx: i32, dimy: i32, dimz: i32, stride: i32,
                     point: vec3<f32>) -> Corners {
    let HALF: f32 = 0.5;
    var r: Corners;

    if (point.x < -HALF || point.x + HALF >= f32(dimx) ||
        point.y < -HALF || point.y + HALF >= f32(dimy) ||
        point.z < -HALF || point.z + HALF >= f32(dimz)) {
        r.status = -1;
        return r;
    }

    // as in dipy.core.interpolation: within half a voxel of the border,
    // only the border voxel is used
    let fl = floor(point);
    let fli = vec3<i32>(fl);
    let hi = point - fl;
    let lo = vec3<f32>(1.0) - hi;
    let c0 = max(vec3<i32>(0), fli);
    let c1 = min(vec3<i32>(dimx, dimy, dimz) - vec3<i32>(1), fli + vec3<i32>(1));

    let sx = dimy * dimz * stride;
    let sy = dimz * stride;
    let x0 = c0.x * sx;
    let x1 = c1.x * sx;
    let y0 = c0.y * sy;
    let y1 = c1.y * sy;
    let z0 = c0.z * stride;
    let z1 = c1.z * stride;

    r.off_lo = vec4<i32>(x0 + y0 + z0, x0 + y0 + z1, x0 + y1 + z0, x0 + y1 + z1);
    r.off_hi = vec4<i32>(x1 + y0 + z0, x1 + y0 + z1, x1 + y1 + z0, x1 + y1 + z1);
    r.w_lo = vec4<f32>(lo.x * lo.y * lo.z, lo.x * lo.y * hi.z,
                       lo.x * hi.y * lo.z, lo.x * hi.y * hi.z);
    r.w_hi = vec4<f32>(hi.x * lo.y * lo.z, hi.x * lo.y * hi.z,
                       hi.x * hi.y * lo.z, hi.x * hi.y * hi.z);
    r.status = 0;
    return r;
}

// Interpolate all dimt channels of dataf at point into wg_sh_mem[wg_offset..].
fn trilinear_interp_dataf(dimx: i32, dimy: i32, dimz: i32, dimt: i32,
                          point: vec3<f32>, wg_offset: u32, tidx: u32) -> i32 {
    let c = trilinear_corners(dimx, dimy, dimz, dimt, point);
    if (c.status != 0) { return -1; }
    for (var t = i32(tidx); t < dimt; t += i32(THR_X_SL)) {
        let v_lo = vec4<f32>(dataf[c.off_lo.x + t], dataf[c.off_lo.y + t],
                             dataf[c.off_lo.z + t], dataf[c.off_lo.w + t]);
        let v_hi = vec4<f32>(dataf[c.off_hi.x + t], dataf[c.off_hi.y + t],
                             dataf[c.off_hi.z + t], dataf[c.off_hi.w + t]);
        wg_sh_mem[wg_offset + u32(t)] = dot(c.w_lo, v_lo) + dot(c.w_hi, v_hi);
    }
    return 0;
}

fn check_point_fn(tc_threshold: f32, point: vec3<f32>,
                  dimx: i32, dimy: i32, dimz: i32) -> i32 {
    let c = trilinear_corners(dimx, dimy, dimz, 1, point);
    if (c.status != 0) {
        return OUTSIDEIMAGE;
    }
    let v_lo = vec4<f32>(metric_map[c.off_lo.x], metric_map[c.off_lo.y],
                         metric_map[c.off_lo.z], metric_map[c.off_lo.w]);
    let v_hi = vec4<f32>(metric_map[c.off_hi.x], metric_map[c.off_hi.y],
                         metric_map[c.off_hi.z], metric_map[c.off_hi.w]);
    if (dot(c.w_lo, v_lo) + dot(c.w_hi, v_hi) > tc_threshold) {
        return TRACKPOINT;
    }
    return ENDPOINT;
}

// ── probabilistic direction getter ──────────────────────────────────

struct GetDirProbResult {
    ndir: i32,
    state: PhiloxState,
}

fn get_direction_prob(st: PhiloxState, dir: vec3<f32>, point: vec3<f32>,
                      sh_offset: u32, tidx: u32, tidy: u32) -> GetDirProbResult {
    var rng = st;
    let dimt = params.dimt;

    subgroupBarrier();
    let rv = trilinear_interp_dataf(params.dimx, params.dimy, params.dimz, dimt,
                                    point, sh_offset, tidx);
    subgroupBarrier();
    if (rv != 0) {
        return GetDirProbResult(0, rng);
    }

    let absol_thresh = params.pmf_threshold * sg_max_reduce_wg(dimt, sh_offset, REAL_MIN, tidx);
    subgroupBarrier();

    let cos_similarity = cos(params.max_angle);
    for (var i = i32(tidx); i < dimt; i += i32(THR_X_SL)) {
        let sv = load_sphere_verts_f3(u32(i));
        let dot_val = dir.x * sv.x + dir.y * sv.y + dir.z * sv.z;
        if (wg_sh_mem[sh_offset + u32(i)] < absol_thresh ||
            select(dot_val, abs(dot_val), SPHERE_SYMM == 1u) < cos_similarity) {
            wg_sh_mem[sh_offset + u32(i)] = 0.0;
        }
    }
    subgroupBarrier();

    prefix_sum_sh(sh_offset, dimt, tidx);

    let last_cdf = wg_sh_mem[sh_offset + u32(dimt - 1)];
    if (last_cdf == 0.0) {
        return GetDirProbResult(0, rng);
    }

    // lane 0 draws the random number and holds the authoritative RNG state
    var selected_cdf: f32 = 0.0;
    if (tidx == 0u) {
        let ur = philox_uniform(rng);
        rng = ur.state;
        selected_cdf = ur.value * last_cdf;
    }
    selected_cdf = subgroupBroadcastFirst(selected_cdf);

    // first index with cdf > selected_cdf (see kernel.cu)
    var low: i32 = 0;
    var high: i32 = dimt - 1;
    while ((high - low) >= i32(THR_X_SL)) {
        let mid = (low + high) / 2;
        if (wg_sh_mem[sh_offset + u32(mid)] <= selected_cdf) {
            low = mid;
        } else {
            high = mid;
        }
    }

    var ballot_pred = false;
    if (low + i32(tidx) <= high) {
        ballot_pred = selected_cdf < wg_sh_mem[sh_offset + u32(low) + tidx];
    }
    let msk = subgroupBallot(ballot_pred).x;
    var ind_prob: i32;
    if (msk != 0u) {
        ind_prob = low + i32(countTrailingZeros(msk));
    } else {
        ind_prob = high;
    }

    if (tidx == 0u) {
        let sv = load_sphere_verts_f3(u32(ind_prob));
        if (dir.x * sv.x + dir.y * sv.y + dir.z * sv.z > 0.0) {
            wg_new_dir[tidy] = sv;
        } else {
            wg_new_dir[tidy] = -sv;
        }
    }
    return GetDirProbResult(1, rng);
}

// ── tracker: follow one direction from a seed ───────────────────────
// Point i is stored at point index sline_center + i * stride; nsteps is the
// number of points kept, counting the seed (see tracker_d in kernel.cu).

// stopped is set when the stopping map ended the streamline (a valid
// ending on the CPU).
struct TrackerResult {
    nsteps: i32,
    first_dir: vec3<f32>,
    state: PhiloxState,
    stopped: bool,
}

fn tracker_prob_fn(st: PhiloxState, seed: vec3<f32>, first_step: vec3<f32>,
                   sline_center: u32, stride: i32, sh_offset: u32,
                   tidx: u32, tidy: u32) -> TrackerResult {
    var rng = st;
    var point = seed;
    var direction = first_step;
    var first_dir = first_step;
    var stopped = false;

    var i: i32 = 1;
    for (; i < params.max_sline_len; i++) {
        let gdr = get_direction_prob(rng, direction, point, sh_offset, tidx, tidy);
        rng = gdr.state;
        subgroupBarrier();
        direction = wg_new_dir[tidy];
        subgroupBarrier();

        if (gdr.ndir == 0) { break; }
        if (i == 1) { first_dir = direction; }

        point += direction * params.step_size;

        if (tidx == 0u) {
            store_sline_f3(u32(i32(sline_center) + i * stride), point);
        }
        subgroupBarrier();

        if (check_point_fn(params.tc_threshold, point,
                           params.dimx, params.dimy, params.dimz) != TRACKPOINT) {
            stopped = true;
            break;
        }
    }
    return TrackerResult(min(i, params.max_sline_len - 1), first_dir, rng, stopped);
}

// ── kernel: generate streamlines from precomputed seed directions ───
// Same buffers and layout as genStreamlinesProb_k in kernel.cu.

@compute @workgroup_size(32, 2, 1)
fn genStreamlinesProb_k(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) gid: vec3<u32>
) {
    let tidx = tid.x;
    let tidy = tid.y;
    let slid = gid.x * BLOCK_Y + tidy;

    if (i32(slid) >= params.nseed) { return; }

    let sh_offset = tidy * N32DIMT;
    let seed = load_seeds_f3(slid);
    let max_len = u32(params.max_sline_len);

    let first_sl = slineOutOff[slid];
    let ndir = slineOutOff[slid + 1u] - first_sl;

    for (var i = 0; i < ndir; i++) {
        let sl = u32(first_sl + i);
        // 64-bit streamline index sl_base + sl
        let sl_lo = params.sl_base_lo + sl;
        let sl_hi = params.sl_base_hi + select(0u, 1u, sl_lo < sl);
        let st = philox_init(params.rng_seed_lo, params.rng_seed_hi, sl_lo, sl_hi);

        let first_step = load_shDir0_f3(sl);
        let sline_center = sl * max_len * 2u + max_len;
        if (tidx == 0u) {
            store_sline_f3(sline_center, seed);
        }
        subgroupBarrier();

        // forward, then backward from the direction of the first forward step
        let trF = tracker_prob_fn(st, seed, first_step, sline_center, 1,
                                  sh_offset, tidx, tidy);
        var back = first_step;
        if (trF.nsteps > 1) {
            back = trF.first_dir;
        }
        let trB = tracker_prob_fn(trF.state, seed, -back, sline_center, -1,
                                  sh_offset, tidx, tidy);

        if (tidx == 0u) {
            slineStart[sl] = params.max_sline_len - (trB.nsteps - 1);
            // negative length: invalid ending (see _streamline_generator)
            let len = trF.nsteps + trB.nsteps - 1;
            slineLen[sl] = select(-len, len, trF.stopped && trB.stopped);
        }
    }
}


// ── kernel: pack the streamlines for the readback ───────────────────
// Each streamline reserves 2 * max_sline_len points in sline but only uses
// |slineLen| of them: copying them contiguously, at outOffsets[sl], shrinks
// the device-to-host copy by about the ratio of the two.

@group(2) @binding(0) var<storage, read> outOffsets: array<u32>;
@group(2) @binding(1) var<storage, read_write> packed: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn compactStreamlines_k(@builtin(global_invocation_id) gid: vec3<u32>,
                        @builtin(num_workgroups) nwg: vec3<u32>) {
    let sl = gid.y * nwg.x * 64u + gid.x;
    if (sl + 1u >= arrayLength(&outOffsets)) { return; }
    let src = sl * u32(params.max_sline_len) * 2u + u32(slineStart[sl]);
    let dst = outOffsets[sl];
    let n = u32(abs(slineLen[sl]));
    for (var i = 0u; i < n; i++) {
        let a = (src + i) * 3u;
        let b = (dst + i) * 3u;
        packed[b] = sline[a];
        packed[b + 1u] = sline[a + 1u];
        packed[b + 2u] = sline[a + 2u];
    }
}
