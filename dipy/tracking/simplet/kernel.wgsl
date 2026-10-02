// WebGPU (WGSL) backend for dipy.tracking.simpletracker
//
// Constants prepended by dipy/tracking/webgpusimplet.py:
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
    rng_seed_lo: i32,
    rng_seed_hi: i32,
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
@group(1) @binding(2) var<storage, read_write> slineSeed: array<i32>;
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

fn load_sline_f3(idx: u32) -> vec3<f32> {
    let base = idx * 3u;
    return vec3<f32>(sline[base], sline[base + 1u], sline[base + 2u]);
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

// ── Philox4x32-10 RNG (matches curandStatePhilox4_32_10_t) ──────────
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

fn philox_init(seed_lo: u32, seed_hi: u32, subsequence: u32) -> PhiloxState {
    var s: PhiloxState;
    s.key = vec2<u32>(seed_lo, seed_hi);
    s.counter = vec4<u32>(0u, subsequence, 0u, 0u);
    s.output = philox4x32_10(s.counter, s.key);
    s.idx = 0u;
    return s;
}

fn philox_next(s: PhiloxState) -> PhiloxState {
    var r = s;
    r.counter.x += 1u;
    if (r.counter.x == 0u) {
        r.counter.y += 1u;
        if (r.counter.y == 0u) {
            r.counter.z += 1u;
            if (r.counter.z == 0u) {
                r.counter.w += 1u;
            }
        }
    }
    r.output = philox4x32_10(r.counter, r.key);
    r.idx = 0u;
    return r;
}

struct PhiloxUniformResult {
    state: PhiloxState,
    value: f32,
}

// uniform f32 in (0, 1], matches curand_uniform
fn philox_uniform(s: PhiloxState) -> PhiloxUniformResult {
    var r = s;
    if (r.idx >= 4u) {
        r = philox_next(r);
    }
    var bits: u32;
    switch (r.idx) {
        case 0u: { bits = r.output.x; }
        case 1u: { bits = r.output.y; }
        case 2u: { bits = r.output.z; }
        default: { bits = r.output.w; }
    }
    r.idx += 1u;
    let value = f32(bits) * 2.3283064365386963e-10 + 2.3283064365386963e-10;
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

struct TrilinearSetup {
    status: i32,
    wgh: array<array<f32, 2>, 3>,
    coo: array<array<i32, 2>, 3>,
}

fn trilinear_setup(dimx: i32, dimy: i32, dimz: i32, point: vec3<f32>) -> TrilinearSetup {
    let HALF: f32 = 0.5;
    var r: TrilinearSetup;

    if (point.x < -HALF || point.x + HALF >= f32(dimx) ||
        point.y < -HALF || point.y + HALF >= f32(dimy) ||
        point.z < -HALF || point.z + HALF >= f32(dimz)) {
        r.status = -1;
        return r;
    }

    let fl = floor(point);

    r.wgh[0][1] = point.x - fl.x;
    r.wgh[0][0] = 1.0 - r.wgh[0][1];
    r.coo[0][0] = max(0, i32(fl.x));
    r.coo[0][1] = min(dimx - 1, r.coo[0][0] + 1);

    r.wgh[1][1] = point.y - fl.y;
    r.wgh[1][0] = 1.0 - r.wgh[1][1];
    r.coo[1][0] = max(0, i32(fl.y));
    r.coo[1][1] = min(dimy - 1, r.coo[1][0] + 1);

    r.wgh[2][1] = point.z - fl.z;
    r.wgh[2][0] = 1.0 - r.wgh[2][1];
    r.coo[2][0] = max(0, i32(fl.z));
    r.coo[2][1] = min(dimz - 1, r.coo[2][0] + 1);

    r.status = 0;
    return r;
}

fn interpolation_helper_dataf(s: TrilinearSetup, dimy: i32, dimz: i32, dimt: i32, t: i32) -> f32 {
    var tmp: f32 = 0.0;
    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            for (var k = 0; k < 2; k++) {
                let idx = s.coo[0][i] * dimy * dimz * dimt +
                          s.coo[1][j] * dimz * dimt +
                          s.coo[2][k] * dimt + t;
                tmp += s.wgh[0][i] * s.wgh[1][j] * s.wgh[2][k] * dataf[idx];
            }
        }
    }
    return tmp;
}

fn interpolation_helper_metric(s: TrilinearSetup, dimy: i32, dimz: i32) -> f32 {
    var tmp: f32 = 0.0;
    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            for (var k = 0; k < 2; k++) {
                let idx = s.coo[0][i] * dimy * dimz + s.coo[1][j] * dimz + s.coo[2][k];
                tmp += s.wgh[0][i] * s.wgh[1][j] * s.wgh[2][k] * metric_map[idx];
            }
        }
    }
    return tmp;
}

// Interpolate all dimt channels of dataf at point into wg_sh_mem[wg_offset..].
fn trilinear_interp_dataf(dimx: i32, dimy: i32, dimz: i32, dimt: i32,
                          point: vec3<f32>, wg_offset: u32, tidx: u32) -> i32 {
    let s = trilinear_setup(dimx, dimy, dimz, point);
    if (s.status != 0) { return -1; }
    for (var t = i32(tidx); t < dimt; t += i32(THR_X_SL)) {
        wg_sh_mem[wg_offset + u32(t)] = interpolation_helper_dataf(s, dimy, dimz, dimt, t);
    }
    return 0;
}

fn check_point_fn(tc_threshold: f32, point: vec3<f32>,
                  dimx: i32, dimy: i32, dimz: i32) -> i32 {
    let s = trilinear_setup(dimx, dimy, dimz, point);
    if (s.status != 0) {
        return OUTSIDEIMAGE;
    }
    if (interpolation_helper_metric(s, dimy, dimz) > tc_threshold) {
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

    var low: i32 = 0;
    var high: i32 = dimt - 1;
    while ((high - low) >= i32(THR_X_SL)) {
        let mid = (low + high) / 2;
        if (wg_sh_mem[sh_offset + u32(mid)] < selected_cdf) {
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
        ind_prob = dimt - 1;
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

struct TrackerResult {
    nsteps: i32,
    state: PhiloxState,
}

fn tracker_prob_fn(st: PhiloxState, seed: vec3<f32>, first_step: vec3<f32>,
                   sline_base: u32, sh_offset: u32, tidx: u32, tidy: u32) -> TrackerResult {
    var rng = st;
    var tissue_class: i32 = TRACKPOINT;
    var point = seed;
    var direction = first_step;

    if (tidx == 0u) {
        store_sline_f3(sline_base, point);
    }
    subgroupBarrier();

    var i: i32 = 1;
    for (; i < params.max_sline_len; i++) {
        let gdr = get_direction_prob(rng, direction, point, sh_offset, tidx, tidy);
        rng = gdr.state;
        subgroupBarrier();
        direction = wg_new_dir[tidy];
        subgroupBarrier();

        if (gdr.ndir == 0) { break; }

        point += direction * params.step_size;

        if (tidx == 0u) {
            store_sline_f3(sline_base + u32(i), point);
        }
        subgroupBarrier();

        tissue_class = check_point_fn(params.tc_threshold, point,
                                      params.dimx, params.dimy, params.dimz);
        if (tissue_class != TRACKPOINT) {
            break;
        }
    }
    return TrackerResult(i, rng);
}

// ── kernel: generate streamlines from precomputed seed directions ───
//
// slineOutOff : (nseed+1,) exclusive prefix sum of the number of
//               directions per seed (from get_num_streamlines_prob)
// shDir0      : (sum(ndir), 3) initial direction of each streamline
// sline       : (sum(ndir) * max_sline_len * 2, 3) output points

@compute @workgroup_size(32, 2, 1)
fn genStreamlinesProb_k(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) gid: vec3<u32>
) {
    let tidx = tid.x;
    let tidy = tid.y;
    let slid = gid.x * BLOCK_Y + tidy;

    if (i32(slid) >= params.nseed) { return; }

    let global_id = gid.x * BLOCK_Y * THR_X_SL + THR_X_SL * tidy + tidx;
    var st = philox_init(u32(params.rng_seed_lo), u32(params.rng_seed_hi), global_id + 1u);

    let sh_offset = tidy * N32DIMT;
    let seed = load_seeds_f3(slid);

    let ndir = slineOutOff[slid + 1u] - slineOutOff[slid];
    var sline_off = slineOutOff[slid];

    for (var i = 0; i < ndir; i++) {
        let first_step = load_shDir0_f3(u32(sline_off));
        let sline_base = u32(sline_off) * u32(params.max_sline_len) * 2u;

        if (tidx == 0u) {
            slineSeed[sline_off] = i32(slid);
        }

        // backward
        let trB = tracker_prob_fn(st, seed, -first_step, sline_base, sh_offset, tidx, tidy);
        st = trB.state;
        let stepsB = trB.nsteps;

        // reverse backward streamline
        for (var j = i32(tidx); j < stepsB / 2; j += i32(THR_X_SL)) {
            let a = sline_base + u32(j);
            let b = sline_base + u32(stepsB - 1 - j);
            let pa = load_sline_f3(a);
            let pb = load_sline_f3(b);
            store_sline_f3(a, pb);
            store_sline_f3(b, pa);
        }
        // forward
        let trF = tracker_prob_fn(st, seed, first_step, sline_base + u32(stepsB - 1),
                                  sh_offset, tidx, tidy);
        st = trF.state;

        if (tidx == 0u) {
            slineLen[sline_off] = stepsB - 1 + trF.nsteps;
        }
        sline_off += 1;
    }
}
