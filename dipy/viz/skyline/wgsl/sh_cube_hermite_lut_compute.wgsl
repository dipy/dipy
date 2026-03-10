// GPU compute shader: bake cube-mapped Hermite LUT for SH glyphs.
//
// Two-pass design:
//   pass1_eval   – evaluate SH on internal padded grid (N+6)² per face
//   pass2_hermite – 4th-order finite-difference → (val, du, dv, d²uv)
//
// Uses a flat 1D dispatch to avoid exceeding the 65535 workgroup limit.
// Each thread derives (glyph, face, row, col) from its flat global ID.

@group(0) @binding(0) var<storage, read> s_sh_coeffs: array<f32>;
@group(0) @binding(1) var<storage, read_write> s_hermite_lut: array<vec4<f32>>;

struct ComputeUniforms {
    n_glyphs: u32,
    n_coeffs: u32,
    lut_res: u32,       // N  (output face edge = N+2)
    l_max: u32,
    items_per_glyph_p1: u32,   // s_int * s_int * 6  (pass 1)
    items_per_glyph_p2: u32,   // s_out * s_out * 6  (pass 2)
    _pad2: u32,
    _pad3: u32,
}
@group(0) @binding(2) var<uniform> uniforms: ComputeUniforms;

@group(0) @binding(3) var<storage, read_write> s_scratch: array<f32>;

const PI: f32 = 3.14159265359;
const SQRT_2: f32 = 1.4142135623730951;

// ── SH evaluation ───────────────────────────────────────────────

fn factorial_ratio(l: i32, m: i32) -> f32 {
    if (m == 0 || l == 0) { return 1.0; }
    var result = 1.0;
    let start = l - abs(m) + 1;
    let stop  = l + abs(m);
    if (stop < start) { return 1.0; }
    for (var k = start; k <= stop; k++) { result /= f32(k); }
    return result;
}

fn legendre_polynomial(l: i32, m_in: i32, x: f32) -> f32 {
    let am = abs(m_in);
    let somx2 = sqrt(max(1.0 - x * x, 0.0));
    var pmm = 1.0;
    var fact = 1.0;
    for (var i = 1; i <= am; i++) {
        pmm *= -fact * somx2;
        fact += 2.0;
    }
    if (l == am) { return pmm; }
    var pmm1 = x * f32(2 * am + 1) * pmm;
    if (l == am + 1) { return pmm1; }
    for (var ll = am + 2; ll <= l; ll++) {
        let pll = (x * f32(2*ll-1) * pmm1 - f32(ll+am-1) * pmm) / f32(ll-am);
        pmm = pmm1;
        pmm1 = pll;
    }
    return pmm1;
}

fn sh_normalization(l: i32, m: i32) -> f32 {
    let factor = (2.0 * f32(l) + 1.0) / (4.0 * PI);
    return sqrt(factor * factorial_ratio(l, m));
}

fn spherical_harmonic(l: i32, m: i32, cos_theta: f32, phi: f32) -> f32 {
    let norm = sh_normalization(l, m);
    let plm  = legendre_polynomial(l, abs(m), cos_theta);
    if (m == 0) { return norm * plm; }
    if (m > 0)  { return SQRT_2 * norm * plm * cos(f32(m) * phi); }
    return SQRT_2 * norm * plm * sin(f32(-m) * phi);
}

fn evaluate_sh(offset: u32, dir: vec3<f32>, l_max: i32) -> f32 {
    let len_ = length(dir);
    if (len_ < 1e-8) { return 0.0; }
    let d = dir / len_;
    let cos_theta = clamp(d.z, -1.0, 1.0);
    let phi = atan2(d.y, d.x);
    var sum = 0.0;
    var idx = 0u;
    for (var l = 0; l <= l_max; l++) {
        for (var m = -l; m <= l; m++) {
            sum += s_sh_coeffs[offset + idx] * spherical_harmonic(l, m, cos_theta, phi);
            idx++;
        }
    }
    return sum;
}

// ── Cube-face direction ─────────────────────────────────────────

fn cube_direction(face: u32, u_: f32, v_: f32) -> vec3<f32> {
    var d: vec3<f32>;
    switch face {
        case 0u: { d = vec3<f32>( 1.0, -v_, -u_); }
        case 1u: { d = vec3<f32>(-1.0, -v_,  u_); }
        case 2u: { d = vec3<f32>( u_,  1.0,  v_); }
        case 3u: { d = vec3<f32>( u_, -1.0, -v_); }
        case 4u: { d = vec3<f32>( u_, -v_,  1.0); }
        default: { d = vec3<f32>(-u_, -v_, -1.0); }
    }
    return normalize(d);
}

// ── Pass 1: evaluate radius on the internal padded grid ─────────
//
// Internal grid per face: (N + 6)²
// Total items per glyph = 6 * s_int * s_int
// 2D dispatch: workgroup(x,y) → flat_wg = y * 65535 + x
//              flat_id = flat_wg * 256 + local_id

@compute @workgroup_size(256)
fn pass1_eval(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let N       = uniforms.lut_res;
    let g_int   = 3u;
    let s_int   = N + 2u * g_int;
    let l_max   = i32(uniforms.l_max);
    let n_coeffs = uniforms.n_coeffs;
    let n_glyphs = uniforms.n_glyphs;
    let items_per_glyph = uniforms.items_per_glyph_p1;

    let flat_wg = wg_id.y * 65535u + wg_id.x;
    let flat_id = flat_wg * 256u + lid.x;
    let glyph_id = flat_id / items_per_glyph;
    let local_id = flat_id % items_per_glyph;

    if (glyph_id >= n_glyphs) { return; }

    let face_stride = s_int * s_int;
    let face = local_id / face_stride;
    let within_face = local_id % face_stride;
    let row = within_face / s_int;
    let col = within_face % s_int;

    if (face >= 6u) { return; }

    let step = 2.0 / f32(N - 1u);
    let u_ = -1.0 + (f32(col) - f32(g_int)) * step;
    let v_ = -1.0 + (f32(row) - f32(g_int)) * step;

    let dir = cube_direction(face, u_, v_);
    let coeff_offset = glyph_id * n_coeffs;
    let radius = evaluate_sh(coeff_offset, dir, l_max);

    let scratch_glyph_stride = 6u * face_stride;
    let idx = glyph_id * scratch_glyph_stride
            + face * face_stride
            + row * s_int
            + col;
    s_scratch[idx] = radius;
}

// ── Pass 2: finite-difference hermite derivatives ───────────────
//
// Output grid per face: (N+2)²
// Total items per glyph = 6 * s_out * s_out

@compute @workgroup_size(256)
fn pass2_hermite(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let N       = uniforms.lut_res;
    let g_out   = 1u;
    let s_out   = N + 2u * g_out;
    let g_int   = 3u;
    let s_int   = N + 2u * g_int;
    let off     = g_int - g_out;              // = 2
    let n_glyphs = uniforms.n_glyphs;
    let items_per_glyph = uniforms.items_per_glyph_p2;

    let flat_wg = wg_id.y * 65535u + wg_id.x;
    let flat_id = flat_wg * 256u + lid.x;
    let glyph_id = flat_id / items_per_glyph;
    let local_id = flat_id % items_per_glyph;

    if (glyph_id >= n_glyphs) { return; }

    let out_face_stride = s_out * s_out;
    let face    = local_id / out_face_stride;
    let within_face = local_id % out_face_stride;
    let row_out = within_face / s_out;
    let col_out = within_face % s_out;

    if (face >= 6u) { return; }

    let ri = row_out + off;
    let ci = col_out + off;

    let scratch_face_stride = s_int * s_int;
    let scratch_glyph_stride = 6u * scratch_face_stride;
    let base = glyph_id * scratch_glyph_stride + face * scratch_face_stride;

    let val = s_scratch[base + ri * s_int + ci];

    // 4th-order central finite differences  (c1 = 8/12, c2 = -1/12)
    let c1: f32 = 0.6666666666666666;
    let c2: f32 = -0.0833333333333333;

    let du = c1 * (s_scratch[base + ri * s_int + ci + 1u]
                  - s_scratch[base + ri * s_int + ci - 1u])
           + c2 * (s_scratch[base + ri * s_int + ci + 2u]
                  - s_scratch[base + ri * s_int + ci - 2u]);

    let dv = c1 * (s_scratch[base + (ri+1u) * s_int + ci]
                  - s_scratch[base + (ri-1u) * s_int + ci])
           + c2 * (s_scratch[base + (ri+2u) * s_int + ci]
                  - s_scratch[base + (ri-2u) * s_int + ci]);

    // Cross derivative d²/dudv
    let du_rp1 = c1 * (s_scratch[base + (ri+1u)*s_int + ci+1u]
                      - s_scratch[base + (ri+1u)*s_int + ci-1u])
               + c2 * (s_scratch[base + (ri+1u)*s_int + ci+2u]
                      - s_scratch[base + (ri+1u)*s_int + ci-2u]);
    let du_rm1 = c1 * (s_scratch[base + (ri-1u)*s_int + ci+1u]
                      - s_scratch[base + (ri-1u)*s_int + ci-1u])
               + c2 * (s_scratch[base + (ri-1u)*s_int + ci+2u]
                      - s_scratch[base + (ri-1u)*s_int + ci-2u]);
    let du_rp2 = c1 * (s_scratch[base + (ri+2u)*s_int + ci+1u]
                      - s_scratch[base + (ri+2u)*s_int + ci-1u])
               + c2 * (s_scratch[base + (ri+2u)*s_int + ci+2u]
                      - s_scratch[base + (ri+2u)*s_int + ci-2u]);
    let du_rm2 = c1 * (s_scratch[base + (ri-2u)*s_int + ci+1u]
                      - s_scratch[base + (ri-2u)*s_int + ci-1u])
               + c2 * (s_scratch[base + (ri-2u)*s_int + ci+2u]
                      - s_scratch[base + (ri-2u)*s_int + ci-2u]);
    let d2uv = c1 * (du_rp1 - du_rm1) + c2 * (du_rp2 - du_rm2);

    let out_glyph_stride = 6u * out_face_stride;
    let out_idx = glyph_id * out_glyph_stride
                + face * out_face_stride
                + row_out * s_out
                + col_out;
    s_hermite_lut[out_idx] = vec4<f32>(val, du, dv, d2uv);
}
