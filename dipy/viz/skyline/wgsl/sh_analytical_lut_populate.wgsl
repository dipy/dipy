@group(0) @binding(0) var<storage, read> s_sh_coeffs: array<f32>;
@group(0) @binding(1) var<storage, read_write> s_radius_lut: array<f32>;

struct ComputeUniforms {
    n_glyphs: u32,
    n_coeffs: u32,
    phi_res: u32,
    theta_res: u32,
    l_max: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}
@group(0) @binding(2) var<uniform> uniforms: ComputeUniforms;

const PI: f32 = 3.14159265359;
const SQRT_2: f32 = 1.4142135623730951;

fn factorial(n: i32) -> f32 {
    if (n <= 0) { return 1.0; }
    if (n == 1) { return 1.0; }
    if (n == 2) { return 2.0; }
    if (n == 3) { return 6.0; }
    if (n == 4) { return 24.0; }
    if (n == 5) { return 120.0; }
    if (n == 6) { return 720.0; }
    if (n == 7) { return 5040.0; }
    if (n == 8) { return 40320.0; }
    return 1.0;
}

fn factorial_ratio(l: i32, m: i32) -> f32 {
    if (m == 0 || l == 0) { return 1.0; }

    var result = 1.0;
    let start = l - m + 1;
    let stop = l + m;

    if (stop < start) { return 1.0; }

    for (var k: i32 = start; k <= stop; k++) {
        result /= f32(k);
    }
    return result;
}

fn legendre_polynomial(l: i32, m: i32, x: f32) -> f32 {
    if (l == 0) { return 1.0; }

    let abs_m = abs(m);

    if (l == abs_m) {
        var pmm = 1.0;
        let somx2 = sqrt((1.0 - x) * (1.0 + x));
        var fact = 1.0;
        for (var i: i32 = 1; i <= abs_m; i++) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
        return pmm;
    }

    var pmm = 1.0;
    let somx2 = sqrt((1.0 - x) * (1.0 + x));
    var fact = 1.0;
    for (var i: i32 = 1; i <= abs_m; i++) {
        pmm *= -fact * somx2;
        fact += 2.0;
    }

    if (l == abs_m + 1) {
        return x * f32(2 * abs_m + 1) * pmm;
    }

    var pmm1 = x * f32(2 * abs_m + 1) * pmm;

    for (var ll: i32 = abs_m + 2; ll <= l; ll++) {
        let pll = (x * f32(2 * ll - 1) * pmm1 - f32(ll + abs_m - 1) * pmm) / f32(ll - abs_m);
        pmm = pmm1;
        pmm1 = pll;
    }

    return pmm1;
}

fn sh_normalization(l: i32, m: i32) -> f32 {
    let abs_m = abs(m);
    let factor = (2.0 * f32(l) + 1.0) / (4.0 * PI);
    let frac = factorial_ratio(l, abs_m);
    return sqrt(factor * frac);
}

fn spherical_harmonic(l: i32, m: i32, cos_theta: f32, phi: f32) -> f32 {
    let norm = sh_normalization(l, m);
    let plm = legendre_polynomial(l, abs(m), cos_theta);

    if (m == 0) {
        return norm * plm;
    } else if (m > 0) {
        return SQRT_2 * norm * plm * cos(f32(m) * phi);
    } else {
        return SQRT_2 * norm * plm * sin(f32(-m) * phi);
    }
}

fn evaluate_sh(coeff_offset: u32, cos_theta: f32, phi: f32, l_max: i32) -> f32 {
    var sum = 0.0;
    var idx = 0;

    for (var l: i32 = 0; l <= l_max; l++) {
        for (var m: i32 = -l; m <= l; m++) {
            let coeff = s_sh_coeffs[coeff_offset + u32(idx)];
            let y = spherical_harmonic(l, m, cos_theta, phi);
            sum += coeff * y;
            idx++;
        }
    }

    return sum;
}

@compute @workgroup_size(1, 8, 8)
fn compute_lut(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let glyph_id = global_id.x;
    let theta_idx = global_id.y;
    let phi_idx = global_id.z;

    let n_glyphs = uniforms.n_glyphs;
    let phi_res = uniforms.phi_res;
    let theta_res = uniforms.theta_res;
    let n_coeffs = uniforms.n_coeffs;
    let l_max = i32(uniforms.l_max);

    if (glyph_id >= n_glyphs || theta_idx >= theta_res || phi_idx >= phi_res) {
        return;
    }

    let theta = f32(theta_idx) * PI / f32(max(theta_res - 1u, 1u));
    let phi = (f32(phi_idx) / f32(phi_res)) * (2.0 * PI) - PI;
    let cos_theta = cos(theta);
    let coeffs_offset = glyph_id * n_coeffs;
    let radius = evaluate_sh(coeffs_offset, cos_theta, phi, l_max);
    let lut_idx = glyph_id * (phi_res * theta_res) + theta_idx * phi_res + phi_idx;
    s_radius_lut[lut_idx] = radius;
}
