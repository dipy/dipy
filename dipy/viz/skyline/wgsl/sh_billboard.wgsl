{$ if use_float16 == 'true' $}
enable f16;
{$ endif $}

{$ include 'pygfx.std.wgsl' $}
{$ include 'pygfx.light_phong.wgsl' $}
{$ include 'fury.utils.wgsl' $}

const NUM_COEFFS = i32({{ n_coeffs }});
const L_MAX = i32({{ l_max }});
const COLOR_TYPE = i32({{ color_type }});
const SH_STRIDE: u32 = u32(L_MAX + 1);
const SH_TABLE_SIZE: u32 = SH_STRIDE * u32(L_MAX + 1);
const SH_TRIG_SIZE: u32 = u32(L_MAX + 1);

const LUT_PHI_RES = u32({{ radius_lut_phi }});
const LUT_STRIDE = u32({{ radius_lut_stride }});

const LUT_N_CHUNKS = u32({{ lut_n_chunks }});
const LUT_GLYPHS_PER_CHUNK = u32({{ lut_glyphs_per_chunk }});

const USE_HERMITE_LUT = bool({{ use_hermite_lut }});

fn get_lut_chunk_info(glyph_id: u32) -> vec2<u32> {
    if (LUT_N_CHUNKS <= 1u || LUT_GLYPHS_PER_CHUNK == 0u) {
        return vec2<u32>(0u, glyph_id);
    }
    let chunk_idx = glyph_id / LUT_GLYPHS_PER_CHUNK;
    let local_glyph = glyph_id % LUT_GLYPHS_PER_CHUNK;
    return vec2<u32>(chunk_idx, local_glyph);
}

fn read_hermite_value(glyph_id: u32, local_offset: u32) -> vec4<f32> {
    let chunk_info = get_lut_chunk_info(glyph_id);
    let chunk_idx = chunk_info.x;
    let local_glyph = chunk_info.y;
    let actual_offset = local_glyph * LUT_STRIDE + local_offset;

    {$ if use_float16 == 'true' $}
        if (LUT_N_CHUNKS <= 1u) {
            return vec4<f32>(s_sh_hermite_lut_0[actual_offset]);
        }

        if (chunk_idx == 0u) { return vec4<f32>(s_sh_hermite_lut_0[actual_offset]); }
        else if (chunk_idx == 1u) { return vec4<f32>(s_sh_hermite_lut_1[actual_offset]); }
        else if (chunk_idx == 2u) { return vec4<f32>(s_sh_hermite_lut_2[actual_offset]); }
        else if (chunk_idx == 3u) { return vec4<f32>(s_sh_hermite_lut_3[actual_offset]); }
        else if (chunk_idx == 4u) { return vec4<f32>(s_sh_hermite_lut_4[actual_offset]); }
        else if (chunk_idx == 5u) { return vec4<f32>(s_sh_hermite_lut_5[actual_offset]); }
        else if (chunk_idx == 6u) { return vec4<f32>(s_sh_hermite_lut_6[actual_offset]); }
        else { return vec4<f32>(s_sh_hermite_lut_7[actual_offset]); }
    {$ else $}
        if (LUT_N_CHUNKS <= 1u) {
            return vec4<f32>(s_sh_hermite_lut_0[actual_offset]);
        }

        if (chunk_idx == 0u) { return vec4<f32>(s_sh_hermite_lut_0[actual_offset]); }
        else if (chunk_idx == 1u) { return vec4<f32>(s_sh_hermite_lut_1[actual_offset]); }
        else if (chunk_idx == 2u) { return vec4<f32>(s_sh_hermite_lut_2[actual_offset]); }
        else if (chunk_idx == 3u) { return vec4<f32>(s_sh_hermite_lut_3[actual_offset]); }
        else if (chunk_idx == 4u) { return vec4<f32>(s_sh_hermite_lut_4[actual_offset]); }
        else if (chunk_idx == 5u) { return vec4<f32>(s_sh_hermite_lut_5[actual_offset]); }
        else if (chunk_idx == 6u) { return vec4<f32>(s_sh_hermite_lut_6[actual_offset]); }
        else { return vec4<f32>(s_sh_hermite_lut_7[actual_offset]); }
    {$ endif $}
}

struct VertexInput {
    @builtin(vertex_index) index : u32,
};

fn factorial_ratio(l: i32, m: i32) -> f32 {
    if (m == 0 || l == 0) {
        return 1.0;
    }
    var result = 1.0;
    let start = l - m + 1;
    let stop = l + m;
    if (stop < start) {
        return 1.0;
    }
    for (var k: i32 = start; k <= stop; k++) {
        result /= f32(k);
    }
    return result;
}

fn legendre_index(l: i32, m: i32) -> u32 {
    return u32(l) * SH_STRIDE + u32(m);
}

fn get_active_coeff_count() -> i32 {
    if (NUM_COEFFS == 0) {
        return 0;
    }
    if (u_material.n_coeffs != -1) {
        if (u_material.n_coeffs < NUM_COEFFS) {
            return u_material.n_coeffs;
        }
        return NUM_COEFFS;
    }
    return NUM_COEFFS;
}

fn clamp_radius(value: f32) -> f32 {
    return abs(value) * u_material.scale;
}

fn hermite_basis(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    let two_t3_minus_three_t2 = 2.0 * t3 - 3.0 * t2;
    return vec4<f32>(
        two_t3_minus_three_t2 + 1.0,
        t3 - 2.0 * t2 + t,
        -two_t3_minus_three_t2,
        t2 * (t - 1.0)
    );
}

fn hermite_basis_deriv(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let six_t2_minus_six_t = 6.0 * t2 - 6.0 * t;
    return vec4<f32>(
        six_t2_minus_six_t,
        3.0 * t2 - 4.0 * t + 1.0,
        -six_t2_minus_six_t,
        3.0 * t2 - 2.0 * t
    );
}

fn sample_hermite_cube(glyph_id: u32, direction: vec3<f32>) -> f32 {
    if (LUT_PHI_RES == 0u || LUT_STRIDE == 0u) {
        return 0.0;
    }

    let abs_dir = abs(direction);
    var face_idx = 0u;
    var sc = 0.0;
    var tc = 0.0;
    var ma = 0.0;

    if (abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) {
        if (direction.x > 0.0) { face_idx = 0u; sc = -direction.z; tc = -direction.y; ma = abs_dir.x; }
        else { face_idx = 1u; sc = direction.z; tc = -direction.y; ma = abs_dir.x; }
    } else if (abs_dir.y >= abs_dir.z) {
        if (direction.y > 0.0) { face_idx = 2u; sc = direction.x; tc = direction.z; ma = abs_dir.y; }
        else { face_idx = 3u; sc = direction.x; tc = -direction.z; ma = abs_dir.y; }
    } else {
        if (direction.z > 0.0) { face_idx = 4u; sc = direction.x; tc = -direction.y; ma = abs_dir.z; }
        else { face_idx = 5u; sc = -direction.x; tc = -direction.y; ma = abs_dir.z; }
    }

    let u_norm = (sc / ma + 1.0) * 0.5;
    let v_norm = (tc / ma + 1.0) * 0.5;

    let lut_res = LUT_PHI_RES;

    let s = 1.0 + u_norm * (f32(lut_res) - 3.0);
    let t = 1.0 + v_norm * (f32(lut_res) - 3.0);

    let i0 = i32(floor(s));
    let j0 = i32(floor(t));

    let alpha = fract(s);
    let beta = fract(t);

    let face_offset = face_idx * lut_res * lut_res;

    let p00 = read_hermite_value(glyph_id, face_offset + u32(j0) * lut_res + u32(i0));
    let p10 = read_hermite_value(glyph_id, face_offset + u32(j0) * lut_res + u32(i0 + 1));
    let p01 = read_hermite_value(glyph_id, face_offset + u32(j0 + 1) * lut_res + u32(i0));
    let p11 = read_hermite_value(glyph_id, face_offset + u32(j0 + 1) * lut_res + u32(i0 + 1));

    let Hu = hermite_basis(alpha);
    let Hv = hermite_basis(beta);

    let Huv_00 = vec4<f32>(Hu.x * Hv.x, Hu.y * Hv.x, Hu.x * Hv.y, Hu.y * Hv.y);
    let Huv_10 = vec4<f32>(Hu.z * Hv.x, Hu.w * Hv.x, Hu.z * Hv.y, Hu.w * Hv.y);
    let Huv_01 = vec4<f32>(Hu.x * Hv.z, Hu.y * Hv.z, Hu.x * Hv.w, Hu.y * Hv.w);
    let Huv_11 = vec4<f32>(Hu.z * Hv.z, Hu.w * Hv.z, Hu.z * Hv.w, Hu.w * Hv.w);

    return dot(p00, Huv_00) + dot(p10, Huv_10) + dot(p01, Huv_01) + dot(p11, Huv_11);
}

struct HermiteCubeResult {
    r: f32,
    grad_s2: vec3<f32>,
}

fn sample_hermite_cube_with_gradient(glyph_id: u32, direction: vec3<f32>) -> HermiteCubeResult {
    var result: HermiteCubeResult;
    result.r = 0.0;
    result.grad_s2 = vec3<f32>(0.0);

    if (LUT_PHI_RES == 0u || LUT_STRIDE == 0u) {
        return result;
    }

    let abs_dir = abs(direction);
    var face_idx = 0u;
    var sc = 0.0;
    var tc = 0.0;
    var ma = 0.0;

    if (abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) {
        if (direction.x > 0.0) { face_idx = 0u; sc = -direction.z; tc = -direction.y; ma = abs_dir.x; }
        else { face_idx = 1u; sc = direction.z; tc = -direction.y; ma = abs_dir.x; }
    } else if (abs_dir.y >= abs_dir.z) {
        if (direction.y > 0.0) { face_idx = 2u; sc = direction.x; tc = direction.z; ma = abs_dir.y; }
        else { face_idx = 3u; sc = direction.x; tc = -direction.z; ma = abs_dir.y; }
    } else {
        if (direction.z > 0.0) { face_idx = 4u; sc = direction.x; tc = -direction.y; ma = abs_dir.z; }
        else { face_idx = 5u; sc = -direction.x; tc = -direction.y; ma = abs_dir.z; }
    }

    let u_norm = (sc / ma + 1.0) * 0.5;
    let v_norm = (tc / ma + 1.0) * 0.5;

    let lut_res = LUT_PHI_RES;
    let s = 1.0 + u_norm * (f32(lut_res) - 3.0);
    let t = 1.0 + v_norm * (f32(lut_res) - 3.0);

    let i0 = i32(floor(s));
    let j0 = i32(floor(t));

    let alpha = fract(s);
    let beta = fract(t);

    let face_offset = face_idx * lut_res * lut_res;

    let p00 = read_hermite_value(glyph_id, face_offset + u32(j0) * lut_res + u32(i0));
    let p10 = read_hermite_value(glyph_id, face_offset + u32(j0) * lut_res + u32(i0 + 1));
    let p01 = read_hermite_value(glyph_id, face_offset + u32(j0 + 1) * lut_res + u32(i0));
    let p11 = read_hermite_value(glyph_id, face_offset + u32(j0 + 1) * lut_res + u32(i0 + 1));

    let Hu = hermite_basis(alpha);
    let Hv = hermite_basis(beta);
    let dHu = hermite_basis_deriv(alpha);
    let dHv = hermite_basis_deriv(beta);

    let Huv_00 = vec4<f32>(Hu.x * Hv.x, Hu.y * Hv.x, Hu.x * Hv.y, Hu.y * Hv.y);
    let Huv_10 = vec4<f32>(Hu.z * Hv.x, Hu.w * Hv.x, Hu.z * Hv.y, Hu.w * Hv.y);
    let Huv_01 = vec4<f32>(Hu.x * Hv.z, Hu.y * Hv.z, Hu.x * Hv.w, Hu.y * Hv.w);
    let Huv_11 = vec4<f32>(Hu.z * Hv.z, Hu.w * Hv.z, Hu.z * Hv.w, Hu.w * Hv.w);

    result.r = dot(p00, Huv_00) + dot(p10, Huv_10) + dot(p01, Huv_01) + dot(p11, Huv_11);

    let dHuv_00 = vec4<f32>(dHu.x * Hv.x, dHu.y * Hv.x, dHu.x * Hv.y, dHu.y * Hv.y);
    let dHuv_10 = vec4<f32>(dHu.z * Hv.x, dHu.w * Hv.x, dHu.z * Hv.y, dHu.w * Hv.y);
    let dHuv_01 = vec4<f32>(dHu.x * Hv.z, dHu.y * Hv.z, dHu.x * Hv.w, dHu.y * Hv.w);
    let dHuv_11 = vec4<f32>(dHu.z * Hv.z, dHu.w * Hv.z, dHu.z * Hv.w, dHu.w * Hv.w);

    let dr_dalpha = dot(p00, dHuv_00) + dot(p10, dHuv_10) + dot(p01, dHuv_01) + dot(p11, dHuv_11);

    let Hudv_00 = vec4<f32>(Hu.x * dHv.x, Hu.y * dHv.x, Hu.x * dHv.y, Hu.y * dHv.y);
    let Hudv_10 = vec4<f32>(Hu.z * dHv.x, Hu.w * dHv.x, Hu.z * dHv.y, Hu.w * dHv.y);
    let Hudv_01 = vec4<f32>(Hu.x * dHv.z, Hu.y * dHv.z, Hu.x * dHv.w, Hu.y * dHv.w);
    let Hudv_11 = vec4<f32>(Hu.z * dHv.z, Hu.w * dHv.z, Hu.z * dHv.w, Hu.w * dHv.w);

    let dr_dbeta = dot(p00, Hudv_00) + dot(p10, Hudv_10) + dot(p01, Hudv_01) + dot(p11, Hudv_11);

    let scale = (f32(lut_res) - 3.0) * 0.5;
    let dr_dsc_ma = dr_dalpha * scale;
    let dr_dtc_ma = dr_dbeta * scale;

    let sc_val = sc / ma;
    let tc_val = tc / ma;

    let P_len_sq = 1.0 + sc_val * sc_val + tc_val * tc_val;
    let P_len = sqrt(P_len_sq);

    let dP_dsc_table = array<vec3<f32>, 6>(
        vec3<f32>(0.0, 0.0, -1.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(-1.0, 0.0, 0.0),
    );
    let dP_dtc_table = array<vec3<f32>, 6>(
        vec3<f32>(0.0, -1.0, 0.0),
        vec3<f32>(0.0, -1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 0.0, -1.0),
        vec3<f32>(0.0, -1.0, 0.0),
        vec3<f32>(0.0, -1.0, 0.0),
    );
    let dP_dsc = dP_dsc_table[face_idx];
    let dP_dtc = dP_dtc_table[face_idx];

    let omega = direction;
    let inv_P_len = 1.0 / P_len;
    let sc_over_P = sc_val * inv_P_len;
    let tc_over_P = tc_val * inv_P_len;
    let dw_dsc = (dP_dsc - omega * sc_over_P) * inv_P_len;
    let dw_dtc = (dP_dtc - omega * tc_over_P) * inv_P_len;

    let a = dot(dw_dsc, dw_dsc);
    let b = dot(dw_dsc, dw_dtc);
    let c = dot(dw_dtc, dw_dtc);
    let inv_det = 1.0 / (a * c - b * b);

    let coef_sc = (c * dr_dsc_ma - b * dr_dtc_ma) * inv_det;
    let coef_tc = (a * dr_dtc_ma - b * dr_dsc_ma) * inv_det;
    result.grad_s2 = coef_sc * dw_dsc + coef_tc * dw_dtc;

    return result;
}

fn get_radius(glyph_id: u32, coeff_offset: i32, direction: vec3<f32>, coeff_limit: i32) -> f32 {
    if (USE_HERMITE_LUT) {
        return sample_hermite_cube(glyph_id, direction);
    }
    return evaluate_radius_direct(coeff_offset, direction, coeff_limit);
}

fn evaluate_radius_direct(coeff_offset: i32, direction: vec3<f32>, coeff_limit: i32) -> f32 {
    if (coeff_limit <= 0) {
        return 0.0;
    }
    if (L_MAX > 4) {
        return evaluate_radius(coeff_offset, direction, coeff_limit);
    }

    let x = direction.x;
    let y = direction.y;
    let z = direction.z;

    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;

    var sum = 0.0;
    var idx = coeff_offset;
    var used = 0;

    if (used < coeff_limit) {
        let Y_00 = 0.28209479177387814;
        sum += s_coeffs[idx] * Y_00;
        idx += 1;
        used += 1;
    }

    if (L_MAX >= 1 && used < coeff_limit) {
        let c_1m1 = 0.4886025119029199;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_1m1 * y;
            idx += 1;
            used += 1;
        }
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_1m1 * z;
            idx += 1;
            used += 1;
        }
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_1m1 * x;
            idx += 1;
            used += 1;
        }
    }

    if (L_MAX >= 2 && used < coeff_limit) {
        let c_2m2 = 1.0925484305920792;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_2m2 * xy;
            idx += 1;
            used += 1;
        }
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_2m2 * yz;
            idx += 1;
            used += 1;
        }
        let c_20 = 0.31539156525252005;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_20 * (3.0 * z2 - 1.0);
            idx += 1;
            used += 1;
        }
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_2m2 * xz;
            idx += 1;
            used += 1;
        }
        let c_22 = 0.5462742152960396;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_22 * (x2 - y2);
            idx += 1;
            used += 1;
        }
    }

    if (L_MAX >= 3 && used < coeff_limit) {
        let c_3m3 = 0.5900435899266435;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_3m3 * y * (3.0 * x2 - y2);
            idx += 1;
            used += 1;
        }
        let c_3m2 = 2.890611442640554;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_3m2 * xy * z;
            idx += 1;
            used += 1;
        }
        let c_3m1 = 0.4570457994644658;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_3m1 * y * (5.0 * z2 - 1.0);
            idx += 1;
            used += 1;
        }
        let c_30 = 0.3731763325901154;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_30 * z * (5.0 * z2 - 3.0);
            idx += 1;
            used += 1;
        }
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_3m1 * x * (5.0 * z2 - 1.0);
            idx += 1;
            used += 1;
        }
        let c_32 = 1.4453057213202769;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_32 * z * (x2 - y2);
            idx += 1;
            used += 1;
        }
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_3m3 * x * (x2 - 3.0 * y2);
            idx += 1;
            used += 1;
        }
    }

    if (L_MAX >= 4 && used < coeff_limit) {
        let x3 = x * x2;
        let y3 = y * y2;
        let z3 = z * z2;
        let z4 = z2 * z2;

        let c_4m4 = 2.5033429417967046;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_4m4 * xy * (x2 - y2);
            idx += 1;
            used += 1;
        }
        let c_4m3 = 1.7701307697799304;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_4m3 * yz * (3.0 * x2 - y2);
            idx += 1;
            used += 1;
        }
        let c_4m2 = 0.9461746957575601;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_4m2 * xy * (7.0 * z2 - 1.0);
            idx += 1;
            used += 1;
        }
        let c_4m1 = 0.6690465435572892;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_4m1 * yz * (7.0 * z2 - 3.0);
            idx += 1;
            used += 1;
        }
        let c_40 = 0.10578554691520431;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_40 * (35.0 * z4 - 30.0 * z2 + 3.0);
            idx += 1;
            used += 1;
        }
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_4m1 * xz * (7.0 * z2 - 3.0);
            idx += 1;
            used += 1;
        }
        let c_42 = 0.47308734787878004;
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_42 * (x2 - y2) * (7.0 * z2 - 1.0);
            idx += 1;
            used += 1;
        }
        if (used < coeff_limit) {
            sum += s_coeffs[idx] * c_4m3 * xz * (x2 - 3.0 * y2);
            idx += 1;
            used += 1;
        }
        let c_44 = 0.6258357354491761;
        if (used < coeff_limit) {
            let x4 = x2 * x2;
            let y4 = y2 * y2;
            sum += s_coeffs[idx] * c_44 * (x4 - 6.0 * x2 * y2 + y4);
            idx += 1;
            used += 1;
        }
    }

    return sum;
}

fn evaluate_radius(coeff_offset: i32, direction: vec3<f32>, coeff_limit: i32) -> f32 {
    if (coeff_limit <= 0) {
        return 0.0;
    }

    var idx = coeff_offset;
    var used = 0;
    var sum = 0.0;

    let phi = atan2(direction.y, direction.x);
    let cos_theta = clamp(direction.z, -1.0, 1.0);

    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    var cos_table: array<f32, SH_TRIG_SIZE>;
    var sin_table: array<f32, SH_TRIG_SIZE>;
    cos_table[0u] = 1.0;
    sin_table[0u] = 0.0;
    if (L_MAX >= 1) {
        let cos_phi = cos(phi);
        let sin_phi = sin(phi);
        cos_table[1u] = cos_phi;
        sin_table[1u] = sin_phi;
        for (var m: i32 = 2; m <= L_MAX; m = m + 1) {
            let prev_cos = cos_table[u32(m - 1)];
            let prev_sin = sin_table[u32(m - 1)];
            cos_table[u32(m)] = prev_cos * cos_phi - prev_sin * sin_phi;
            sin_table[u32(m)] = prev_sin * cos_phi + prev_cos * sin_phi;
        }
    }

    var norm_p_table: array<f32, SH_TABLE_SIZE>;

    for (var m: i32 = 0; m <= L_MAX; m = m + 1) {
        var pmm = 1.0;
        if (m > 0) {
            var fact = 1.0;
            let somx2 = sin_theta;
            for (var i: i32 = 1; i <= m; i = i + 1) {
                pmm *= -fact * somx2;
                fact += 2.0;
            }
        }

        var ratio = factorial_ratio(m, m);
        let idx_mm = legendre_index(m, m);
        let norm_mm = sqrt(((2.0 * f32(m) + 1.0) / (4.0 * PI)) * ratio);
        norm_p_table[idx_mm] = pmm * norm_mm;

        var prev_prev = pmm;
        var prev = pmm;
        if (m < L_MAX) {
            var pmmp1 = cos_theta * (2.0 * f32(m) + 1.0) * pmm;
            let l1 = m + 1;
            ratio = ratio * f32(l1 - m) / f32(l1 + m);
            let norm1 = sqrt(((2.0 * f32(l1) + 1.0) / (4.0 * PI)) * ratio);
            norm_p_table[legendre_index(l1, m)] = pmmp1 * norm1;
            prev_prev = pmm;
            prev = pmmp1;

            for (var l: i32 = m + 2; l <= L_MAX; l = l + 1) {
                let ll_f = f32(l);
                let value = ((2.0 * ll_f - 1.0) * cos_theta * prev -
                    (ll_f + f32(m) - 1.0) * prev_prev) /
                    (ll_f - f32(m));
                prev_prev = prev;
                prev = value;
                ratio = ratio * f32(l - m) / f32(l + m);
                let norm_l = sqrt(((2.0 * ll_f + 1.0) / (4.0 * PI)) * ratio);
                norm_p_table[legendre_index(l, m)] = value * norm_l;
            }
        }
    }

    for (var l: i32 = 0; l <= L_MAX; l = l + 1) {
        for (var m: i32 = -l; m <= l; m = m + 1) {
            if (used >= coeff_limit) {
                return sum;
            }

            let abs_m = abs(m);
            let base = norm_p_table[legendre_index(l, abs_m)];
            var sh = base;
            if (abs_m > 0) {
                let cos_m = cos_table[u32(abs_m)];
                let sin_m = sin_table[u32(abs_m)];
                if (m > 0) {
                    sh = base * SQRT_2 * cos_m;
                } else {
                    sh = base * SQRT_2 * sin_m;
                }
            }

            sum += s_coeffs[idx] * sh;
            idx += 1;
            used += 1;
        }
    }

    return sum;
}

const BISECTION_STEPS: i32 = 10;
const NEWTON_STEPS: i32 = 6;
const RAY_BRACKET_STEPS_MAX: i32 = 96;
const RAY_BRACKET_STEPS_MIN: i32 = 32;

const LOD_HIGH_THRESHOLD: f32 = 64.0;
const LOD_LOW_THRESHOLD: f32 = 16.0;

struct IntersectionResult {
    hit: bool,
    position: vec3<f32>,
    direction: vec3<f32>,
    raw_radius: f32,
    grad_s2: vec3<f32>,
};

fn surface_difference(
    coeff_offset: i32,
    center: vec3<f32>,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    t: f32,
    coeff_limit: i32,
    glyph_id: u32,
) -> f32 {
    let position = ray_origin + ray_dir * t;
    let offset = position - center;
    let dist = length(offset);
    if (dist < 1e-5) {
        let fallback_local = normalize((u_wobject.world_transform_inv * vec4<f32>(normalize(ray_dir), 0.0)).xyz);
        let raw_radius = get_radius(glyph_id, coeff_offset, fallback_local, coeff_limit);
        return -clamp_radius(raw_radius);
    }
    let direction = offset / dist;
    let local_dir = normalize((u_wobject.world_transform_inv * vec4<f32>(direction, 0.0)).xyz);
    let raw_radius = get_radius(glyph_id, coeff_offset, local_dir, coeff_limit);
    let radius = clamp_radius(raw_radius);
    return dist - radius;
}

struct SurfaceEvalResult {
    F: f32,
    F_prime: f32,
    r: f32,
    omega: vec3<f32>,
    rho: f32,
    grad_s2: vec3<f32>,
}

fn evaluate_surface_analytic(
    coeff_offset: i32,
    center: vec3<f32>,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    t: f32,
    coeff_limit: i32,
    glyph_id: u32,
) -> SurfaceEvalResult {
    var result: SurfaceEvalResult;

    let position = ray_origin + ray_dir * t;
    let offset = position - center;
    let rho = length(offset);

    if (rho < 1e-5) {
        let omega = normalize(ray_dir);
        result.omega = omega;
        result.rho = rho;

        let local_omega = normalize((u_wobject.world_transform_inv * vec4<f32>(omega, 0.0)).xyz);
        if (USE_HERMITE_LUT) {
            result.r = sample_hermite_cube(glyph_id, local_omega);
        } else {
            result.r = get_radius(glyph_id, coeff_offset, local_omega, coeff_limit);
        }
        result.grad_s2 = vec3<f32>(0.0);

        let radius = clamp_radius(result.r);
        result.F = -radius;
        result.F_prime = 1.0;
        return result;
    }

    let omega = offset / rho;
    result.omega = omega;
    result.rho = rho;

    let local_omega = normalize((u_wobject.world_transform_inv * vec4<f32>(omega, 0.0)).xyz);
    if (USE_HERMITE_LUT) {
        result.r = sample_hermite_cube(glyph_id, local_omega);
    } else {
        result.r = get_radius(glyph_id, coeff_offset, local_omega, coeff_limit);
    }
    result.grad_s2 = vec3<f32>(0.0);

    let radius = clamp_radius(result.r);
    result.F = rho - radius;

    let omega_dot_d = dot(omega, ray_dir);
    result.F_prime = omega_dot_d;

    return result;
}

fn evaluate_implicit(
    coeff_offset: i32,
    center: vec3<f32>,
    point: vec3<f32>,
    coeff_limit: i32,
    glyph_id: u32,
) -> f32 {
    let offset = point - center;
    let dist = length(offset);
    if (dist < 1e-5) {
        let fallback_dir = normalize(vec3<f32>(0.577, 0.577, 0.577));
        let local_fallback = normalize((u_wobject.world_transform_inv * vec4<f32>(fallback_dir, 0.0)).xyz);
        let raw_radius = get_radius(glyph_id, coeff_offset, local_fallback, coeff_limit);
        let radius = clamp_radius(raw_radius);
        return dist - radius;
    }
    let direction = offset / dist;
    let local_dir = normalize((u_wobject.world_transform_inv * vec4<f32>(direction, 0.0)).xyz);
    let raw_radius = get_radius(glyph_id, coeff_offset, local_dir, coeff_limit);
    let radius = clamp_radius(raw_radius);
    return dist - radius;
}

fn estimate_surface_normal_analytic(
    omega: vec3<f32>,
    _rho: f32,
    grad_s2: vec3<f32>,
    raw_radius: f32,
) -> vec3<f32> {

    let abs_r = abs(raw_radius);
    if (abs_r < 1e-5) {
        return omega;
    }

    let sign_r = select(-1.0, 1.0, raw_radius >= 0.0);
    let gradient = omega - sign_r * grad_s2 / abs_r;
    let len = length(gradient);

    if (len < 1e-5) {
        return omega;
    }

    return gradient / len;
}

fn estimate_surface_normal(
    coeff_offset: i32,
    center: vec3<f32>,
    point: vec3<f32>,
    coeff_limit: i32,
    glyph_id: u32,
) -> vec3<f32> {
    let scale = max(length(point - center) * 1e-3, 1e-3);
    let offsets = array<vec3<f32>, 3>(
        vec3<f32>(scale, 0.0, 0.0),
        vec3<f32>(0.0, scale, 0.0),
        vec3<f32>(0.0, 0.0, scale),
    );

    var gradient = vec3<f32>(0.0);
    for (var axis: i32 = 0; axis < 3; axis = axis + 1) {
        let o = offsets[axis];
        let pos_plus = evaluate_implicit(coeff_offset, center, point + o, coeff_limit, glyph_id);
        let pos_minus = evaluate_implicit(coeff_offset, center, point - o, coeff_limit, glyph_id);
        gradient[axis] = (pos_plus - pos_minus) / (2.0 * scale);
    }

    if (length(gradient) < 1e-5) {
        return normalize(point - center);
    }
    return normalize(gradient);
}

fn find_surface_intersection(
    coeff_offset: i32,
    center: vec3<f32>,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    bound_radius: f32,
    coeff_limit: i32,
    glyph_id: u32,
    screen_size: f32,
) -> IntersectionResult {
    var result: IntersectionResult;
    result.hit = false;
    result.position = vec3<f32>(0.0);
    result.direction = vec3<f32>(0.0);
    result.raw_radius = 0.0;
    result.grad_s2 = vec3<f32>(0.0);

    if (bound_radius <= 1e-5) {
        return result;
    }

    let lod_factor = clamp(
        (screen_size - LOD_LOW_THRESHOLD) / (LOD_HIGH_THRESHOLD - LOD_LOW_THRESHOLD),
        0.0, 1.0
    );
    let ray_bracket_steps = i32(mix(
        f32(RAY_BRACKET_STEPS_MIN),
        f32(RAY_BRACKET_STEPS_MAX),
        lod_factor
    ));

    let oc = ray_origin - center;
    let a = dot(ray_dir, ray_dir);
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - bound_radius * bound_radius;
    var discriminant = b * b - a * c;
    if (discriminant < 0.0) {
        if (discriminant > -1e-4) {
            discriminant = 0.0;
        } else {
            return result;
        }
    }
    let sqrt_disc = sqrt(max(discriminant, 0.0));
    let inv_a = 1.0 / a;
    var t_near = (-b - sqrt_disc) * inv_a;
    var t_far = (-b + sqrt_disc) * inv_a;
    if (t_far < 0.0) {
        return result;
    }
    if (t_near < 0.0) {
        t_near = 0.0;
    }

    var start = t_near;
    var end = t_far;
    if (end <= start) {
        end = start + bound_radius;
    }

    var prev_t = start;
    var prev_val = surface_difference(coeff_offset, center, ray_origin, ray_dir, start, coeff_limit, glyph_id);

    var bracket_low = start;
    var bracket_high = end;
    var found = false;

    for (var step: i32 = 1; step <= ray_bracket_steps; step = step + 1) {
        let t = start + (end - start) * f32(step) / f32(ray_bracket_steps);
        let val = surface_difference(coeff_offset, center, ray_origin, ray_dir, t, coeff_limit, glyph_id);

        if (prev_val > 0.0 && val <= 0.0) {
            bracket_low = prev_t;
            bracket_high = t;
            found = true;
            break;
        }

        prev_t = t;
        prev_val = val;
    }

    if (!found) {
        return result;
    }

    for (var i: i32 = 0; i < BISECTION_STEPS; i = i + 1) {
        let mid = 0.5 * (bracket_low + bracket_high);
        let val = surface_difference(coeff_offset, center, ray_origin, ray_dir, mid, coeff_limit, glyph_id);
        if (val <= 0.0) {
            bracket_high = mid;
        } else {
            bracket_low = mid;
        }
    }

    var t = 0.5 * (bracket_low + bracket_high);
    for (var i: i32 = 0; i < NEWTON_STEPS; i = i + 1) {
        let diff = surface_difference(coeff_offset, center, ray_origin, ray_dir, t, coeff_limit, glyph_id);
        let epsilon = max(1e-4 * (bracket_high - bracket_low + 1.0), 1e-4);
        let diff_next = surface_difference(
            coeff_offset,
            center,
            ray_origin,
            ray_dir,
            t + epsilon,
            coeff_limit,
            glyph_id,
        );
        let derivative = (diff_next - diff) / epsilon;

        if (abs(derivative) < 1e-5) {
            break;
        }

        let new_t = clamp(t - diff / derivative, bracket_low, bracket_high);

        if (abs(new_t - t) < 1e-4 * max(bracket_high - bracket_low, 1.0)) {
            t = new_t;
            break;
        }

        t = new_t;
    }

    let eval_final = evaluate_surface_analytic(coeff_offset, center, ray_origin, ray_dir, t, coeff_limit, glyph_id);

    result.hit = true;
    result.position = ray_origin + ray_dir * t;
    result.direction = eval_final.omega;
    result.raw_radius = eval_final.r;
    result.grad_s2 = eval_final.grad_s2;
    return result;
}

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    let billboard_index = i32(in.index) / 6;
    let vertex_in_quad = i32(in.index) % 6;

    var raw_center = load_s_positions(billboard_index * 6);

    // --- Slice-based visibility: discard glyphs not on active slice ---
    {$ if use_slicing == 'true' $}
    {
        let slice_pos = vec3<f32>(
            u_material.active_slice_x,
            u_material.active_slice_y,
            u_material.active_slice_z,
        );
        let visibility = vec3<i32>(u_material.vis_x, u_material.vis_y, u_material.vis_z);
        let w_slice_center = (u_wobject.world_transform * vec4<f32>(raw_center, 1.0)).xyz;
        var is_visible = false;

        if (!all(visibility == vec3<i32>(-1))) {
            if (abs(w_slice_center.x - slice_pos.x) <= abs(u_wobject.world_transform[0][0]) * 0.5 && visibility.x != 0) {
                is_visible = true;
            }

            if (abs(w_slice_center.y - slice_pos.y) <= abs(u_wobject.world_transform[1][1]) * 0.5 && visibility.y != 0) {
                is_visible = true;
            }

            if (abs(w_slice_center.z - slice_pos.z) <= abs(u_wobject.world_transform[2][2]) * 0.5 && visibility.z != 0) {
                is_visible = true;
            }
        } else {
            is_visible = true;
        }

        if (!is_visible) {
            var discard_out: Varyings;
            discard_out.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
            return discard_out;
        }
    }
    {$ endif $}

    var w_center = u_wobject.world_transform * vec4<f32>(raw_center.xyz, 1.0);

    var local_pos: vec2<f32>;
    switch vertex_in_quad {
        case 0: { local_pos = vec2<f32>(-0.5, -0.5); }
        case 1: { local_pos = vec2<f32>(0.5, -0.5); }
        case 2: { local_pos = vec2<f32>(-0.5, 0.5); }
        case 3: { local_pos = vec2<f32>(0.5, -0.5); }
        case 4: { local_pos = vec2<f32>(0.5, 0.5); }
        default: { local_pos = vec2<f32>(-0.5, 0.5); }
    }

    let cam_right = vec3<f32>(u_stdinfo.cam_transform_inv[0].xyz);
    let cam_up = vec3<f32>(u_stdinfo.cam_transform_inv[1].xyz);
    let raw_size = load_s_normals(billboard_index * 6);
    let size = raw_size.xy;

    let billboard_offset = local_pos.x * cam_right * size.x + local_pos.y * cam_up * size.y;
    let world_pos = w_center.xyz + billboard_offset;

    let clip_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(world_pos, 1.0);

    var tex_coord: vec2<f32>;
    switch vertex_in_quad {
        case 0: { tex_coord = vec2<f32>(0.0, 0.0); }
        case 1: { tex_coord = vec2<f32>(1.0, 0.0); }
        case 2: { tex_coord = vec2<f32>(0.0, 1.0); }
        case 3: { tex_coord = vec2<f32>(1.0, 0.0); }
        case 4: { tex_coord = vec2<f32>(1.0, 1.0); }
        default: { tex_coord = vec2<f32>(0.0, 1.0); }
    }

    var varyings: Varyings;
    varyings.position = vec4<f32>(clip_pos);
    varyings.world_pos = vec3<f32>(world_pos);
    let color = load_s_colors(billboard_index * 6);
    varyings.color = vec4<f32>(color, 1.0);
    varyings.texcoord_vert = vec2<f32>(tex_coord);
    varyings.billboard_center = vec3<f32>(w_center.xyz);
    varyings.billboard_right = vec3<f32>(cam_right.xyz);
    varyings.billboard_up = vec3<f32>(cam_up.xyz);
    varyings.billboard_size = vec2<f32>(size);
    varyings.billboard_index = f32(billboard_index);

    return varyings;
}

struct ReflectedLight {
    direct_diffuse: vec3<f32>,
    direct_specular: vec3<f32>,
    indirect_diffuse: vec3<f32>,
    indirect_specular: vec3<f32>,
};

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let uv = varyings.texcoord_vert.xy;
    let coord = uv * 2.0 - vec2<f32>(1.0);
    let radius_sq = dot(coord, coord);
    if (radius_sq > 1.0) {
        discard;
    }

    let smooth_edge = fwidth(radius_sq);
    let mask = clamp(1.0 - smoothstep(1.0 - smooth_edge, 1.0 + smooth_edge, radius_sq), 0.0, 1.0);

    let center = varyings.billboard_center;
    let plane_pos = varyings.world_pos;
    let cam_pos = u_stdinfo.cam_transform_inv[3].xyz;
    let cam_forward = normalize((u_stdinfo.cam_transform_inv * vec4<f32>(0.0, 0.0, -1.0, 0.0)).xyz);
    let ortho = is_orthographic();

    var ray_origin = cam_pos;
    var ray_dir = plane_pos - cam_pos;
    if (ortho) {
        ray_origin = plane_pos;
        ray_dir = -cam_forward;
    } else {
        let dir_len = length(ray_dir);
        if (dir_len > 0.0) {
            ray_dir = ray_dir / dir_len;
        } else {
            ray_dir = cam_forward;
        }
    }
    ray_dir = normalize(ray_dir);

    let coeff_limit = get_active_coeff_count();
    if (coeff_limit <= 0 || NUM_COEFFS == 0) {
        discard;
    }

    let billboard_index = max(i32(round(varyings.billboard_index)), 0);
    let coeff_offset = billboard_index * NUM_COEFFS;
    let glyph_id = u32(billboard_index);

    let bound_radius = 0.5 * max(varyings.billboard_size.x, varyings.billboard_size.y);

    let dist_to_cam = length(center - cam_pos);
    let proj_scale = abs(u_stdinfo.projection_transform[1][1]);
    let screen_height = u_stdinfo.physical_size.y;
    let screen_size = (2.0 * bound_radius / max(dist_to_cam, 0.01)) * proj_scale * screen_height * 0.5;

    let intersection = find_surface_intersection(
        coeff_offset,
        center,
        ray_origin,
        ray_dir,
        bound_radius,
        coeff_limit,
        glyph_id,
        screen_size,
    );

    if (!intersection.hit) {
        discard;
    }

    var direction = normalize(intersection.direction);
    let raw_radius = intersection.raw_radius;
    let surface_radius = clamp_radius(raw_radius);

    var world_pos = intersection.position;
    var dist_to_center = length(world_pos - center);
    if (abs(dist_to_center - surface_radius) > max(1e-3, 1e-3 * surface_radius)) {
        world_pos = center + direction * surface_radius;
        dist_to_center = surface_radius;
    }

    var world_normal = vec3<f32>(0.0);

    if (USE_HERMITE_LUT) {
        let local_dir = normalize((u_wobject.world_transform_inv * vec4<f32>(direction, 0.0)).xyz);

        let grad_sample = sample_hermite_cube_with_gradient(glyph_id, local_dir);
        let local_normal = estimate_surface_normal_analytic(
            local_dir,
            dist_to_center,
            grad_sample.grad_s2,
            raw_radius,
        );

        world_normal = normalize((vec4<f32>(local_normal, 0.0) * u_wobject.world_transform_inv).xyz);
    } else {
        world_normal = estimate_surface_normal(
            coeff_offset,
            center,
            world_pos,
            coeff_limit,
            glyph_id,
        );
    }

    if (dot(ray_dir, world_normal) > 0.0) {
        world_normal = -world_normal;
    }

    let clip_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(world_pos, 1.0);
    var view_dir = normalize(cam_pos - world_pos);
    if (ortho) {
        view_dir = normalize(-ray_dir);
    }

    let local_color_dir = normalize((u_wobject.world_transform_inv * vec4<f32>(direction, 0.0)).xyz);

    var glyph_color = vec3<f32>(1.0);
    if (COLOR_TYPE == 0) {
        if (raw_radius < 0.0) {
            glyph_color = vec3<f32>(0.0, 0.0, 1.0);
        } else {
            glyph_color = vec3<f32>(1.0, 0.0, 0.0);
        }
    } else {
        glyph_color = abs(normalize(local_color_dir));
    }

    glyph_color = clamp(glyph_color * varyings.color.rgb, vec3<f32>(0.0), vec3<f32>(1.0));

    var diffuse_color = vec4<f32>(srgb2physical(glyph_color), 1.0);
    diffuse_color.a *= u_material.opacity * mask;
    do_alpha_test(diffuse_color.a);

    let physical_albedo = diffuse_color.rgb;
    let specular_strength = 1.0;

    var reflected_light: ReflectedLight = ReflectedLight(
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
    );

    var geometry: GeometricContext;
    geometry.position = world_pos;
    geometry.normal = world_normal;
    geometry.view_dir = view_dir;

    var material: BlinnPhongMaterial;
    material.diffuse_color = physical_albedo;
    material.specular_color = srgb2physical(u_material.specular_color.rgb);
    material.specular_shininess = u_material.shininess;
    material.specular_strength = specular_strength;

    let ambient_color = u_ambient_light.color.rgb;
    var irradiance = getAmbientLightIrradiance(ambient_color);
    RE_IndirectDiffuse(irradiance, geometry, material, &reflected_light);

    {$ include 'pygfx.light_punctual.wgsl' $}

    var emissive_color = srgb2physical(u_material.emissive_color.rgb) * u_material.emissive_intensity;

    var physical_color = reflected_light.direct_diffuse +
        reflected_light.direct_specular +
        reflected_light.indirect_diffuse +
        reflected_light.indirect_specular +
        emissive_color;

    if (all(physical_color == vec3<f32>(0.0))) {
        let fallback_light = normalize(vec3<f32>(0.3, 0.5, 0.8));
        let fallback_diffuse = max(dot(world_normal, fallback_light), 0.0);
        physical_color = physical_albedo * clamp(0.3 + 0.7 * fallback_diffuse, 0.0, 1.0);
    }

    var out: FragmentOutput;
    out.color = vec4<f32>(physical_color, diffuse_color.a);
    let ndc = clip_pos / clip_pos.w;
    out.depth = ndc.z;
    return out;
}
