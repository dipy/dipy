{$ include 'pygfx.std.wgsl' $}
{$ include 'pygfx.light_phong.wgsl' $}
{$ include 'dipy.utils.wgsl' $}

struct VertexInput {
    @builtin(vertex_index) index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    let billboard_index = i32(in.index) / 6;
    let vertex_in_quad = i32(in.index) % 6;

    var local_pos: vec2<f32>;
    switch vertex_in_quad {
        case 0: { local_pos = vec2<f32>(-0.5, -0.5); }
        case 1: { local_pos = vec2<f32>(0.5, -0.5); }
        case 2: { local_pos = vec2<f32>(-0.5, 0.5); }
        case 3: { local_pos = vec2<f32>(0.5, -0.5); }
        case 4: { local_pos = vec2<f32>(0.5, 0.5); }
        default: { local_pos = vec2<f32>(-0.5, 0.5); }
    }

    let raw_center = load_s_positions(billboard_index * 6);
    let world_center = u_wobject.world_transform * vec4<f32>(raw_center.xyz, 1.0);

    let cam_right = vec3<f32>(u_stdinfo.cam_transform_inv[0].xyz);
    let cam_up = vec3<f32>(u_stdinfo.cam_transform_inv[1].xyz);

    let raw_size = load_s_normals(billboard_index * 6);
    let size = raw_size.xy;

    let billboard_offset = local_pos.x * cam_right * size.x + local_pos.y * cam_up * size.y;
    let world_pos = world_center.xyz + billboard_offset;

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
    varyings.billboard_center = vec3<f32>(world_center.x, world_center.y, world_center.z);
    varyings.billboard_right = vec3<f32>(cam_right.x, cam_right.y, cam_right.z);
    varyings.billboard_up = vec3<f32>(cam_up.x, cam_up.y, cam_up.z);
    varyings.billboard_size = vec2<f32>(size);

    return varyings;
}

struct ReflectedLight {
    direct_diffuse: vec3<f32>,
    direct_specular: vec3<f32>,
    indirect_diffuse: vec3<f32>,
    indirect_specular: vec3<f32>,
};

@fragment
fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let uv = varyings.texcoord_vert.xy;
    let coord = uv * 2.0 - vec2<f32>(1.0);
    let radius_sq = dot(coord, coord);
    if (radius_sq > 1.0) {
        discard;
    }

    let smooth_edge = fwidth(radius_sq);
    let mask = clamp(1.0 - smoothstep(1.0 - smooth_edge, 1.0 + smooth_edge, radius_sq), 0.0, 1.0);

    let radius = 0.5 * varyings.billboard_size.x;
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

    let oc = ray_origin - center;
    let b = dot(ray_dir, oc);
    let c = dot(oc, oc) - radius * radius;
    var discriminant = b * b - c;
    if (discriminant < 0.0) {
        if (discriminant > -1e-4) {
            discriminant = 0.0;
        } else {
            discard;
        }
    }
    let sqrt_disc = sqrt(discriminant);
    var t = -b - sqrt_disc;
    if (t < 0.0) {
        t = -b + sqrt_disc;
    }
    if (t < 0.0) {
        discard;
    }
    let world_pos = ray_origin + ray_dir * t;
    var world_normal = normalize(world_pos - center);
    if (!is_front) {
        world_normal = -world_normal;
    }

    let clip_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(world_pos, 1.0);
    var view_dir = -ray_dir;
    if (ortho) {
        view_dir = ray_dir;
    }
    view_dir = normalize(view_dir);

    var diffuse_color = vec4<f32>(srgb2physical(varyings.color.rgb), varyings.color.a);
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

    {$ include 'pygfx.light_punctual.wgsl' $}

    let ambient_color = u_ambient_light.color.rgb;
    var irradiance = getAmbientLightIrradiance(ambient_color);
    RE_IndirectDiffuse(irradiance, geometry, material, &reflected_light);

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
