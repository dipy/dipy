const PI: f32 = 3.14159265359;
const SQRT_2: f32 = 1.41421356237;

fn orient2rgb(v: vec3<f32>) -> vec3<f32> {
    let r = sqrt(dot(v, v));

    if (r != 0.0) {
      return abs(v / r);
    }
    return vec3<f32>(1.0);
}

fn scaled_color(v: vec3<f32>) -> vec3<f32> {
    return abs(normalize(v));
}

fn visible_cross_section(center: vec3<f32>, cross_section: vec3<f32>, visibility: vec3<i32>) -> bool {
    let xVal = center.x == cross_section.x && visibility.x != 0;
    let yVal = center.y == cross_section.y && visibility.y != 0;
    let zVal = center.z == cross_section.z && visibility.z != 0;

    return xVal || yVal || zVal;
}

fn is_point_on_plane_equation(plane: vec4<f32>, point: vec3<f32>, scale: f32) -> bool {
    let distance = dot(plane.xyz, point) + plane.w;

    let normalizedDistance = distance / length(plane.xyz);

    return abs(normalizedDistance) <= scale / 2.0;
}

fn crosssect_plane(plane: vec4<f32>, p0: vec3<f32>, p1: vec3<f32>) -> f32 {
    let num = -1 * (plane.x * p0.x + plane.y * p0.y + plane.z * p0.z + plane.w);
    let denom = plane.x * (p1.x - p0.x) + plane.y * (p1.y - p0.y) + plane.z * (p1.z - p0.z);
    if (denom == 0.0) {
        return -1.0;
    }
    let t = num / denom;
    return t;
}

fn intersect_plane(t: f32, p0: vec3<f32>, p1: vec3<f32>) -> vec3<f32> {
    return p0 + t * (p1 - p0);
}

fn perpendicular_point(point_on_plane: vec3<f32>, plane_normal: vec3<f32>, distance: f32) -> vec3<f32> {
    let direction = normalize(plane_normal);
    let offset = direction * distance;
    let new_point = point_on_plane + offset;
    return new_point;
}

fn visible_range(center: vec3<i32>, low_range: vec3<i32>, high_range: vec3<i32>) -> bool {
    let xVal = center.x >= low_range.x && center.x <= high_range.x;
    let yVal = center.y >= low_range.y && center.y <= high_range.y;
    let zVal = center.z >= low_range.z && center.z <= high_range.z;

    return xVal && yVal && zVal;
}

fn get_voxel_id(id: i32, data_shape: vec3<i32>, visible: vec3<i32>) -> vec3<i32> {
    var slice_id = id;

    let z_slice = data_shape.x * data_shape.y;
    let x_slice = data_shape.y * data_shape.z;
    let y_slice = data_shape.x * data_shape.z;

    if (slice_id < z_slice) {
        let x = slice_id / data_shape.y;
        let y = slice_id % data_shape.y;
        return vec3<i32>(x, y, visible.z);
    }

    slice_id = slice_id - z_slice;
    if (slice_id < y_slice) {
        let x = slice_id / data_shape.z;
        let z = slice_id % data_shape.z;
        return vec3<i32>(x, visible.y, z);
    }

    slice_id = slice_id - y_slice;
    let y = slice_id / data_shape.z;
    let z = slice_id % data_shape.z;
    return vec3<i32>(visible.x, y, z);

}

fn flatten_from_3d(coord: vec3<i32>, data_shape: vec3<i32>) -> i32 {
    return coord.x * data_shape.y * data_shape.z + coord.y * data_shape.z + coord.z;
}

fn flatten_to_3d(index: i32, data_shape: vec3<i32>) -> vec3<i32> {
    let z = index % data_shape.z;
    let y = (index / data_shape.z) % data_shape.y;
    let x = index / (data_shape.y * data_shape.z);
    return vec3<i32>(x, y, z);
}
