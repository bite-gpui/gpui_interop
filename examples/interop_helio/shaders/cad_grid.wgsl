// Backdrop for the CAD scene: a full-screen triangle that paints a dark
// gradient and the floor grid, with no depth attachment.
//
// The grid is computed per pixel in world space: the fragment reconstructs its
// view ray from the inverse view-projection and intersects it with the floor
// plane, so the lines converge with perspective instead of being a texture.
// `fwidth` keeps them one pixel wide at every distance, and the distance fade
// dissolves them into the background well before the horizon aliases.

struct Camera {
    view:         mat4x4<f32>,
    proj:         mat4x4<f32>,
    view_proj:    mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far:   vec4<f32>,
};

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       ndc:           vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // A triangle that covers the viewport with three vertices: (-1,-1), (3,-1), (-1,3).
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let position = positions[vertex_index];
    var out: VertexOutput;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.ndc = position;
    return out;
}

/// One anti-aliased line set, 1.0 on a line and 0.0 away from it.
fn grid_lines(coord: vec2<f32>, spacing: f32) -> f32 {
    let scaled = coord / spacing;
    let width = max(fwidth(scaled), vec2<f32>(1e-5));
    let distance = abs(fract(scaled - 0.5) - 0.5);
    let line = distance / width;
    return 1.0 - min(min(line.x, line.y), 1.0);
}

fn background(direction: vec3<f32>) -> vec3<f32> {
    let horizon = vec3<f32>(0.017, 0.020, 0.027);
    let zenith = vec3<f32>(0.004, 0.005, 0.008);
    var color = mix(horizon, zenith, smoothstep(-0.1, 0.9, direction.y));

    // A faint pool of light behind the cube, so the backdrop is not flat.
    let forward = normalize(cameras[0].forward_far.xyz);
    let glow = pow(clamp(dot(direction, forward), 0.0, 1.0), 20.0);
    color += vec3<f32>(0.020, 0.026, 0.038) * glow;
    return color;
}

const FLOOR_Y: f32 = -1.12;
const MINOR_SPACING: f32 = 0.25;
const MAJOR_SPACING: f32 = 1.0;
const MINOR_COLOR: vec3<f32> = vec3<f32>(0.030, 0.035, 0.045);
const MAJOR_COLOR: vec3<f32> = vec3<f32>(0.050, 0.058, 0.072);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let eye = cameras[0].position_near.xyz;
    let near_point = cameras[0].inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let direction = normalize(near_point.xyz / near_point.w - eye);

    var color = background(direction);

    // Intersect the floor plane; directions above the horizon miss it.
    if (direction.y < -1e-4) {
        let distance_along_ray = (FLOOR_Y - eye.y) / direction.y;
        if (distance_along_ray > 0.0) {
            let hit = eye + direction * distance_along_ray;
            let from_center = length(hit.xz);
            let fade = 1.0 - smoothstep(6.0, 30.0, from_center);

            let major = grid_lines(hit.xz, MAJOR_SPACING) * fade;
            let minor = grid_lines(hit.xz, MINOR_SPACING) * fade;
            color = mix(color, MAJOR_COLOR, major * 0.42);
            color = mix(color, MINOR_COLOR, minor * 0.22);

            // A soft contact shadow where the cube meets the floor. There is no
            // shadow pass: this is the whole "grounding" cue the scene gets.
            let contact = exp(-from_center * from_center * 0.55);
            color *= 1.0 - 0.45 * contact;
        }
    }

    return vec4<f32>(color, 1.0);
}
