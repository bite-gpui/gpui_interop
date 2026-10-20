// The CAD cube: one pass, one draw, Cook-Torrance metal.
//
// Everything the fragment shader needs is in two buffers that are written
// before the first frame — the camera uniform Helio already uploads, and a
// 128-byte parameter block (base colour, roughness/metallic, three fixed
// lights). The draw itself is static geometry, so a frame costs the camera
// write Helio makes anyway plus one `draw_indexed`.
//
// There is no shadow, GI, SSR or post-process pass in this graph. Reflections
// come from `environment`, a small analytic studio (sky/ground gradient with
// two soft panels) sampled along the reflected direction, and the highlight
// comes from the same GGX lobe the three analytic lights use.

struct Camera {
    view:          mat4x4<f32>,
    proj:          mat4x4<f32>,
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far:   vec4<f32>,
};

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;

struct Params {
    base_color:         vec4<f32>,
    roughness_metallic: vec4<f32>,
    key:                vec4<f32>, // xyz: direction towards the light, w: intensity
    key_color:          vec4<f32>,
    fill:               vec4<f32>,
    fill_color:         vec4<f32>,
    rim:                vec4<f32>,
    rim_color:          vec4<f32>,
};

@group(0) @binding(1) var<uniform> params: Params;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position:  vec4<f32>,
    @location(0)       world_position: vec3<f32>,
    @location(1)       normal:         vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = cameras[0].view_proj * vec4<f32>(in.position, 1.0);
    out.world_position = in.position;
    out.normal = in.normal;
    return out;
}

const PI: f32 = 3.14159265359;

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denominator = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denominator * denominator, 1e-6);
}

fn geometry_schlick(n_dot_x: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    return n_dot_x / max(n_dot_x * (1.0 - k) + k, 1e-6);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn direct_light(
    n: vec3<f32>,
    v: vec3<f32>,
    light: vec4<f32>,
    light_color: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    let l = normalize(light.xyz);
    let n_dot_l = dot(n, l);
    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_schlick(n_dot_v, roughness) * geometry_schlick(n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);
    let specular = d * g * f / (4.0 * n_dot_v * n_dot_l + 1e-4);
    let diffuse = (vec3<f32>(1.0) - f) * (1.0 - metallic) * albedo / PI;
    return (diffuse + specular) * light_color * light.w * n_dot_l;
}

/// A dim studio built around the reference render: a bright warm bounce where
/// the left facet's mirror ray lands, a much darker cool one on the right, and
/// narrow panels for the two top chamfers. The gaps between them are what give
/// the chamfers their dark bands; a uniform environment washes them out and is
/// what made the first version look soft.
fn environment(direction: vec3<f32>) -> vec3<f32> {
    let up = clamp(direction.y, -1.0, 1.0);
    let azimuth = normalize(direction.xz + vec2<f32>(1e-5));

    // Dim studio walls.
    var color = vec3<f32>(0.050, 0.052, 0.060);

    // Floor bounce: bright to the front-left, dim to the front-right, dark in
    // the gap between them (where the front vertical chamfer looks).
    let below = 1.0 - smoothstep(-0.15, 0.10, up);
    let height = mix(0.55, 1.0, smoothstep(-0.9, 0.05, up));
    let warm = pow(max(dot(azimuth, vec2<f32>(-0.7, 0.7)), 0.0), 3.0);
    let cool = pow(max(dot(azimuth, vec2<f32>(0.2, -0.98)), 0.0), 5.0);
    color += below
        * height
        * (vec3<f32>(0.55, 0.45, 0.20) * warm + vec3<f32>(0.16, 0.17, 0.19) * cool);

    // A cool sky above the horizon.
    color = mix(color, vec3<f32>(0.26, 0.28, 0.31), smoothstep(0.0, 0.7, up));

    // The softboxes: front-left, right, behind, and one over each top chamfer.
    let left_box = pow(max(dot(direction, normalize(vec3<f32>(-0.57, -0.71, 0.39))), 0.0), 10.0);
    color += vec3<f32>(0.45, 0.38, 0.22) * left_box;
    let right_box = pow(max(dot(direction, normalize(vec3<f32>(0.40, -0.73, -0.59))), 0.0), 12.0);
    color += vec3<f32>(0.30, 0.26, 0.14) * right_box;
    let top_box = pow(max(dot(direction, normalize(vec3<f32>(-0.52, 0.30, -0.80))), 0.0), 10.0);
    color += vec3<f32>(0.45, 0.40, 0.24) * top_box;
    // Narrow, bright strip lights across each top chamfer. These are the edge
    // highlights: the 45-degree facets mirror them along their whole length,
    // while the flat faces mirror the much dimmer sky and floor.
    // The mirror directions of the two top chamfers at this camera angle.
    let top_left_box = pow(max(dot(direction, normalize(vec3<f32>(-0.66, 0.46, 0.62))), 0.0), 16.0);
    color += vec3<f32>(3.50, 3.05, 1.70) * top_left_box;
    let top_right_box = pow(max(dot(direction, normalize(vec3<f32>(0.61, 0.46, -0.65))), 0.0), 16.0);
    color += vec3<f32>(2.20, 1.95, 1.10) * top_right_box;

    return color;
}

fn tonemap(color: vec3<f32>) -> vec3<f32> {
    // Narkowicz's ACES approximation: cheap, filmic, and it keeps the
    // highlight from clipping to a flat white disc.
    let mapped = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    return clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let v = normalize(cameras[0].position_near.xyz - in.world_position);

    let roughness = clamp(params.roughness_metallic.x, 0.03, 1.0);
    let metallic = clamp(params.roughness_metallic.y, 0.0, 1.0);
    let base = params.base_color.rgb;

    // Conductors have no diffuse albedo; their base colour *is* F0.
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let albedo = base * (1.0 - metallic);

    var color = vec3<f32>(0.0);
    color += direct_light(n, v, params.key, params.key_color.rgb, albedo, roughness, metallic, f0);
    color += direct_light(n, v, params.fill, params.fill_color.rgb, albedo, roughness, metallic, f0);
    color += direct_light(n, v, params.rim, params.rim_color.rgb, albedo, roughness, metallic, f0);

    // Cheap image-based lighting: mirror the view vector, sample the studio,
    // and weight it by a roughness-aware Fresnel. This is what gives the three
    // visible facets their distinct values and the bevelled edges their glint.
    let reflected = reflect(-v, n);
    let n_dot_v = max(dot(n, v), 0.0);
    let fresnel = f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(1.0 - n_dot_v, 5.0);
    let studio = environment(reflected);
    // A broad left/right and up/down bias in the studio, plus cheap contact
    // occlusion at the base. Without these, wide flat facets lit by smooth
    // panels read as flat colour fields and the cube looks soft.
    let bias = 1.0 + 0.55 * (-reflected.x) + 0.15 * reflected.y;
    let ao = mix(0.72, 1.0, smoothstep(-1.15, 0.15, in.world_position.y));
    color += studio * fresnel * mix(1.0, 0.55, roughness) * bias * ao;

    // A little floor bounce, for dielectric parts only.
    color += albedo * 0.04 * studio;

    return vec4<f32>(tonemap(color), 1.0);
}
