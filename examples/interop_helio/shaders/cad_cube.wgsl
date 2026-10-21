// The CAD cube: one pass, one draw, a Cook-Torrance white metal.
//
// Everything the fragment shader needs is in two buffers written before the
// first frame — the camera uniform Helio already uploads, and a small
// parameter block (base colour and roughness/metallic). The draw itself is
// static geometry, so a frame costs the camera write Helio makes anyway plus
// one `draw_indexed`.
//
// There is no shadow, GI, SSR or post-process pass, and no analytic light of
// any kind: the image is entirely ambient. Reflections come from
// `environment`, a soft studio sampled along the reflected direction. Its wide
// panels are the only highlights, so they read as gentle sheen rather than
// spot lights.

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

/// A soft, mostly ambient studio: a cool sky, a neutral floor bounce and four
/// wide panels. There are no spot lights in the scene; the panels' reflections
/// are the only highlights, so they are wide and gentle, and the ambient walls
/// are bright enough to keep white metal white.
fn environment(direction: vec3<f32>) -> vec3<f32> {
    let up = clamp(direction.y, -1.0, 1.0);
    let azimuth = normalize(direction.xz + vec2<f32>(1e-5));

    // Bright, neutral ambient walls.
    var color = vec3<f32>(0.16, 0.165, 0.175);

    // Floor bounce: slightly brighter to the front-left, softer to the right.
    let below = 1.0 - smoothstep(-0.15, 0.10, up);
    let height = mix(0.60, 1.0, smoothstep(-0.9, 0.05, up));
    let warm = pow(max(dot(azimuth, vec2<f32>(-0.7, 0.7)), 0.0), 3.0);
    let cool = pow(max(dot(azimuth, vec2<f32>(0.2, -0.98)), 0.0), 4.0);
    color += below
        * height
        * (vec3<f32>(0.55, 0.56, 0.58) * warm + vec3<f32>(0.24, 0.25, 0.26) * cool);

    // A cool sky above the horizon.
    color = mix(color, vec3<f32>(0.40, 0.42, 0.45), smoothstep(0.0, 0.7, up));

    // Broad panels for the floor-facing and top-facing reflections.
    let left_box = pow(max(dot(direction, normalize(vec3<f32>(-0.57, -0.71, 0.39))), 0.0), 6.0);
    color += vec3<f32>(0.45, 0.46, 0.47) * left_box;
    let right_box = pow(max(dot(direction, normalize(vec3<f32>(0.40, -0.73, -0.59))), 0.0), 8.0);
    color += vec3<f32>(0.30, 0.31, 0.32) * right_box;
    let top_box = pow(max(dot(direction, normalize(vec3<f32>(-0.80, 0.30, -0.45))), 0.0), 6.0);
    color += vec3<f32>(0.50, 0.51, 0.52) * top_box;

    // The two top chamfers' reflections. These are the edge highlights: broad
    // and bright, so they read as ambient sheen rather than spot lights.
    let top_left_box = pow(max(dot(direction, normalize(vec3<f32>(-0.66, 0.46, 0.62))), 0.0), 8.0);
    color += vec3<f32>(1.60, 1.62, 1.65) * top_left_box;
    let top_right_box = pow(max(dot(direction, normalize(vec3<f32>(0.61, 0.46, -0.65))), 0.0), 8.0);
    color += vec3<f32>(1.00, 1.01, 1.03) * top_right_box;

    // The front vertical chamfer mirrors into the gap between the floor
    // bounces; a little fill gives it depth rather than a flat dark wedge.
    let front_fill = pow(max(dot(direction, normalize(vec3<f32>(0.44, -0.77, 0.44))), 0.0), 6.0);
    color += vec3<f32>(0.07, 0.07, 0.075) * front_fill;

    // A narrow strip reflected near the front edge of the right face.
    let right_glint = pow(max(dot(azimuth, normalize(vec2<f32>(0.78, -0.63))), 0.0), 60.0)
        * (1.0 - smoothstep(-0.1, 0.15, up));
    color += vec3<f32>(0.28, 0.29, 0.30) * right_glint;

    // A soft fill behind the camera for the corner blend patches.
    let corner_fill = pow(max(dot(direction, normalize(vec3<f32>(0.60, 0.55, 0.60))), 0.0), 2.0);
    color += vec3<f32>(0.55, 0.56, 0.58) * corner_fill;

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

    // Everything is ambient now: mirror the view vector, sample the studio,
    // and weight it by a roughness-aware Fresnel. The panels in the studio are
    // what give the facets their distinct values and the bevels their sheen.
    var color = vec3<f32>(0.0);
    let reflected = reflect(-v, n);
    let n_dot_v = max(dot(n, v), 0.0);
    let fresnel = f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(1.0 - n_dot_v, 5.0);
    let studio = environment(reflected);
    // A broad left/right and up/down bias in the studio, plus cheap contact
    // occlusion at the base. Without these, wide flat facets lit by smooth
    // panels read as flat colour fields and the cube looks soft.
    let bias = 1.0 + 0.55 * (-reflected.x) + 0.15 * reflected.y;
    let ao = mix(0.68, 1.0, smoothstep(-1.15, 0.15, in.world_position.y));
    color += studio * fresnel * mix(1.0, 0.55, roughness) * bias * ao;

    // A thin grazing rim, so the silhouette reads as a rolled highlight
    // rather than a hard cut, without washing out the near-vertical sides.
    color += vec3<f32>(0.80, 0.82, 0.85) * pow(1.0 - n_dot_v, 8.0) * 0.18;

    return vec4<f32>(tonemap(color), 1.0);
}
