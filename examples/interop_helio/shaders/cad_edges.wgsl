// CAD feature-edge overlay.
//
// Every edge is a screen-space quad, not a hardware line: wgpu lines are one
// pixel wide, and the reference draws them thick enough to read as pickable
// edges. Each vertex carries both endpoints of its edge so the quad's
// perpendicular is computed from the same vector at all four corners; using
// just the "other" endpoint would flip it at the far end and fold the quad.
//
// The quads are depth-tested against the cube, so only visible edges survive
// and hidden ones are culled exactly like a CAD viewport's hidden-line pass.

struct Camera {
    view:          mat4x4<f32>,
    proj:          mat4x4<f32>,
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far:   vec4<f32>,
};

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
/// xy: viewport size in pixels, z: line half-width in pixels.
@group(0) @binding(1) var<uniform> viewport: vec4<f32>;

struct VertexInput {
    @location(0) position:   vec3<f32>,
    @location(1) edge_start: vec3<f32>,
    @location(2) edge_end:   vec3<f32>,
    @location(3) side:       f32,
};

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    let size = max(viewport.xy, vec2<f32>(1.0));
    let clip_start = cameras[0].view_proj * vec4<f32>(in.edge_start, 1.0);
    let clip_end = cameras[0].view_proj * vec4<f32>(in.edge_end, 1.0);
    let delta = (clip_end.xy / clip_end.w - clip_start.xy / clip_start.w) * size;

    var normal = vec2<f32>(-delta.y, delta.x);
    let length_squared = dot(normal, normal);
    normal = select(
        vec2<f32>(1.0, 0.0),
        normal / sqrt(length_squared),
        length_squared > 1e-12,
    );

    let clip = cameras[0].view_proj * vec4<f32>(in.position, 1.0);
    // 2 / size converts pixels to NDC; multiplying by w keeps the width
    // constant after the perspective divide.
    let offset = normal * (2.0 * viewport.z / size) * clip.w * in.side;
    return vec4<f32>(clip.xy + offset, clip.z, clip.w);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Linear; the sRGB target encodes it into a pale warm line.
    return vec4<f32>(1.0, 0.90, 0.65, 1.0);
}
