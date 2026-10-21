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
    @location(0) position: vec3<f32>,
    @location(1) previous: vec3<f32>,
    @location(2) next:     vec3<f32>,
    @location(3) side:     f32,
};

fn screen_normal(size: vec2<f32>, start: vec4<f32>, end: vec4<f32>) -> vec2<f32> {
    let delta = (end.xy / end.w - start.xy / start.w) * size;
    return normalize(vec2<f32>(-delta.y, delta.x) + vec2<f32>(1e-6, 1e-6));
}

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    let size = max(viewport.xy, vec2<f32>(1.0));
    let clip = cameras[0].view_proj * vec4<f32>(in.position, 1.0);
    let clip_previous = cameras[0].view_proj * vec4<f32>(in.previous, 1.0);
    let clip_next = cameras[0].view_proj * vec4<f32>(in.next, 1.0);

    let incoming = screen_normal(size, clip_previous, clip);
    let outgoing = screen_normal(size, clip, clip_next);
    let miter = incoming + outgoing;
    let miter_length = length(miter);
    let corner = miter_length > 1e-4;
    let direction = select(outgoing, miter / max(miter_length, 1e-4), corner);
    // 1 / cos(half the turn) holds the width around the corner; clamping keeps
    // a very sharp vertex from growing an unbounded spike.
    let scale = select(1.0, 1.0 / max(dot(direction, incoming), 0.34), corner);

    let offset = direction * (2.0 * viewport.z / size) * clip.w * in.side * scale;
    return vec4<f32>(clip.xy + offset, clip.z, clip.w);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Linear; the sRGB target encodes it into a pale neutral highlight.
    return vec4<f32>(0.42, 0.44, 0.46, 1.0);
}
