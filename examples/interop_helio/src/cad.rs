//! A CAD-viewer scene built entirely from custom Helio passes.
//!
//! Two passes, no engine graph:
//!
//! 1. [`GridPass`] covers the viewport with a dark gradient and a perspective
//!    grid floor. It is a full-screen triangle whose fragment shader rebuilds
//!    the view ray from the camera uniform and intersects it with the floor
//!    plane; there is no grid texture.
//! 2. [`CadPass`] draws one chamfered, gold, metallic cube with Cook-Torrance
//!    direct lighting and a small analytic studio environment for reflections.
//!    Its twelve edges are flat chamfers and its eight corners are smooth
//!    multi-sided blends, with the curvature carried by a barycentric normal
//!    field rather than a bulge. It then overlays the cube's feature edges as
//!    soft CAD-style highlights; the overlay is depth-tested against the cube,
//!    so hidden edges are culled like a CAD viewport's.
//!
//! There is no shadow, GI, ambient-occlusion, SSR, atmosphere or post-process
//! pass, and no per-frame CPU work beyond recording two draws: the mesh
//! (generated once on the CPU at construction), the vertex/index buffers, the
//! light/material block and both pipelines are all created once. Helio already
//! writes the camera uniform each frame, so nothing in this module touches a
//! buffer after the first frame. Both passes draw into one shared 4x
//! multisampled colour target, which the cube pass resolves into the exported
//! image, so the antialiasing costs one resolve and no separate AA pass.

use std::sync::Arc;

use helio::{GraphRebuilder, PassBuildContext, RenderGraph, RenderPass};
use helio_core::{
    PassContext, PrepareContext, RenderFrameStorage, ResourceRegistry, Result as HelioResult,
};
use wgpu30::util::DeviceExt;

use helio_core::depth;

/// The cube: half-extent (so the edge is `2 * CUBE_HALF`) and the width of
/// the 45-degree chamfer cut along every edge.
const CUBE_HALF: f32 = 1.1;
const CUBE_CHAMFER: f32 = 0.16;

/// Vertex layout: position(12) | normal(12).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CadVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

/// One corner of an edge's screen-space quad.
///
/// `previous` and `next` are the neighbouring points on the same polyline, so
/// the vertex shader can miter the join instead of letting each segment's
/// quad end square-on (which crosses its neighbour at sharp corners).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EdgeVertex {
    position: [f32; 3],
    previous: [f32; 3],
    next: [f32; 3],
    side: f32,
}

/// The 128-byte material/light block, written once at construction.
///
/// `vec4` members only, so the WGSL struct matches without padding games.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CadParams {
    /// Linear base colour, which is also F0 for a conductor.
    base_color: [f32; 4],
    /// `x`: roughness, `y`: metallic.
    roughness_metallic: [f32; 4],
    /// `xyz`: direction towards the key light, `w`: intensity.
    key: [f32; 4],
    key_color: [f32; 4],
    fill: [f32; 4],
    fill_color: [f32; 4],
    rim: [f32; 4],
    rim_color: [f32; 4],
}

impl CadParams {
    fn gold() -> Self {
        Self {
            // Linear gold: a warm yellow-orange in the sRGB target.
            base_color: [1.00, 0.62, 0.025, 1.0],
            roughness_metallic: [0.11, 1.0, 0.0, 0.0],
            // Upper-front-left key, cool right fill, warm rim from behind.
            key: [-0.30, 0.80, 0.55, 3.0],
            key_color: [1.00, 0.94, 0.85, 1.0],
            fill: [0.80, 0.15, 0.20, 1.2],
            fill_color: [0.55, 0.68, 1.00, 1.0],
            rim: [0.10, 0.30, -0.90, 1.5],
            rim_color: [1.00, 0.75, 0.45, 1.0],
        }
    }
}

// ── Mesh ─────────────────────────────────────────────────────────────────────

type Point = [f32; 3];

fn vec3_sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_dot(a: Point, b: Point) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_normalize(v: Point) -> Point {
    let length = vec3_dot(v, v).sqrt().max(1e-6);
    [v[0] / length, v[1] / length, v[2] / length]
}

/// Appends a triangle whose vertices carry their own normals, flipping its
/// winding if it would face away from `reference`.
fn push_smooth_triangle(
    vertices: &mut Vec<CadVertex>,
    indices: &mut Vec<u32>,
    mut points: [Point; 3],
    mut normals: [Point; 3],
    reference: Point,
) {
    let facing = vec3_dot(
        vec3_cross(
            vec3_sub(points[1], points[0]),
            vec3_sub(points[2], points[0]),
        ),
        reference,
    );
    if facing < 0.0 {
        points.swap(1, 2);
        normals.swap(1, 2);
    }
    let base = vertices.len() as u32;
    for (position, normal) in points.into_iter().zip(normals) {
        vertices.push(CadVertex { position, normal });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2]);
}

/// Appends a quad, flipping its winding if it would face away from `normal`.
fn push_quad(
    vertices: &mut Vec<CadVertex>,
    indices: &mut Vec<u32>,
    mut corners: [Point; 4],
    normal: Point,
) {
    let facing = vec3_dot(
        vec3_cross(
            vec3_sub(corners[1], corners[0]),
            vec3_sub(corners[2], corners[0]),
        ),
        normal,
    );
    if facing < 0.0 {
        corners.reverse();
    }
    let base = vertices.len() as u32;
    for position in corners {
        vertices.push(CadVertex { position, normal });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// A chamfered box with smooth vertex blends: the twelve edges are flat
/// 45-degree strips on planes like `x + y = 2 * half - chamfer`, and the eight
/// corners are flush patches whose normals blend smoothly from each strip's
/// normal on the shared edges to the space diagonal in the middle.
///
/// The blend is in the normal field rather than in a bulge, which is what the
/// reference's CAD corner looks like: a multi-sided patch that distributes the
/// shading across the corner instead of leaving the bevels to meet in a flat,
/// pinched, creased triangle. Generated once, at construction.
fn chamfered_cube(half: f32, chamfer: f32) -> (Vec<CadVertex>, Vec<u32>) {
    let face_half = half - chamfer;
    let mut vertices = Vec::with_capacity(4096);
    let mut indices = Vec::with_capacity(6144);

    // Six flat faces. `u x v = normal`, so the table's order is already
    // counter-clockwise; `push_quad` checks instead of trusting that.
    let faces: [(Point, Point, Point); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    for (normal, u_axis, v_axis) in faces {
        let center = [normal[0] * half, normal[1] * half, normal[2] * half];
        let corner = |u: f32, v: f32| {
            [
                center[0] + u_axis[0] * u + v_axis[0] * v,
                center[1] + u_axis[1] * u + v_axis[1] * v,
                center[2] + u_axis[2] * u + v_axis[2] * v,
            ]
        };
        push_quad(
            &mut vertices,
            &mut indices,
            [
                corner(-face_half, -face_half),
                corner(face_half, -face_half),
                corner(face_half, face_half),
                corner(-face_half, face_half),
            ],
            normal,
        );
    }

    // Twelve edge strips, one per pair of faces and sign combination. Each
    // strip is flat; its ends are straight because the corner patch is flush
    // with it.
    let axes: [Point; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let across_half = chamfer * std::f32::consts::FRAC_1_SQRT_2;
    for (i, j) in [(0usize, 1usize), (0, 2), (1, 2)] {
        let k = 3 - i - j;
        for si in [-1.0_f32, 1.0] {
            for sj in [-1.0_f32, 1.0] {
                let normal = vec3_normalize([
                    axes[i][0] * si + axes[j][0] * sj,
                    axes[i][1] * si + axes[j][1] * sj,
                    axes[i][2] * si + axes[j][2] * sj,
                ]);
                let along = axes[k];
                let across = vec3_cross(normal, along);
                let mut center = [0.0_f32; 3];
                center[i] = si * (half + face_half) * 0.5;
                center[j] = sj * (half + face_half) * 0.5;
                let corner = |a: f32, b: f32| {
                    [
                        center[0] + along[0] * a + across[0] * b,
                        center[1] + along[1] * a + across[1] * b,
                        center[2] + along[2] * a + across[2] * b,
                    ]
                };
                push_quad(
                    &mut vertices,
                    &mut indices,
                    [
                        corner(-face_half, -across_half),
                        corner(face_half, -across_half),
                        corner(face_half, across_half),
                        corner(-face_half, across_half),
                    ],
                    normal,
                );
            }
        }
    }

    // Eight flat corner blend patches. Each sits flush with the three strips,
    // so it does not bulge into a knob, and its normals come from one smooth
    // field: every shared edge carries its strip's normal, the middle carries
    // the space diagonal, and quadratic barycentric weights (`ab`, `bc`,
    // `ca`) blend the three. A three-sector fan would leave creases along the
    // sector borders -- the V-shaped shading artifact -- whereas this field is
    // continuous across the whole patch except at its three corners, where two
    // strips meet and a crease is correct.
    const PATCH_RINGS: usize = 8;
    for sx in [-1.0_f32, 1.0] {
        for sy in [-1.0_f32, 1.0] {
            for sz in [-1.0_f32, 1.0] {
                let corners = [
                    [sx * half, sy * face_half, sz * face_half],
                    [sx * face_half, sy * half, sz * face_half],
                    [sx * face_half, sy * face_half, sz * half],
                ];
                // Normals of the three chamfer strips meeting at this corner;
                // index 0 belongs to edge AB, 1 to BC and 2 to CA.
                let strip_normals = [
                    vec3_normalize([sx, sy, 0.0]),
                    vec3_normalize([0.0, sy, sz]),
                    vec3_normalize([sx, 0.0, sz]),
                ];
                let diagonal = vec3_normalize([sx, sy, sz]);

                let position_at = |a: f32, b: f32, c: f32| {
                    [
                        corners[0][0] * a + corners[1][0] * b + corners[2][0] * c,
                        corners[0][1] * a + corners[1][1] * b + corners[2][1] * c,
                        corners[0][2] * a + corners[1][2] * b + corners[2][2] * c,
                    ]
                };
                let normal_at = |a: f32, b: f32, c: f32| {
                    // Near an edge, use that edge's strip normal; by the middle
                    // of the patch, use the space diagonal. Keeping the
                    // transition narrow leaves the patch reading as a bright
                    // CAD facet instead of a dark well that inherits the
                    // vertical chamfer's shadow.
                    let nearest = if a <= b && a <= c {
                        strip_normals[1]
                    } else if b <= c {
                        strip_normals[2]
                    } else {
                        strip_normals[0]
                    };
                    let distance = a.min(b).min(c);
                    let x = (distance / 0.10).clamp(0.0, 1.0);
                    let blend = x * x * (3.0 - 2.0 * x);
                    vec3_normalize([
                        nearest[0] + (diagonal[0] - nearest[0]) * blend,
                        nearest[1] + (diagonal[1] - nearest[1]) * blend,
                        nearest[2] + (diagonal[2] - nearest[2]) * blend,
                    ])
                };

                for ring in 0..PATCH_RINGS {
                    let c0 = ring as f32 / PATCH_RINGS as f32;
                    let c1 = (ring + 1) as f32 / PATCH_RINGS as f32;
                    let columns = PATCH_RINGS - ring;
                    for column in 0..columns {
                        let b0 = column as f32 / PATCH_RINGS as f32;
                        let b1 = (column + 1) as f32 / PATCH_RINGS as f32;
                        let a0 = 1.0 - b0 - c0;
                        let a1 = 1.0 - b1 - c0;
                        let a2 = 1.0 - b0 - c1;
                        push_smooth_triangle(
                            &mut vertices,
                            &mut indices,
                            [
                                position_at(a0, b0, c0),
                                position_at(a1, b1, c0),
                                position_at(a2, b0, c1),
                            ],
                            [
                                normal_at(a0, b0, c0),
                                normal_at(a1, b1, c0),
                                normal_at(a2, b0, c1),
                            ],
                            diagonal,
                        );
                        if column + 1 < columns {
                            let a3 = 1.0 - b1 - c1;
                            push_smooth_triangle(
                                &mut vertices,
                                &mut indices,
                                [
                                    position_at(a1, b1, c0),
                                    position_at(a3, b1, c1),
                                    position_at(a2, b0, c1),
                                ],
                                [
                                    normal_at(a1, b1, c0),
                                    normal_at(a3, b1, c1),
                                    normal_at(a2, b0, c1),
                                ],
                                diagonal,
                            );
                        }
                    }
                }
            }
        }
    }

    (vertices, indices)
}

/// The feature edges of [`chamfered_cube`], expanded into screen-space quads.
///
/// The sharp edges form closed polylines: the six face perimeters and the
/// eight corner-patch boundaries. Each polyline is expanded with mitered joins
/// so a corner is one clean angle, not two overlapping quads.
fn chamfered_cube_edges(half: f32, chamfer: f32) -> Vec<EdgeVertex> {
    let face_half = half - chamfer;
    let mut loops: Vec<Vec<Point>> = Vec::with_capacity(14);

    let faces: [(Point, Point, Point); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    for (normal, u_axis, v_axis) in faces {
        let center = [normal[0] * half, normal[1] * half, normal[2] * half];
        let corner = |u: f32, v: f32| {
            [
                center[0] + u_axis[0] * u + v_axis[0] * v,
                center[1] + u_axis[1] * u + v_axis[1] * v,
                center[2] + u_axis[2] * u + v_axis[2] * v,
            ]
        };
        loops.push(vec![
            corner(-face_half, -face_half),
            corner(face_half, -face_half),
            corner(face_half, face_half),
            corner(-face_half, face_half),
        ]);
    }

    for sx in [-1.0_f32, 1.0] {
        for sy in [-1.0_f32, 1.0] {
            for sz in [-1.0_f32, 1.0] {
                loops.push(vec![
                    [sx * half, sy * face_half, sz * face_half],
                    [sx * face_half, sy * half, sz * face_half],
                    [sx * face_half, sy * face_half, sz * half],
                ]);
            }
        }
    }

    let mut vertices = Vec::new();
    for points in loops {
        let count = points.len();
        for index in 0..count {
            let next_index = (index + 1) % count;
            let previous_index = (index + count - 1) % count;
            let start = points[index];
            let end = points[next_index];
            let start_previous = points[previous_index];
            let end_next = points[(next_index + 1) % count];

            // Two triangles per segment. Culling is off in the line pipeline,
            // so the winding only has to cover the quad.
            for (position, previous, next, side) in [
                (start, start_previous, end, -1.0),
                (start, start_previous, end, 1.0),
                (end, start, end_next, -1.0),
                (end, start, end_next, -1.0),
                (start, start_previous, end, 1.0),
                (end, start, end_next, 1.0),
            ] {
                vertices.push(EdgeVertex {
                    position,
                    previous,
                    next,
                    side,
                });
            }
        }
    }
    vertices
}

/// Anti-aliasing: both passes draw into one shared 4x multisampled colour
/// target, and the cube pass resolves it into the exported image. No separate
/// AA pass and no additional 1x colour target are needed.
const SAMPLES: u32 = 4;

/// Half the edge-line width in pixels; the quads are expanded both sides.
const EDGE_HALF_WIDTH: f32 = 0.75;

/// The shared multisampled colour target.
///
/// `cad_graph`'s rebuilder constructs fresh passes on every resize, so this
/// texture's size is fixed for a pass instance's lifetime.
pub(crate) struct MsaaTarget {
    #[allow(dead_code)]
    texture: wgpu30::Texture,
    view: wgpu30::TextureView,
    /// The cube pass writes this so the edge overlay can hide back edges.
    #[allow(dead_code)]
    depth_texture: wgpu30::Texture,
    depth_view: wgpu30::TextureView,
}

impl MsaaTarget {
    fn new(
        device: &wgpu30::Device,
        format: wgpu30::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let texture = device.create_texture(&wgpu30::TextureDescriptor {
            label: Some("CAD MSAA Colour"),
            size: wgpu30::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: SAMPLES,
            dimension: wgpu30::TextureDimension::D2,
            format,
            usage: wgpu30::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu30::TextureViewDescriptor::default());
        let depth_texture = device.create_texture(&wgpu30::TextureDescriptor {
            label: Some("CAD MSAA Depth"),
            size: wgpu30::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: SAMPLES,
            dimension: wgpu30::TextureDimension::D2,
            format: wgpu30::TextureFormat::Depth32Float,
            usage: wgpu30::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu30::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            depth_texture,
            depth_view,
        }
    }

    fn view(&self) -> &wgpu30::TextureView {
        &self.view
    }

    fn depth_view(&self) -> &wgpu30::TextureView {
        &self.depth_view
    }
}

// ── Backdrop ─────────────────────────────────────────────────────────────────

pub struct GridPass {
    pipeline: wgpu30::RenderPipeline,
    bgl: wgpu30::BindGroupLayout,
    msaa: Arc<MsaaTarget>,
    bind_group: Option<wgpu30::BindGroup>,
    bind_group_key: Option<wgpu30::Buffer>,
}

impl GridPass {
    pub fn new(
        device: &wgpu30::Device,
        surface_format: wgpu30::TextureFormat,
        msaa: Arc<MsaaTarget>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu30::ShaderModuleDescriptor {
            label: Some("CAD Grid Shader"),
            source: wgpu30::ShaderSource::Wgsl(include_str!("../shaders/cad_grid.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu30::BindGroupLayoutDescriptor {
            label: Some("CAD Grid BGL"),
            entries: &[wgpu30::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu30::ShaderStages::FRAGMENT,
                ty: wgpu30::BindingType::Buffer {
                    ty: wgpu30::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu30::PipelineLayoutDescriptor {
            label: Some("CAD Grid Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu30::RenderPipelineDescriptor {
            label: Some("CAD Grid Pipeline"),
            layout: Some(&layout),
            vertex: wgpu30::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu30::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu30::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu30::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu30::ColorWrites::ALL,
                })],
                compilation_options: wgpu30::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu30::PrimitiveState {
                topology: wgpu30::PrimitiveTopology::TriangleList,
                front_face: wgpu30::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            // The backdrop covers every pixel, so it needs no depth buffer.
            depth_stencil: None,
            multisample: wgpu30::MultisampleState {
                count: SAMPLES,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            msaa,
            bind_group: None,
            bind_group_key: None,
        }
    }
}

impl RenderPass for GridPass {
    fn name(&self) -> &'static str {
        "CAD Grid"
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        _target: &'a wgpu30::TextureView,
        _depth: &'a wgpu30::TextureView,
        _resources: &'a ResourceRegistry<'a>,
        storage: &'a mut RenderFrameStorage,
    ) -> Option<wgpu30::RenderPassDescriptor<'a>> {
        // Opaque clear of the shared MSAA target: the grid pass is the first
        // writer, and the cube pass loads its samples.
        let color_attachments: &'a [Option<wgpu30::RenderPassColorAttachment<'a>>] = storage
            .retain_boxed_slice(Box::new([Some(wgpu30::RenderPassColorAttachment {
                view: self.msaa.view(),
                resolve_target: None,
                depth_slice: None,
                ops: wgpu30::Operations {
                    load: wgpu30::LoadOp::Clear(wgpu30::Color {
                        r: 0.004,
                        g: 0.005,
                        b: 0.008,
                        a: 1.0,
                    }),
                    store: wgpu30::StoreOp::Store,
                },
            })]));
        Some(wgpu30::RenderPassDescriptor {
            label: Some("CAD Grid"),
            color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // The camera buffer is stable after construction, but a resized or
        // rebuilt renderer hands over a new one; rebuild the group only then.
        if self.bind_group_key.as_ref() != Some(ctx.camera) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu30::BindGroupDescriptor {
                label: Some("CAD Grid BG"),
                layout: &self.bgl,
                entries: &[wgpu30::BindGroupEntry {
                    binding: 0,
                    resource: ctx.camera.as_entire_binding(),
                }],
            }));
            self.bind_group_key = Some(ctx.camera.clone());
        }

        let mut pass = ctx.render_cmds().unwrap();
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

// ── Cube ─────────────────────────────────────────────────────────────────────

pub struct CadPass {
    pipeline: wgpu30::RenderPipeline,
    bgl: wgpu30::BindGroupLayout,
    msaa: Arc<MsaaTarget>,
    /// Written once at construction and never again.
    params: wgpu30::Buffer,
    bind_group: Option<wgpu30::BindGroup>,
    bind_group_key: Option<wgpu30::Buffer>,
    vertices: wgpu30::Buffer,
    indices: wgpu30::Buffer,
    index_count: u32,
    /// Feature-edge quads and their pipeline.
    edges: wgpu30::Buffer,
    edge_count: u32,
    edge_pipeline: wgpu30::RenderPipeline,
    edge_bgl: wgpu30::BindGroupLayout,
    edge_bind_group: Option<wgpu30::BindGroup>,
    /// xy: target size in pixels, z: line half-width.
    viewport: wgpu30::Buffer,
    viewport_size: (u32, u32),
}

impl CadPass {
    pub fn new(
        device: &wgpu30::Device,
        surface_format: wgpu30::TextureFormat,
        msaa: Arc<MsaaTarget>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu30::ShaderModuleDescriptor {
            label: Some("CAD Cube Shader"),
            source: wgpu30::ShaderSource::Wgsl(include_str!("../shaders/cad_cube.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu30::BindGroupLayoutDescriptor {
            label: Some("CAD Cube BGL"),
            entries: &[
                wgpu30::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu30::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu30::BindingType::Buffer {
                        ty: wgpu30::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu30::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu30::ShaderStages::FRAGMENT,
                    ty: wgpu30::BindingType::Buffer {
                        ty: wgpu30::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu30::PipelineLayoutDescriptor {
            label: Some("CAD Cube Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu30::RenderPipelineDescriptor {
            label: Some("CAD Cube Pipeline"),
            layout: Some(&layout),
            vertex: wgpu30::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu30::VertexBufferLayout {
                    array_stride: std::mem::size_of::<CadVertex>() as u64,
                    step_mode: wgpu30::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu30::VertexAttribute {
                            format: wgpu30::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu30::VertexAttribute {
                            format: wgpu30::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                })],
                compilation_options: wgpu30::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu30::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu30::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu30::ColorWrites::ALL,
                })],
                compilation_options: wgpu30::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu30::PrimitiveState {
                topology: wgpu30::PrimitiveTopology::TriangleList,
                front_face: wgpu30::FrontFace::Ccw,
                cull_mode: Some(wgpu30::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu30::DepthStencilState {
                format: wgpu30::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(depth::NEARER),
                stencil: wgpu30::StencilState::default(),
                bias: wgpu30::DepthBiasState::default(),
            }),
            multisample: wgpu30::MultisampleState {
                count: SAMPLES,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });

        let (vertices, indices) = chamfered_cube(CUBE_HALF, CUBE_CHAMFER);
        let vertex_buffer = device.create_buffer_init(&wgpu30::util::BufferInitDescriptor {
            label: Some("CAD Cube VB"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu30::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu30::util::BufferInitDescriptor {
            label: Some("CAD Cube IB"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu30::BufferUsages::INDEX,
        });
        let params = device.create_buffer_init(&wgpu30::util::BufferInitDescriptor {
            label: Some("CAD Cube Params"),
            contents: bytemuck::bytes_of(&CadParams::gold()),
            usage: wgpu30::BufferUsages::UNIFORM,
        });

        let edge_shader = device.create_shader_module(wgpu30::ShaderModuleDescriptor {
            label: Some("CAD Edges Shader"),
            source: wgpu30::ShaderSource::Wgsl(include_str!("../shaders/cad_edges.wgsl").into()),
        });
        let edge_bgl = device.create_bind_group_layout(&wgpu30::BindGroupLayoutDescriptor {
            label: Some("CAD Edges BGL"),
            entries: &[
                wgpu30::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu30::ShaderStages::VERTEX,
                    ty: wgpu30::BindingType::Buffer {
                        ty: wgpu30::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu30::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu30::ShaderStages::VERTEX,
                    ty: wgpu30::BindingType::Buffer {
                        ty: wgpu30::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let edge_layout = device.create_pipeline_layout(&wgpu30::PipelineLayoutDescriptor {
            label: Some("CAD Edges Pipeline Layout"),
            bind_group_layouts: &[Some(&edge_bgl)],
            immediate_size: 0,
        });
        let edge_pipeline = device.create_render_pipeline(&wgpu30::RenderPipelineDescriptor {
            label: Some("CAD Edges Pipeline"),
            layout: Some(&edge_layout),
            vertex: wgpu30::VertexState {
                module: &edge_shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu30::VertexBufferLayout {
                    array_stride: std::mem::size_of::<EdgeVertex>() as u64,
                    step_mode: wgpu30::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu30::VertexAttribute {
                            format: wgpu30::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu30::VertexAttribute {
                            format: wgpu30::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu30::VertexAttribute {
                            format: wgpu30::VertexFormat::Float32x3,
                            offset: 24,
                            shader_location: 2,
                        },
                        wgpu30::VertexAttribute {
                            format: wgpu30::VertexFormat::Float32,
                            offset: 36,
                            shader_location: 3,
                        },
                    ],
                })],
                compilation_options: wgpu30::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu30::FragmentState {
                module: &edge_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu30::ColorTargetState {
                    format: surface_format,
                    // Additive: the line brightens whatever it crosses, so on a
                    // lit chamfer it reads as a soft specular glint and on a
                    // dark edge as a warm, faint line -- not a white wireframe.
                    blend: Some(wgpu30::BlendState {
                        color: wgpu30::BlendComponent {
                            src_factor: wgpu30::BlendFactor::One,
                            dst_factor: wgpu30::BlendFactor::One,
                            operation: wgpu30::BlendOperation::Add,
                        },
                        alpha: wgpu30::BlendComponent {
                            src_factor: wgpu30::BlendFactor::One,
                            dst_factor: wgpu30::BlendFactor::One,
                            operation: wgpu30::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu30::ColorWrites::ALL,
                })],
                compilation_options: wgpu30::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu30::PrimitiveState {
                topology: wgpu30::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu30::DepthStencilState {
                format: wgpu30::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(depth::NEARER_EQUAL),
                stencil: wgpu30::StencilState::default(),
                // Pull the quads towards the camera so they win against the
                // coplanar facet they trace.
                bias: depth::bias_toward_camera(1, 1.0, 0.0),
            }),
            multisample: wgpu30::MultisampleState {
                count: SAMPLES,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });
        let edge_vertices = chamfered_cube_edges(CUBE_HALF, CUBE_CHAMFER);
        let edge_buffer = device.create_buffer_init(&wgpu30::util::BufferInitDescriptor {
            label: Some("CAD Edges VB"),
            contents: bytemuck::cast_slice(&edge_vertices),
            usage: wgpu30::BufferUsages::VERTEX,
        });
        let viewport = device.create_buffer_init(&wgpu30::util::BufferInitDescriptor {
            label: Some("CAD Edges Viewport"),
            contents: bytemuck::cast_slice(&[1.0_f32, 1.0, EDGE_HALF_WIDTH, 0.0]),
            usage: wgpu30::BufferUsages::UNIFORM | wgpu30::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            bgl,
            msaa,
            params,
            bind_group: None,
            bind_group_key: None,
            vertices: vertex_buffer,
            indices: index_buffer,
            index_count: indices.len() as u32,
            edges: edge_buffer,
            edge_count: edge_vertices.len() as u32,
            edge_pipeline,
            edge_bgl,
            edge_bind_group: None,
            viewport,
            viewport_size: (0, 0),
        }
    }
}

impl RenderPass for CadPass {
    fn name(&self) -> &'static str {
        "CAD Cube"
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        target: &'a wgpu30::TextureView,
        _depth: &'a wgpu30::TextureView,
        _resources: &'a ResourceRegistry<'a>,
        storage: &'a mut RenderFrameStorage,
    ) -> Option<wgpu30::RenderPassDescriptor<'a>> {
        // Load the grid pass's samples and resolve them into the exported
        // image when the pass ends. The graph records passes in declaration
        // order and neither pass declares a read or write of the other's
        // resources, so the two are never fused into a subpass chain and the
        // load is a real, ordered load.
        //
        // Our own depth target, at the exported size and sample count: the
        // cube writes it and the edge overlay tests against it, so hidden
        // edges are removed. Helio's depth texture is at the internal render
        // scale and would not match this attachment.
        let depth_attachment = wgpu30::RenderPassDepthStencilAttachment {
            view: self.msaa.depth_view(),
            depth_ops: Some(wgpu30::Operations {
                load: wgpu30::LoadOp::Clear(depth::CLEAR),
                store: wgpu30::StoreOp::Store,
            }),
            stencil_ops: None,
        };
        let color_attachments: &'a [Option<wgpu30::RenderPassColorAttachment<'a>>] = storage
            .retain_boxed_slice(Box::new([Some(wgpu30::RenderPassColorAttachment {
                view: self.msaa.view(),
                resolve_target: Some(target),
                depth_slice: None,
                ops: wgpu30::Operations {
                    load: wgpu30::LoadOp::Load,
                    // The resolve reads the samples; keeping them past that
                    // is pure bandwidth.
                    store: wgpu30::StoreOp::Discard,
                },
            })]));
        Some(wgpu30::RenderPassDescriptor {
            label: Some("CAD Cube"),
            color_attachments,
            depth_stencil_attachment: Some(depth_attachment),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.bind_group_key.as_ref() != Some(ctx.camera) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu30::BindGroupDescriptor {
                label: Some("CAD Cube BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu30::BindGroupEntry {
                        binding: 0,
                        resource: ctx.camera.as_entire_binding(),
                    },
                    wgpu30::BindGroupEntry {
                        binding: 1,
                        resource: self.params.as_entire_binding(),
                    },
                ],
            }));
            self.edge_bind_group =
                Some(ctx.device.create_bind_group(&wgpu30::BindGroupDescriptor {
                    label: Some("CAD Edges BG"),
                    layout: &self.edge_bgl,
                    entries: &[
                        wgpu30::BindGroupEntry {
                            binding: 0,
                            resource: ctx.camera.as_entire_binding(),
                        },
                        wgpu30::BindGroupEntry {
                            binding: 1,
                            resource: self.viewport.as_entire_binding(),
                        },
                    ],
                }));
            self.bind_group_key = Some(ctx.camera.clone());
        }

        // The line shader needs the pixel size to keep its width constant.
        if self.viewport_size != (ctx.width, ctx.height) {
            self.viewport_size = (ctx.width, ctx.height);
            ctx.queue.write_buffer(
                &self.viewport,
                0,
                bytemuck::cast_slice(&[ctx.width as f32, ctx.height as f32, EDGE_HALF_WIDTH, 0.0]),
            );
        }

        let mut pass = ctx.render_cmds().unwrap();
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu30::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);

        // Feature-edge overlay: the cube has already filled depth, so back
        // edges fail the comparison and only the visible ones are drawn.
        pass.set_pipeline(&self.edge_pipeline);
        pass.set_bind_group(0, self.edge_bind_group.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, self.edges.slice(..));
        pass.draw(0..self.edge_count, 0..1);
        Ok(())
    }
}

// ── Graph ────────────────────────────────────────────────────────────────────

/// The whole render graph: backdrop, then cube. Both passes render into one
/// shared 4x MSAA target, which the cube pass resolves into the exported
/// image: two draws and one resolve, with no post-process AA pass.
pub fn cad_graph(ctx: PassBuildContext<'_>) -> RenderGraph {
    let msaa = Arc::new(MsaaTarget::new(
        ctx.device,
        ctx.config.surface_format,
        ctx.config.width,
        ctx.config.height,
    ));
    let mut graph = RenderGraph::new_with_external_device(ctx.device, ctx.queue);
    graph.add_pass(Box::new(GridPass::new(
        ctx.device,
        ctx.config.surface_format,
        Arc::clone(&msaa),
    )));
    graph.add_pass(Box::new(CadPass::new(
        ctx.device,
        ctx.config.surface_format,
        msaa,
    )));

    // A resize rebuilds the graph through this closure; it must lock the new
    // graph itself, because `RendererBuilder::build` panics on an unlocked one.
    let rebuilder: GraphRebuilder = Arc::new(
        |device, queue, config, _debug_state, _camera, _debug_camera, _cull| {
            let msaa = Arc::new(MsaaTarget::new(
                device,
                config.surface_format,
                config.width,
                config.height,
            ));
            let mut graph = RenderGraph::new_with_external_device(device, queue);
            graph.add_pass(Box::new(GridPass::new(
                device,
                config.surface_format,
                Arc::clone(&msaa),
            )));
            graph.add_pass(Box::new(CadPass::new(device, config.surface_format, msaa)));
            graph.lock(config.internal_width(), config.internal_height());
            graph
        },
    );
    graph.set_graph_data(rebuilder);
    graph.lock(ctx.config.internal_width(), ctx.config.internal_height());
    graph
}
