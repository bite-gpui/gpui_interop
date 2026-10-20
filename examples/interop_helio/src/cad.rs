//! A CAD-viewer scene built entirely from custom Helio passes.
//!
//! Two passes, no engine graph:
//!
//! 1. [`GridPass`] covers the viewport with a dark gradient and a perspective
//!    grid floor. It is a full-screen triangle whose fragment shader rebuilds
//!    the view ray from the camera uniform and intersects it with the floor
//!    plane; there is no grid texture.
//! 2. [`CadPass`] draws one rounded, gold, metallic cube with Cook-Torrance
//!    direct lighting and a small analytic studio environment for reflections.
//!
//! There is no shadow, GI, ambient-occlusion, SSR, atmosphere or post-process
//! pass, and no per-frame CPU work beyond recording two draws: the mesh
//! (generated once on the CPU at construction), the vertex/index buffers, the
//! light/material block and both pipelines are all created once. Helio already
//! writes the camera uniform each frame, so nothing in this module touches a
//! buffer after the first frame.

use std::sync::Arc;

use helio::{GraphRebuilder, PassBuildContext, RenderGraph, RenderPass};
use helio_core::{
    PassContext, PrepareContext, RenderFrameStorage, ResourceRegistry, Result as HelioResult,
};
use wgpu30::util::DeviceExt;

/// The cube: half-extent (so the edge is `2 * HALF`), bevel radius and the
/// number of quads per cube edge before the edge-dense warp.
const CUBE_HALF: f32 = 1.1;
const CUBE_RADIUS: f32 = 0.16;
const CUBE_DIVISIONS: u32 = 48;

/// Vertex layout: position(12) | normal(12).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CadVertex {
    position: [f32; 3],
    normal: [f32; 3],
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
            roughness_metallic: [0.18, 1.0, 0.0, 0.0],
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

/// `[-1, 1] -> [-1, 1]`, dense near the ends.
///
/// A uniform grid spends most of its quads on the flat faces and only a couple
/// on a bevel 7% of the face wide. Warping the parameter concentrates samples
/// exactly where the curvature is, so a 48-quad edge resolves the bevel with
/// roughly a dozen quads and leaves the flat centre coarse.
fn warp(t: f32) -> f32 {
    t.signum() * (1.0 - (1.0 - t.abs()).powi(3))
}

/// A rounded box: the Minkowski sum of a box of half-extent `half - radius`
/// and a sphere of `radius`.
///
/// Each of the six outer faces is sampled on a warped grid; a sample `p` is
/// snapped to the inner box (`q`) and pushed back out by `radius` along
/// `p - q`. That offset is exactly the rounded box's surface normal, so the
/// bevel and corner normals are smooth without a normal-smoothing pass.
fn rounded_cube(half: f32, radius: f32, divisions: u32) -> (Vec<CadVertex>, Vec<u32>) {
    let inner = half - radius;
    // (outward normal, u axis, v axis) with u × v = normal, so every face is
    // wound counter-clockwise seen from outside.
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];

    let stride = divisions + 1;
    let mut vertices = Vec::with_capacity((6 * stride * stride) as usize);
    let mut indices = Vec::with_capacity((6 * divisions * divisions * 6) as usize);

    for (normal, u_axis, v_axis) in faces {
        let base = vertices.len() as u32;
        for j in 0..=divisions {
            let v = warp(-1.0 + 2.0 * j as f32 / divisions as f32) * half;
            for i in 0..=divisions {
                let u = warp(-1.0 + 2.0 * i as f32 / divisions as f32) * half;
                let p = [
                    normal[0] * half + u_axis[0] * u + v_axis[0] * v,
                    normal[1] * half + u_axis[1] * u + v_axis[1] * v,
                    normal[2] * half + u_axis[2] * u + v_axis[2] * v,
                ];
                let q = [
                    p[0].clamp(-inner, inner),
                    p[1].clamp(-inner, inner),
                    p[2].clamp(-inner, inner),
                ];
                let offset = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                let length =
                    (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2])
                        .sqrt()
                        .max(1e-6);
                let n = [offset[0] / length, offset[1] / length, offset[2] / length];
                vertices.push(CadVertex {
                    position: [
                        q[0] + n[0] * radius,
                        q[1] + n[1] * radius,
                        q[2] + n[2] * radius,
                    ],
                    normal: n,
                });
            }
        }

        for j in 0..divisions {
            for i in 0..divisions {
                let v00 = base + j * stride + i;
                let v10 = v00 + 1;
                let v01 = v00 + stride;
                let v11 = v01 + 1;
                indices.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
            }
        }
    }

    (vertices, indices)
}

// ── Backdrop ─────────────────────────────────────────────────────────────────

pub struct GridPass {
    pipeline: wgpu30::RenderPipeline,
    bgl: wgpu30::BindGroupLayout,
    bind_group: Option<wgpu30::BindGroup>,
    bind_group_key: Option<wgpu30::Buffer>,
}

impl GridPass {
    pub fn new(device: &wgpu30::Device, surface_format: wgpu30::TextureFormat) -> Self {
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
            multisample: wgpu30::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bgl,
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
        target: &'a wgpu30::TextureView,
        _depth: &'a wgpu30::TextureView,
        _resources: &'a ResourceRegistry<'a>,
        storage: &'a mut RenderFrameStorage,
    ) -> Option<wgpu30::RenderPassDescriptor<'a>> {
        // Opaque clear: the grid pass is the first writer, and the cube pass
        // loads its result.
        let color_attachments: &'a [Option<wgpu30::RenderPassColorAttachment<'a>>] = storage
            .retain_boxed_slice(Box::new([Some(wgpu30::RenderPassColorAttachment {
                view: target,
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
    /// Written once at construction and never again.
    params: wgpu30::Buffer,
    bind_group: Option<wgpu30::BindGroup>,
    bind_group_key: Option<wgpu30::Buffer>,
    vertices: wgpu30::Buffer,
    indices: wgpu30::Buffer,
    index_count: u32,
}

impl CadPass {
    pub fn new(device: &wgpu30::Device, surface_format: wgpu30::TextureFormat) -> Self {
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
            depth_stencil: None,
            multisample: wgpu30::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let (vertices, indices) = rounded_cube(CUBE_HALF, CUBE_RADIUS, CUBE_DIVISIONS);
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

        Self {
            pipeline,
            bgl,
            params,
            bind_group: None,
            bind_group_key: None,
            vertices: vertex_buffer,
            indices: index_buffer,
            index_count: indices.len() as u32,
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
        // Load the grid pass's colour. The graph records passes in declaration
        // order and neither pass declares a read or write of the other's
        // resources, so the two are never fused into a subpass chain and the
        // load is a real, ordered load.
        //
        // No depth attachment: the pass draws one convex mesh with back-face
        // culling, which cannot self-occlude, and Helio's own depth texture is
        // at the internal render scale while this target is at output size.
        let color_attachments: &'a [Option<wgpu30::RenderPassColorAttachment<'a>>] = storage
            .retain_boxed_slice(Box::new([Some(wgpu30::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu30::Operations {
                    load: wgpu30::LoadOp::Load,
                    store: wgpu30::StoreOp::Store,
                },
            })]));
        Some(wgpu30::RenderPassDescriptor {
            label: Some("CAD Cube"),
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
            self.bind_group_key = Some(ctx.camera.clone());
        }

        let mut pass = ctx.render_cmds().unwrap();
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu30::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
        Ok(())
    }
}

// ── Graph ────────────────────────────────────────────────────────────────────

/// The whole render graph: backdrop, then cube. Both passes render straight
/// into the exported target, so the scene costs two draws and no blits.
pub fn cad_graph(ctx: PassBuildContext<'_>) -> RenderGraph {
    let mut graph = RenderGraph::new_with_external_device(ctx.device, ctx.queue);
    graph.add_pass(Box::new(GridPass::new(
        ctx.device,
        ctx.config.surface_format,
    )));
    graph.add_pass(Box::new(CadPass::new(
        ctx.device,
        ctx.config.surface_format,
    )));

    // A resize rebuilds the graph through this closure; it must lock the new
    // graph itself, because `RendererBuilder::build` panics on an unlocked one.
    let rebuilder: GraphRebuilder = Arc::new(
        |device, queue, config, _debug_state, _camera, _debug_camera, _cull| {
            let mut graph = RenderGraph::new_with_external_device(device, queue);
            graph.add_pass(Box::new(GridPass::new(device, config.surface_format)));
            graph.add_pass(Box::new(CadPass::new(device, config.surface_format)));
            graph.lock(config.internal_width(), config.internal_height());
            graph
        },
    );
    graph.set_graph_data(rebuilder);
    graph.lock(ctx.config.internal_width(), ctx.config.internal_height());
    graph
}
