//! `interop_helio`: a Helio scene inside a GPUI window.
//!
//! ```sh
//! cargo run -p interop_helio
//! ```
//!
//! Helio is not on crates.io yet, so this pins the revision
//! `4356a855e33b691aac10621e712bd973d8c4b68b` — the one the upstream
//! `Helio_Standalone_Demo` uses. Its renderer is built against **wgpu 30**;
//! GPUI's renderer is still **wgpu 29**. Cargo keeps both majors in one graph
//! because nothing in either renderer's type system crosses the boundary.
//!
//! # Windowing
//!
//! GPUI owns the window and selects the platform itself; in this session that is Wayland
//! (`WAYLAND_DISPLAY=wayland-0`). Helio is **not** tied to X11 or Wayland: it renders into an
//! offscreen Vulkan texture on its own device. X11 appears in this repository's debugging notes
//! only because it was used to inject pointer events while fixing the Bevy drag; it is not a
//! requirement of this example.
//!
//! # The hand-off
//!
//! Helio owns its wgpu 30 instance, adapter, device and queue, just as the
//! standalone demo does. It renders into an ordinary offscreen `wgpu 30`
//! texture. This first scaffold then:
//!
//! 1. copies that texture into a read-back buffer and waits for the GPU,
//! 2. copies the rows on the CPU into a tightly packed image, and
//! 3. writes it with `queue.write_texture` into a `wgpu 29` texture on GPUI's
//!    device, which GPUI samples with `paint_texture`.
//!
//! That is deliberately the simple bridge, not the fast one: it costs a
//! GPU→CPU→GPU round trip per frame. The next step is the dma-buf export the
//! other producer demos use, which needs the `WgpuSharedBuffer` Vulkan
//! allocation ported from wgpu 29's HAL to wgpu 30's HAL. Keeping the copy
//! here means the Helio integration can be proved independently of that port.
//!
//! # Scene
//!
//! The scene is the standalone demo's: a sky, a reflective floor, four
//! columns with spheres, four orbiting point lights, and a metallic crystal
//! spinner. It exercises Helio's default graph — GI, shadows, reflections,
//! post-processing — rather than a single forward pass.
//!
//! Drag to orbit the camera, scroll to zoom. Pause stops the clock but keeps
//! the last frame on screen. `--stats` prints the frame rate and the two copy
//! timings; `--probe` prints the copied image's centre pixel once.
//!
//! # Cost of the scaffold
//!
//! The read-back waits for the full 39-pass graph to finish. On an Intel HD
//! 520 at 677x686 that is 30–46 ms/frame in a debug build, so the window runs
//! at ~13 fps; the dma-buf export removes the wait and the CPU copy.

#[cfg(target_os = "linux")]
fn main() {
    demo::run();
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!(
        "interop_helio is a Linux example: Helio's renderer and the dma-buf path this stack grows \
         from are Vulkan-shaped. Run it on Linux."
    );
}

#[cfg(target_os = "linux")]
mod demo {
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use glam::{Mat4, Quat, Vec3};
    use gpui::{
        AnyElement, App, Bounds, Context, Corners, Entity, IntoElement, Render, ScrollDelta,
        Window, WindowBounds, WindowOptions, div, gpu_canvas, prelude::*, px, rgb, size,
    };
    use gpui_interop::{SurfaceEvent, SurfaceInput, SurfacePoint, attach, surface_input};
    use gpui_wgpu::{ImportedTextureExt as _, WgpuRenderer};
    use helio::{
        Camera, GpuLight, GpuMaterial, GroupMask, LightId, LightType, MaterialId, MeshId,
        MeshUpload, Movability, ObjectDescriptor, ObjectId, PackedVertex, Renderer,
        RendererBuilder, RendererConfig, SceneActor, SkyActor,
    };
    use helio_default_graphs::build_default_graph_external;
    use interop_demo::{
        ACCENT, LEFT_WIDTH, MUTED, control, error_panel, histogram, histogram_caption,
        histogram_style, metric_row, metrics_panel_style, overlay_badge, overlay_hint, pacing_note,
        page, panel, section_label, title, viewport_frame,
    };

    /// Helio's output format. GPUI samples `Rgba8UnormSrgb` directly.
    const FORMAT: wgpu30::TextureFormat = wgpu30::TextureFormat::Rgba8UnormSrgb;
    const GPUI_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    /// Orbit/zoom, the same feel as the other two demos.
    const SENSITIVITY: f32 = 0.007;
    const ZOOM_PER_NOTCH: f32 = 1.12;
    const PIXELS_PER_LINE: f32 = 40.0;

    /// The camera's defaults, and the scene's focus.
    const DEFAULT_YAW: f32 = 0.55;
    const DEFAULT_PITCH: f32 = 0.32;
    const DEFAULT_DISTANCE: f32 = 11.5;
    const TARGET: Vec3 = Vec3::new(0.0, 2.0, 0.0);

    struct Cli {
        list: bool,
        preference: Option<gpui::GpuPreference>,
        adapter: Option<usize>,
        probe: bool,
        stats: bool,
    }

    fn parse_args() -> Cli {
        let mut cli = Cli {
            list: false,
            preference: None,
            adapter: None,
            probe: false,
            stats: false,
        };
        let mut args = std::env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--list-gpus" => cli.list = true,
                "--probe" => cli.probe = true,
                "--stats" => cli.stats = true,
                "--gpu" => {
                    let value = args
                        .next()
                        .unwrap_or_else(|| usage_error("--gpu needs an adapter index"));
                    let index: usize = value.parse().unwrap_or_else(|_| {
                        usage_error(&format!("--gpu: {value:?} is not an adapter index"))
                    });
                    cli.preference = Some(gpui::GpuPreference::Custom(Arc::new(
                        move |gpus: &[gpui::GpuAdapterInfo]| (index < gpus.len()).then_some(index),
                    )));
                    cli.adapter = Some(index);
                }
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => usage_error(&format!("unknown argument {other:?}")),
            }
        }
        cli
    }

    fn usage_error(message: &str) -> ! {
        eprintln!("interop_helio: {message}\n");
        print_help();
        std::process::exit(2);
    }

    fn print_help() {
        println!(concat!(
            "interop_helio — a Helio renderer inside a GPUI window\n",
            "\n",
            "Usage: interop_helio [OPTIONS]\n",
            "\n",
            "  --list-gpus          enumerate the adapters this process can render on, then exit\n",
            "  --gpu <INDEX>        use the adapter at INDEX from --list-gpus\n",
            "  --stats              print the frame rate and the copy timings once a second\n",
            "  --probe              print the copied image's centre pixel and mean once\n",
            "  -h, --help           show this help\n",
            "\n",
            "The first scaffold copies Helio's wgpu 30 frame to GPUI's wgpu 29 texture on the CPU;\n",
            "the dma-buf export is the next step.",
        ));
    }

    fn list_gpus(cx: &App) {
        let gpus = cx.enumerate_gpus();
        if gpus.is_empty() {
            println!("no adapters reported");
            return;
        }
        println!(
            "{:<3} {:<46} {:<8} {:<11} path",
            "#", "adapter", "backend", "type"
        );
        for (index, gpu) in gpus.iter().enumerate() {
            let software = if gpu.is_software { " (software)" } else { "" };
            println!(
                "{index:<3} {:<46} {:<8} {:<11} cpu-copy{software}",
                gpu.name,
                format!("{:?}", gpu.backend),
                format!("{:?}", gpu.device_type),
            );
        }
        println!(
            "\nHelio owns its own wgpu 30 device; the window's renderer is wgpu 29. Select with \
             --gpu <#>."
        );
    }

    /// One snapshot of the numbers the right panel shows.
    #[derive(Clone, Default)]
    struct Metrics {
        frames: u64,
        interval_ms: f32,
        fps: f32,
        render_ms: f32,
        readback_ms: f32,
        upload_ms: f32,
        width: u32,
        height: u32,
        adapter: String,
        intervals: VecDeque<f32>,
        last: Option<Instant>,
    }

    impl Metrics {
        fn record(
            &mut self,
            render: Duration,
            readback: Duration,
            upload: Duration,
            width: u32,
            height: u32,
        ) {
            self.frames += 1;
            self.render_ms = render.as_secs_f32() * 1000.0;
            self.readback_ms = readback.as_secs_f32() * 1000.0;
            self.upload_ms = upload.as_secs_f32() * 1000.0;
            self.width = width;
            self.height = height;

            let now = Instant::now();
            if let Some(last) = self.last.replace(now) {
                self.intervals
                    .push_back((now - last).as_secs_f32() * 1000.0);
                while self.intervals.len() > 120 {
                    self.intervals.pop_front();
                }
            }
            self.interval_ms = self.intervals.back().copied().unwrap_or(0.0);
            self.fps = if self.intervals.is_empty() {
                0.0
            } else {
                let average = self.intervals.iter().sum::<f32>() / self.intervals.len() as f32;
                1000.0 / average
            };
        }
    }

    /// The Helio producer plus the two textures that bridge wgpu 30 to wgpu 29.
    struct HelioScene {
        device: Arc<wgpu30::Device>,
        queue: Arc<wgpu30::Queue>,
        renderer: Renderer,
        target: wgpu30::Texture,
        target_view: wgpu30::TextureView,
        readback: wgpu30::Buffer,
        width: u32,
        height: u32,
        spinner: ObjectId,
        lights: [LightId; 4],
        started: Instant,
        adapter: String,
        gpui_device: Arc<wgpu::Device>,
        gpui_queue: Arc<wgpu::Queue>,
        gpui_texture: wgpu::Texture,
        gpui_view: wgpu::TextureView,
        pixels: Vec<u8>,
        frames: u64,
        probe: bool,
    }

    impl HelioScene {
        fn build(
            gpu: &mut gpui::GpuCanvasContext,
            width: u32,
            height: u32,
            adapter_index: Option<usize>,
            probe: bool,
        ) -> anyhow::Result<Self> {
            let interop = attach(gpu.try_device::<WgpuRenderer>())
                .map_err(|unavailable| anyhow::anyhow!("{unavailable:?}"))?;
            let (gpui_device, gpui_queue) = interop
                .adapter()
                .wgpu()
                .ok_or_else(|| anyhow::anyhow!("the window's renderer is not wgpu"))?;

            let (gpui_texture, gpui_view) = gpui_target(&gpui_device, width, height)?;

            let instance = wgpu30::Instance::new(wgpu30::InstanceDescriptor {
                backends: wgpu30::Backends::VULKAN | wgpu30::Backends::GL,
                flags: wgpu30::InstanceFlags::empty(),
                ..wgpu30::InstanceDescriptor::new_without_display_handle()
            });
            let adapters = pollster::block_on(
                instance.enumerate_adapters(wgpu30::Backends::VULKAN | wgpu30::Backends::GL),
            );
            if adapters.is_empty() {
                anyhow::bail!("Helio found no wgpu 30 adapter");
            }
            let index = adapter_index
                .filter(|index| *index < adapters.len())
                .unwrap_or_else(|| {
                    adapters
                        .iter()
                        .position(|adapter| {
                            adapter.get_info().device_type != wgpu30::DeviceType::Cpu
                        })
                        .unwrap_or(0)
                });
            let adapter = &adapters[index];
            let info = adapter.get_info();
            eprintln!(
                "interop_helio: Helio on {} ({:?}), GPUI on wgpu 29, CPU copy",
                info.name, info.backend
            );

            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu30::DeviceDescriptor {
                    label: Some("interop_helio"),
                    required_features: helio::required_wgpu_features(adapter.features()),
                    required_limits: helio::required_wgpu_limits(adapter.limits()),
                    experimental_features: helio::required_experimental_features(
                        adapter.features(),
                    ),
                    ..Default::default()
                }))?;
            let (device, queue) = (Arc::new(device), Arc::new(queue));

            let mut renderer = RendererBuilder::new(RendererConfig::new(width, height, FORMAT))
                .with_graph(Box::new(|d, q, s, c, ds, cb, csb| {
                    build_default_graph_external(d, q, s, c, ds, cb, csb, None)
                }))
                .build(device.clone(), queue.clone(), width, height, FORMAT);

            let (spinner, lights) = populate(&mut renderer);

            let (target, target_view) = helio_target(&device, width, height)?;
            let readback = readback_buffer(&device, width, height);
            let pixels = vec![0u8; width as usize * height as usize * 4];

            Ok(Self {
                device,
                queue,
                renderer,
                target,
                target_view,
                readback,
                width,
                height,
                spinner,
                lights,
                started: Instant::now(),
                adapter: info.name,
                gpui_device,
                gpui_queue,
                gpui_texture,
                gpui_view,
                pixels,
                frames: 0,
                probe,
            })
        }

        fn resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
            let width = width.max(1);
            let height = height.max(1);
            if (width, height) == (self.width, self.height) {
                return Ok(());
            }

            self.renderer.set_render_size(width, height);
            let (target, target_view) = helio_target(&self.device, width, height)?;
            self.target = target;
            self.target_view = target_view;
            self.readback = readback_buffer(&self.device, width, height);
            let (texture, view) = gpui_target(&self.gpui_device, width, height)?;
            self.gpui_texture = texture;
            self.gpui_view = view;
            self.pixels = vec![0u8; width as usize * height as usize * 4];
            self.width = width;
            self.height = height;
            Ok(())
        }

        fn camera(&self, yaw: f32, pitch: f32, distance: f32) -> Camera {
            let eye = TARGET
                + Vec3::new(
                    distance * pitch.cos() * yaw.sin(),
                    distance * pitch.sin(),
                    distance * pitch.cos() * yaw.cos(),
                );
            Camera::perspective_look_at(
                eye,
                TARGET,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                self.width as f32 / self.height as f32,
                0.1,
                200.0,
            )
        }

        fn animate(&mut self, elapsed: f32) {
            let rot = Quat::from_rotation_y(elapsed * 0.8)
                * Quat::from_rotation_x((elapsed * 0.5).sin() * 0.3);
            let _ = self.renderer.scene_mut().update_object_transform(
                self.spinner,
                Mat4::from_rotation_translation(rot, Vec3::new(0.0, 2.5, 0.0)),
            );

            let colors = [
                [1.0, 0.3, 0.3],
                [0.3, 1.0, 0.3],
                [0.3, 0.5, 1.0],
                [1.0, 0.8, 0.2],
            ];
            for (i, &light) in self.lights.iter().enumerate() {
                let angle = elapsed * 0.6 + i as f32 * std::f32::consts::FRAC_PI_2;
                let (x, z) = (angle.cos() * 4.0, angle.sin() * 4.0);
                let _ = self.renderer.scene_mut().update_light(
                    light,
                    pt_light(
                        [x, 5.0 + (elapsed * 1.2 + i as f32).sin() * 0.5, z],
                        colors[i],
                        10.0,
                        10.0,
                    ),
                );
            }
        }

        /// Render one frame, copy it across the version boundary, and return the three timings.
        fn render(&mut self, camera: &Camera) -> anyhow::Result<(Duration, Duration, Duration)> {
            let render_start = Instant::now();
            self.renderer
                .render(camera, &self.target_view)
                .map_err(|error| anyhow::anyhow!("Helio render: {error}"))?;
            let render = render_start.elapsed();

            let readback_start = Instant::now();
            let padded = padded_bytes_per_row(self.width);
            let mut encoder =
                self.device
                    .create_command_encoder(&wgpu30::CommandEncoderDescriptor {
                        label: Some("interop_helio_readback"),
                    });
            encoder.copy_texture_to_buffer(
                wgpu30::TexelCopyTextureInfo {
                    texture: &self.target,
                    mip_level: 0,
                    origin: wgpu30::Origin3d::ZERO,
                    aspect: wgpu30::TextureAspect::All,
                },
                wgpu30::TexelCopyBufferInfo {
                    buffer: &self.readback,
                    layout: wgpu30::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(self.height),
                    },
                },
                wgpu30::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(Some(encoder.finish()));
            self.device
                .poll(wgpu30::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(10)),
                })
                .map_err(|error| anyhow::anyhow!("waiting for the readback copy: {error}"))?;

            let slice = self.readback.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu30::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            self.device
                .poll(wgpu30::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(10)),
                })
                .map_err(|error| anyhow::anyhow!("waiting for the readback map: {error}"))?;
            receiver
                .recv()
                .map_err(|error| anyhow::anyhow!("readback channel: {error}"))?
                .map_err(|error| anyhow::anyhow!("mapping the readback buffer: {error}"))?;

            let row = self.width as usize * 4;
            {
                let mapped = slice
                    .get_mapped_range()
                    .map_err(|error| anyhow::anyhow!("mapping the readback range: {error}"))?;
                for y in 0..self.height as usize {
                    let source = y * padded as usize;
                    self.pixels[y * row..(y + 1) * row]
                        .copy_from_slice(&mapped[source..source + row]);
                }
            }
            self.readback.unmap();
            let readback = readback_start.elapsed();

            self.frames += 1;
            if self.probe && self.frames == 60 {
                let row = self.width as usize * 4;
                let centre_at = (self.height as usize / 2) * row + (self.width as usize / 2) * 4;
                let centre = &self.pixels[centre_at..centre_at + 4];
                let sum: u64 = self
                    .pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|[r, g, b, _]| u64::from(*r) + u64::from(*g) + u64::from(*b))
                    .sum();
                let samples = self.pixels.len() as u64 / 4 * 3;
                eprintln!(
                    "interop_helio: probe {}x{}: centre {:?}, mean channel {:.2}",
                    self.width,
                    self.height,
                    centre,
                    sum as f64 / samples.max(1) as f64,
                );
            }

            let upload_start = Instant::now();
            self.gpui_queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.gpui_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &self.pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.width * 4),
                    rows_per_image: Some(self.height),
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            let upload = upload_start.elapsed();

            Ok((render, readback, upload))
        }

        fn paint(&self, gpu: &mut gpui::GpuCanvasContext) -> anyhow::Result<()> {
            let handle = self.gpui_view.to_imported_handle()?;
            gpu.paint_texture(handle, Corners::default(), 1.0, false);
            Ok(())
        }
    }

    fn gpui_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> anyhow::Result<(wgpu::Texture, wgpu::TextureView)> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("interop_helio_gpui_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: GPUI_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("interop_helio_gpui_target_view"),
            ..Default::default()
        });
        Ok((texture, view))
    }

    fn helio_target(
        device: &wgpu30::Device,
        width: u32,
        height: u32,
    ) -> anyhow::Result<(wgpu30::Texture, wgpu30::TextureView)> {
        let texture = device.create_texture(&wgpu30::TextureDescriptor {
            label: Some("interop_helio_target"),
            size: wgpu30::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu30::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu30::TextureUsages::RENDER_ATTACHMENT | wgpu30::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu30::TextureViewDescriptor {
            label: Some("interop_helio_target_view"),
            ..Default::default()
        });
        Ok((texture, view))
    }

    fn readback_buffer(device: &wgpu30::Device, width: u32, height: u32) -> wgpu30::Buffer {
        let padded = padded_bytes_per_row(width);
        device.create_buffer(&wgpu30::BufferDescriptor {
            label: Some("interop_helio_readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu30::BufferUsages::COPY_DST | wgpu30::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    fn padded_bytes_per_row(width: u32) -> u32 {
        let unpadded = width * 4;
        unpadded.div_ceil(wgpu30::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu30::COPY_BYTES_PER_ROW_ALIGNMENT
    }

    /// The standalone demo's scene, unchanged in spirit: sky, floor, columns, spheres, four
    /// moving point lights, and a metallic crystal spinner.
    fn populate(renderer: &mut Renderer) -> (ObjectId, [LightId; 4]) {
        let gold = renderer.scene_mut().insert_material(make_mat(
            [0.95, 0.75, 0.25, 1.0],
            0.25,
            0.85,
            [0.0; 3],
            0.0,
        ));
        let marble = renderer.scene_mut().insert_material(make_mat(
            [0.85, 0.83, 0.80, 1.0],
            0.55,
            0.0,
            [0.0; 3],
            0.0,
        ));
        let crystal = renderer.scene_mut().insert_material(make_mat(
            [0.3, 0.6, 1.0, 1.0],
            0.05,
            0.1,
            [0.2, 0.4, 1.0],
            2.0,
        ));
        let floor = renderer.scene_mut().insert_material(make_mat(
            [0.22, 0.22, 0.25, 1.0],
            0.7,
            0.05,
            [0.0; 3],
            0.0,
        ));
        renderer.scene_mut().insert_actor(SceneActor::sky(
            SkyActor::new().with_sky_color([0.15, 0.25, 0.45]),
        ));

        let ground = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(plane_mesh(12.0)))
            .as_mesh()
            .unwrap();
        let _ = add_obj(renderer, ground, floor, Mat4::IDENTITY, 12.0, None);

        let pillar = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh({
                let mut mesh = cube_mesh(0.15);
                for vertex in &mut mesh.vertices {
                    vertex.position[1] *= 13.33;
                }
                mesh
            }))
            .as_mesh()
            .unwrap();
        let sphere = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(sphere_mesh(0.4)))
            .as_mesh()
            .unwrap();

        let colors = [
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.5, 1.0],
            [1.0, 0.8, 0.2],
        ];
        let mut lights = Vec::new();
        for (i, angle) in [
            0.0,
            90.0f32.to_radians(),
            180.0f32.to_radians(),
            270.0f32.to_radians(),
        ]
        .into_iter()
        .enumerate()
        {
            let (x, z) = (angle.cos() * 4.0, angle.sin() * 4.0);
            let _ = add_obj(
                renderer,
                pillar,
                marble,
                Mat4::from_translation(Vec3::new(x, 2.0, z)),
                0.15,
                None,
            );
            let _ = add_obj(
                renderer,
                sphere,
                gold,
                Mat4::from_translation(Vec3::new(x, 4.3, z)),
                0.4,
                None,
            );
            lights.push(
                renderer
                    .scene_mut()
                    .insert_actor(SceneActor::light(pt_light(
                        [x, 5.0, z],
                        colors[i],
                        8.0,
                        8.0,
                    )))
                    .as_light()
                    .unwrap(),
            );
        }

        let cube = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(cube_mesh(0.6)))
            .as_mesh()
            .unwrap();
        let spinner = add_obj(
            renderer,
            cube,
            crystal,
            Mat4::from_translation(Vec3::new(0.0, 2.5, 0.0)),
            0.6,
            Some(Movability::Movable),
        );

        (spinner, lights.try_into().unwrap())
    }

    fn make_mat(
        color: [f32; 4],
        roughness: f32,
        metal: f32,
        emit: [f32; 3],
        ei: f32,
    ) -> GpuMaterial {
        GpuMaterial {
            base_color: color,
            emissive: [emit[0], emit[1], emit[2], ei],
            roughness_metallic: [roughness, metal, 1.5, 0.5],
            tex_base_color: GpuMaterial::NO_TEXTURE,
            tex_normal: GpuMaterial::NO_TEXTURE,
            tex_roughness: GpuMaterial::NO_TEXTURE,
            tex_emissive: GpuMaterial::NO_TEXTURE,
            tex_occlusion: GpuMaterial::NO_TEXTURE,
            workflow: 0,
            flags: 0,
            material_class: 0,
            class_params: [0.0; 4],
        }
    }

    fn pt_light(pos: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> GpuLight {
        GpuLight {
            position_range: [pos[0], pos[1], pos[2], range],
            direction_outer: [0.0, 0.0, -1.0, 0.0],
            color_intensity: [color[0], color[1], color[2], intensity],
            shadow_index: 0,
            light_type: LightType::Point as u32,
            inner_angle: 0.0,
            _pad: 0,
            ..Default::default()
        }
    }

    fn add_obj(
        renderer: &mut Renderer,
        mesh: MeshId,
        material: MaterialId,
        transform: Mat4,
        radius: f32,
        movability: Option<Movability>,
    ) -> ObjectId {
        let id = renderer
            .scene_mut()
            .insert_actor(SceneActor::object(ObjectDescriptor {
                mesh,
                material,
                transform,
                bounds: [
                    transform.w_axis.x,
                    transform.w_axis.y,
                    transform.w_axis.z,
                    radius,
                ],
                flags: 3,
                groups: GroupMask::NONE,
                movability,
                user_tag: 0,
            }));
        id.as_object().unwrap()
    }

    type FaceQuads = (Vec<[usize; 4]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[f32; 3]>);
    type FaceArrays = (
        [[f32; 3]; 24],
        [[f32; 3]; 24],
        [[f32; 2]; 24],
        [[f32; 3]; 24],
    );

    fn cube_mesh(half: f32) -> MeshUpload {
        use PackedVertex as PV;
        let e = half;
        let (pos, norm, uv, tan): FaceArrays = {
            let f = |[a, b, cc, d]: [usize; 4], n: [f32; 3], t: [f32; 3]| {
                let u = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
                ([a, b, cc, d], [n; 4], u, [t; 4])
            };
            let (qi, n, u, t): FaceQuads = vec![
                f([0, 1, 2, 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
                f([5, 4, 7, 6], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]),
                f([4, 0, 3, 7], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
                f([1, 5, 6, 2], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
                f([3, 2, 6, 7], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
                f([4, 5, 1, 0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]),
            ]
            .into_iter()
            .fold(
                (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
                |(mut p, mut n, mut u, mut t), (q, ns, us, ts)| {
                    p.push(q);
                    n.extend(ns);
                    u.extend(us);
                    t.extend(ts);
                    (p, n, u, t)
                },
            );
            let cs = [
                [-e, -e, e],
                [e, -e, e],
                [e, e, e],
                [-e, e, e],
                [-e, -e, -e],
                [e, -e, -e],
                [e, e, -e],
                [-e, e, -e],
            ];
            let pts: Vec<[f32; 3]> = qi.iter().flat_map(|q| q.iter().map(|&i| cs[i])).collect();
            (
                pts.try_into().unwrap(),
                n.try_into().unwrap(),
                u.try_into().unwrap(),
                t.try_into().unwrap(),
            )
        };
        let vertices: Vec<PackedVertex> = (0..24)
            .map(|i| PV::from_components(pos[i], norm[i], uv[i], tan[i], 1.0))
            .collect();
        let mut indices = Vec::with_capacity(36);
        for f in 0..6 {
            let b = (f * 4) as u32;
            indices.extend([b, b + 1, b + 2, b, b + 2, b + 3]);
        }
        MeshUpload { vertices, indices }
    }

    fn sphere_mesh(radius: f32) -> MeshUpload {
        let (lat, lon) = (16, 32);
        let mut verts = Vec::new();
        let mut idx = Vec::new();
        for i in 0..=lat {
            let phi = std::f32::consts::PI * i as f32 / lat as f32;
            let (sp, cp) = phi.sin_cos();
            for j in 0..=lon {
                let theta = 2.0 * std::f32::consts::PI * j as f32 / lon as f32;
                let (st, ct) = theta.sin_cos();
                let n = [sp * ct, cp, sp * st];
                verts.push(PackedVertex::from_components(
                    [n[0] * radius, n[1] * radius, n[2] * radius],
                    n,
                    [j as f32 / lon as f32, i as f32 / lat as f32],
                    [-n[2], 0.0, n[0]],
                    1.0,
                ));
            }
        }
        for i in 0..lat {
            for j in 0..lon {
                let a = (i * (lon + 1) + j) as u32;
                let b = a + (lon + 1) as u32;
                idx.extend([a, a + 1, b, b, a + 1, b + 1]);
            }
        }
        MeshUpload {
            vertices: verts,
            indices: idx,
        }
    }

    fn plane_mesh(half: f32) -> MeshUpload {
        let p = |x, z| {
            PackedVertex::from_components(
                [x, 0.0, z],
                [0.0, 1.0, 0.0],
                [x / half * 0.5 + 0.5, z / half * 0.5 + 0.5],
                [1.0, 0.0, 0.0],
                1.0,
            )
        };
        MeshUpload {
            vertices: vec![
                p(-half, -half),
                p(half, -half),
                p(half, half),
                p(-half, half),
            ],
            indices: vec![0, 2, 1, 0, 3, 2],
        }
    }

    /// The right column: title, rate and the two copies.
    struct MetricsPanel {
        metrics: Rc<RefCell<Option<Metrics>>>,
    }

    impl Render for MetricsPanel {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let metrics = self.metrics.borrow().clone().unwrap_or_default();
            metrics_section(&metrics)
        }
    }

    /// The left column's cached interval histogram.
    struct HistogramPanel {
        metrics: Rc<RefCell<Option<Metrics>>>,
    }

    impl Render for HistogramPanel {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let metrics = self.metrics.borrow().clone().unwrap_or_default();
            histogram_section(&metrics)
        }
    }

    struct Demo {
        scene: Rc<RefCell<Option<HelioScene>>>,
        error: Rc<RefCell<Option<String>>>,
        last: Rc<Cell<Option<SurfacePoint>>>,
        grab: Rc<Cell<bool>>,
        paused: Rc<Cell<bool>>,
        yaw: Rc<Cell<f32>>,
        pitch: Rc<Cell<f32>>,
        distance: Rc<Cell<f32>>,
        metrics: Rc<RefCell<Metrics>>,
        panel_metrics: Rc<RefCell<Option<Metrics>>>,
        metrics_panel: Entity<MetricsPanel>,
        histogram_panel: Entity<HistogramPanel>,
        displayed_at: Option<Instant>,
        adapter: Option<usize>,
        probe: bool,
        stats: bool,
        logged: Rc<RefCell<Instant>>,
    }

    impl Demo {
        fn new(cx: &mut Context<Self>, adapter: Option<usize>, probe: bool, stats: bool) -> Self {
            let panel_metrics = Rc::new(RefCell::new(None));
            let metrics_panel = cx.new(|_| MetricsPanel {
                metrics: panel_metrics.clone(),
            });
            let histogram_panel = cx.new(|_| HistogramPanel {
                metrics: panel_metrics.clone(),
            });
            Self {
                scene: Rc::new(RefCell::new(None)),
                error: Rc::new(RefCell::new(None)),
                last: Rc::new(Cell::new(None)),
                grab: Rc::new(Cell::new(false)),
                paused: Rc::new(Cell::new(false)),
                yaw: Rc::new(Cell::new(DEFAULT_YAW)),
                pitch: Rc::new(Cell::new(DEFAULT_PITCH)),
                distance: Rc::new(Cell::new(DEFAULT_DISTANCE)),
                metrics: Rc::new(RefCell::new(Metrics::default())),
                panel_metrics,
                metrics_panel,
                histogram_panel,
                displayed_at: None,
                adapter,
                probe,
                stats,
                logged: Rc::new(RefCell::new(Instant::now())),
            }
        }
    }

    impl Render for Demo {
        fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            let failed = self.error.borrow().is_some();
            if !failed && !self.paused.get() {
                window.request_animation_frame();
            }

            if self
                .displayed_at
                .is_none_or(|at| at.elapsed() >= Duration::from_millis(500))
            {
                let mut metrics = self.metrics.borrow().clone();
                if let Some(scene) = self.scene.borrow().as_ref() {
                    metrics.adapter = scene.adapter.clone();
                }
                *self.panel_metrics.borrow_mut() = Some(metrics);
                self.displayed_at = Some(Instant::now());
                self.metrics_panel.update(cx, |_, cx| cx.notify());
                self.histogram_panel.update(cx, |_, cx| cx.notify());
            }

            page(
                controls_panel(
                    cx,
                    self.paused.clone(),
                    self.yaw.clone(),
                    self.pitch.clone(),
                    self.distance.clone(),
                    self.histogram_panel.clone(),
                ),
                viewport(
                    self.scene.clone(),
                    self.error.clone(),
                    self.metrics.clone(),
                    self.last.clone(),
                    self.grab.clone(),
                    self.paused.clone(),
                    self.yaw.clone(),
                    self.pitch.clone(),
                    self.distance.clone(),
                    self.adapter,
                    self.probe,
                    self.stats,
                    self.logged.clone(),
                ),
                self.metrics_panel.clone().cached(metrics_panel_style()),
            )
        }
    }

    fn controls_panel(
        cx: &mut Context<Demo>,
        paused: Rc<Cell<bool>>,
        yaw: Rc<Cell<f32>>,
        pitch: Rc<Cell<f32>>,
        distance: Rc<Cell<f32>>,
        histogram_panel: Entity<HistogramPanel>,
    ) -> AnyElement {
        let pause_label = if paused.get() { "Resume" } else { "Pause" };
        let pause =
            control("pause", pause_label).on_click(cx.listener(move |_this, _, _window, cx| {
                paused.set(!paused.get());
                cx.notify();
            }));

        let reset = control("reset", "Reset view").on_click(cx.listener(move |_this, _, _, cx| {
            yaw.set(DEFAULT_YAW);
            pitch.set(DEFAULT_PITCH);
            distance.set(DEFAULT_DISTANCE);
            cx.notify();
        }));

        panel(LEFT_WIDTH)
            .child(section_label("Controls"))
            .child(pause)
            .child(reset)
            .child(div().h(px(8.)).flex_none())
            .child(histogram_panel.cached(histogram_style()))
            .into_any_element()
    }

    fn histogram_section(metrics: &Metrics) -> AnyElement {
        let intervals: Vec<f32> = metrics.intervals.iter().copied().collect();
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap_1()
            .child(section_label("Frame interval"))
            .child(histogram(&intervals, None))
            .child(histogram_caption(false))
            .child(pacing_note(
                metrics.interval_ms,
                metrics.render_ms,
                metrics.interval_ms - metrics.render_ms,
                true,
            ))
            .into_any_element()
    }

    fn metrics_section(metrics: &Metrics) -> AnyElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap_1()
            .child(title(
                "Helio, inside GPUI",
                "Helio's wgpu 30 renderer, bridged to GPUI's wgpu 29 texture.",
            ))
            .child(div().h(px(2.)).flex_none())
            .child(
                div()
                    .text_2xl()
                    .text_color(rgb(ACCENT))
                    .child(format!("{:.1} FPS", metrics.fps)),
            )
            .child(div().text_xs().text_color(rgb(MUTED)).child(format!(
                "{:.2} ms · Helio {:.2} · copy {:.2} · upload {:.2}",
                metrics.interval_ms, metrics.render_ms, metrics.readback_ms, metrics.upload_ms
            )))
            .child(div().h(px(4.)).flex_none())
            .child(section_label("Frame"))
            .child(metric_row(
                "frame interval",
                format!("{:.2} ms", metrics.interval_ms),
            ))
            .child(metric_row(
                "Helio render",
                format!("{:.2} ms", metrics.render_ms),
            ))
            .child(metric_row(
                "readback",
                format!("{:.2} ms", metrics.readback_ms),
            ))
            .child(metric_row("upload", format!("{:.2} ms", metrics.upload_ms)))
            .child(div().h(px(4.)).flex_none())
            .child(section_label("Producer"))
            .child(metric_row(
                "size",
                format!("{} × {}", metrics.width, metrics.height),
            ))
            .child(metric_row("adapter", metrics.adapter.clone()))
            .child(metric_row("hand-off", "CPU copy · wgpu 30 → 29".to_owned()))
            .child(metric_row("frames", metrics.frames.to_string()))
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn viewport(
        scene: Rc<RefCell<Option<HelioScene>>>,
        error: Rc<RefCell<Option<String>>>,
        metrics: Rc<RefCell<Metrics>>,
        last: Rc<Cell<Option<SurfacePoint>>>,
        grab: Rc<Cell<bool>>,
        paused: Rc<Cell<bool>>,
        yaw: Rc<Cell<f32>>,
        pitch: Rc<Cell<f32>>,
        distance: Rc<Cell<f32>>,
        adapter: Option<usize>,
        probe: bool,
        stats: bool,
        logged: Rc<RefCell<Instant>>,
    ) -> AnyElement {
        let drawing_error = error.clone();
        let overlay_error = error;
        let paused_for_hint = paused.clone();
        let yaw_for_canvas = yaw.clone();
        let pitch_for_canvas = pitch.clone();
        let distance_for_camera = distance.clone();
        let canvas = gpu_canvas(move |gpu| {
            let bounds = gpu.bounds();
            let scale = gpu.scale_factor();
            let width = (bounds.size.width.0 * scale).round().max(1.0) as u32;
            let height = (bounds.size.height.0 * scale).round().max(1.0) as u32;

            let mut slot = scene.borrow_mut();
            if slot.is_none() && drawing_error.borrow().is_none() {
                match HelioScene::build(gpu, width, height, adapter, probe) {
                    Ok(built) => *slot = Some(built),
                    Err(build) => {
                        let message = format!("{build:#}");
                        eprintln!("interop_helio: cannot build the scene: {message}");
                        *drawing_error.borrow_mut() = Some(message);
                        return;
                    }
                }
            }
            let Some(scene) = slot.as_mut() else {
                return;
            };

            if let Err(error) = scene.resize(width, height) {
                let message = format!("{error:#}");
                eprintln!("interop_helio: cannot resize the target: {message}");
                *drawing_error.borrow_mut() = Some(message);
                return;
            }

            if !paused.get() {
                let elapsed = scene.started.elapsed().as_secs_f32();
                scene.animate(elapsed);
                let camera = scene.camera(
                    yaw_for_canvas.get(),
                    pitch_for_canvas.get(),
                    distance_for_camera.get(),
                );
                match scene.render(&camera) {
                    Ok((render, readback, upload)) => {
                        metrics.borrow_mut().record(
                            render,
                            readback,
                            upload,
                            scene.width,
                            scene.height,
                        );
                    }
                    Err(error) => {
                        let message = format!("{error:#}");
                        eprintln!("interop_helio: cannot render: {message}");
                        *drawing_error.borrow_mut() = Some(message);
                        return;
                    }
                }
            }

            if stats {
                let now = Instant::now();
                let mut last = logged.borrow_mut();
                if now.duration_since(*last) >= Duration::from_secs(1) {
                    *last = now;
                    let metrics = metrics.borrow();
                    eprintln!(
                        "interop_helio: window {} frames — {:.1} fps, {:.2} ms interval, \
                         helio {:.2} ms, readback {:.2} ms, upload {:.2} ms, {}x{}",
                        metrics.frames,
                        metrics.fps,
                        metrics.interval_ms,
                        metrics.render_ms,
                        metrics.readback_ms,
                        metrics.upload_ms,
                        scene.width,
                        scene.height,
                    );
                }
            }

            if let Err(error) = scene.paint(gpu) {
                let message = format!("{error:#}");
                eprintln!("interop_helio: cannot hand the frame to GPUI: {message}");
                *drawing_error.borrow_mut() = Some(message);
            }
        })
        .on_scroll_wheel(move |event, window, _cx| {
            let notches = match event.delta {
                ScrollDelta::Lines(lines) => lines.y,
                ScrollDelta::Pixels(pixels) => pixels.y.as_f32() / PIXELS_PER_LINE,
            };
            let next = (distance.get() * ZOOM_PER_NOTCH.powf(notches)).clamp(4.0, 30.0);
            distance.set(next);
            window.refresh();
        })
        .size_full();

        let handler = move |input: SurfaceInput| match input.event {
            SurfaceEvent::PointerDown { position, .. } => {
                grab.set(true);
                last.set(Some(position));
            }
            SurfaceEvent::PointerMove { position, .. } if grab.get() => {
                if let Some(previous) = last.replace(Some(position)) {
                    yaw.set(yaw.get() + (position.x - previous.x) * SENSITIVITY);
                    pitch.set(
                        (pitch.get() - (position.y - previous.y) * SENSITIVITY).clamp(-1.2, 1.2),
                    );
                }
            }
            SurfaceEvent::PointerMove { .. } => {}
            SurfaceEvent::PointerUp { .. } => {
                grab.set(false);
                last.set(None);
            }
        };

        viewport_frame()
            .child(surface_input(canvas, handler))
            .children(overlay_error.borrow().clone().map(|message| {
                error_panel(
                    "The Helio viewport could not start",
                    message,
                    "Helio needs a Vulkan adapter; --list-gpus shows the adapters this process sees.",
                )
            }))
            .child(overlay_badge("helio · wgpu 30 → CPU copy → GPUI"))
            .child(overlay_hint(if paused_for_hint.get() {
                "paused · GPUI presents the last frame"
            } else {
                "drag to orbit · scroll to zoom"
            }))
            .into_any_element()
    }

    pub fn run() {
        let cli = parse_args();
        let app = gpui::application();
        let app = match cli.preference {
            Some(preference) => app.with_gpu_preference(preference),
            None => app,
        };
        app.run(move |cx: &mut App| {
            if cli.list {
                list_gpus(cx);
                std::process::exit(0);
            }
            let bounds = Bounds::centered(None, size(px(1280.), px(760.)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                move |_, cx| cx.new(|cx| Demo::new(cx, cli.adapter, cli.probe, cli.stats)),
            )
            .expect("a window to composite into");
            cx.activate(true);
        });
    }
}
