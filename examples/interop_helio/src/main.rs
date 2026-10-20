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
//! standalone demo does. It renders into an **exportable** image: the image
//! and its memory are allocated by `gpui_interop`'s Vulkan exporter (the same
//! allocation `WgpuSharedBuffer` makes for wgpu 29), wrapped into a wgpu 30
//! texture with `texture_from_raw`, and exported as a dma-buf with its DRM
//! modifier and plane layout. GPUI imports it on its own wgpu 29 device with
//! `paint_surface`, so no pixel crosses the CPU.
//!
//! Two things make this cross the version boundary without a shim inside
//! `gpui_wgpu`:
//!
//! - Only **raw Vulkan handles and file descriptors** cross it. An `ash::Device`
//!   from wgpu 30's HAL is the same `ash::Device` `gpui_interop` was compiled
//!   against; no `wgpu` 30 type is visible to wgpu 29.
//! - The device is created through wgpu 30's HAL, not `request_device`, so the
//!   one extension wgpu-hal does not enable by itself —
//!   `VK_KHR_external_semaphore_fd` — is added before `vkCreateDevice`. The
//!   renderer's own device already enables it (with
//!   `VK_EXT_image_drm_format_modifier`), so the consumer half needs no change.
//!
//! The frame is ordered by a `sync_file` exported from a binary semaphore that
//! an empty submission on Helio's queue signals after the graph has run. A ring
//! of two images recycles a slot only after the renderer signals the `eventfd`
//! it was handed, so drawing frame N+1 never races GPUI sampling frame N.
//!
//! The renderer polls that fence for only 16 ms and drops the surface if it has
//! not signalled, so the producer waits for it here before the hand-off — the
//! CPU read-back's wait without the copy. A Helio frame slower than 16 ms then
//! presents as a frame, not as black.
//!
//! # Scene
//!
//! The scene is the standalone demo's: a sky, a reflective floor, four
//! columns with spheres, four orbiting point lights, and a metallic crystal
//! spinner. It exercises Helio's default graph — GI, shadows, reflections,
//! post-processing — rather than a single forward pass.
//!
//! Drag to orbit the camera, scroll to zoom. Pause stops the clock but keeps
//! the last frame on screen. `--stats` prints the frame rate and the render and
//! export timings; `--probe` reads the exported image back once.
//!
//! # Build profile
//!
//! Run it with `--release`. Helio's 39-pass graph and shader compilation are
//! compiled at opt-level 0 in a debug build; the debug numbers measure the
//! build, not the bridge.

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
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::path::{Path, PathBuf};
    use std::rc::Rc;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use ash::vk;
    use glam::{EulerRot, Mat4, Quat, Vec3};
    use gpui::{
        AnyElement, App, Bounds, Context, Entity, IntoElement, Render, ScrollDelta, Window,
        WindowBounds, WindowOptions, div, gpu_canvas, prelude::*, px, rgb, size,
    };
    use gpui_interop::vulkan_export::{
        DmaBufHandle, ExportDevice, SurfaceFormatKind, Tiling, allocate_export_image,
    };
    use gpui_interop::{SurfaceEvent, SurfaceInput, SurfacePoint, attach, surface_input};
    use gpui_wgpu::WgpuRenderer;
    use helio::{
        Camera, GpuLight, GpuMaterial, GraphRebuilder, GroupMask, LightId, LightType, MaterialId,
        MeshId, MeshUpload, Movability, ObjectDescriptor, ObjectId, PackedVertex, RenderGraph,
        Renderer, RendererBuilder, RendererConfig, SceneActor, SkyActor,
    };
    use helio_default_graphs::{build_default_graph_external, build_forward_only_graph_external};
    use helio_pass_simple_cube::SimpleCubePass;
    use interop_demo::{
        ACCENT, LEFT_WIDTH, MUTED, control, error_panel, histogram, histogram_caption,
        histogram_style, metric_row, metrics_panel_style, overlay_badge, overlay_hint, pacing_note,
        page, panel, section_label, title, viewport_frame,
    };

    /// Helio's output format. GPUI imports the exported image as `rgba8`.
    const FORMAT: wgpu30::TextureFormat = wgpu30::TextureFormat::Rgba8UnormSrgb;
    /// The Vulkan form of `FORMAT`: `_SRGB`, the form a shader's linear output is stored in.
    const VK_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
    /// How many exportable images the producer cycles: one presented, one GPUI may still be
    /// sampling, and one free to render into. A release that never comes costs a timeout, not the
    /// frame loop.
    const SLOTS: usize = 3;
    /// How long a handed-out slot waits for the renderer's release before it is reclaimed.
    const RELEASE_TIMEOUT: Duration = Duration::from_millis(250);
    /// How long the producer waits for its own acquire fence before handing a frame over.
    ///
    /// The renderer polls the fence for only 16 ms (`ACQUIRE_FENCE_TIMEOUT` in `gpui_wgpu`) and
    /// drops the surface for that frame otherwise. A Helio frame is regularly slower than that, so
    /// the hand-off would lose almost every frame; waiting here — the CPU read-back's wait without
    /// the copy — is what keeps the viewport stable.
    const HANDOFF_TIMEOUT: Duration = Duration::from_secs(2);

    /// Orbit/zoom, the same feel as the other two demos.
    const SENSITIVITY: f32 = 0.007;
    const ZOOM_PER_NOTCH: f32 = 1.12;
    const PIXELS_PER_LINE: f32 = 40.0;

    /// The camera's defaults, and the scene's focus.
    const DEFAULT_YAW: f32 = 0.55;
    const DEFAULT_PITCH: f32 = 0.32;
    const DEFAULT_DISTANCE: f32 = 11.5;
    const TARGET: Vec3 = Vec3::new(0.0, 2.0, 0.0);
    /// `interop_bevy`'s field: 7 × 7 cubes of 1.0 at 1.75 spacing.
    const GRID: usize = 7;
    const SPACING: f32 = 1.75;
    const CUBE: f32 = 1.0;

    struct Cli {
        list: bool,
        preference: Option<gpui::GpuPreference>,
        probe: bool,
        stats: bool,
        scene: SceneKind,
        /// `None` keeps the scene's default: shadows on for the standalone scene, off for the
        /// field (Bevy's light has shadow maps disabled).
        shadows: Option<bool>,
        graph: GraphKind,
    }

    /// Which render graph Helio runs.
    ///
    /// The default graph is the 39-pass standalone one. `Forward` is Helio's forward-mode graph:
    /// fewer passes, but still its shadows, GI and fog. `Simple` is a single pass drawing one
    /// debug cube — the micro baseline, comparable to `interop_wgpu`'s hand-written scene.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    enum GraphKind {
        #[default]
        Default,
        Forward,
        Simple,
    }

    fn parse_graph(value: &str) -> GraphKind {
        match value {
            "default" | "full" => GraphKind::Default,
            "forward" => GraphKind::Forward,
            "simple" | "cube" => GraphKind::Simple,
            other => usage_error(&format!(
                "--graph: {other:?} is not default, forward, or simple"
            )),
        }
    }

    /// Which scene to build.
    ///
    /// `Field` is `interop_bevy`'s scene — a 7 × 7 field of cubes, one directional light, no floor
    /// or sky — so the two producers can be compared on the same content. `Clear` is the control:
    /// the same render graph with no actors, which separates the graph's own cost from the scene.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    enum SceneKind {
        /// The `Helio_Standalone_Demo` scene: sky, reflective floor, columns, spheres, four moving
        /// point lights and a crystal spinner.
        #[default]
        Full,
        /// `interop_bevy`'s 7 × 7 field of cubes.
        Field,
        /// No actors: the graph's own per-frame cost.
        Clear,
    }

    fn parse_scene(value: &str) -> SceneKind {
        match value {
            "full" | "standalone" => SceneKind::Full,
            "field" | "bevy" | "ember" => SceneKind::Field,
            "clear" => SceneKind::Clear,
            other => usage_error(&format!("--scene: {other:?} is not full, field, or clear")),
        }
    }

    fn parse_args() -> Cli {
        let mut cli = Cli {
            list: false,
            preference: None,
            probe: false,
            stats: false,
            scene: SceneKind::default(),
            shadows: None,
            graph: GraphKind::default(),
        };
        let mut args = std::env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--list-gpus" => cli.list = true,
                "--probe" => cli.probe = true,
                "--stats" => cli.stats = true,
                "--scene" => {
                    let value = args
                        .next()
                        .unwrap_or_else(|| usage_error("--scene needs full, field, or clear"));
                    cli.scene = parse_scene(&value);
                }
                "--shadows" => {
                    let value = args
                        .next()
                        .unwrap_or_else(|| usage_error("--shadows needs on or off"));
                    cli.shadows = Some(match value.as_str() {
                        "on" | "true" => true,
                        "off" | "false" => false,
                        other => usage_error(&format!("--shadows: {other:?} is not on or off")),
                    });
                }
                "--graph" => {
                    let value = args.next().unwrap_or_else(|| {
                        usage_error("--graph needs default, forward, or simple")
                    });
                    cli.graph = parse_graph(&value);
                }
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
            "  --gpu <INDEX>        select the window adapter at INDEX from --list-gpus (Helio follows it)\n",
            "  --stats              print the frame-rate and timing breakdown once a second\n",
            "  --probe              read the exported image back once and print its centre pixel\n",
            "  --scene <KIND>       full (standalone demo), field (49 cubes, Bevy's scene), clear\n",
            "  --shadows <ON|OFF>   override the scene's shadow maps (full: on, field: off)\n",
            "  --graph <KIND>       default (39 passes), forward (fewer), simple (one cube)\n",
            "  -h, --help           show this help\n",
            "\n",
            "Helio's wgpu 30 frame leaves as a dma-buf and GPUI imports it on its own wgpu 29\n",
            "device; no pixels cross the CPU. Run with --release for meaningful timings.",
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
                "{index:<3} {:<46} {:<8} {:<11} dma-buf{software}",
                gpu.name,
                format!("{:?}", gpu.backend),
                format!("{:?}", gpu.device_type),
            );
        }
        println!(
            "\nHelio opens its own wgpu 30 device and exports a dma-buf; the window's renderer is \
             wgpu 29. Select the window's adapter with --gpu <#>."
        );
    }

    /// One snapshot of the numbers the right panel shows.
    #[derive(Clone, Default)]
    struct Metrics {
        frames: u64,
        interval_ms: f32,
        fps: f32,
        render_ms: f32,
        export_ms: f32,
        fence_ms: f32,
        /// The part of the interval outside the producer: GPUI's own layout, text, compositing and
        /// the wait for the next animation frame.
        gpui_ms: f32,
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
            export: Duration,
            fence: Duration,
            width: u32,
            height: u32,
        ) {
            self.frames += 1;
            self.render_ms = render.as_secs_f32() * 1000.0;
            self.export_ms = export.as_secs_f32() * 1000.0;
            self.fence_ms = fence.as_secs_f32() * 1000.0;
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
            self.gpui_ms =
                (self.interval_ms - self.render_ms - self.export_ms - self.fence_ms).max(0.0);
            self.fps = if self.intervals.is_empty() {
                0.0
            } else {
                let average = self.intervals.iter().sum::<f32>() / self.intervals.len() as f32;
                1000.0 / average
            };
        }
    }

    /// What moves each frame, per scene.
    enum Animation {
        Full {
            spinner: ObjectId,
            lights: [LightId; 4],
            shadow_index: u32,
        },
        Field {
            cubes: Vec<ObjectId>,
        },
        Clear,
    }

    /// The Helio producer and its ring of exportable images.
    ///
    /// Declaration order is load-bearing: the slots own the raw images and their drop callbacks, so
    /// they are dropped before the `Renderer` and the `Arc<wgpu::Device>` that created them.
    struct HelioScene {
        slots: Vec<Slot>,
        renderer: Renderer,
        queue: Arc<wgpu30::Queue>,
        device: Arc<wgpu30::Device>,
        /// The handle last handed to GPUI; re-presented while paused.
        presented: Option<DmaBufHandle>,
        /// The slot `presented` belongs to. It is never chosen for rendering while it is the
        /// presented image: GPUI may still be sampling a re-presentation even after the release
        /// from the previous one has arrived.
        presented_slot: Option<usize>,
        width: u32,
        height: u32,
        animation: Animation,
        /// What the orbit camera looks at; the field looks at the origin.
        target: Vec3,
        started: Instant,
        adapter: String,
        frames: u64,
        dropped: u64,
        probe: bool,
        /// `INTEROP_HELIO_DUMP=/path/frame.bmp` writes one rendered frame once, for diagnosing what
        /// the producer actually drew rather than what the metrics say.
        dump: Option<PathBuf>,
    }

    /// One exportable image, and the release the renderer signals when it has sampled it.
    struct Slot {
        target: HelioTarget,
        release: Option<Arc<OwnedFd>>,
        handed_at: Option<Instant>,
    }

    impl Slot {
        /// Whether the renderer has finished sampling this slot, or its release has timed out.
        fn released(&self) -> bool {
            match (&self.release, self.handed_at) {
                (None, _) => true,
                (Some(release), Some(at)) => {
                    release_signalled(release) || at.elapsed() >= RELEASE_TIMEOUT
                }
                (Some(_), None) => true,
            }
        }
    }

    impl HelioScene {
        fn build(
            gpu: &mut gpui::GpuCanvasContext,
            width: u32,
            height: u32,
            probe: bool,
            scene: SceneKind,
            shadows: Option<bool>,
            graph: GraphKind,
        ) -> anyhow::Result<Self> {
            let interop = attach(gpu.try_device::<WgpuRenderer>())
                .map_err(|unavailable| anyhow::anyhow!("{unavailable:?}"))?;
            // The consumer's GPU identity. A dma-buf is importable only on the physical device it
            // was exported from, and a DRM modifier is meaningful only to that device's driver.
            let renderer_gpu = interop.adapter().gpu_id().ok_or_else(|| {
                anyhow::anyhow!(
                    "the window's renderer is not Vulkan: a dma-buf surface needs a Vulkan consumer"
                )
            })?;

            let instance = wgpu30::Instance::new(wgpu30::InstanceDescriptor {
                backends: wgpu30::Backends::VULKAN,
                flags: wgpu30::InstanceFlags::empty(),
                ..wgpu30::InstanceDescriptor::new_without_display_handle()
            });
            let adapters =
                pollster::block_on(instance.enumerate_adapters(wgpu30::Backends::VULKAN));
            if adapters.is_empty() {
                anyhow::bail!("Helio found no Vulkan adapter");
            }
            let adapter = adapters
                .iter()
                .find(|adapter| adapter.get_info().name == renderer_gpu.name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "no Vulkan adapter named {}: Helio must render on the window's GPU for a \
                         dma-buf to import",
                        renderer_gpu.name,
                    )
                })?;
            let info = adapter.get_info();
            eprintln!(
                "interop_helio: Helio and the window share {} ({:?}), dma-buf hand-off",
                info.name, info.backend
            );

            let (device, queue) = helio_device(adapter)?;
            let (device, queue) = (Arc::new(device), Arc::new(queue));

            let builder = RendererBuilder::new(RendererConfig::new(width, height, FORMAT));
            let builder = match graph {
                GraphKind::Default => builder.with_graph(Box::new(|d, q, s, c, ds, cb, csb| {
                    build_default_graph_external(d, q, s, c, ds, cb, csb, None)
                })),
                GraphKind::Forward => builder.with_graph(Box::new(|d, q, s, c, ds, cb, csb| {
                    build_forward_only_graph_external(d, q, s, c, ds, cb, csb, None)
                })),
                GraphKind::Simple => builder.with_graph(Box::new(|d, q, _s, c, _ds, _cb, _csb| {
                    simple_graph(d, q, c)
                })),
            };
            let mut renderer = builder
                // Helio shares a device it did not create: no owned-device readbacks, no teardown.
                .with_external_device()
                .build(device.clone(), queue.clone(), width, height, FORMAT);

            let (animation, target) = match scene {
                SceneKind::Full => (
                    populate_full(&mut renderer, shadows.unwrap_or(true)),
                    TARGET,
                ),
                SceneKind::Field => (
                    populate_field(&mut renderer, shadows.unwrap_or(false)),
                    Vec3::ZERO,
                ),
                SceneKind::Clear => (Animation::Clear, Vec3::ZERO),
            };
            let slots = slots(&device, width, height)?;

            Ok(Self {
                slots,
                renderer,
                queue,
                device,
                presented: None,
                presented_slot: None,
                width,
                height,
                animation,
                target,
                started: Instant::now(),
                adapter: info.name,
                frames: 0,
                dropped: 0,
                probe,
                dump: std::env::var_os("INTEROP_HELIO_DUMP").map(PathBuf::from),
            })
        }

        fn resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
            let width = width.max(1);
            let height = height.max(1);
            if (width, height) == (self.width, self.height) {
                return Ok(());
            }

            self.renderer.set_render_size(width, height);
            self.slots = slots(&self.device, width, height)?;
            self.presented = None;
            self.presented_slot = None;
            self.width = width;
            self.height = height;
            Ok(())
        }

        fn camera(&self, yaw: f32, pitch: f32, distance: f32) -> Camera {
            let eye = self.target
                + Vec3::new(
                    distance * pitch.cos() * yaw.sin(),
                    distance * pitch.sin(),
                    distance * pitch.cos() * yaw.cos(),
                );
            Camera::perspective_look_at(
                eye,
                self.target,
                Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                self.width as f32 / self.height as f32,
                0.1,
                200.0,
            )
        }

        fn animate(&mut self, elapsed: f32) {
            match &self.animation {
                Animation::Full {
                    spinner,
                    lights,
                    shadow_index,
                } => {
                    let rot = Quat::from_rotation_y(elapsed * 0.8)
                        * Quat::from_rotation_x((elapsed * 0.5).sin() * 0.3);
                    let _ = self.renderer.scene_mut().update_object_transform(
                        *spinner,
                        Mat4::from_rotation_translation(rot, Vec3::new(0.0, 2.5, 0.0)),
                    );

                    let colors = [
                        [1.0, 0.3, 0.3],
                        [0.3, 1.0, 0.3],
                        [0.3, 0.5, 1.0],
                        [1.0, 0.8, 0.2],
                    ];
                    for (i, &light) in lights.iter().enumerate() {
                        let angle = elapsed * 0.6 + i as f32 * std::f32::consts::FRAC_PI_2;
                        let (x, z) = (angle.cos() * 4.0, angle.sin() * 4.0);
                        let _ = self.renderer.scene_mut().update_light(
                            light,
                            pt_light(
                                [x, 5.0 + (elapsed * 1.2 + i as f32).sin() * 0.5, z],
                                colors[i],
                                10.0,
                                10.0,
                                *shadow_index,
                            ),
                        );
                    }
                }
                Animation::Field { cubes } => {
                    let half = (GRID - 1) as f32 / 2.0;
                    for (index, &cube) in cubes.iter().enumerate() {
                        let (x, y) = (index % GRID, index / GRID);
                        let (fx, fy) = (x as f32 - half, y as f32 - half);
                        let phase = elapsed * 1.25 + fx * 0.42 + fy * 0.31;
                        let transform = Mat4::from_scale_rotation_translation(
                            Vec3::splat(0.40 + 0.05 * (phase * 0.9).sin()),
                            Quat::from_euler(EulerRot::YXZ, phase * 0.62, phase * 0.44, 0.0),
                            Vec3::new(fx * SPACING, fy * SPACING, phase.sin() * 0.42),
                        );
                        let _ = self
                            .renderer
                            .scene_mut()
                            .update_object_transform(cube, transform);
                    }
                }
                Animation::Clear => {}
            }
        }

        /// Render one frame into a free slot and export it, or `None` when every slot is still in
        /// flight — a dropped frame rather than a write into a buffer the compositor is sampling.
        fn render(
            &mut self,
            camera: &Camera,
        ) -> anyhow::Result<Option<(Duration, Duration, Duration)>> {
            let Some(index) = (0..self.slots.len())
                .find(|&index| Some(index) != self.presented_slot && self.slots[index].released())
            else {
                self.dropped += 1;
                return Ok(None);
            };

            let render_start = Instant::now();
            self.renderer
                .render(camera, self.slots[index].target.view())
                .map_err(|error| anyhow::anyhow!("Helio render: {error}"))?;
            let render = render_start.elapsed();

            if self.probe && self.frames + 1 == 60 {
                self.probe_slot(index)?;
            }
            if self.frames + 1 == 30
                && let Some(path) = self.dump.clone()
            {
                self.dump_slot(index, &path)?;
            }

            let export_start = Instant::now();
            let mut handle = self.slots[index].target.submit(&self.queue)?;
            let export = export_start.elapsed();

            // The renderer polls this fence for at most 16 ms and drops the surface if it has not
            // signalled. Helio's graph is slower than that, so complete the frame here before the
            // hand-off: the read-back path's wait, without the copy.
            let fence_start = Instant::now();
            let ready = handle
                .acquire_fence
                .as_ref()
                .is_none_or(|fence| wait_for_fence(fence, HANDOFF_TIMEOUT));
            let fence = fence_start.elapsed();
            if !ready {
                anyhow::bail!(
                    "the frame's acquire fence did not signal within {HANDOFF_TIMEOUT:?}"
                );
            }
            // The GPU has finished; the renderer need not wait on the fence again.
            handle.acquire_fence = None;

            let release = release_descriptor()?;
            handle.release = Some(Arc::clone(&release));
            let slot = &mut self.slots[index];
            slot.release = Some(release);
            slot.handed_at = Some(Instant::now());
            self.presented = Some(handle);
            self.presented_slot = Some(index);
            self.frames += 1;

            Ok(Some((render, export, fence)))
        }

        /// Composite the last exported frame into the canvas. Cheap: GPUI imports the dma-buf on
        /// its own device, and the handle carries the acquire fence.
        fn paint(&self, gpu: &mut gpui::GpuCanvasContext) {
            if let Some(handle) = &self.presented {
                gpu.paint_surface(handle.clone());
            }
        }

        /// Read the exported image back once, to prove it is not blank. This is the only path that
        /// touches the CPU, and it is opt-in.
        /// Copy the slot's image back to the CPU, tightly packed RGBA from the top row down.
        fn read_slot(&self, index: usize) -> anyhow::Result<Vec<u8>> {
            let padded = padded_bytes_per_row(self.width);
            let buffer = readback_buffer(&self.device, self.width, self.height);
            let mut encoder =
                self.device
                    .create_command_encoder(&wgpu30::CommandEncoderDescriptor {
                        label: Some("interop_helio_probe"),
                    });
            encoder.copy_texture_to_buffer(
                wgpu30::TexelCopyTextureInfo {
                    texture: self.slots[index].target.texture(),
                    mip_level: 0,
                    origin: wgpu30::Origin3d::ZERO,
                    aspect: wgpu30::TextureAspect::All,
                },
                wgpu30::TexelCopyBufferInfo {
                    buffer: &buffer,
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
                .map_err(|error| anyhow::anyhow!("waiting for the probe copy: {error}"))?;

            let slice = buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu30::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            self.device
                .poll(wgpu30::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(10)),
                })
                .map_err(|error| anyhow::anyhow!("waiting for the probe map: {error}"))?;
            receiver
                .recv()
                .map_err(|error| anyhow::anyhow!("probe channel: {error}"))?
                .map_err(|error| anyhow::anyhow!("mapping the probe buffer: {error}"))?;

            let row = self.width as usize * 4;
            let mapped = slice
                .get_mapped_range()
                .map_err(|error| anyhow::anyhow!("mapping the probe range: {error}"))?;
            let mut pixels = vec![0u8; row * self.height as usize];
            for y in 0..self.height as usize {
                let source = y * padded as usize;
                pixels[y * row..(y + 1) * row].copy_from_slice(&mapped[source..source + row]);
            }
            drop(mapped);
            buffer.unmap();
            Ok(pixels)
        }

        fn probe_slot(&self, index: usize) -> anyhow::Result<()> {
            let pixels = self.read_slot(index)?;
            let row = self.width as usize * 4;
            let centre_at = (self.height as usize / 2) * row + (self.width as usize / 2) * 4;
            let centre = &pixels[centre_at..centre_at + 4];
            let sum: u64 = pixels
                .as_chunks::<4>()
                .0
                .iter()
                .map(|[r, g, b, _]| u64::from(*r) + u64::from(*g) + u64::from(*b))
                .sum();
            let samples = pixels.len() as u64 / 4 * 3;
            eprintln!(
                "interop_helio: probe {}x{}: centre {:?}, mean channel {:.2}",
                self.width,
                self.height,
                centre,
                sum as f64 / samples.max(1) as f64,
            );
            Ok(())
        }

        fn dump_slot(&self, index: usize, path: &Path) -> anyhow::Result<()> {
            let pixels = self.read_slot(index)?;
            write_bmp(path, self.width, self.height, &pixels)?;
            eprintln!(
                "interop_helio: wrote {}x{} to {}",
                self.width,
                self.height,
                path.display()
            );
            Ok(())
        }
    }

    /// The micro graph: one pass drawing `SimpleCubePass`'s debug cube.
    ///
    /// Helio's `build_simple_graph` creates the same pass but its rebuilder returns an unlocked
    /// graph, so a resize (or the renderer's first rebuild) panics in `execute`. This version locks
    /// the initial graph and every rebuild.
    fn simple_graph(
        device: &Arc<wgpu30::Device>,
        queue: &Arc<wgpu30::Queue>,
        config: RendererConfig,
    ) -> RenderGraph {
        let mut graph = RenderGraph::new_with_external_device(device, queue);
        graph.add_pass(Box::new(SimpleCubePass::new(device, FORMAT)));
        let rebuilder: GraphRebuilder = Arc::new(
            move |device, queue, _scene, config, _debug_state, _camera, _cull| {
                let mut graph = RenderGraph::new_with_external_device(device, queue);
                graph.add_pass(Box::new(SimpleCubePass::new(device, FORMAT)));
                graph.lock(config.internal_width(), config.internal_height());
                graph
            },
        );
        graph.set_graph_data(rebuilder);
        graph.lock(config.internal_width(), config.internal_height());
        graph
    }

    /// Create the wgpu 30 device with the one export extension wgpu-hal does not enable itself.
    ///
    /// `request_device` would enable `VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`
    /// and `VK_EXT_image_drm_format_modifier` (wgpu-hal adds those when the driver supports them),
    /// but not `VK_KHR_external_semaphore_fd`, which the acquire fence needs. `open_with_callback`
    /// mutates the extension list before `vkCreateDevice`, and `create_device_from_hal` adopts the
    /// result back into wgpu — the same escape hatch `gpui_wgpu` uses on the consumer side, and it
    /// keeps every `wgpu` 30 type on Helio's side of the boundary.
    fn helio_device(adapter: &wgpu30::Adapter) -> anyhow::Result<(wgpu30::Device, wgpu30::Queue)> {
        use wgpu30::hal::vulkan::Api;

        let features = helio::required_wgpu_features(adapter.features())
            | wgpu30::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF;
        if !adapter
            .features()
            .contains(wgpu30::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
        {
            anyhow::bail!(
                "{} lacks VK_EXT_external_memory_dma_buf or VK_EXT_image_drm_format_modifier; a \
                 dma-buf export is not available on it",
                adapter.get_info().name,
            );
        }
        let descriptor = wgpu30::DeviceDescriptor {
            label: Some("interop_helio"),
            required_features: features,
            required_limits: helio::required_wgpu_limits(adapter.limits()),
            memory_hints: wgpu30::MemoryHints::MemoryUsage,
            trace: wgpu30::Trace::Off,
            experimental_features: helio::required_experimental_features(adapter.features()),
        };

        // SAFETY: the HAL device is opened from this adapter with exactly the descriptor's features
        // and limits, and the callback only adds an extension the physical device reports. The
        // `OpenDevice` is adopted back into wgpu before this function returns.
        unsafe {
            let hal = adapter
                .as_hal::<Api>()
                .ok_or_else(|| anyhow::anyhow!("Helio's adapter is not Vulkan"))?;
            if !hal
                .physical_device_capabilities()
                .supports_extension(vk::KHR_EXTERNAL_SEMAPHORE_FD_NAME)
            {
                anyhow::bail!(
                    "{} does not support VK_KHR_external_semaphore_fd; the frame cannot carry an \
                     acquire fence",
                    adapter.get_info().name,
                );
            }
            let open = hal
                .open_with_callback(
                    features,
                    &descriptor.required_limits,
                    &descriptor.memory_hints,
                    Some(Box::new(|args| {
                        args.extensions.push(vk::KHR_EXTERNAL_SEMAPHORE_FD_NAME);
                    })),
                )
                .map_err(|error| anyhow::anyhow!("opening the Helio device: {error}"))?;
            adapter
                .create_device_from_hal::<Api>(open, &descriptor)
                .map_err(|error| anyhow::anyhow!("adopting the Helio device: {error}"))
        }
    }

    /// A shareable image on Helio's device, wrapped back into wgpu 30 so the renderer draws into it.
    struct HelioTarget {
        texture: wgpu30::Texture,
        view: wgpu30::TextureView,
        handle: DmaBufHandle,
        device: ash::Device,
        semaphore: vk::Semaphore,
        external_semaphore_fd: ash::khr::external_semaphore_fd::Device,
    }

    impl HelioTarget {
        fn new(device: &wgpu30::Device, width: u32, height: u32) -> anyhow::Result<Self> {
            // SAFETY: the HAL device is borrowed only for this call; the image and memory outlive
            // it inside the wgpu texture's drop callback.
            let hal_guard = unsafe { device.as_hal::<wgpu30::hal::vulkan::Api>() }
                .ok_or_else(|| anyhow::anyhow!("a dma-buf target needs a Vulkan wgpu device"))?;
            let hal: &wgpu30::hal::vulkan::Device = &hal_guard;
            let instance = hal.shared_instance().raw_instance();
            let raw_device = hal.raw_device();
            for required in [
                vk::EXT_IMAGE_DRM_FORMAT_MODIFIER_NAME,
                vk::KHR_EXTERNAL_MEMORY_FD_NAME,
                vk::EXT_EXTERNAL_MEMORY_DMA_BUF_NAME,
                vk::KHR_EXTERNAL_SEMAPHORE_FD_NAME,
            ] {
                if !hal.enabled_device_extensions().contains(&required) {
                    anyhow::bail!(
                        "Helio's device does not have {} enabled; it cannot export a dma-buf",
                        required.to_string_lossy(),
                    );
                }
            }

            let external_memory_fd =
                ash::khr::external_memory_fd::Device::new(instance, raw_device);
            let drm_format_modifier =
                ash::ext::image_drm_format_modifier::Device::new(instance, raw_device);
            let export_device = HelioExportDevice {
                hal,
                external_memory_fd,
                drm_format_modifier,
            };
            let (image, memory, handle) = allocate_export_image(
                &export_device,
                width,
                height,
                SurfaceFormatKind::rgba8(),
                VK_FORMAT,
                Tiling::Tiled,
                true,
            )?;
            // Which modifier the driver gave the export target. A linear modifier is legal on some
            // drivers but a notoriously slow render target; a non-preferred tiled one can be slow
            // too. The graph's GPU tail is the fence wait, so this is the first thing to check when
            // the tail is much larger than the same scene on a wgpu-allocated texture.
            let linear = handle.modifier == DmaBufHandle::LINEAR;
            eprintln!(
                "interop_helio: export target {}x{}: modifier {:#018x}{}, offset {}, stride {}",
                width,
                height,
                handle.modifier,
                if linear { " (LINEAR)" } else { "" },
                handle.planes[0].offset,
                handle.planes[0].stride,
            );

            // The acquire fence. A `SYNC_FD` export moves the signal into a `sync_file`, which is
            // exactly what `DmaBufHandle::acquire_fence` takes; the same semaphore is reused once
            // the export has consumed its signal.
            let mut export_semaphore = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            // SAFETY: the semaphore is created on this device and destroyed when this value drops.
            let semaphore = match unsafe {
                raw_device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut export_semaphore),
                    None,
                )
            } {
                Ok(semaphore) => semaphore,
                Err(error) => {
                    // SAFETY: neither image nor memory is wrapped or bound to anything else yet.
                    unsafe {
                        raw_device.free_memory(memory, None);
                        raw_device.destroy_image(image, None);
                    }
                    return Err(anyhow::anyhow!("creating the acquire semaphore: {error:?}"));
                }
            };

            let size = wgpu30::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };
            let hal_descriptor = wgpu30::hal::TextureDescriptor {
                label: Some("interop_helio_target"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu30::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu30::TextureUses::COLOR_TARGET | wgpu30::TextureUses::COPY_SRC,
                memory_flags: wgpu30::hal::MemoryFlags::empty(),
                view_formats: Vec::new(),
            };
            // The texture owns the raw image and memory: wgpu does not free either through
            // `destroy_texture` when a drop callback is supplied, so the callback does, in reverse
            // order, after the last view is gone.
            let owner = raw_device.clone();
            let drop_callback: wgpu30::hal::DropCallback = Box::new(move || {
                // SAFETY: created by this device and freed exactly once, by the texture's drop.
                unsafe {
                    owner.destroy_image(image, None);
                    owner.free_memory(memory, None);
                }
            });
            // SAFETY: the image was created with this descriptor's extent, format, usage and
            // tiling on this device, and its memory is bound.
            let hal_texture = unsafe {
                hal.texture_from_raw(
                    image,
                    &hal_descriptor,
                    Some(drop_callback),
                    wgpu30::hal::vulkan::TextureMemory::External,
                )
            };
            // SAFETY: the HAL texture came from this device and the descriptor matches.
            let texture = unsafe {
                device.create_texture_from_hal::<wgpu30::hal::vulkan::Api>(
                    hal_texture,
                    &wgpu30::TextureDescriptor {
                        label: Some("interop_helio_target"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu30::TextureDimension::D2,
                        format: FORMAT,
                        usage: wgpu30::TextureUsages::RENDER_ATTACHMENT
                            | wgpu30::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    },
                    wgpu30::TextureUses::UNINITIALIZED,
                )
            };
            let view = texture.create_view(&wgpu30::TextureViewDescriptor {
                label: Some("interop_helio_target_view"),
                ..Default::default()
            });

            Ok(Self {
                texture,
                view,
                handle,
                device: raw_device.clone(),
                semaphore,
                external_semaphore_fd: ash::khr::external_semaphore_fd::Device::new(
                    instance, raw_device,
                ),
            })
        }

        fn texture(&self) -> &wgpu30::Texture {
            &self.texture
        }

        fn view(&self) -> &wgpu30::TextureView {
            &self.view
        }

        /// Signal the acquire semaphore after every submission Helio made for this frame, then
        /// export the signal as the `sync_file` the renderer waits on.
        ///
        /// Helio submits internally (a clear, then the graph), so the semaphore cannot ride the
        /// graph's own submission. An empty submission on the same queue is ordered after all of
        /// them, and is the only ordering `vkQueueSubmit` offers a producer that does not own the
        /// submission the frame was recorded in.
        fn submit(&mut self, queue: &wgpu30::Queue) -> anyhow::Result<DmaBufHandle> {
            let hal_queue = unsafe { queue.as_hal::<wgpu30::hal::vulkan::Api>() }
                .ok_or_else(|| anyhow::anyhow!("a dma-buf export needs a Vulkan wgpu queue"))?;
            hal_queue.add_signal_semaphore(self.semaphore, None);
            queue.submit(std::iter::empty::<wgpu30::CommandBuffer>());

            // SAFETY: the semaphore is live and was given to this device; `SYNC_FD` moves the
            // signal into the returned descriptor, whose ownership passes to the handle.
            let fd = unsafe {
                self.external_semaphore_fd.get_semaphore_fd(
                    &vk::SemaphoreGetFdInfoKHR::default()
                        .semaphore(self.semaphore)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD),
                )
            }
            .map_err(|error| anyhow::anyhow!("exporting the acquire fence: {error:?}"))?;
            // SAFETY: `fd` is a fresh descriptor, owned by the handle from here.
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };

            let mut handle = self.handle.clone();
            handle.acquire_fence = Some(Arc::new(fd));
            Ok(handle)
        }
    }

    impl Drop for HelioTarget {
        fn drop(&mut self) {
            // SAFETY: the semaphore was created by this value's device and is destroyed once. A
            // `SYNC_FD` export has already moved any pending signal into the fence's descriptor.
            unsafe { self.device.destroy_semaphore(self.semaphore, None) };
        }
    }

    /// The Vulkan handles Helio's wgpu 30 device lends the exporter in `gpui_interop`.
    struct HelioExportDevice<'a> {
        hal: &'a wgpu30::hal::vulkan::Device,
        external_memory_fd: ash::khr::external_memory_fd::Device,
        drm_format_modifier: ash::ext::image_drm_format_modifier::Device,
    }

    impl ExportDevice for HelioExportDevice<'_> {
        fn instance(&self) -> &ash::Instance {
            self.hal.shared_instance().raw_instance()
        }

        fn physical_device(&self) -> vk::PhysicalDevice {
            self.hal.raw_physical_device()
        }

        fn device(&self) -> &ash::Device {
            self.hal.raw_device()
        }

        fn external_memory_fd(&self) -> &ash::khr::external_memory_fd::Device {
            &self.external_memory_fd
        }

        fn drm_format_modifier(&self) -> &ash::ext::image_drm_format_modifier::Device {
            &self.drm_format_modifier
        }

        fn renderer_bound(&self) -> bool {
            // The adapter was matched to the window's `GpuId`, so this is the physical device the
            // renderer samples on; a modifier read from it is the one the importer can take.
            true
        }
    }

    fn slots(device: &wgpu30::Device, width: u32, height: u32) -> anyhow::Result<Vec<Slot>> {
        (0..SLOTS)
            .map(|_| {
                Ok(Slot {
                    target: HelioTarget::new(device, width, height)?,
                    release: None,
                    handed_at: None,
                })
            })
            .collect()
    }

    /// A fresh `eventfd` the renderer signals once it has finished sampling a frame.
    fn release_descriptor() -> anyhow::Result<Arc<OwnedFd>> {
        // SAFETY: `eventfd` returns a fresh descriptor this process owns.
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if fd < 0 {
            return Err(anyhow::anyhow!(
                "creating a release descriptor: {}",
                std::io::Error::last_os_error()
            ));
        }
        // SAFETY: `fd` is fresh and ownership moves into the value.
        Ok(Arc::new(unsafe { OwnedFd::from_raw_fd(fd) }))
    }

    /// Whether the renderer has signalled `release` — a non-blocking `poll`.
    fn release_signalled(release: &OwnedFd) -> bool {
        wait_for_fence(release, Duration::ZERO)
    }

    /// Wait, up to `timeout`, for a `sync_file` (or `eventfd`) to become readable; `true` if it
    /// signalled in time. A `sync_file` becomes readable when its fence signals.
    fn wait_for_fence(fence: &OwnedFd, timeout: Duration) -> bool {
        let timeout_ms = i32::try_from(timeout.as_millis()).unwrap_or(i32::MAX);
        let mut descriptor = libc::pollfd {
            fd: fence.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        loop {
            match unsafe { libc::poll(&mut descriptor, 1, timeout_ms) } {
                ready if ready > 0 => return true,
                0 => return false,
                _ => {
                    if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    return false;
                }
            }
        }
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

    /// Write a tightly packed RGBA image as a 24-bit BMP, for looking at what the producer drew.
    fn write_bmp(path: &Path, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
        let row = width as usize * 3;
        let padded = (row + 3) & !3;
        let image_size = padded * height as usize;
        let file_size = 54 + image_size;
        let mut out = Vec::with_capacity(file_size);
        out.extend_from_slice(b"BM");
        out.extend_from_slice(&(file_size as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&54u32.to_le_bytes());
        out.extend_from_slice(&40u32.to_le_bytes());
        out.extend_from_slice(&(width as i32).to_le_bytes());
        out.extend_from_slice(&(height as i32).to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&24u16.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(image_size as u32).to_le_bytes());
        out.extend_from_slice(&2835i32.to_le_bytes());
        out.extend_from_slice(&2835i32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        // BMP rows are bottom-up.
        for y in (0..height as usize).rev() {
            let start = y * width as usize * 4;
            for x in 0..width as usize {
                let index = start + x * 4;
                out.extend_from_slice(&[rgba[index + 2], rgba[index + 1], rgba[index]]);
            }
            out.resize(out.len() + (padded - row), 0);
        }
        std::fs::write(path, out)
    }

    fn padded_bytes_per_row(width: u32) -> u32 {
        let unpadded = width * 4;
        unpadded.div_ceil(wgpu30::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu30::COPY_BYTES_PER_ROW_ALIGNMENT
    }

    /// The standalone demo's scene, unchanged in spirit: sky, floor, columns, spheres, four
    /// moving point lights, and a metallic crystal spinner.
    fn populate_full(renderer: &mut Renderer, shadows: bool) -> Animation {
        let shadow_index = if shadows { 0 } else { u32::MAX };
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
                        shadow_index,
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

        Animation::Full {
            spinner,
            lights: lights.try_into().unwrap(),
            shadow_index,
        }
    }

    /// `interop_bevy`'s scene: 7 × 7 cubes of 1.0 at 1.75 spacing, one directional light, no floor
    /// and no sky. This is the content match, so a timing difference is the producer, not the scene.
    fn populate_field(renderer: &mut Renderer, shadows: bool) -> Animation {
        let shadow_index = if shadows { 0 } else { u32::MAX };
        renderer
            .scene_mut()
            .insert_actor(SceneActor::light(directional_light(
                Vec3::new(0.38, 0.82, 0.44).normalize().to_array(),
                [1.0, 1.0, 1.0],
                3000.0,
                shadow_index,
            )));

        let cube = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(cube_mesh(CUBE * 0.5)))
            .as_mesh()
            .unwrap();
        let half = (GRID - 1) as f32 / 2.0;
        let (_, from, to) = interop_demo::PALETTES[0];
        let mut cubes = Vec::with_capacity(GRID * GRID);
        for y in 0..GRID {
            for x in 0..GRID {
                let (fx, fy) = (x as f32 - half, y as f32 - half);
                let t = ((fx + fy + GRID as f32) / (2.0 * GRID as f32)).clamp(0.0, 1.0);
                let color = [
                    from[0] + (to[0] - from[0]) * t,
                    from[1] + (to[1] - from[1]) * t,
                    from[2] + (to[2] - from[2]) * t,
                    1.0,
                ];
                let material = renderer
                    .scene_mut()
                    .insert_material(make_mat(color, 1.0, 0.0, [0.0; 3], 0.0));
                cubes.push(add_obj(
                    renderer,
                    cube,
                    material,
                    Mat4::from_translation(Vec3::new(fx * SPACING, fy * SPACING, 0.0)),
                    CUBE * 0.5,
                    Some(Movability::Movable),
                ));
            }
        }
        Animation::Field { cubes }
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

    fn directional_light(
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        shadow_index: u32,
    ) -> GpuLight {
        GpuLight {
            direction_outer: [direction[0], direction[1], direction[2], -1.0],
            color_intensity: [color[0], color[1], color[2], intensity],
            shadow_index,
            light_type: LightType::Directional as u32,
            ..Default::default()
        }
    }

    fn pt_light(
        pos: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
        shadow_index: u32,
    ) -> GpuLight {
        GpuLight {
            position_range: [pos[0], pos[1], pos[2], range],
            direction_outer: [0.0, 0.0, -1.0, 0.0],
            color_intensity: [color[0], color[1], color[2], intensity],
            shadow_index,
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
        probe: bool,
        stats: bool,
        scene_kind: SceneKind,
        shadows: Option<bool>,
        graph: GraphKind,
        logged: Rc<RefCell<Instant>>,
    }

    impl Demo {
        fn new(
            cx: &mut Context<Self>,
            probe: bool,
            stats: bool,
            scene: SceneKind,
            shadows: Option<bool>,
            graph: GraphKind,
        ) -> Self {
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
                probe,
                stats,
                scene_kind: scene,
                shadows,
                graph,
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
                    self.scene_kind,
                    self.shadows,
                    self.graph,
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
                "Helio's wgpu 30 renderer, exported as a dma-buf to GPUI's wgpu 29 device.",
            ))
            .child(div().h(px(2.)).flex_none())
            .child(
                div()
                    .text_2xl()
                    .text_color(rgb(ACCENT))
                    .child(format!("{:.1} FPS", metrics.fps)),
            )
            .child(div().text_xs().text_color(rgb(MUTED)).child(format!(
                "{:.2} ms · Helio {:.2} · fence {:.2}",
                metrics.interval_ms, metrics.render_ms, metrics.fence_ms
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
            .child(metric_row("export", format!("{:.2} ms", metrics.export_ms)))
            .child(metric_row(
                "fence wait",
                format!("{:.2} ms", metrics.fence_ms),
            ))
            .child(metric_row(
                "GPUI + pacing",
                format!("{:.2} ms", metrics.gpui_ms),
            ))
            .child(div().h(px(4.)).flex_none())
            .child(section_label("Producer"))
            .child(metric_row(
                "size",
                format!("{} × {}", metrics.width, metrics.height),
            ))
            .child(metric_row("adapter", metrics.adapter.clone()))
            .child(metric_row("hand-off", "dma-buf · zero copy".to_owned()))
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
        scene_kind: SceneKind,
        shadows: Option<bool>,
        graph: GraphKind,
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
                match HelioScene::build(gpu, width, height, probe, scene_kind, shadows, graph) {
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
                    Ok(Some((render, export, fence))) => {
                        metrics.borrow_mut().record(
                            render,
                            export,
                            fence,
                            scene.width,
                            scene.height,
                        );
                    }
                    Ok(None) => {}
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
                        "interop_helio: window {} frames ({} dropped) — {:.1} fps, {:.2} ms \
                         interval, helio {:.2} ms, export {:.2} ms, fence {:.2} ms, gpui {:.2} \
                         ms, {}x{}",
                        metrics.frames,
                        scene.dropped,
                        metrics.fps,
                        metrics.interval_ms,
                        metrics.render_ms,
                        metrics.export_ms,
                        metrics.fence_ms,
                        metrics.gpui_ms,
                        scene.width,
                        scene.height,
                    );
                }
            }

            scene.paint(gpu);
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
            .child(overlay_badge("helio · wgpu 30 → dma-buf → GPUI"))
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
                move |_, cx| {
                    cx.new(|cx| {
                        Demo::new(cx, cli.probe, cli.stats, cli.scene, cli.shadows, cli.graph)
                    })
                },
            )
            .expect("a window to composite into");
            cx.activate(true);
        });
    }
}
