//! `interop_bevy`: a Bevy scene inside a GPUI window.
//!
//! ```sh
//! cargo run -p interop_bevy
//! ```
//!
//! Bevy is built **headless**: `default-features = false`, no `bevy_winit`, no window, no event
//! loop of its own. The app is an ordinary Bevy `App` — `RenderPlugin`, `CameraPlugin`,
//! `CorePipelinePlugin`, `PbrPlugin`, the asset and glTF plugins — with a `Camera3d` whose
//! `RenderTarget` is an offscreen `Image`.
//!
//! The image is created on **the window renderer's own wgpu device**. The render resources are
//! handed to Bevy through `RenderCreation::Manual`, so Bevy's meshes, materials, pipelines and
//! render passes all run on the device GPUI already draws with. A render-world system reads the
//! image's `GpuImage` and publishes its `TextureView`; GPUI composites that view with
//! `paint_texture`. There is no export, no fence, and no CPU copy — the pixels never leave the
//! device.
//!
//! # What the device has to expose
//!
//! A PBR frame needs more than a hand-written pipeline: 5–8 storage buffers per stage, a 64 KiB
//! uniform binding, 16 inter-stage variables, and adapter-specific storage-texture formats
//! (`R16Float`/`Rgba16Float`). `gpui_wgpu` raises those limits to the adapter's reported values on
//! native and enables `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` where available. On a downlevel
//! device Bevy's bind-group layouts fail validation, its render world quits, and the image stays
//! all zeros; `--probe` detects that and `bevy_log` prints the validation error.
//!
//! # Who owns the loop
//!
//! The windowed example pumps Bevy one `SubApps::update` per GPUI paint: GPUI owns the cadence,
//! Bevy does not spawn a thread or block. That is the embedding model, and it is why `winit` is
//! not compiled in at all. The `--stats` line prints the GPUI and Bevy tick counters together;
//! they stay equal, so the frame interval is GPUI's presentation cadence, not a second Bevy tick.
//! When the platform reports the window inactive, GPUI's frame source caps it at 33.3 ms; the
//! `window active` and `present wait` fields make that visible.
//!
//! `--bench <secs>` is the other model: it builds the same headless Bevy app but lets
//! `ScheduleRunnerPlugin` own the loop, with no GPUI and no presentation. Each frame polls the
//! device to completion, so the number is a real GPU frame rate rather than a submission rate. It
//! prints the rate and the adapter Bevy chose, then exits. `--scene clear` (or `one`) is the
//! control: it removes the 49-cube field so the engine's own per-frame cost can be separated from
//! the scene's. A windowed run with `--stats` prints the presented rate plus the pipeline's
//! evaluate/layout/paint split, the draw and present-wait split, and whether the platform still
//! considers the window active; `--probe` reads the rendered image back once and prints its centre
//! pixel and mean, which distinguishes "Bevy drew nothing" from "GPUI sampled the wrong handle".
//!
//! `--transfer` is the dma-buf shape: Bevy renders into a `WgpuSharedBuffer` through a manual
//! texture view, the buffer's `sync_file` is exported after each update, and `paint_surface`
//! imports it. The default (no flag) is the same-device shape, where GPUI samples the image's
//! `TextureView` with `paint_texture`. The transfer export costs ~0.14–0.8 ms/frame here and the
//! import ~0.02 ms, so transport is not what separates a hand-written wgpu pipeline from Bevy.
//!
//! In release on the test laptop the engine floor is ~5.9 ms/frame on the Intel HD 520 and
//! ~3.6 ms on the NVIDIA 930M; the 49-cube field adds only ~0.4 ms. The same field as a
//! hand-written wgpu producer costs 1.29 ms and 0.41 ms, which is where the producer gap lives.
//! Both paths present at the display's 60 Hz in a window.
//!
//! # Sizing
//!
//! The canvas is a physical-pixel producer: each paint reads its bounds and the window's scale
//! factor, and the target follows. The image path changes the `Image` descriptor in place and Bevy
//! recreates the texture; the transfer path rebuilds the exportable buffer and its manual view.
//! The windowed viewport is therefore resizable without stretching the source. The headless bench
//! keeps a fixed 960 × 600 so producer runs compare.
//!
//! `--scene clear` and `--scene one` apply to `--bench`, where they isolate the engine from the
//! field; the window always uses whatever `--scene` was passed.
//!
//! ```sh
//! cargo run -p interop_bevy -- --bench 10
//! cargo run -p interop_bevy -- --gpu 1 --bench 10
//! cargo run -p interop_bevy -- --gpu 1 --scene clear --bench 10
//! cargo run -p interop_bevy -- --gpu 1 --stats
//! cargo run -p interop_bevy -- --gpu 1 --transfer --stats
//! ```
//!
//! Compare that with a normal windowed run's frame metrics: the bench is the producer ceiling
//! (Bevy render + GPU completion), while the window's `frame` interval is producer plus GPUI's
//! composite and presentation. The same-device path means there is no transfer to attribute.
//!
//! # Which GPU
//!
//! `--list-gpus`, `--gpu <index>`, `--gpu-id <hex>` and `--gpu-name <text>` select the adapter,
//! exactly as `interop_wgpu` does. In a windowed run the preference selects the window's renderer
//! before it makes a device. In `--bench` it selects the adapter Bevy requests by name; the bench
//! prints the name it actually got.

#[cfg(target_os = "linux")]
fn main() {
    demo::run();
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!(
        "interop_bevy is a Linux example: it renders Bevy offscreen on the window renderer's wgpu \
         device and hands the texture view to GPUI. Run it on Linux."
    );
}

#[cfg(target_os = "linux")]
mod demo {
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use bevy::DefaultPlugins;
    use bevy::a11y::AccessibilityPlugin;
    use bevy::app::TerminalCtrlCHandlerPlugin;
    use bevy::app::{
        App as BevyApp, AppExit, Last, PluginGroup as _, ScheduleRunnerPlugin, Startup, SubApps,
        Update,
    };
    use bevy::asset::{Assets, Handle, RenderAssetUsages};
    use bevy::camera::{Camera3d, ClearColor, ManualTextureViewHandle, RenderTarget};
    use bevy::color::Color;
    use bevy::core_pipeline::tonemapping::Tonemapping;
    use bevy::diagnostic::DiagnosticsPlugin;
    use bevy::input::InputPlugin;
    // Bevy's derive macros emit `bevy_ecs::…`, and their manifest lookup only sees a direct
    // `bevy_ecs` dependency. This example depends on the `bevy` umbrella, so the re-export is
    // aliased to the name the macros expand to.
    use bevy::ecs as bevy_ecs;
    use bevy::ecs::component::Component;
    use bevy::ecs::message::MessageWriter;
    use bevy::ecs::resource::Resource;
    use bevy::ecs::schedule::IntoScheduleConfigs;
    use bevy::ecs::system::{Commands, Query, Res, ResMut};
    use bevy::image::Image;
    use bevy::light::{DirectionalLight, GlobalAmbientLight};
    use bevy::math::{EulerRot, Quat, UVec2, Vec3};
    use bevy::mesh::{Mesh, Mesh3d};
    use bevy::pbr::{MeshMaterial3d, StandardMaterial};
    use bevy::prelude::default;
    use bevy::render::render_asset::RenderAssets;
    use bevy::render::renderer::{
        RenderAdapter, RenderAdapterInfo, RenderDevice, RenderInstance, RenderQueue, WgpuWrapper,
    };
    use bevy::render::settings::{RenderCreation, WgpuSettings};
    use bevy::render::texture::{GpuImage, ManualTextureView, ManualTextureViews};
    use bevy::render::{Render as BevyRender, RenderApp, RenderPlugin, RenderSystems};
    use bevy::time::Time;
    use bevy::transform::components::Transform;
    use bevy::window::{ExitCondition, WindowPlugin};
    use gpui::{
        AnyElement, App, Bounds, Context, Corners, Entity, FocusId, FramePipeline,
        FramePipelineExt, IntoElement, PhaseMetrics, PreparedRoots, Render,
        StandardImmediatePipeline, Window as GpuiWindow, WindowBounds, WindowMetrics,
        WindowOptions, div, gpu_canvas, prelude::*, px, rgb, size,
    };
    use gpui_engine::DmaBufHandle;
    use gpui_interop::{WgpuSharedBuffer, attach};
    use gpui_wgpu::{ImportedTextureExt as _, WgpuRenderer};
    use interop_demo::{
        ACCENT, LEFT_WIDTH, MUTED, control, error_panel, histogram, histogram_caption,
        histogram_style, metric_row, metrics_panel_style, overlay_badge, overlay_hint, pacing_note,
        page, panel, section_label, title, viewport_frame,
    };

    /// The headless bench's fixed size. The windowed viewport follows its canvas bounds instead,
    /// so the two are only directly comparable when the window happens to be this size.
    const VIEW_WIDTH: u32 = 960;
    const VIEW_HEIGHT: u32 = 600;

    /// The headless bench size, kept fixed so producer runs compare.
    const BENCH_WIDTH: u32 = VIEW_WIDTH;
    const BENCH_HEIGHT: u32 = VIEW_HEIGHT;

    const GRID: i32 = 7;

    /// The CLI. `selector` is kept raw as well as lowered to a `GpuPreference`: the windowed path
    /// uses the preference before a renderer exists, while `--bench` has no GPUI application and
    /// resolves the same choice against Bevy's own adapter enumeration.
    struct Cli {
        list: bool,
        preference: Option<gpui::GpuPreference>,
        selector: Option<Selector>,
        bench: Option<Duration>,
        stats: bool,
        probe: bool,
        scene: SceneKind,
        transfer: bool,
    }

    /// What the bench renders. `Clear` is the control: a camera and a clear pass with no meshes,
    /// so the engine's own per-frame cost can be separated from the PBR field.
    #[derive(Clone, Copy, Resource, Debug, Default)]
    enum SceneKind {
        #[default]
        Field,
        One,
        Clear,
    }

    fn parse_scene(value: &str) -> SceneKind {
        match value {
            "field" => SceneKind::Field,
            "one" => SceneKind::One,
            "clear" => SceneKind::Clear,
            other => usage_error(&format!("--scene: {other:?} is not field, one, or clear")),
        }
    }

    #[derive(Clone)]
    enum Selector {
        Index(usize),
        Device(u32),
        Name(String),
    }

    fn parse_args() -> Cli {
        let mut cli = Cli {
            list: false,
            preference: None,
            selector: None,
            bench: None,
            stats: false,
            probe: false,
            scene: SceneKind::default(),
            transfer: false,
        };
        let mut args = std::env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--list-gpus" => cli.list = true,
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
                    cli.selector = Some(Selector::Index(index));
                }
                "--gpu-id" => {
                    let value = args
                        .next()
                        .unwrap_or_else(|| usage_error("--gpu-id needs a PCI device id"));
                    let Some(device) = gpui::GpuId::parse_pci_id(&value) else {
                        usage_error(&format!(
                            "--gpu-id: {value:?} is not four hexadecimal digits"
                        ));
                    };
                    cli.preference =
                        Some(gpui::GpuPreference::Device(gpui::GpuId::by_device(device)));
                    cli.selector = Some(Selector::Device(device));
                }
                "--gpu-name" => {
                    let value = args
                        .next()
                        .unwrap_or_else(|| usage_error("--gpu-name needs text"));
                    let needle = value.to_lowercase();
                    cli.preference = Some(gpui::GpuPreference::Custom(Arc::new(
                        move |gpus: &[gpui::GpuAdapterInfo]| {
                            gpus.iter()
                                .position(|gpu| gpu.name.to_lowercase().contains(&needle))
                        },
                    )));
                    cli.selector = Some(Selector::Name(value));
                }
                "--bench" => {
                    let value = args.next().unwrap_or_else(|| {
                        usage_error("--bench needs a duration in seconds, e.g. --bench 10")
                    });
                    let seconds: f64 = value.parse().unwrap_or_else(|_| {
                        usage_error(&format!("--bench: {value:?} is not a number of seconds"))
                    });
                    cli.bench = Some(Duration::from_secs_f64(seconds.max(0.1)));
                }
                "--stats" => cli.stats = true,
                "--probe" => cli.probe = true,
                "--scene" => {
                    let value = args
                        .next()
                        .unwrap_or_else(|| usage_error("--scene needs field, one, or clear"));
                    cli.scene = parse_scene(&value);
                }
                "--transfer" => cli.transfer = true,
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
        eprintln!("interop_bevy: {message}\n");
        print_help();
        std::process::exit(2);
    }

    fn print_help() {
        println!(concat!(
            "interop_bevy — a headless Bevy scene, composited by GPUI on the same wgpu device\n",
            "\n",
            "Usage: interop_bevy [OPTIONS]\n",
            "\n",
            "  --list-gpus          enumerate the adapters this process can render on, then exit\n",
            "  --gpu <INDEX>        use the adapter at INDEX from --list-gpus\n",
            "  --gpu-id <HEX>       use the adapter with this four-digit hexadecimal PCI device id\n",
            "  --gpu-name <TEXT>    use the first adapter whose name contains TEXT (case-insensitive)\n",
            "  --bench <SECS>       run headless Bevy alone through its schedule runner; print the\n",
            "                       producer rate, then exit without opening a window or presenting\n",
            "  --stats              print the windowed frame rate to stderr once a second\n",
            "  --probe              read the rendered image back once and print its centre pixel\n",
            "  --scene <KIND>       bench scene: field (49 cubes), one, or clear (camera + clear only)\n",
            "  --transfer           render into a dma-buf and import it with paint_surface instead of\n",
            "                       sampling an Image on the same device (windowed mode)\n",
            "  -h, --help           show this help",
        ));
    }

    /// Print the adapters with the index, class and PCI id the selectors take.
    fn list_gpus(cx: &App) {
        let gpus = cx.enumerate_gpus();
        if gpus.is_empty() {
            println!("no adapters reported");
            return;
        }
        println!(
            "{:<3} {:<46} {:<8} {:<11} {:<18} path",
            "#", "adapter", "backend", "type", "pci"
        );
        for (index, gpu) in gpus.iter().enumerate() {
            let pci = match (gpu.vendor_id(), gpu.device_id()) {
                (Some(vendor), Some(device)) => format!("{vendor:04x}:{device:04x}"),
                (None, Some(device)) => format!("????:{device:04x}"),
                _ => "-".to_owned(),
            };
            let software = if gpu.is_software { " (software)" } else { "" };
            // A windowed run takes the same-device path whatever the backend: Bevy and GPUI share
            // one device, so the frame never leaves it.
            println!(
                "{index:<3} {:<46} {:<8} {:<11} {pci:<18} same-device{software}",
                gpu.name,
                format!("{:?}", gpu.backend),
                format!("{:?}", gpu.device_type),
            );
        }
        println!(
            "\nSelect with --gpu <#>, --gpu-id <device id, e.g. 1916>, or --gpu-name <text>. \
             --bench resolves the same selector against Bevy's adapter list."
        );
    }

    /// The one thing that crosses from Bevy's render world to GPUI: the image's texture view.
    ///
    /// A `TextureView` is a handle; cloning it is an `Arc` increment, and the texture behind it is
    /// on the window renderer's own device, so GPUI can sample it directly.
    #[derive(Resource, Clone, Default)]
    struct SharedFrame(Arc<Mutex<Option<FrameTexture>>>);

    /// The raw texture and its view. The view is what GPUI samples; the texture is what the probe
    /// copies back, because a view cannot be a copy source on its own.
    #[derive(Clone)]
    struct FrameTexture {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    }

    /// Where the camera renders. The manual variant is a view over a `WgpuSharedBuffer`, which is
    /// exported as a dma-buf after each Bevy update.
    #[derive(Resource, Clone)]
    enum SessionTarget {
        Image(Handle<Image>),
        Manual(ManualTextureViewHandle),
    }

    enum SceneOutput {
        Image(SharedFrame),
        Transfer(Box<TransferOutput>),
    }

    struct TransferOutput {
        buffer: WgpuSharedBuffer,
        /// The handle exported after the most recent update.
        handle: Option<DmaBufHandle>,
    }

    /// The offscreen image every camera renders into. Kept in the render world so the publisher
    /// can find the prepared `GpuImage` without a second asset path.
    #[derive(Resource, Clone)]
    struct TargetImage(Handle<Image>);

    /// A cube's place in the wave, so the animation is a function of time alone.
    #[derive(Component)]
    struct Floating {
        phase: f32,
    }

    /// Render one Bevy frame per GPUI paint, then wait for the GPU so the view GPUI samples is the
    /// frame that was just submitted.
    struct BevyScene {
        sub_apps: SubApps,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        output: SceneOutput,
        gpu: String,
        adapter: wgpu::AdapterInfo,
        probe: bool,
        probed: bool,
        frames: u64,
        width: u32,
        height: u32,
    }

    impl BevyScene {
        fn build(
            gpu: &mut gpui::GpuCanvasContext,
            width: u32,
            height: u32,
            config: ViewConfig,
        ) -> anyhow::Result<Self> {
            let interop = attach(gpu.try_device::<WgpuRenderer>())
                .map_err(|unavailable| anyhow::anyhow!("{unavailable:?}"))?;
            let (device, queue) = interop
                .adapter()
                .wgpu()
                .ok_or_else(|| anyhow::anyhow!("the window's renderer is not wgpu"))?;
            let gpu_name = interop.adapter().gpu_id().map(|id| id.name);
            let (creation, adapter) = manual_creation(&device, &queue, gpu_name.as_deref())?;
            eprintln!(
                "interop_bevy: bevy render resources on {} ({:?}), {}",
                adapter.name,
                adapter.backend,
                if config.transfer {
                    "exporting a dma-buf each frame"
                } else {
                    "sharing the window device"
                },
            );

            let mut app = base_app(creation);
            let output = if config.transfer {
                let buffer = WgpuSharedBuffer::new(
                    &device,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                )?;
                let manual = ManualTextureViewHandle(1);
                app.world_mut().init_resource::<ManualTextureViews>();
                app.world_mut().resource_mut::<ManualTextureViews>().insert(
                    manual,
                    ManualTextureView::with_default_format(
                        buffer.view().clone().into(),
                        UVec2::new(width, height),
                    ),
                );
                add_scene(&mut app, width, height, None, config.scene, Some(manual));
                SceneOutput::Transfer(Box::new(TransferOutput {
                    buffer,
                    handle: None,
                }))
            } else {
                let frame = SharedFrame::default();
                add_scene(
                    &mut app,
                    width,
                    height,
                    Some(frame.clone()),
                    config.scene,
                    None,
                );
                SceneOutput::Image(frame)
            };
            app.finish();
            app.cleanup();
            let sub_apps = std::mem::take(app.sub_apps_mut());

            Ok(Self {
                sub_apps,
                device,
                queue,
                output,
                gpu: adapter.name.clone(),
                adapter,
                probe: config.probe,
                probed: false,
                frames: 0,
                width,
                height,
            })
        }

        /// Follow the canvas's physical size. The image's descriptor is changed in place, so the
        /// camera keeps its target handle and Bevy's render asset recreates the texture; the
        /// transfer path replaces the exportable buffer and its manual view instead.
        fn resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
            let width = width.max(1);
            let height = height.max(1);
            if (width, height) == (self.width, self.height) {
                return Ok(());
            }

            match &mut self.output {
                SceneOutput::Image(_) => {
                    let session = self
                        .sub_apps
                        .main
                        .world()
                        .resource::<SessionTarget>()
                        .clone();
                    let SessionTarget::Image(handle) = session else {
                        return Ok(());
                    };
                    let world = self.sub_apps.main.world_mut();
                    let mut images = world.resource_mut::<Assets<Image>>();
                    let Some(mut image) = images.get_mut(&handle) else {
                        return Ok(());
                    };
                    image.texture_descriptor.size = wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    };
                }
                SceneOutput::Transfer(transfer) => {
                    let buffer = WgpuSharedBuffer::new(
                        &self.device,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8UnormSrgb,
                    )?;
                    let session = self
                        .sub_apps
                        .main
                        .world()
                        .resource::<SessionTarget>()
                        .clone();
                    let SessionTarget::Manual(manual) = session else {
                        return Ok(());
                    };
                    self.sub_apps
                        .main
                        .world_mut()
                        .resource_mut::<ManualTextureViews>()
                        .insert(
                            manual,
                            ManualTextureView::with_default_format(
                                buffer.view().clone().into(),
                                UVec2::new(width, height),
                            ),
                        );
                    transfer.buffer = buffer;
                    transfer.handle = None;
                }
            }

            self.width = width;
            self.height = height;
            Ok(())
        }

        /// Advance Bevy once and wait for its frame. The durations are the engine's CPU/queue work,
        /// the GPU completion GPUI would otherwise discover at presentation, and (on the transfer
        /// path) the dma-buf export.
        fn step(&mut self) -> anyhow::Result<(Duration, Duration, Duration)> {
            let update = Instant::now();
            self.sub_apps.update();
            self.frames += 1;
            let update = update.elapsed();

            let wait = Instant::now();
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(10)),
                })
                .map_err(|error| anyhow::anyhow!("waiting for Bevy's frame: {error}"))?;
            let wait = wait.elapsed();

            let transport = Instant::now();
            if let SceneOutput::Transfer(transfer) = &mut self.output {
                // Bevy has already submitted its render on this queue. An empty submission after
                // it is enough to signal the export semaphore once that work is complete.
                let encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("interop_bevy_transfer"),
                    });
                transfer.handle = Some(
                    transfer
                        .buffer
                        .submit(&self.queue, Some(encoder.finish()))?,
                );
            }
            Ok((update, wait, transport.elapsed()))
        }

        /// Hand this frame to GPUI: the same-device view, or the dma-buf exported after the update.
        fn paint(&mut self, gpu: &mut gpui::GpuCanvasContext) -> anyhow::Result<()> {
            match &mut self.output {
                SceneOutput::Image(frame) => {
                    let Some(frame) = frame.0.lock().ok().and_then(|slot| slot.clone()) else {
                        return Ok(());
                    };
                    let handle = frame.view.to_imported_handle()?;
                    gpu.paint_texture(handle, Corners::default(), 1.0, false);
                }
                SceneOutput::Transfer(transfer) => {
                    // A paused frame does not advance Bevy, so there is no newly exported handle;
                    // exporting the buffer again with an empty submission gives this paint a fresh
                    // fence for the frame it is already holding.
                    if transfer.handle.is_none() {
                        let encoder =
                            self.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("interop_bevy_transfer"),
                                });
                        transfer.handle = Some(
                            transfer
                                .buffer
                                .submit(&self.queue, Some(encoder.finish()))?,
                        );
                    }
                    if let Some(handle) = transfer.handle.take() {
                        gpu.paint_surface(handle);
                    }
                }
            }
            Ok(())
        }

        /// Read the rendered image back once and print its centre pixel and mean. A black viewport
        /// is either "Bevy drew nothing" or "GPUI sampled the wrong handle"; this distinguishes
        /// them without a screenshot tool. The transfer path exports only render-attachment usage,
        /// so it has no copy source to read; the window itself is the check there.
        fn probe(&mut self, force: bool) -> anyhow::Result<()> {
            if !force && (!self.probe || self.probed || self.frames < 30) {
                return Ok(());
            }
            let SceneOutput::Image(frame) = &self.output else {
                self.probed = true;
                eprintln!("interop_bevy: probe is only available on the image path");
                return Ok(());
            };
            let Some(frame) = frame.0.lock().ok().and_then(|slot| slot.clone()) else {
                return Ok(());
            };
            self.probed = true;

            let width = frame.texture.width();
            let height = frame.texture.height();
            let unpadded = width * 4;
            let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let size = u64::from(padded) * u64::from(height);
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("interop_bevy_probe"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("interop_bevy_probe"),
                });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &frame.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(10)),
            })?;

            let slice = buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(10)),
            })?;
            receiver.recv()??;

            let data = slice.get_mapped_range();
            let stride = padded as usize;
            let centre_offset = (height as usize / 2) * stride + (width as usize / 2) * 4;
            let centre = [
                data[centre_offset],
                data[centre_offset + 1],
                data[centre_offset + 2],
                data[centre_offset + 3],
            ];
            let mut sum = 0u64;
            let mut samples = 0u64;
            for row in 0..height as usize {
                let base = row * stride;
                for column in 0..width as usize {
                    let offset = base + column * 4;
                    sum += u64::from(data[offset])
                        + u64::from(data[offset + 1])
                        + u64::from(data[offset + 2]);
                    samples += 3;
                }
            }
            drop(data);
            buffer.unmap();

            eprintln!(
                "interop_bevy: probe {}x{}: centre {centre:?}, mean channel {:.2}",
                width,
                height,
                sum as f64 / samples.max(1) as f64,
            );
            Ok(())
        }
    }

    /// Build the headless app: no `winit`, no window, and the caller's choice of who creates the
    /// render resources.
    fn base_app(creation: RenderCreation) -> BevyApp {
        let mut app = BevyApp::new();
        app.add_plugins(
            DefaultPlugins
                // A camera can target an image without a window, but several Bevy pieces still
                // expect `WindowPlugin` to exist. No primary window, and never exit just because
                // there is none.
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                // Compile pipelines on the first frame. An example should show the scene, not a
                // dozen black frames while the task pool works through the shaders.
                .set(RenderPlugin {
                    render_creation: creation,
                    synchronous_pipeline_compilation: true,
                    ..default()
                })
                // The window is GPUI's, and this producer has no input, diagnostics, accessibility
                // tree or glTF loader of its own. Dropping those plugins keeps the per-frame ECS
                // cost to the render path. The terminal Ctrl-C handler is also Bevy's: with GPUI
                // owning the process, a second Ctrl-C reaches it after the first has already
                // started a shutdown, and Bevy logs "Received more than one ctrl+c". Let the
                // process's default signal handling own that instead.
                .disable::<TerminalCtrlCHandlerPlugin>()
                .disable::<DiagnosticsPlugin>()
                .disable::<InputPlugin>()
                .disable::<AccessibilityPlugin>(),
        );
        app
    }

    /// Add the offscreen image, the camera and the scene, plus (for the windowed path) the
    /// render-world publisher.
    fn add_scene(
        app: &mut BevyApp,
        width: u32,
        height: u32,
        publisher: Option<SharedFrame>,
        scene: SceneKind,
        manual: Option<ManualTextureViewHandle>,
    ) {
        if let Some(manual) = manual {
            app.insert_resource(SessionTarget::Manual(manual));
        } else {
            let mut image = Image::new_uninit(
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::RENDER_WORLD,
            );
            // `new_uninit` already asks for `TEXTURE_BINDING | COPY_DST | COPY_SRC`; the camera
            // needs the image to be a render attachment, and GPUI needs the binding bit to sample
            // it.
            image.texture_descriptor.usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;

            let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
            app.insert_resource(TargetImage(target.clone()));
            app.insert_resource(SessionTarget::Image(target));
        }

        app.insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 180.0,
            ..default()
        });
        app.insert_resource(scene);
        // A dark blue that is recognisably not the GPUI background, so a rendered-but-empty
        // image is distinguishable from one the canvas never sampled.
        app.insert_resource(ClearColor(Color::srgb(0.035, 0.055, 0.10)));
        app.add_systems(Startup, spawn_scene);
        app.add_systems(Update, animate);

        if let Some(frame) = publisher {
            let target = match app.world().resource::<SessionTarget>() {
                SessionTarget::Image(handle) => handle.clone(),
                SessionTarget::Manual(_) => unreachable!("a publisher is image mode only"),
            };
            app.insert_resource(frame.clone());
            let render = app.sub_app_mut(RenderApp);
            render.insert_resource(frame);
            render.insert_resource(TargetImage(target));
            // The image is prepared before the graph runs, so publishing in `Render` is enough for
            // the main thread, which reads the slot after `SubApps::update` has returned.
            render.add_systems(BevyRender, publish_frame.in_set(RenderSystems::Render));
        }
    }

    /// The render-world half of the hand-off.
    ///
    /// `GpuImage.texture_view` is Bevy's wrapper around a `wgpu::TextureView`; dereferencing it
    /// gives the raw handle GPUI's importer takes.
    fn publish_frame(
        target: Res<TargetImage>,
        images: Res<RenderAssets<GpuImage>>,
        frame: Res<SharedFrame>,
    ) {
        let Some(image) = images.get(&target.0) else {
            return;
        };
        if let Ok(mut slot) = frame.0.lock() {
            *slot = Some(FrameTexture {
                texture: (*image.texture).clone(),
                view: (*image.texture_view).clone(),
            });
        }
    }

    /// A 7 × 7 field of lit cubes over a plane-less dark background. The materials are standard
    /// PBR, the light is directional, and the camera is fixed; only the cubes move.
    fn spawn_scene(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        session: Res<SessionTarget>,
        scene: Res<SceneKind>,
    ) {
        let cube = meshes.add(bevy::math::primitives::Cuboid::new(1.0, 1.0, 1.0));
        let half = (GRID - 1) as f32 / 2.0;

        let count = match *scene {
            SceneKind::Field => GRID,
            SceneKind::One => 1,
            SceneKind::Clear => 0,
        };
        if count == 1 {
            commands.spawn((
                Mesh3d(cube),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.55, 0.75, 1.0),
                    metallic: 0.35,
                    perceptual_roughness: 0.28,
                    ..default()
                })),
                Transform::from_xyz(0.0, 0.0, 0.0),
                Floating { phase: 0.0 },
            ));
        } else {
            for y in 0..count {
                for x in 0..count {
                    let (fx, fy) = (x as f32 - half, y as f32 - half);
                    let t = ((fx + fy + GRID as f32) / (2.0 * GRID as f32)).clamp(0.0, 1.0);
                    // `interop_wgpu`'s "Ember" palette: a warm-to-cool gradient across the field.
                    let colour = Color::srgb(
                        1.00 + (0.15 - 1.00) * t,
                        0.42 + (0.55 - 0.42) * t,
                        0.15 + (1.00 - 0.15) * t,
                    );
                    commands.spawn((
                        Mesh3d(cube.clone()),
                        MeshMaterial3d(materials.add(StandardMaterial {
                            base_color: colour,
                            metallic: 0.35,
                            perceptual_roughness: 0.28,
                            ..default()
                        })),
                        Transform::from_xyz(fx * 1.75, fy * 1.75, 0.0),
                        Floating {
                            phase: fx * 0.42 + fy * 0.31,
                        },
                    ));
                }
            }
        }

        commands.spawn((
            DirectionalLight {
                illuminance: 12_000.0,
                shadow_maps_enabled: false,
                ..default()
            },
            Transform::from_xyz(6.0, 10.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));

        commands.spawn((
            Camera3d::default(),
            match &*session {
                SessionTarget::Image(handle) => RenderTarget::Image(handle.clone().into()),
                SessionTarget::Manual(handle) => RenderTarget::TextureView(*handle),
            },
            // AcesFitted is a shader-only curve: `TonyMcMapface` would want the LUT assets that
            // the `tonemapping_luts` feature embeds, which a default-features-off app does not
            // carry.
            Tonemapping::AcesFitted,
            Transform::from_xyz(5.24, 4.49, 8.56).looking_at(Vec3::ZERO, Vec3::Y),
        ));
    }

    fn animate(time: Res<Time>, mut cubes: Query<(&mut Transform, &Floating)>) {
        let elapsed = time.elapsed_secs();
        for (mut transform, floating) in &mut cubes {
            let phase = elapsed * 1.25 + floating.phase;
            transform.translation.z = phase.sin() * 0.42;
            transform.rotation = Quat::from_euler(EulerRot::YXZ, phase * 0.62, phase * 0.44, 0.0);
            transform.scale = Vec3::splat(0.40 + 0.05 * (phase * 0.9).sin());
        }
    }

    /// Build render resources on GPUI's device.
    ///
    /// The device and queue are the window renderer's, so they are used as-is. Bevy also wants the
    /// adapter and instance behind them; GPUI does not hand those out through
    /// `GpuCanvasContext::try_device`, so a matching adapter is found on a fresh instance. It is
    /// used only for capability queries and pipeline compilation, never to create a device.
    fn manual_creation(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        gpu_name: Option<&str>,
    ) -> anyhow::Result<(RenderCreation, wgpu::AdapterInfo)> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN | wgpu::Backends::GL;
        let backends = descriptor.backends;
        let instance = wgpu::Instance::new(descriptor);

        let mut adapters = bevy::tasks::block_on(instance.enumerate_adapters(backends));
        let index = gpu_name
            .and_then(|name| {
                adapters
                    .iter()
                    .position(|adapter| adapter.get_info().name == name)
            })
            .or_else(|| {
                adapters
                    .iter()
                    .position(|adapter| adapter.get_info().device_type != wgpu::DeviceType::Cpu)
            })
            .or_else(|| (!adapters.is_empty()).then_some(0))
            .ok_or_else(|| anyhow::anyhow!("no wgpu adapter to describe the window's device"))?;
        let adapter = adapters.swap_remove(index);
        let info = adapter.get_info();

        let creation = RenderCreation::manual(
            RenderDevice::from((**device).clone()),
            RenderQueue(Arc::new(WgpuWrapper::new((**queue).clone()))),
            RenderAdapterInfo(WgpuWrapper::new(info.clone())),
            RenderAdapter(Arc::new(WgpuWrapper::new(adapter))),
            RenderInstance(Arc::new(WgpuWrapper::new(instance))),
        );
        Ok((creation, info))
    }

    /// Resolve a CLI selector to an adapter name for Bevy's automatic renderer.
    fn bench_adapter(selector: &Selector) -> Option<String> {
        let gpus = gpui_wgpu::enumerate_adapters();
        match selector {
            Selector::Index(index) => gpus.get(*index).map(|gpu| gpu.name.clone()),
            Selector::Device(device) => gpus
                .iter()
                .find(|gpu| gpu.device_id() == Some(*device))
                .map(|gpu| gpu.name.clone()),
            Selector::Name(text) => {
                let needle = text.to_lowercase();
                gpus.iter()
                    .find(|gpu| gpu.name.to_lowercase().contains(&needle))
                    .map(|gpu| gpu.name.clone())
            }
        }
    }

    /// The unbounded producer measurement: headless Bevy, its own `ScheduleRunnerPlugin` loop, no
    /// GPUI and no presentation.
    #[derive(Resource)]
    struct Bench {
        limit: Duration,
        warmup: u32,
        started: Option<Instant>,
        frames: u64,
        report: Arc<Mutex<Option<String>>>,
    }

    fn bench_tick(
        device: Res<RenderDevice>,
        adapter: Res<RenderAdapterInfo>,
        mut bench: ResMut<Bench>,
        mut exit: MessageWriter<AppExit>,
    ) {
        // `ScheduleRunnerPlugin` calls `App::update`, which runs the render app and submits; the
        // poll makes the loop wait for the GPU, so this is a frame rate rather than a submission
        // rate. On a lost device the error is reported instead of counting bogus frames.
        if let Err(error) = device.wgpu_device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(10)),
        }) {
            *bench.report.lock().unwrap() = Some(format!(
                "interop_bevy: bench stopped — waiting for the GPU failed: {error}"
            ));
            exit.write(AppExit::Success);
            return;
        }

        let Some(started) = bench.started else {
            if bench.warmup > 0 {
                bench.warmup -= 1;
            } else {
                bench.started = Some(Instant::now());
            }
            return;
        };

        bench.frames += 1;
        let elapsed = started.elapsed();
        if elapsed >= bench.limit {
            let seconds = elapsed.as_secs_f64().max(1e-9);
            let fps = bench.frames as f64 / seconds;
            *bench.report.lock().unwrap() = Some(format!(
                "interop_bevy: bench {} frames in {:.3}s — {:.1} fps, {:.3} ms/frame on {} ({:?}); \
                 bevy only, no GPUI and no presentation",
                bench.frames,
                seconds,
                fps,
                seconds * 1000.0 / bench.frames.max(1) as f64,
                adapter.name,
                adapter.backend,
            ));
            exit.write(AppExit::Success);
        }
    }

    fn bench(limit: Duration, adapter_name: Option<String>, scene: SceneKind) {
        let settings = WgpuSettings {
            // Bevy only matches `adapter_name` when the backend list is explicit.
            backends: Some(wgpu::Backends::all()),
            adapter_name,
            ..default()
        };
        let mut app = base_app(RenderCreation::Automatic(Box::new(settings)));
        add_scene(&mut app, BENCH_WIDTH, BENCH_HEIGHT, None, scene, None);

        let report = Arc::new(Mutex::new(None));
        app.insert_resource(Bench {
            limit,
            // Let the pipeline cache and the asset extraction settle before the clock starts.
            warmup: 15,
            started: None,
            frames: 0,
            report: report.clone(),
        });
        app.add_systems(Last, bench_tick);
        // The headless counterpart of `winit`: a fixed cadence would only add sleep, so this is
        // `Duration::ZERO` and the loop runs as fast as the GPU completes.
        app.add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO));
        let _ = app.run();

        if let Some(line) = report.lock().unwrap().take() {
            eprintln!("{line}");
        }
    }

    /// The metrics the view renders. Intervals are wall-clock between completed Bevy frames, so
    /// they include GPUI's own work on the same thread.
    #[derive(Clone, Default)]
    struct Metrics {
        frames: u64,
        interval_ms: f32,
        update_ms: f32,
        wait_ms: f32,
        transport_ms: f32,
        paint_ms: f32,
        evaluate_ms: f32,
        layout_ms: f32,
        pipeline_paint_ms: f32,
        draw_ms: f32,
        present_wait_ms: f32,
        active: bool,
        fps: f32,
        intervals: VecDeque<f32>,
        last: Option<Instant>,
        gpui_frames: u64,
    }

    impl Metrics {
        fn record(
            &mut self,
            update: Duration,
            wait: Duration,
            transport: Duration,
            paint: Duration,
        ) {
            self.frames += 1;
            self.update_ms = update.as_secs_f32() * 1000.0;
            self.wait_ms = wait.as_secs_f32() * 1000.0;
            self.transport_ms = transport.as_secs_f32() * 1000.0;
            self.paint_ms = paint.as_secs_f32() * 1000.0;

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

    /// Root-pass durations, copied out of `PhaseMetrics` once a frame.
    #[derive(Clone, Copy, Default)]
    struct PhaseCosts {
        evaluate_ms: f32,
        layout_ms: f32,
        paint_ms: f32,
        draw_ms: f32,
        active: bool,
        frames: u64,
    }

    /// Records the root passes and forwards every one to `inner`. The same decorator
    /// `interop_wgpu` uses: `StandardImmediatePipeline.instrumented(PhaseMetrics)` accumulates the
    /// durations, and this frame's cost is the difference from the previous frame's snapshot.
    struct MetricsPipeline<P> {
        inner: P,
        phases: Rc<RefCell<PhaseMetrics>>,
        costs: Rc<RefCell<PhaseCosts>>,
        last: PhaseMetrics,
        started: Option<Instant>,
        active: bool,
        frames: u64,
    }

    impl<P> MetricsPipeline<P> {
        fn new(
            inner: P,
            phases: Rc<RefCell<PhaseMetrics>>,
            costs: Rc<RefCell<PhaseCosts>>,
        ) -> Self {
            Self {
                inner,
                phases,
                costs,
                last: PhaseMetrics::default(),
                started: None,
                active: false,
                frames: 0,
            }
        }
    }

    impl<P: FramePipeline> FramePipeline for MetricsPipeline<P> {
        fn should_render(&mut self, is_dirty: bool, metrics: &WindowMetrics) -> bool {
            self.active = metrics.is_active;
            self.inner.should_render(is_dirty, metrics)
        }

        fn begin_frame(&mut self, window: &mut GpuiWindow, cx: &mut App) {
            self.last = *self.phases.borrow();
            self.started = Some(Instant::now());
            // Count here, before the canvas paints, so the "gpui" and "bevy" counters in the
            // canvas compare the same frame rather than being one apart by sampling order.
            self.frames += 1;
            self.costs.borrow_mut().frames = self.frames;
            self.inner.begin_frame(window, cx);
        }

        fn evaluate_roots(&mut self, window: &mut GpuiWindow, cx: &mut App) -> PreparedRoots {
            self.inner.evaluate_roots(window, cx)
        }

        fn layout_roots(
            &mut self,
            window: &mut GpuiWindow,
            roots: &mut PreparedRoots,
            cx: &mut App,
        ) {
            self.inner.layout_roots(window, roots, cx);
        }

        fn paint_roots(&mut self, window: &mut GpuiWindow, roots: PreparedRoots, cx: &mut App) {
            self.inner.paint_roots(window, roots, cx);
        }

        fn finish_frame(&mut self, window: &mut GpuiWindow, cx: &mut App) {
            self.inner.finish_frame(window, cx);
        }

        fn complete_frame(&mut self, window: &mut GpuiWindow, cx: &mut App) -> Option<FocusId> {
            self.inner.complete_frame(window, cx)
        }

        fn end_frame(&mut self, window: &mut GpuiWindow, cx: &mut App, focus: Option<FocusId>) {
            self.inner.end_frame(window, cx, focus);

            let phases = *self.phases.borrow();
            let since =
                |now: std::time::Duration, then: std::time::Duration| now.saturating_sub(then);
            let mut costs = self.costs.borrow_mut();
            costs.evaluate_ms = since(phases.evaluate, self.last.evaluate).as_secs_f32() * 1000.0;
            costs.layout_ms = since(phases.layout, self.last.layout).as_secs_f32() * 1000.0;
            costs.paint_ms = since(phases.paint, self.last.paint).as_secs_f32() * 1000.0;
            costs.draw_ms = self
                .started
                .take()
                .map(|started| started.elapsed().as_secs_f32() * 1000.0)
                .unwrap_or(0.0);
            costs.active = self.active;
            drop(costs);
            self.last = phases;
        }
    }

    /// The right column's live text, updated at 5 Hz rather than every animation frame.
    #[derive(Clone, Default)]
    struct HeaderInfo {
        gpu: String,
        backend: String,
        hand_off: String,
    }

    /// The right column: the title, the live metrics and the producer's path. Its text is cached,
    /// so the shapes are rebuilt only when the snapshot is refreshed (5×/s).
    struct MetricsPanel {
        info: Rc<RefCell<HeaderInfo>>,
        metrics: Rc<RefCell<Option<Metrics>>>,
    }

    impl Render for MetricsPanel {
        fn render(
            &mut self,
            _window: &mut GpuiWindow,
            _cx: &mut Context<Self>,
        ) -> impl IntoElement {
            let info = self.info.borrow().clone();
            let metrics = self.metrics.borrow().clone().unwrap_or_default();
            metrics_section(&info, &metrics)
        }
    }

    /// The left column's cached interval histogram.
    struct HistogramPanel {
        metrics: Rc<RefCell<Option<Metrics>>>,
    }

    impl Render for HistogramPanel {
        fn render(
            &mut self,
            _window: &mut GpuiWindow,
            _cx: &mut Context<Self>,
        ) -> impl IntoElement {
            let metrics = self.metrics.borrow().clone().unwrap_or_default();
            histogram_section(&metrics)
        }
    }

    #[derive(Clone, Copy)]
    struct ViewConfig {
        log_stats: bool,
        probe: bool,
        scene: SceneKind,
        transfer: bool,
    }

    struct BevyDemo {
        scene: Rc<RefCell<Option<BevyScene>>>,
        error: Rc<RefCell<Option<String>>>,
        metrics: Rc<RefCell<Metrics>>,
        panel_metrics: Rc<RefCell<Option<Metrics>>>,
        header_info: Rc<RefCell<HeaderInfo>>,
        metrics_panel: Entity<MetricsPanel>,
        histogram_panel: Entity<HistogramPanel>,
        phase_costs: Rc<RefCell<PhaseCosts>>,
        displayed_at: Option<Instant>,
        config: ViewConfig,
        logged: Rc<RefCell<Instant>>,
        /// Whether Bevy is stopped. Paused, the canvas keeps presenting the last frame.
        paused: Rc<Cell<bool>>,
        /// A one-shot read-back requested from the left panel.
        probe_requested: Rc<Cell<bool>>,
    }

    impl BevyDemo {
        fn new(
            cx: &mut Context<Self>,
            phase_costs: Rc<RefCell<PhaseCosts>>,
            config: ViewConfig,
        ) -> Self {
            let panel_metrics = Rc::new(RefCell::new(None));
            let header_info = Rc::new(RefCell::new(HeaderInfo::default()));
            let metrics_panel = cx.new(|_| MetricsPanel {
                info: header_info.clone(),
                metrics: panel_metrics.clone(),
            });
            let histogram_panel = cx.new(|_| HistogramPanel {
                metrics: panel_metrics.clone(),
            });
            Self {
                scene: Rc::new(RefCell::new(None)),
                error: Rc::new(RefCell::new(None)),
                metrics: Rc::new(RefCell::new(Metrics::default())),
                panel_metrics,
                header_info,
                metrics_panel,
                histogram_panel,
                phase_costs,
                displayed_at: None,
                config,
                logged: Rc::new(RefCell::new(Instant::now())),
                paused: Rc::new(Cell::new(false)),
                probe_requested: Rc::new(Cell::new(false)),
            }
        }
    }

    impl Render for BevyDemo {
        fn render(&mut self, window: &mut GpuiWindow, cx: &mut Context<Self>) -> impl IntoElement {
            let failed = self.error.borrow().is_some();
            if !failed && !self.paused.get() {
                // The canvas must run every frame: its paint is what advances Bevy and samples
                // the produced texture. Paused, only a control asks for another frame.
                window.request_animation_frame();
            }

            if self
                .displayed_at
                .is_none_or(|at| at.elapsed() >= Duration::from_millis(200))
            {
                let (gpu, backend, hand_off) = {
                    let scene = self.scene.borrow();
                    match scene.as_ref() {
                        Some(scene) => (
                            scene.gpu.clone(),
                            format!("{:?}", scene.adapter.backend),
                            if self.config.transfer {
                                "dma-buf + sync_file".to_owned()
                            } else {
                                "same-device texture".to_owned()
                            },
                        ),
                        None => ("—".to_owned(), "—".to_owned(), "—".to_owned()),
                    }
                };
                *self.header_info.borrow_mut() = HeaderInfo {
                    gpu,
                    backend,
                    hand_off,
                };

                let mut metrics = self.metrics.borrow().clone();
                let costs = *self.phase_costs.borrow();
                metrics.evaluate_ms = costs.evaluate_ms;
                metrics.layout_ms = costs.layout_ms;
                metrics.pipeline_paint_ms = costs.paint_ms;
                metrics.draw_ms = costs.draw_ms;
                metrics.present_wait_ms = (metrics.interval_ms - costs.draw_ms).max(0.0);
                metrics.active = costs.active;
                metrics.gpui_frames = costs.frames;
                *self.panel_metrics.borrow_mut() = Some(metrics);
                self.displayed_at = Some(Instant::now());
                self.metrics_panel.update(cx, |_, cx| cx.notify());
                self.histogram_panel.update(cx, |_, cx| cx.notify());
            }

            page(
                controls_panel(
                    cx,
                    self.paused.clone(),
                    self.probe_requested.clone(),
                    self.histogram_panel.clone(),
                ),
                viewport(
                    self.scene.clone(),
                    self.error.clone(),
                    self.metrics.clone(),
                    self.phase_costs.clone(),
                    self.config,
                    self.logged.clone(),
                    self.paused.clone(),
                    self.probe_requested.clone(),
                ),
                self.metrics_panel.clone().cached(metrics_panel_style()),
            )
        }
    }

    /// The left column: the pause and probe controls, and the cached interval histogram.
    fn controls_panel(
        cx: &mut Context<BevyDemo>,
        paused: Rc<Cell<bool>>,
        probe_requested: Rc<Cell<bool>>,
        histogram_panel: Entity<HistogramPanel>,
    ) -> AnyElement {
        let pause_label = if paused.get() { "Resume" } else { "Pause" };
        let pause =
            control("pause", pause_label).on_click(cx.listener(move |_this, _, window, cx| {
                paused.set(!paused.get());
                window.request_animation_frame();
                cx.notify();
            }));
        let probe = control("probe", "Probe").on_click(cx.listener(move |_this, _, window, cx| {
            probe_requested.set(true);
            window.request_animation_frame();
            cx.notify();
        }));

        panel(LEFT_WIDTH)
            .child(section_label("Controls"))
            .child(pause)
            .child(probe)
            .child(div().h(px(8.)).flex_none())
            .child(histogram_panel.cached(histogram_style()))
            .into_any_element()
    }

    /// The left column's histogram block: the interval bars and where the interval went.
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
                metrics.draw_ms,
                metrics.present_wait_ms,
                metrics.active,
            ))
            .into_any_element()
    }

    /// The right column: title, frame rate, the pipeline's breakdown and the engine's own timings.
    fn metrics_section(info: &HeaderInfo, metrics: &Metrics) -> AnyElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap_1()
            .child(title(
                "Bevy, inside GPUI",
                "A headless Bevy app renders into a texture on the window renderer's own device.",
            ))
            .child(div().h(px(2.)).flex_none())
            .child(
                div()
                    .text_2xl()
                    .text_color(rgb(ACCENT))
                    .child(format!("{:.1} FPS", metrics.fps)),
            )
            .child(div().text_xs().text_color(rgb(MUTED)).child(format!(
                "{:.2} ms · bevy {:.2} · gpu {:.2} · layout {:.2}",
                metrics.interval_ms, metrics.update_ms, metrics.wait_ms, metrics.layout_ms
            )))
            .child(div().h(px(4.)).flex_none())
            .child(section_label("Frame"))
            .child(metric_row(
                "frame interval",
                format!("{:.2} ms", metrics.interval_ms),
            ))
            .child(metric_row("draw", format!("{:.2} ms", metrics.draw_ms)))
            .child(metric_row(
                "present wait",
                format!("{:.2} ms", metrics.present_wait_ms),
            ))
            .child(metric_row(
                "evaluate",
                format!("{:.2} ms", metrics.evaluate_ms),
            ))
            .child(metric_row("layout", format!("{:.2} ms", metrics.layout_ms)))
            .child(metric_row(
                "pipeline paint",
                format!("{:.2} ms", metrics.pipeline_paint_ms),
            ))
            .child(div().h(px(4.)).flex_none())
            .child(section_label("Engine"))
            .child(metric_row(
                "gpui frames",
                format!("{}", metrics.gpui_frames),
            ))
            .child(metric_row("bevy frames", format!("{}", metrics.frames)))
            .child(metric_row(
                "engine step",
                format!("{:.2} ms", metrics.update_ms),
            ))
            .child(metric_row("gpu wait", format!("{:.2} ms", metrics.wait_ms)))
            .child(metric_row(
                "transport",
                format!("{:.2} ms", metrics.transport_ms),
            ))
            .child(metric_row(
                "texture paint",
                format!("{:.2} ms", metrics.paint_ms),
            ))
            .child(div().h(px(4.)).flex_none())
            .child(section_label("Adapter"))
            .child(metric_row(
                "gpu",
                format!("{} · {}", info.gpu, info.backend),
            ))
            .child(metric_row("hand-off", info.hand_off.clone()))
            .child(metric_row(
                "present rate",
                format!("{:.1} fps", metrics.fps),
            ))
            .child(metric_row(
                "window",
                if metrics.active { "active" } else { "inactive" }.to_owned(),
            ))
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn viewport(
        scene: Rc<RefCell<Option<BevyScene>>>,
        error: Rc<RefCell<Option<String>>>,
        metrics: Rc<RefCell<Metrics>>,
        phase_costs: Rc<RefCell<PhaseCosts>>,
        config: ViewConfig,
        logged: Rc<RefCell<Instant>>,
        paused: Rc<Cell<bool>>,
        probe_requested: Rc<Cell<bool>>,
    ) -> AnyElement {
        let drawing_error = error.clone();
        let overlay_error = error;
        let paused_for_canvas = paused.clone();
        let canvas = gpu_canvas(move |gpu| {
            // The canvas is a physical-pixel producer: follow its current bounds, not the size it
            // happened to have when the app was built. A resize recreates the backing texture.
            let bounds = gpu.bounds();
            let scale = gpu.scale_factor();
            let width = (bounds.size.width.0 * scale).round().max(1.0) as u32;
            let height = (bounds.size.height.0 * scale).round().max(1.0) as u32;

            let mut slot = scene.borrow_mut();
            if slot.is_none() && drawing_error.borrow().is_none() {
                match BevyScene::build(gpu, width, height, config) {
                    Ok(built) => *slot = Some(built),
                    Err(build) => {
                        let message = format!("{build:#}");
                        eprintln!("interop_bevy: cannot build the Bevy scene: {message}");
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
                eprintln!("interop_bevy: cannot resize the render target: {message}");
                *drawing_error.borrow_mut() = Some(message);
                return;
            }

            let stepped = !paused_for_canvas.get();
            let (update, wait, transport) = if stepped {
                match scene.step() {
                    Ok(timings) => timings,
                    Err(error) => {
                        let message = format!("{error:#}");
                        eprintln!("interop_bevy: cannot render a Bevy frame: {message}");
                        *drawing_error.borrow_mut() = Some(message);
                        return;
                    }
                }
            } else {
                (Duration::ZERO, Duration::ZERO, Duration::ZERO)
            };

            if let Err(error) = scene.probe(probe_requested.take()) {
                eprintln!("interop_bevy: probe failed: {error:#}");
            }

            let paint = Instant::now();
            if let Err(error) = scene.paint(gpu) {
                let message = format!("{error:#}");
                eprintln!("interop_bevy: cannot hand the frame to GPUI: {message}");
                *drawing_error.borrow_mut() = Some(message);
                return;
            }
            if stepped {
                metrics
                    .borrow_mut()
                    .record(update, wait, transport, paint.elapsed());
            }

            if config.log_stats {
                let now = Instant::now();
                let mut last = logged.borrow_mut();
                if now.duration_since(*last) >= Duration::from_secs(1) {
                    *last = now;
                    let metrics = metrics.borrow();
                    let costs = *phase_costs.borrow();
                    eprintln!(
                        "interop_bevy: window {} frames — {:.1} fps, {:.2} ms interval, \
                         draw {:.2} ms, present wait {:.2} ms, active {}, \
                         gpui {} / bevy {} ticks, \
                         bevy {:.2} ms, gpu {:.2} ms, transport {:.2} ms, texture {:.2} ms, \
                         evaluate {:.2} ms, layout {:.2} ms, pipeline paint {:.2} ms",
                        metrics.frames,
                        metrics.fps,
                        metrics.interval_ms,
                        costs.draw_ms,
                        (metrics.interval_ms - costs.draw_ms).max(0.0),
                        costs.active,
                        costs.frames,
                        metrics.frames,
                        metrics.update_ms,
                        metrics.wait_ms,
                        metrics.transport_ms,
                        metrics.paint_ms,
                        costs.evaluate_ms,
                        costs.layout_ms,
                        costs.paint_ms,
                    );
                }
            }
        })
        .size_full();

        viewport_frame()
            .child(canvas)
            .children(overlay_error.borrow().clone().map(|message| {
                error_panel(
                    "The Bevy viewport could not start",
                    message,
                    "Run with --list-gpus to see the adapters; --transfer switches to the \
                     dma-buf hand-off.",
                )
            }))
            .child(overlay_badge(if config.transfer {
                "bevy · dma-buf · 0 CPU copies"
            } else {
                "bevy · same-device texture · 0 CPU copies"
            }))
            .child(overlay_hint(if paused.get() {
                "paused · GPUI presents the last frame"
            } else {
                "headless bevy 0.19 · pumped by GPUI's paint"
            }))
            .into_any_element()
    }

    pub fn run() {
        let cli = parse_args();
        if let Some(limit) = cli.bench {
            let adapter = cli.selector.as_ref().and_then(bench_adapter);
            if cli.selector.is_some() && adapter.is_none() {
                eprintln!(
                    "interop_bevy: no adapter matches the requested selector; letting Bevy choose"
                );
            }
            bench(limit, adapter, cli.scene);
            if cli.transfer {
                eprintln!(
                    "interop_bevy: --transfer applies to the windowed path; the bench uses Bevy's \
                     own device and does not export"
                );
            }
            return;
        }

        let list = cli.list;
        let config = ViewConfig {
            log_stats: cli.stats,
            probe: cli.probe,
            scene: cli.scene,
            transfer: cli.transfer,
        };
        let phases = Rc::new(RefCell::new(PhaseMetrics::default()));
        let costs = Rc::new(RefCell::new(PhaseCosts::default()));
        let app = gpui::application().with_frame_pipeline({
            let costs = costs.clone();
            move |_window_id| {
                Box::new(MetricsPipeline::new(
                    StandardImmediatePipeline.instrumented(phases.clone()),
                    phases.clone(),
                    costs.clone(),
                ))
            }
        });
        let app = match cli.preference {
            Some(preference) => app.with_gpu_preference(preference),
            None => app,
        };
        app.run(move |cx: &mut App| {
            if list {
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
                    let costs = costs.clone();
                    cx.new(move |cx| BevyDemo::new(cx, costs, config))
                },
            )
            .expect("a window to composite into");
            cx.activate(true);
        });
    }
}
