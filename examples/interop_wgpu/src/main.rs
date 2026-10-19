//! `interop_wgpu`: a wgpu scene inside a GPUI window, with GPUI controls in front of it.
//!
//! ```sh
//! cargo run -p interop_wgpu
//! ```
//!
//! The window is a real 3D viewport: an instanced field of lit cubes, turning on a clock or dragged
//! by the pointer, with depth testing and a perspective camera. An ordinary `wgpu` pipeline renders
//! it — the same API a standalone wgpu application uses — into an **exportable** image, and
//! `WgpuSharedBuffer` hands GPUI the frame as a dma-buf with its DRM modifier, plane layout and a
//! `sync_file` fence. The window's renderer imports that native handle and composites it; no CPU
//! copy, and no wgpu type crosses the boundary.
//!
//! The chrome is the shared demo shell: real buttons (pause, speed, palette, reset) and a frame
//! interval histogram in the left column, live metrics in the right, and two non-interactive
//! overlays on the viewport. The columns and overlays sit outside the producer's hitbox, so a click
//! on a control is a click and a drag anywhere else reaches the scene.
//!
//! Drag the viewport to orbit the camera, scroll to zoom, or use the controls. Pausing stops the
//! clock but not interaction: a drag still turns the camera, and the next frame is exported.
//!
//! # Two hand-offs, one scene
//!
//! On a Vulkan device that can export, the frame leaves as a dma-buf with its modifier, plane layout
//! and a `sync_file` fence. On an adapter that cannot export — a GL device, or a Vulkan driver
//! without the external-memory extensions — the scene renders into a `WgpuSurface` and
//! `paint_texture` samples the same-device view instead. GPUI's own drawing needs neither export nor
//! copy in either case; the metrics panel says which path the frame took.
//!
//! # Raw throughput
//!
//! `--bench <secs>` renders the scene into the chosen output, waits for the GPU each frame, prints
//! the producer's rate and exits, without presenting a frame. On the dma-buf path that includes the
//! export and fence; on the same-device path it is the wgpu render alone. It is the unbounded
//! number, independent of vsync and the window loop.
//!
//! The consumer's half is a normal run's `paint` (the renderer importing/compositing the frame) and
//! `wait` (presentation — on an Optimus laptop, the PRIME transfer to the display). Compare the
//! two GPUs both ways: `--bench` for the producer ceiling, a normal run for the import and present
//! cost.
//!
//! ```sh
//! cargo run -p interop_wgpu -- --gpu 0 --bench 10
//! cargo run -p interop_wgpu -- --gpu 1 --bench 10
//! ```
//!
//! # Choosing the GPU
//!
//! `--list-gpus` enumerates the adapters this process can render on and exits; `--gpu <index>`
//! selects one from that list, and `--gpu-id <hex>` or `--gpu-name <text>` name one directly. Each
//! lowers to a `GpuPreference` on the application, before the first window makes a device. The
//! adapter the renderer actually chose is printed at startup and shown in the metrics panel.
//!
//! ```sh
//! cargo run -p interop_wgpu -- --list-gpus
//! cargo run -p interop_wgpu -- --gpu 1
//! ```

#[cfg(target_os = "linux")]
fn main() {
    demo::run();
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!(
        "interop_wgpu is a Linux example: it allocates an exportable Vulkan image through the wgpu \
         device and exports it as a dma-buf for the window's renderer. Run it on Linux."
    );
}

#[cfg(target_os = "linux")]
mod demo {
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use gpui::{
        AnyElement, App, Bounds, Context, Corners, Entity, FocusId, FramePipeline,
        FramePipelineExt, IntoElement, PhaseMetrics, PreparedRoots, Render, ScrollDelta,
        StandardImmediatePipeline, Window, WindowBounds, WindowMetrics, WindowOptions, div,
        gpu_canvas, prelude::*, px, rgb, size,
    };
    use gpui_interop::{
        SurfaceEvent, SurfaceInput, SurfacePoint, WgpuSharedBuffer, WgpuSurface, attach,
        surface_input,
    };
    use gpui_wgpu::{ImportedTextureExt as _, WgpuRenderer};
    use interop_demo::{
        ACCENT, LEFT_WIDTH, MUTED, control, error_panel, histogram, histogram_caption,
        histogram_style, metric_row, metrics_panel_style, overlay_badge, overlay_hint, pacing_note,
        page, panel, section_label, title, viewport_frame,
    };
    use wgpu::util::DeviceExt as _;

    /// The headless bench's fixed size, so producer runs are comparable between the two demos.
    /// The windowed viewport follows its canvas's physical size instead.
    const BENCH_WIDTH: u32 = 960;
    const BENCH_HEIGHT: u32 = 600;

    /// The grid of cubes: `GRID` × `GRID`, centred on the origin.
    const GRID: i32 = 7;
    const INSTANCES: u32 = (GRID * GRID) as u32;

    /// How far the camera orbits per physical pixel dragged, and how far the wheel moves it per notch.
    const SENSITIVITY: f32 = 0.007;
    const ZOOM_PER_NOTCH: f32 = 1.12;
    const PIXELS_PER_LINE: f32 = 40.0;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    /// Three colour pairs the instances interpolate between, cycled by the left panel's palette
    /// button.
    const PALETTES: [(&str, [f32; 3], [f32; 3]); 3] = [
        ("Ember", [1.00, 0.42, 0.15], [0.15, 0.55, 1.00]),
        ("Tide", [0.10, 0.90, 0.72], [0.16, 0.32, 1.00]),
        ("Mono", [0.92, 0.93, 1.00], [0.34, 0.36, 0.46]),
    ];

    /// The same WGSL a standalone wgpu application would use: a cube mesh with normals, per-instance
    /// model matrices and colours, one directional light, and the camera's view-projection.
    const SHADER: &str = r#"
        struct Uniforms {
            view_projection: mat4x4<f32>,
        };
        @group(0) @binding(0) var<uniform> uniforms: Uniforms;

        struct VertexOut {
            @builtin(position) position: vec4<f32>,
            @location(0) colour: vec3<f32>,
            @location(1) normal: vec3<f32>,
        };

        @vertex
        fn vs_main(
            @location(0) position: vec3<f32>,
            @location(1) normal: vec3<f32>,
            @location(2) m0: vec4<f32>,
            @location(3) m1: vec4<f32>,
            @location(4) m2: vec4<f32>,
            @location(5) m3: vec4<f32>,
            @location(6) colour: vec4<f32>,
        ) -> VertexOut {
            let model = mat4x4<f32>(m0, m1, m2, m3);
            let world = model * vec4<f32>(position, 1.0);
            var out: VertexOut;
            out.position = uniforms.view_projection * world;
            out.colour = colour.rgb;
            out.normal = normalize((model * vec4<f32>(normal, 0.0)).xyz);
            return out;
        }

        @fragment
        fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
            let light = normalize(vec3<f32>(0.38, 0.82, 0.44));
            let diffuse = max(dot(normalize(in.normal), light), 0.0);
            // A little cool ambient and a stronger key light: the cubes read as volumes rather than
            // flat tiles, and the shadows between them come from the depth test.
            let lit = in.colour * (0.20 + 0.80 * diffuse) + vec3<f32>(0.03, 0.05, 0.09);
            return vec4<f32>(lit, 1.0);
        }
    "#;

    /// The command line. The example parses its three selectors itself rather than pulling in a
    /// dependency: `--list-gpus` is the enumeration, and each selector lowers to a `GpuPreference`.
    struct Cli {
        list: bool,
        preference: Option<gpui::GpuPreference>,
        bench: Option<Duration>,
    }

    fn parse_args() -> Cli {
        let mut cli = Cli {
            list: false,
            preference: None,
            bench: None,
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
                    // Resolved against the list the renderer itself enumerates, so the index the
                    // application prints is the index it selects.
                    cli.preference = Some(gpui::GpuPreference::Custom(Arc::new(
                        move |gpus: &[gpui::GpuAdapterInfo]| (index < gpus.len()).then_some(index),
                    )));
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
        eprintln!("interop_wgpu: {message}\n");
        print_help();
        std::process::exit(2);
    }

    fn print_help() {
        println!(concat!(
            "interop_wgpu — a wgpu scene in a GPUI window, handed over as a dma-buf\n",
            "\n",
            "Usage: interop_wgpu [OPTIONS]\n",
            "\n",
            "  --list-gpus          enumerate the adapters this process can render on, then exit\n",
            "  --gpu <INDEX>        use the adapter at INDEX from --list-gpus\n",
            "  --gpu-id <HEX>       use the adapter with this four-digit hexadecimal PCI device id\n",
            "  --gpu-name <TEXT>    use the first adapter whose name contains TEXT (case-insensitive)\n",
            "  --bench <SECS>       render flat out without presenting; print producer throughput, then exit\n",
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
            // The demo attempts the dma-buf hand-off on a Vulkan adapter; a wgpu device on
            // another backend can render the same scene, but it has no exportable image to hand
            // over. Whether a Vulkan driver can export is only known once a device exists, so a
            // Vulkan adapter that cannot falls back to the same-device view at startup.
            let path = if gpu.backend == gpui::GpuBackend::Vulkan {
                "dma-buf"
            } else {
                "same-device"
            };
            println!(
                "{index:<3} {:<46} {:<8} {:<11} {pci:<18} {path}{software}",
                gpu.name,
                format!("{:?}", gpu.backend),
                format!("{:?}", gpu.device_type),
            );
        }
        println!(
            "\nVulkan adapters are tried as dma-buf producers and fall back to the same-device \
             view when the driver cannot export; the others always use the same-device view. \
             Select with --gpu <#>, --gpu-id <device id, e.g. 1916>, or --gpu-name <text>."
        );
    }

    /// How many frame intervals the histogram keeps.
    const MAX_SAMPLES: usize = 120;

    /// Frame intervals, the pipeline work each interval contained, and the window state the frame
    /// source saw. Newest last.
    #[derive(Default)]
    struct FrameStats {
        intervals_ms: VecDeque<f32>,
        work_ms: VecDeque<f32>,
        frames: u64,
        asked: u64,
        active: bool,
        draw_ms: f32,
        evaluate_ms: f32,
        layout_ms: f32,
        paint_ms: f32,
        started: Option<Instant>,
        last: Option<Instant>,
        pending_interval_ms: Option<f32>,
    }

    impl FrameStats {
        /// Record that the pipeline was asked about a frame, and whether the platform considered
        /// the window active. The frame source throttles inactive windows before this is
        /// consulted, so a stalled rate shows up here as `asked` staying put.
        fn record_ask(&mut self, metrics: &WindowMetrics) {
            self.asked += 1;
            self.active = metrics.is_active;
        }

        /// Open a frame: count it and remember the interval since the previous one.
        fn begin(&mut self, now: Instant) {
            self.frames += 1;
            self.started = Some(now);
            self.pending_interval_ms = self
                .last
                .replace(now)
                .map(|last| (now - last).as_secs_f32() * 1000.0);
        }

        /// Close a frame: record how long the pipeline drove it against the interval it opened
        /// with. The draw spans `begin_frame` to `end_frame`, so it includes the root passes and
        /// the finishing work but not presentation, which happens after the pipeline returns.
        fn end(
            &mut self,
            evaluate: std::time::Duration,
            layout: std::time::Duration,
            paint: std::time::Duration,
        ) {
            self.draw_ms = self
                .started
                .take()
                .map(|started| started.elapsed().as_secs_f32() * 1000.0)
                .unwrap_or(0.0);
            self.evaluate_ms = evaluate.as_secs_f32() * 1000.0;
            self.layout_ms = layout.as_secs_f32() * 1000.0;
            self.paint_ms = paint.as_secs_f32() * 1000.0;

            // The first frame has no previous frame to measure from, so it contributes no sample.
            if let Some(interval) = self.pending_interval_ms.take() {
                self.intervals_ms.push_back(interval);
                self.work_ms.push_back(self.draw_ms);
                while self.intervals_ms.len() > MAX_SAMPLES {
                    self.intervals_ms.pop_front();
                    self.work_ms.pop_front();
                }
            }
        }

        /// The most recent interval, in milliseconds.
        fn latest_ms(&self) -> f32 {
            self.intervals_ms.back().copied().unwrap_or(0.0)
        }

        /// The root passes of the most recent frame, in milliseconds.
        fn latest_passes_ms(&self) -> f32 {
            self.evaluate_ms + self.layout_ms + self.paint_ms
        }

        /// The part of the interval the pipeline did not spend driving the frame: presentation,
        /// vsync, and the frame source's pacing.
        fn latest_wait_ms(&self) -> f32 {
            (self.latest_ms() - self.draw_ms).max(0.0)
        }

        /// The rate implied by the average interval.
        fn fps(&self) -> f32 {
            let count = self.intervals_ms.len();
            if count == 0 {
                return 0.0;
            }
            let average = self.intervals_ms.iter().sum::<f32>() / count as f32;
            if average <= 0.0 {
                0.0
            } else {
                1000.0 / average
            }
        }
    }

    /// The numbers the view renders, copied out of the pipeline's cells once per frame.
    #[derive(Clone)]
    struct Metrics {
        fps: f32,
        frame_ms: f32,
        draw_ms: f32,
        wait_ms: f32,
        passes_ms: f32,
        gpui_frames: u64,
        asked: u64,
        produced: u64,
        evaluate_ms: f32,
        layout_ms: f32,
        paint_ms: f32,
        active: bool,
        intervals: Vec<f32>,
        work: Vec<f32>,
        /// The adapter the window's renderer chose.
        gpu: String,
        /// How the frames reach GPUI: `dma-buf + sync_file` or `same-device view`.
        path: &'static str,
    }

    impl Metrics {
        fn new(stats: &FrameStats, produced: u64) -> Self {
            Self {
                fps: stats.fps(),
                frame_ms: stats.latest_ms(),
                draw_ms: stats.draw_ms,
                wait_ms: stats.latest_wait_ms(),
                passes_ms: stats.latest_passes_ms(),
                gpui_frames: stats.frames,
                asked: stats.asked,
                produced,
                evaluate_ms: stats.evaluate_ms,
                layout_ms: stats.layout_ms,
                paint_ms: stats.paint_ms,
                active: stats.active,
                intervals: stats.intervals_ms.iter().copied().collect(),
                work: stats.work_ms.iter().copied().collect(),
                gpu: String::new(),
                path: "",
            }
        }
    }

    /// Measures the interval between frames, the pass work inside each one, and forwards every
    /// pass to `inner`.
    ///
    /// This is the out-of-tree decorator pattern the crate documents: a pipeline can observe one
    /// policy decision and leave the drawing to the pipeline it wraps. The built-in
    /// [`InstrumentedPipeline`](gpui::InstrumentedPipeline), stacked underneath, accumulates the
    /// root-pass durations; taking the difference between two frames gives this frame's work, so
    /// the panel can say whether a slow frame is layout, paint, or waiting on presentation.
    ///
    /// Note what is *not* here: a frame the frame source throttles never reaches
    /// [`FramePipeline::should_render`], so a cap applied before the pipeline (an inactive window's
    /// `inactive_frame_interval`, thermal pressure) appears as a longer interval and more `wait`,
    /// not as pipeline work.
    struct MetricsPipeline<P> {
        inner: P,
        stats: Rc<RefCell<FrameStats>>,
        phases: Rc<RefCell<PhaseMetrics>>,
        last_phases: PhaseMetrics,
    }

    impl<P> MetricsPipeline<P> {
        fn new(
            inner: P,
            stats: Rc<RefCell<FrameStats>>,
            phases: Rc<RefCell<PhaseMetrics>>,
        ) -> Self {
            Self {
                inner,
                stats,
                phases,
                last_phases: PhaseMetrics::default(),
            }
        }
    }

    impl<P: FramePipeline> FramePipeline for MetricsPipeline<P> {
        fn should_render(&mut self, is_dirty: bool, metrics: &WindowMetrics) -> bool {
            self.stats.borrow_mut().record_ask(metrics);
            self.inner.should_render(is_dirty, metrics)
        }

        fn begin_frame(&mut self, window: &mut Window<'_>, cx: &mut App) {
            self.last_phases = *self.phases.borrow();
            self.stats.borrow_mut().begin(Instant::now());
            self.inner.begin_frame(window, cx);
        }

        fn evaluate_roots(&mut self, window: &mut Window<'_>, cx: &mut App) -> PreparedRoots {
            self.inner.evaluate_roots(window, cx)
        }

        fn layout_roots(
            &mut self,
            window: &mut Window<'_>,
            roots: &mut PreparedRoots,
            cx: &mut App,
        ) {
            self.inner.layout_roots(window, roots, cx);
        }

        fn paint_roots(&mut self, window: &mut Window<'_>, roots: PreparedRoots, cx: &mut App) {
            self.inner.paint_roots(window, roots, cx);
        }

        fn finish_frame(&mut self, window: &mut Window<'_>, cx: &mut App) {
            self.inner.finish_frame(window, cx);
        }

        fn complete_frame(&mut self, window: &mut Window<'_>, cx: &mut App) -> Option<FocusId> {
            self.inner.complete_frame(window, cx)
        }

        fn end_frame(&mut self, window: &mut Window<'_>, cx: &mut App, focus: Option<FocusId>) {
            self.inner.end_frame(window, cx, focus);

            let phases = *self.phases.borrow();
            let since =
                |now: std::time::Duration, then: std::time::Duration| now.saturating_sub(then);
            self.stats.borrow_mut().end(
                since(phases.evaluate, self.last_phases.evaluate),
                since(phases.layout, self.last_phases.layout),
                since(phases.paint, self.last_phases.paint),
            );
            self.last_phases = phases;
        }
    }

    pub fn run() {
        let cli = parse_args();
        let list = cli.list;
        let bench = cli.bench;

        let stats = Rc::new(RefCell::new(FrameStats::default()));
        let phases = Rc::new(RefCell::new(PhaseMetrics::default()));
        let app = gpui::application().with_frame_pipeline({
            let stats = stats.clone();
            move |_window_id| {
                Box::new(MetricsPipeline::new(
                    StandardImmediatePipeline.instrumented(phases.clone()),
                    stats.clone(),
                    phases.clone(),
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
                // The enumeration is the whole run: exit rather than waiting for an event loop
                // that has no window to drive.
                std::process::exit(0);
            }
            let view_stats = stats;
            let bounds = Bounds::centered(None, size(px(1280.), px(760.)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                move |_, cx| cx.new(move |cx| Showcase::new(view_stats, bench, cx)),
            )
            .expect("a window to composite into");
            cx.activate(true);
        });
    }

    /// The left column's histogram block, cached so the animation's per-frame render of the root
    /// does not re-lay-out its bars. Embedded with `.cached()`, it is a single definite-size
    /// layout node until the root notifies it (5×/s, when the snapshot is refreshed).
    struct HistogramPanel {
        metrics: Rc<RefCell<Option<Metrics>>>,
    }

    impl Render for HistogramPanel {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            match self.metrics.borrow().as_ref() {
                Some(metrics) => histogram_section(metrics),
                None => div().into_any_element(),
            }
        }
    }

    /// The right column: the demo's title, the live metrics and the producer's path. Cached for the
    /// same reason as the histogram.
    struct MetricsPanel {
        metrics: Rc<RefCell<Option<Metrics>>>,
    }

    impl Render for MetricsPanel {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            match self.metrics.borrow().as_ref() {
                Some(metrics) => metrics_section(metrics),
                None => div().into_any_element(),
            }
        }
    }

    /// The window's view. The scene is built lazily at paint time — the renderer's device exists only
    /// then — so the view holds a shared slot the canvas fills on the first frame.
    struct Showcase {
        scene: Rc<RefCell<Option<Scene>>>,
        /// Why the scene could not be built, when it could not: shown in place of the stats.
        error: Rc<RefCell<Option<String>>>,
        /// Whether the clock is stopped. Controls and input still render a frame.
        paused: Rc<Cell<bool>>,
        /// How fast the clock runs, cycled by the left panel.
        speed: Rc<Cell<f32>>,
        /// Which colour pair the instances interpolate between.
        palette: Rc<Cell<u8>>,
        /// The pointer's last surface-local position, so a move becomes the delta the camera orbits by.
        last: Rc<Cell<Option<SurfacePoint>>>,
        /// Whether this viewport founded the gesture — a drag that merely passes over it does not turn it.
        grab: Rc<Cell<bool>>,
        /// A couple of frames up front build the scene (or give up); after that the animation drives.
        requested: u32,
        /// The frame-interval and pass-timing collector the installed pipeline fills.
        stats: Rc<RefCell<FrameStats>>,
        /// The left column's histogram, cached as a child view.
        histogram_panel: Entity<HistogramPanel>,
        /// The right column's metrics, cached as a child view.
        metrics_panel: Entity<MetricsPanel>,
        /// The snapshot the cached panels render.
        panel_metrics: Rc<RefCell<Option<Metrics>>>,
        /// The metrics as last displayed in the right panel. Refreshed a few times a second rather than
        /// every frame: each changed string is a text-layout cache miss.
        display: Option<Metrics>,
        displayed_at: Instant,
        /// Raw producer bench: render into the output, wait for the GPU, and report throughput
        /// without presenting anything. A fixed limit rather than a flag so the scene build can
        /// take it; `None` is the normal animated view.
        bench: Option<Duration>,
    }

    impl Showcase {
        fn new(
            stats: Rc<RefCell<FrameStats>>,
            bench: Option<Duration>,
            cx: &mut Context<Self>,
        ) -> Self {
            let panel_metrics = Rc::new(RefCell::new(None));
            let histogram_panel = cx.new(|_| HistogramPanel {
                metrics: panel_metrics.clone(),
            });
            let metrics_panel = cx.new(|_| MetricsPanel {
                metrics: panel_metrics.clone(),
            });
            Self {
                scene: Rc::new(RefCell::new(None)),
                error: Rc::new(RefCell::new(None)),
                paused: Rc::new(Cell::new(false)),
                speed: Rc::new(Cell::new(1.0)),
                palette: Rc::new(Cell::new(0)),
                last: Rc::new(Cell::new(None)),
                grab: Rc::new(Cell::new(false)),
                requested: 0,
                stats,
                histogram_panel,
                metrics_panel,
                panel_metrics,
                display: None,
                displayed_at: Instant::now(),
                bench,
            }
        }
    }

    impl Render for Showcase {
        fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            let built = self.scene.borrow().is_some();
            let failed = self.error.borrow().is_some();
            // A moving scene needs a frame every tick; a paused one only when something changed — a
            // button, a drag, a scroll — and those paths ask for their own repaint. A failed build
            // stops the loop: the error panel is painted on the frame already requested.
            if !failed && (!self.paused.get() || (!built && self.requested < 2)) {
                if !built {
                    self.requested += 1;
                }
                window.request_animation_frame();
            }

            let (produced, gpu, path) = {
                let scene = self.scene.borrow();
                match scene.as_ref() {
                    Some(scene) => (scene.frames, scene.gpu.clone(), scene.path),
                    None => (0, "—".to_owned(), "—"),
                }
            };
            let refresh = self.display.is_none() || self.displayed_at.elapsed().as_millis() >= 200;
            if refresh {
                let mut metrics = Metrics::new(&self.stats.borrow(), produced);
                metrics.gpu = gpu;
                metrics.path = path;
                *self.panel_metrics.borrow_mut() = Some(metrics.clone());
                self.display = Some(metrics);
                self.displayed_at = Instant::now();
                // The cached panels are the only parts of the root that have to be told their
                // contents changed; the root itself is already dirty every animation frame.
                self.histogram_panel.update(cx, |_, cx| cx.notify());
                self.metrics_panel.update(cx, |_, cx| cx.notify());
            }

            let paused = self.paused.clone();
            let speed = self.speed.clone();
            let palette = self.palette.clone();

            page(
                controls_panel(
                    cx,
                    paused.clone(),
                    speed.clone(),
                    palette.clone(),
                    self.histogram_panel.clone(),
                ),
                viewport(
                    self.scene.clone(),
                    self.error.clone(),
                    self.last.clone(),
                    self.grab.clone(),
                    paused,
                    speed,
                    palette,
                    self.bench,
                ),
                self.metrics_panel.clone().cached(metrics_panel_style()),
            )
        }
    }

    /// The left column: real GPUI controls, and beneath them the cached interval histogram.
    fn controls_panel(
        cx: &mut Context<Showcase>,
        paused: Rc<Cell<bool>>,
        speed: Rc<Cell<f32>>,
        palette: Rc<Cell<u8>>,
        histogram_panel: Entity<HistogramPanel>,
    ) -> AnyElement {
        let pause_label = if paused.get() { "Resume" } else { "Pause" };
        let pause = control("pause", pause_label).on_click(cx.listener(move |_this, _, _, cx| {
            paused.set(!paused.get());
            cx.notify();
        }));

        let speed_label = format!("Speed {:.1}×", speed.get());
        let speed_button =
            control("speed", speed_label).on_click(cx.listener(move |_this, _, _, cx| {
                let next = match speed.get() {
                    s if s < 1.0 => 1.0,
                    s if s < 2.0 => 2.0,
                    _ => 0.5,
                };
                speed.set(next);
                cx.notify();
            }));

        let palette_index = palette.get() as usize % PALETTES.len();
        let palette_button = control(
            "palette",
            format!("Palette · {}", PALETTES[palette_index].0),
        )
        .on_click(cx.listener(move |_this, _, _, cx| {
            palette.set((palette.get() + 1) % PALETTES.len() as u8);
            cx.notify();
        }));

        let reset = control("reset", "Reset view").on_click(cx.listener(|this, _, _, cx| {
            if let Some(scene) = this.scene.borrow_mut().as_mut() {
                scene.reset_view();
            }
            cx.notify();
        }));

        panel(LEFT_WIDTH)
            .child(section_label("Controls"))
            .child(pause)
            .child(speed_button)
            .child(palette_button)
            .child(reset)
            .child(div().h(px(8.)).flex_none())
            .child(histogram_panel.cached(histogram_style()))
            .into_any_element()
    }

    /// The left column's histogram block: the interval bars, what they mean, and where the most
    /// recent interval went.
    fn histogram_section(metrics: &Metrics) -> AnyElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap_1()
            .child(section_label("Frame interval"))
            .child(histogram(&metrics.intervals, Some(&metrics.work)))
            .child(histogram_caption(true))
            .child(pacing_note(
                metrics.frame_ms,
                metrics.draw_ms,
                metrics.wait_ms,
                metrics.active,
            ))
            .into_any_element()
    }

    /// The right column: the title, the frame rate, the pipeline's own breakdown, and the producer.
    fn metrics_section(metrics: &Metrics) -> AnyElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap_1()
            .child(title(
                "wgpu, inside GPUI",
                "An instanced wgpu scene handed to the window renderer with no CPU copy.",
            ))
            .child(div().h(px(2.)).flex_none())
            .child(
                div()
                    .text_2xl()
                    .text_color(rgb(ACCENT))
                    .child(format!("{:.1} FPS", metrics.fps)),
            )
            .child(div().text_xs().text_color(rgb(MUTED)).child(format!(
                "{:.2} ms · draw {:.2} ms · wait {:.2} ms",
                metrics.frame_ms, metrics.draw_ms, metrics.wait_ms
            )))
            .child(div().h(px(4.)).flex_none())
            .child(section_label("Frame pipeline · per frame"))
            .child(metric_row(
                "frame interval",
                format!("{:.2} ms", metrics.frame_ms),
            ))
            .child(metric_row(
                "root passes",
                format!("{:.3} ms", metrics.passes_ms),
            ))
            .child(metric_row(
                "evaluate",
                format!("{:.3} ms", metrics.evaluate_ms),
            ))
            .child(metric_row("layout", format!("{:.3} ms", metrics.layout_ms)))
            .child(metric_row("paint", format!("{:.3} ms", metrics.paint_ms)))
            .child(div().h(px(4.)).flex_none())
            .child(section_label("Counters"))
            .child(metric_row("GPUI frames", metrics.gpui_frames.to_string()))
            .child(metric_row("asked", metrics.asked.to_string()))
            .child(metric_row("produced", metrics.produced.to_string()))
            .child(metric_row(
                "window",
                if metrics.active { "active" } else { "inactive" }.to_owned(),
            ))
            .child(div().h(px(4.)).flex_none())
            .child(section_label("Producer"))
            .child(metric_row("adapter", metrics.gpu.clone()))
            .child(metric_row("hand-off", metrics.path.to_owned()))
            .into_any_element()
    }

    /// The viewport: the exported surface, wrapped in the input gateway, with two non-interactive
    /// overlays that insert no hitbox and so never steal a drag.
    #[allow(clippy::too_many_arguments)]
    fn viewport(
        scene: Rc<RefCell<Option<Scene>>>,
        error: Rc<RefCell<Option<String>>>,
        last: Rc<Cell<Option<SurfacePoint>>>,
        grab: Rc<Cell<bool>>,
        paused: Rc<Cell<bool>>,
        speed: Rc<Cell<f32>>,
        palette: Rc<Cell<u8>>,
        bench: Option<Duration>,
    ) -> AnyElement {
        let drawing = scene.clone();
        let zooming = scene.clone();
        let overlay_error = error.clone();
        let painting_error = error;
        let surface = gpu_canvas(move |gpu| {
            // The canvas is a physical-pixel producer: follow its current bounds, not the size it
            // had when the app was built. A resize recreates the exported buffer (or the
            // same-device surface) and the depth attachment, so the image is never stretched.
            let bounds = gpu.bounds();
            let scale = gpu.scale_factor();
            let width = (bounds.size.width.0 * scale).round().max(1.0) as u32;
            let height = (bounds.size.height.0 * scale).round().max(1.0) as u32;

            let mut slot = drawing.borrow_mut();
            if slot.is_none() && painting_error.borrow().is_none() {
                match Scene::build(gpu, width, height, bench) {
                    Ok(built) => *slot = Some(built),
                    Err(build) => {
                        let message = format!("{build:#}");
                        eprintln!("interop_wgpu: cannot build the scene: {message}");
                        log::error!("interop_wgpu: cannot build the scene: {message}");
                        *painting_error.borrow_mut() = Some(message);
                        return;
                    }
                }
            }
            let Some(scene) = slot.as_mut() else {
                return;
            };

            // Raw bench: render into the output and wait for the GPU each frame, without handing
            // anything to the canvas. The scene is built on the paint that reaches here, so the
            // measurement starts after construction. This is the producer side (render + export on
            // the dma-buf path); the consumer's import and the Optimus/PRIME transfer show up in a
            // normal run's `paint` and `wait`, not here.
            if let Some(limit) = scene.bench.take() {
                let started = Instant::now();
                let mut frames = 0u64;
                while started.elapsed() < limit {
                    if let Err(error) = scene.bench_frame(paused.get(), speed.get(), palette.get())
                    {
                        eprintln!("interop_wgpu: bench stopped: {error:#}");
                        break;
                    }
                    frames += 1;
                }
                let elapsed = started.elapsed().as_secs_f64().max(1e-9);
                eprintln!(
                    "interop_wgpu: bench {frames} frames in {:.3}s — {:.1} fps, {:.3} ms/frame, \
                     {} (producer only; no presentation)",
                    elapsed,
                    frames as f64 / elapsed,
                    elapsed * 1000.0 / frames.max(1) as f64,
                    scene.path,
                );
                unsafe { libc::_exit(0) }
            }

            if let Err(error) = scene.resize(width, height) {
                let message = format!("{error:#}");
                eprintln!("interop_wgpu: cannot resize the render target: {message}");
                log::error!("interop_wgpu: cannot resize the render target: {message}");
                *painting_error.borrow_mut() = Some(message);
                return;
            }

            match scene.render(gpu, paused.get(), speed.get(), palette.get()) {
                Ok(()) => {}
                Err(error) => {
                    let message = format!("{error:#}");
                    eprintln!("interop_wgpu: cannot render a frame: {message}");
                    log::error!("interop_wgpu: cannot render a frame: {message}");
                    *painting_error.borrow_mut() = Some(message);
                }
            }
        })
        .on_scroll_wheel(move |event, window, _cx| {
            let notches = match event.delta {
                ScrollDelta::Lines(lines) => lines.y,
                ScrollDelta::Pixels(pixels) => pixels.y.as_f32() / PIXELS_PER_LINE,
            };
            if let Some(scene) = zooming.borrow_mut().as_mut() {
                scene.zoom(ZOOM_PER_NOTCH.powf(notches));
            }
            window.refresh();
        })
        .size_full();

        let handler = move |input: SurfaceInput| {
            let mut slot = scene.borrow_mut();
            let Some(scene) = slot.as_mut() else {
                return;
            };
            match input.event {
                SurfaceEvent::PointerDown { position, .. } => {
                    grab.set(true);
                    last.set(Some(position));
                }
                SurfaceEvent::PointerMove { position, .. } if grab.get() => {
                    if let Some(previous) = last.replace(Some(position)) {
                        scene.orbit(
                            (position.x - previous.x) * SENSITIVITY,
                            (position.y - previous.y) * SENSITIVITY,
                        );
                    }
                }
                SurfaceEvent::PointerMove { .. } => {}
                SurfaceEvent::PointerUp { .. } => {
                    grab.set(false);
                    last.set(None);
                }
            }
        };

        viewport_frame()
            .child(surface_input(surface, handler))
            .children(overlay_error.borrow().clone().map(|message| {
                error_panel(
                    "The wgpu viewport could not start",
                    message,
                    "Run with --list-gpus to see the adapters. A Vulkan adapter uses the dma-buf \
                     path; the others fall back to a same-device view.",
                )
            }))
            .child(overlay_badge("wgpu · 0 CPU copies"))
            .child(overlay_hint("drag to orbit · scroll to zoom"))
            .into_any_element()
    }

    /// Where a frame goes: the exported dma-buf when the wgpu device is a Vulkan device that can
    /// export one, or the same-device view when it is not (a GL adapter, say). GPUI's own drawing
    /// needs neither export nor copy — it samples whichever view the scene rendered into.
    enum Output {
        Exported(Box<WgpuSharedBuffer>),
        SameDevice(WgpuSurface),
    }

    impl Output {
        fn view(&self) -> &wgpu::TextureView {
            match self {
                Self::Exported(buffer) => buffer.view(),
                Self::SameDevice(surface) => surface.view(),
            }
        }
    }

    /// The producer's half, built on the renderer's own wgpu device.
    struct Scene {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        output: Output,
        /// How the frames reach GPUI, for the metrics panel: `dma-buf` or `same-device`.
        path: &'static str,
        /// The target's current size, in physical pixels.
        width: u32,
        height: u32,
        depth: wgpu::TextureView,
        pipeline: wgpu::RenderPipeline,
        vertices: wgpu::Buffer,
        indices: wgpu::Buffer,
        instances: wgpu::Buffer,
        uniforms: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
        /// The clock, in seconds; frozen while paused.
        time: f32,
        yaw: f32,
        pitch: f32,
        distance: f32,
        frames: u64,
        gpu: String,
        /// A raw bench duration, taken by the first paint that reaches the canvas.
        bench: Option<Duration>,
    }

    impl Scene {
        fn build(
            gpu: &mut gpui::GpuCanvasContext,
            width: u32,
            height: u32,
            bench: Option<Duration>,
        ) -> anyhow::Result<Self> {
            // The bench keeps a fixed size so the two demos' producer numbers compare; the window
            // follows the canvas's physical size instead.
            let (width, height) = if bench.is_some() {
                (BENCH_WIDTH, BENCH_HEIGHT)
            } else {
                (width.max(1), height.max(1))
            };
            let interop = attach(gpu.try_device::<WgpuRenderer>())
                .map_err(|unavailable| anyhow::anyhow!("{unavailable:?}"))?;
            let (device, queue) = interop
                .adapter()
                .wgpu()
                .ok_or_else(|| anyhow::anyhow!("the window's renderer is not wgpu"))?;
            let gpu_name = interop
                .adapter()
                .gpu_id()
                .map_or_else(|| "non-Vulkan wgpu adapter".to_owned(), |id| id.name);
            eprintln!("interop_wgpu: rendering on {gpu_name}");
            let (output, path) = match WgpuSharedBuffer::new(&device, width, height, FORMAT) {
                Ok(buffer) => (Output::Exported(Box::new(buffer)), "dma-buf + sync_file"),
                Err(export) => {
                    // A wgpu device on a non-Vulkan backend (or one without the export extensions)
                    // has no VkImage to export, and needs none: the renderer samples a view on the
                    // same device, so the pixels never leave it at all.
                    eprintln!(
                        "interop_wgpu: dma-buf export unavailable ({export:#}); using the \
                         same-device view"
                    );
                    let surface = WgpuSurface::new(&device, width, height)?;
                    (Output::SameDevice(surface), "same-device view")
                }
            };

            let depth = Self::depth_view(&device, width, height);

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("interop_wgpu_shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });

            let vertex_attributes = [
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
            ];
            let instance_attributes = [
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 5,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 64,
                    shader_location: 6,
                },
            ];
            let vertex_layout = wgpu::VertexBufferLayout {
                array_stride: 6 * 4,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &vertex_attributes,
            };
            let instance_layout = wgpu::VertexBufferLayout {
                array_stride: 20 * 4,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &instance_attributes,
            };

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("interop_wgpu_camera_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("interop_wgpu_pipeline_layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("interop_wgpu_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[vertex_layout, instance_layout],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

            let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("interop_wgpu_vertices"),
                contents: &floats(&cube_vertices()),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("interop_wgpu_indices"),
                contents: &u16s(&cube_indices()),
                usage: wgpu::BufferUsages::INDEX,
            });
            let instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("interop_wgpu_instances"),
                size: u64::from(INSTANCES) * 20 * 4,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("interop_wgpu_camera"),
                size: 16 * 4,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("interop_wgpu_camera_bind_group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                }],
            });

            Ok(Self {
                device,
                queue,
                depth,
                pipeline,
                vertices,
                indices,
                instances,
                uniforms,
                bind_group,
                time: 0.0,
                yaw: 0.55,
                pitch: 0.42,
                distance: 11.0,
                frames: 0,
                gpu: gpu_name,
                output,
                path,
                width,
                height,
                bench,
            })
        }

        /// The depth attachment for a target of this size.
        fn depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("interop_wgpu_depth"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        }

        /// Follow the canvas's physical size. Reallocating on a bounds change keeps the producer's
        /// pixels square: GPUI draws the surface into the canvas's bounds, so a target of a
        /// different aspect would be stretched to fill it.
        fn resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
            let width = width.max(1);
            let height = height.max(1);
            if (width, height) == (self.width, self.height) {
                return Ok(());
            }

            match &mut self.output {
                Output::Exported(buffer) => {
                    **buffer = WgpuSharedBuffer::new(&self.device, width, height, FORMAT)?;
                }
                Output::SameDevice(surface) => {
                    *surface = WgpuSurface::new(&self.device, width, height)?;
                }
            }
            self.depth = Self::depth_view(&self.device, width, height);
            self.width = width;
            self.height = height;
            Ok(())
        }

        /// Render one frame and hand it to the canvas — as a dma-buf on the exported path, or as a
        /// same-device view with `paint_texture` on the fallback.
        fn render(
            &mut self,
            gpu: &mut gpui::GpuCanvasContext,
            paused: bool,
            speed: f32,
            palette: u8,
        ) -> anyhow::Result<()> {
            self.draw(Some(gpu), paused, speed, palette)
        }

        /// Render one frame without presenting it, and wait for the GPU to finish. This is the raw
        /// bench's frame: it measures render plus export (on the dma-buf path), not the consumer's
        /// import or the display's transfer.
        fn bench_frame(&mut self, paused: bool, speed: f32, palette: u8) -> anyhow::Result<()> {
            self.draw(None, paused, speed, palette)?;
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(10)),
                })
                .map_err(|error| anyhow::anyhow!("waiting for a bench frame: {error}"))?;
            Ok(())
        }

        fn draw(
            &mut self,
            gpu: Option<&mut gpui::GpuCanvasContext>,
            paused: bool,
            speed: f32,
            palette: u8,
        ) -> anyhow::Result<()> {
            if !paused {
                self.time += 0.016 * speed;
            }

            let eye = [
                self.distance * self.pitch.cos() * self.yaw.sin(),
                self.distance * self.pitch.sin(),
                self.distance * self.pitch.cos() * self.yaw.cos(),
            ];
            let view = look_at(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
            let projection = perspective(0.9, self.width as f32 / self.height as f32, 0.1, 100.0);
            let view_projection = mul(&projection, &view);
            self.queue
                .write_buffer(&self.uniforms, 0, &floats(&view_projection.concat()));

            let (from, to) = {
                let (_, from, to) = PALETTES[usize::from(palette) % PALETTES.len()];
                (from, to)
            };
            let mut data = Vec::with_capacity(INSTANCES as usize * 20);
            let half = (GRID - 1) as f32 / 2.0;
            for y in 0..GRID {
                for x in 0..GRID {
                    let (fx, fy) = (x as f32 - half, y as f32 - half);
                    let phase = self.time * 1.25 + fx * 0.42 + fy * 0.31;
                    let position = [fx * 1.75, fy * 1.75, phase.sin() * 0.42];
                    let scale = 0.40 + 0.05 * (phase * 0.9).sin();
                    let model = model(position, phase * 0.62, phase * 0.44, scale);
                    let t = ((fx + fy + GRID as f32) / (2.0 * GRID as f32)).clamp(0.0, 1.0);
                    let pulse = 0.82 + 0.18 * (phase * 1.7).sin();
                    let colour = [
                        (from[0] + (to[0] - from[0]) * t) * pulse,
                        (from[1] + (to[1] - from[1]) * t) * pulse,
                        (from[2] + (to[2] - from[2]) * t) * pulse,
                    ];
                    for column in model.iter() {
                        data.extend_from_slice(column);
                    }
                    data.extend_from_slice(&[colour[0], colour[1], colour[2], 1.0]);
                }
            }
            self.queue.write_buffer(&self.instances, 0, &floats(&data));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("interop_wgpu_frame"),
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("interop_wgpu_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: self.output.view(),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.005,
                                g: 0.008,
                                b: 0.014,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.vertices.slice(..));
                pass.set_vertex_buffer(1, self.instances.slice(..));
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..36, 0, 0..INSTANCES);
            }

            let queue = Arc::clone(&self.queue);
            match &mut self.output {
                Output::Exported(buffer) => {
                    let handle = buffer.submit(&queue, Some(encoder.finish()))?;
                    if let Some(gpu) = gpu {
                        gpu.paint_surface(handle);
                    }
                }
                Output::SameDevice(surface) => {
                    queue.submit(Some(encoder.finish()));
                    if let Some(gpu) = gpu {
                        gpu.paint_texture(
                            surface.view().to_imported_handle()?,
                            Corners::default(),
                            1.0,
                            false,
                        );
                    }
                }
            }
            self.frames += 1;
            Ok(())
        }

        /// Move the camera by a drag delta, in physical pixels. Pitch is clamped short of the poles so
        /// the up vector never flips the view.
        fn orbit(&mut self, dx: f32, dy: f32) {
            self.yaw += dx;
            self.pitch = (self.pitch - dy).clamp(-1.2, 1.2);
        }

        /// Move the eye, keeping the target. A factor, so a notch feels the same at either distance.
        fn zoom(&mut self, factor: f32) {
            self.distance = (self.distance * factor).clamp(5.0, 30.0);
        }

        fn reset_view(&mut self) {
            self.yaw = 0.55;
            self.pitch = 0.42;
            self.distance = 11.0;
        }
    }

    /// A 4×4 matrix in **column-major** order: `m[column][row]`, which is the layout WGSL's
    /// `mat4x4` takes and the order the instance buffer writes.
    type Mat4 = [[f32; 4]; 4];

    fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
        let mut out = [[0.0; 4]; 4];
        for (column, out_column) in out.iter_mut().enumerate() {
            for (row, value) in out_column.iter_mut().enumerate() {
                *value = a[0][row] * b[column][0]
                    + a[1][row] * b[column][1]
                    + a[2][row] * b[column][2]
                    + a[3][row] * b[column][3];
            }
        }
        out
    }

    fn rotations(yaw: f32, pitch: f32) -> Mat4 {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let y = [
            [cy, 0.0, -sy, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [sy, 0.0, cy, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let x = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cp, sp, 0.0],
            [0.0, -sp, cp, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        mul(&y, &x)
    }

    /// Scale, then rotate, then translate — the instance's model matrix.
    fn model(position: [f32; 3], yaw: f32, pitch: f32, scale: f32) -> Mat4 {
        let mut matrix = rotations(yaw, pitch);
        for column in matrix.iter_mut().take(3) {
            for value in column.iter_mut().take(3) {
                *value *= scale;
            }
        }
        matrix[3] = [position[0], position[1], position[2], 1.0];
        matrix
    }

    fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fov_y * 0.5).tan();
        let range = 1.0 / (near - far);
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far * range, -1.0],
            [0.0, 0.0, far * near * range, 0.0],
        ]
    }

    fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
        let forward = normalize(sub(target, eye));
        let side = normalize(cross(forward, up));
        let up = cross(side, forward);
        [
            [side[0], up[0], -forward[0], 0.0],
            [side[1], up[1], -forward[1], 0.0],
            [side[2], up[2], -forward[2], 0.0],
            [-dot(side, eye), -dot(up, eye), dot(forward, eye), 1.0],
        ]
    }

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let length = dot(v, v).sqrt();
        [v[0] / length, v[1] / length, v[2] / length]
    }

    /// A face's outward normal and its four corners, counter-clockwise from outside.
    type Face = ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 3]);

    /// A unit cube's 24 vertices (position and normal, four per face), wound counter-clockwise when
    /// seen from outside so back-face culling removes the interior.
    fn cube_vertices() -> Vec<f32> {
        let faces: [Face; 6] = [
            (
                [0.0, 0.0, 1.0],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ),
            (
                [0.0, 0.0, -1.0],
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
            ),
            (
                [1.0, 0.0, 0.0],
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ),
            (
                [-1.0, 0.0, 0.0],
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
            ),
            (
                [0.0, 1.0, 0.0],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
            ),
            (
                [0.0, -1.0, 0.0],
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
            ),
        ];
        let mut vertices = Vec::with_capacity(24 * 6);
        for (normal, a, b, c, d) in faces {
            for position in [a, b, c, d] {
                vertices.extend_from_slice(&position);
                vertices.extend_from_slice(&normal);
            }
        }
        vertices
    }

    /// Two triangles per face, six faces.
    fn cube_indices() -> Vec<u16> {
        let mut indices = Vec::with_capacity(36);
        for face in 0..6u16 {
            let base = face * 4;
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        indices
    }

    fn floats(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    fn u16s(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }
}
