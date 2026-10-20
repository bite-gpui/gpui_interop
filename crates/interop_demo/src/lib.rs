//! The GPUI chrome shared by the `interop_wgpu`, `interop_bevy` and `interop_helio` demos.
//!
//! Every demo puts the same shell around a different producer: a left column of real GPUI
//! controls with a frame-interval histogram beneath them, the producer's canvas in the centre, and
//! a right column of live metrics. The producer fills one [`MetricsSnapshot`]; this crate renders
//! it with [`metrics_panel`] and [`metrics_histogram`], so the three demos cannot drift apart
//! visually or in what they measure: the same sections, the same rows, the same order.
//!
//! The palette and the sizes are here rather than in each example for the same reason: a reader
//! comparing the demos should be comparing the engines, not three hand-tuned copies of the same
//! chrome.

use gpui::{
    AnyElement, Div, IntoElement, Stateful, StyleRefinement, div, prelude::*, px, rgb, rgba,
};

/// The window background.
pub const BG: u32 = 0x10141c;
/// A side panel's background.
pub const PANEL_BG: u32 = 0x171d28;
/// A panel's border.
pub const BORDER: u32 = 0x27303e;
/// A heading.
pub const TEXT: u32 = 0xf2f4f8;
/// Body prose.
pub const BODY: u32 = 0xb7c0cf;
/// A subtitle or label.
pub const MUTED: u32 = 0x8b93a3;
/// A caption or footnote.
pub const DIM: u32 = 0x6f7787;
/// A metric value.
pub const VALUE: u32 = 0xd7dce6;
/// The rate and the path.
pub const ACCENT: u32 = 0x9ec1ff;
/// Pipeline work inside an interval.
pub const GREEN: u32 = 0x34d399;
/// A 60 fps interval.
pub const BLUE: u32 = 0x38bdf8;
/// A 30 fps interval.
pub const AMBER: u32 = 0xf59e0b;
/// A slower interval.
pub const RED: u32 = 0xef4444;

/// The three colour pairs the field interpolates between, cycled by the palette control. The same
/// table drives both producers, so the palettes compare directly.
pub const PALETTES: [(&str, [f32; 3], [f32; 3]); 3] = [
    ("Ember", [1.00, 0.42, 0.15], [0.15, 0.55, 1.00]),
    ("Tide", [0.10, 0.90, 0.72], [0.16, 0.32, 1.00]),
    ("Mono", [0.92, 0.93, 1.00], [0.34, 0.36, 0.46]),
];

/// The left column's width, in logical pixels.
pub const LEFT_WIDTH: f32 = 260.0;
/// The right column's width, in logical pixels.
pub const RIGHT_WIDTH: f32 = 280.0;
/// The padding inside a side panel, on each side.
pub const PANEL_PADDING: f32 = 12.0;
/// The height of the histogram block, caption and pacing note included.
pub const HISTOGRAM_HEIGHT: f32 = 168.0;

/// The demo shell: the left controls, the centre viewport and the right metrics.
pub fn page(
    left: impl IntoElement,
    centre: impl IntoElement,
    right: impl IntoElement,
) -> AnyElement {
    div()
        .size_full()
        .bg(rgb(BG))
        .flex()
        .flex_row()
        .gap_4()
        .p_6()
        .child(left)
        .child(centre)
        .child(right)
        .into_any_element()
}

/// A side panel of a fixed width, filling the row's height.
pub fn panel(width: f32) -> Div {
    div()
        .flex_none()
        .w(px(width))
        .h_full()
        .flex()
        .flex_col()
        .gap_3()
        .p_3()
        .rounded_md()
        .bg(rgb(PANEL_BG))
        .border_1()
        .border_color(rgb(BORDER))
}

/// The viewport's frame: it takes the space between the columns and clips the canvas.
pub fn viewport_frame() -> Div {
    div()
        .relative()
        .flex_1()
        .min_w(px(320.))
        .min_h(px(240.))
        .rounded_lg()
        .overflow_hidden()
        .border_1()
        .border_color(rgb(BORDER))
}

/// The style a cached metrics panel is laid out at. Caching skips measuring the contents, so the
/// panel needs a definite width and its full height.
pub fn metrics_panel_style() -> StyleRefinement {
    StyleRefinement::default()
        .w(px(RIGHT_WIDTH))
        .h_full()
        .flex_none()
}

/// The style a cached histogram block is laid out at, inside the left panel's padding.
pub fn histogram_style() -> StyleRefinement {
    StyleRefinement::default()
        .w(px(LEFT_WIDTH - 2.0 * PANEL_PADDING))
        .h(px(HISTOGRAM_HEIGHT))
        .flex_none()
}

/// The demo's name and one line about what it shows.
pub fn title(heading: &'static str, subtitle: &'static str) -> AnyElement {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(div().text_lg().text_color(rgb(TEXT)).child(heading))
        .child(div().text_xs().text_color(rgb(MUTED)).child(subtitle))
        .into_any_element()
}

/// A section caption.
pub fn section_label(label: &'static str) -> AnyElement {
    div()
        .text_xs()
        .text_color(rgb(DIM))
        .child(label.to_uppercase())
        .into_any_element()
}

/// One label/value row.
pub fn metric_row(label: &'static str, value: String) -> AnyElement {
    div()
        .flex()
        .flex_row()
        .justify_between()
        .text_xs()
        .child(div().text_color(rgb(MUTED)).child(label))
        .child(div().text_color(rgb(VALUE)).child(value))
        .into_any_element()
}

/// One control button. Real GPUI widgets: they carry an id and a listener, so they occlude the
/// canvas beneath and take their own clicks.
pub fn control(id: &'static str, label: impl Into<String>) -> Stateful<Div> {
    div()
        .id(id)
        .occlude()
        .px_3()
        .py_2()
        .rounded_md()
        .bg(rgb(0x1a202b))
        .border_1()
        .border_color(rgb(BORDER))
        .text_sm()
        .text_color(rgb(0xe7eaf0))
        .cursor_pointer()
        .hover(|style| style.bg(rgb(0x222b38)).border_color(rgb(0x3a4b63)))
        .child(label.into())
}

/// One bar per recent frame, oldest left. The bar's height is the interval; when `work` is given,
/// its green foot is the pipeline work inside that interval. Blue is a 60 fps interval, amber
/// 30 fps, red slower.
pub fn histogram(intervals: &[f32], work: Option<&[f32]>) -> AnyElement {
    const MAX_MS: f32 = 50.0;
    const HEIGHT: f32 = 96.0;

    let start = intervals.len().saturating_sub(60);
    let bars: Vec<AnyElement> = intervals[start..]
        .iter()
        .enumerate()
        .map(|(offset, &interval)| {
            let work = work
                .and_then(|work| work.get(start + offset))
                .copied()
                .unwrap_or(0.0);
            let total = (interval / MAX_MS).clamp(0.0, 1.0) * HEIGHT;
            let foot = (work / MAX_MS).clamp(0.0, 1.0) * HEIGHT;
            let wait = (total - foot).max(if total < 1.0 { 1.0 } else { 0.0 });
            let colour = if interval <= 16.7 {
                rgb(BLUE)
            } else if interval <= 33.3 {
                rgb(AMBER)
            } else {
                rgb(RED)
            };
            div()
                .flex_1()
                .flex()
                .flex_col()
                .justify_end()
                .child(div().w_full().h(px(wait)).bg(colour))
                .child(div().w_full().h(px(foot)).bg(rgb(GREEN)))
                .into_any_element()
        })
        .collect();

    div()
        .h(px(HEIGHT))
        .w_full()
        .flex()
        .flex_row()
        .items_end()
        .gap(px(1.))
        .children(bars)
        .into_any_element()
}

/// What the bars mean.
pub fn histogram_caption(show_work: bool) -> AnyElement {
    let text = if show_work {
        "bar: interval · green: pipeline draw · newest right"
    } else {
        "bar: frame interval · updates 2/s, newest right"
    };
    div()
        .text_xs()
        .text_color(rgb(DIM))
        .child(text)
        .into_any_element()
}

/// Says where the interval goes. An inactive window is the frame source's own 30 fps cap; an
/// active one at a low rate is either pipeline work or presentation pacing, and the split decides.
pub fn pacing_note(interval_ms: f32, draw_ms: f32, wait_ms: f32, active: bool) -> AnyElement {
    let (note, colour) = if interval_ms <= 0.0 {
        ("no frame measured yet", rgb(DIM))
    } else if !active {
        (
            "inactive window: the frame source caps animation at ~30 fps",
            rgb(AMBER),
        )
    } else if wait_ms > draw_ms {
        (
            "most of the interval is presentation/pacing, not pipeline work",
            rgb(MUTED),
        )
    } else {
        ("the interval is dominated by pipeline work", rgb(MUTED))
    };
    div()
        .text_xs()
        .text_color(colour)
        .child(note)
        .into_any_element()
}

/// A small overlay in the viewport's top-right corner, naming what the producer hands over.
pub fn overlay_badge(text: impl Into<String>) -> AnyElement {
    div()
        .absolute()
        .top(px(12.))
        .right(px(12.))
        .px_2()
        .py_1()
        .rounded_md()
        .bg(rgba(0x0a0f18cc))
        .border_1()
        .border_color(rgba(0xffffff1f))
        .text_xs()
        .text_color(rgb(ACCENT))
        .child(text.into())
        .into_any_element()
}

/// A small non-interactive hint in the viewport's bottom-left corner.
pub fn overlay_hint(text: impl Into<String>) -> AnyElement {
    div()
        .absolute()
        .bottom(px(12.))
        .left(px(12.))
        .text_xs()
        .text_color(rgba(0xffffff99))
        .child(text.into())
        .into_any_element()
}

/// The viewport's fallback when the producer cannot start or a frame cannot be rendered.
pub fn error_panel(heading: &'static str, message: String, hint: &'static str) -> AnyElement {
    div()
        .absolute()
        .size_full()
        .flex()
        .items_center()
        .justify_center()
        .bg(rgba(0x0a0f18ee))
        .child(
            div()
                .max_w(px(560.))
                .p_4()
                .rounded_lg()
                .bg(rgb(0x1a202b))
                .border_1()
                .border_color(rgb(0x6b3a3a))
                .flex()
                .flex_col()
                .gap_2()
                .child(div().text_lg().text_color(rgb(0xffb4a8)).child(heading))
                .child(div().text_xs().text_color(rgb(0xc9ced8)).child(message))
                .child(div().text_xs().text_color(rgb(MUTED)).child(hint)),
        )
        .into_any_element()
}

/// Everything the right column shows, in one schema shared by every demo.
///
/// The producers measure different things internally, but they all map onto the same rows: the
/// engine's own CPU step, the hand-off into GPUI, the producer's GPU wait, GPUI's frame, the
/// producer's output, and the counters. Keeping the schema here is what makes the panels
/// comparable across `interop_wgpu`, `interop_bevy` and `interop_helio`.
#[derive(Clone, Default)]
pub struct MetricsSnapshot {
    pub heading: &'static str,
    pub subtitle: &'static str,
    pub fps: f32,
    /// Wall-clock interval between the producer's frames.
    pub interval_ms: f32,
    /// The engine's own CPU work for the frame (Bevy update, Helio render, wgpu scene build).
    pub producer_ms: f32,
    /// Handing the frame to GPUI (submit, export or transport).
    pub handoff_ms: f32,
    /// Time the producer spent waiting for its own GPU work before the hand-off; zero when the
    /// fence is handed over live.
    pub gpu_wait_ms: f32,
    /// GPUI's whole frame: evaluate, layout, paint and draw.
    pub gpui_ms: f32,
    /// The part of the interval outside GPUI's frame: presentation, vsync and pacing.
    pub present_wait_ms: f32,
    pub evaluate_ms: f32,
    pub layout_ms: f32,
    /// GPUI's pipeline paint (the root passes' paint).
    pub paint_ms: f32,
    /// Frames the producer produced.
    pub produced: u64,
    pub gpui_frames: u64,
    /// Frames the frame source asked GPUI to render.
    pub asked: u64,
    /// Frames the producer dropped because every export slot was in flight.
    pub dropped: u64,
    pub width: u32,
    pub height: u32,
    pub adapter: String,
    /// How frames reach GPUI: `dma-buf + sync_file`, `same-device view`, ...
    pub path: String,
    pub active: bool,
    pub intervals: Vec<f32>,
    /// Per-interval GPUI draw time; drawn as the histogram's green foot.
    pub work: Vec<f32>,
}

/// One label/value row, monospaced enough to line up across the three panels.
fn row(label: &'static str, value: String) -> AnyElement {
    metric_row(label, value)
}

/// The right column, identical in every demo except the title, subtitle and values.
pub fn metrics_panel(metrics: &MetricsSnapshot) -> AnyElement {
    div()
        .size_full()
        .flex()
        .flex_col()
        .gap_1()
        .child(title(metrics.heading, metrics.subtitle))
        .child(div().h(px(2.)).flex_none())
        .child(
            div()
                .text_2xl()
                .text_color(rgb(ACCENT))
                .child(format!("{:.1} FPS", metrics.fps)),
        )
        .child(div().text_xs().text_color(rgb(MUTED)).child(format!(
            "{:.2} ms · producer {:.2} · hand-off {:.2} · GPUI {:.2}",
            metrics.interval_ms, metrics.producer_ms, metrics.handoff_ms, metrics.gpui_ms
        )))
        .child(div().h(px(4.)).flex_none())
        .child(section_label("Frame"))
        .child(row(
            "frame interval",
            format!("{:.2} ms", metrics.interval_ms),
        ))
        .child(row("producer", format!("{:.2} ms", metrics.producer_ms)))
        .child(row("hand-off", format!("{:.2} ms", metrics.handoff_ms)))
        .child(row("GPU wait", format!("{:.2} ms", metrics.gpu_wait_ms)))
        .child(row("GPUI draw", format!("{:.2} ms", metrics.gpui_ms)))
        .child(row(
            "present wait",
            format!("{:.2} ms", metrics.present_wait_ms),
        ))
        .child(row("evaluate", format!("{:.2} ms", metrics.evaluate_ms)))
        .child(row("layout", format!("{:.2} ms", metrics.layout_ms)))
        .child(row("pipeline paint", format!("{:.2} ms", metrics.paint_ms)))
        .child(div().h(px(4.)).flex_none())
        .child(section_label("Producer"))
        .child(row("adapter", metrics.adapter.clone()))
        .child(row("path", metrics.path.clone()))
        .child(row(
            "size",
            format!("{} × {}", metrics.width, metrics.height),
        ))
        .child(row("produced", metrics.produced.to_string()))
        .child(row("dropped", metrics.dropped.to_string()))
        .child(div().h(px(4.)).flex_none())
        .child(section_label("Counters"))
        .child(row("GPUI frames", metrics.gpui_frames.to_string()))
        .child(row("asked", metrics.asked.to_string()))
        .child(row(
            "window",
            if metrics.active { "active" } else { "inactive" }.to_owned(),
        ))
        .into_any_element()
}

/// The left column's histogram block, built from the same snapshot as the right panel.
pub fn metrics_histogram(metrics: &MetricsSnapshot) -> AnyElement {
    div()
        .size_full()
        .flex()
        .flex_col()
        .gap_1()
        .child(section_label("Frame interval"))
        .child(histogram(&metrics.intervals, Some(&metrics.work)))
        .child(histogram_caption(true))
        .child(pacing_note(
            metrics.interval_ms,
            metrics.gpui_ms,
            metrics.present_wait_ms,
            metrics.active,
        ))
        .into_any_element()
}
