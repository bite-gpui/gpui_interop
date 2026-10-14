//! `interop_cube_drag`: a foreign 3D viewport under GPUI chrome, the cube turned by dragging.
//!
//! ```sh
//! cargo run -p gpui_interop --example interop_cube_drag --features example
//! ```
//!
//! The window is a 3D viewport — a cube a producer renders with its own Vulkan pipeline and hands over
//! as a dma-buf, no CPU copy — with ordinary GPUI elements **over** it: a panel of text, and a bar of
//! buttons. Drag the viewport to turn the cube, or press a button; the two do not tread on each other,
//! which is the point of the demo.
//!
//! # The producer sees no GPUI
//!
//! The canvas is wrapped in [`surface_input`], the input gateway, and the handler takes [`SurfaceInput`]:
//! each event's position is in **physical pixels from the viewport's top-left corner**, with the scale
//! factor alongside, and its `SurfaceEvent` is the crate's vocabulary rather than GPUI's. A drag starts
//! on the viewport, which takes the pointer, so the cube keeps turning while the pointer is off past the
//! edge — and because the gateway refreshes the window after each event, the producer only has to change
//! its own state.
//!
//! # How the chrome and the viewport stay out of each other's way
//!
//! Which element an event belongs to is hit-testing's to decide, and a GPUI element in front takes it
//! over its own rectangle:
//!
//! - **A button keeps its own events.** The bar's buttons sit over the viewport and `occlude`, so a
//!   pointer-down on one hit-tests to the button and never to the viewport: a drag that starts on a
//!   button turns nothing, and a click on it is a click.
//! - **The text panel claims nothing.** A plain `div` of text inserts no hitbox, so the viewport under
//!   it is still what the pointer is over — a drag across the panel turns the cube as if the panel were
//!   not there. A rectangle is not the same as wanting the event.
//! - **A drag that starts on the viewport keeps the pointer.** The gateway captures it on pointer-down,
//!   so the cube keeps turning even when the drag passes *under* the panel or the bar, and the buttons
//!   under it do not fire.
//!
//! `interop_cube` is the same cube turned on a clock instead; see the crate's `input` module for what
//! the gateway adapts and what it deliberately leaves to the engine.

#[cfg(target_os = "linux")]
fn main() {
    demo::run();
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!(
        "interop_cube_drag is a Linux example: it renders into a shareable Vulkan image and exports it \
         as a dma-buf for the window's renderer. Run it on Linux."
    );
}

#[cfg(target_os = "linux")]
#[path = "common/cube.rs"]
mod cube;

#[cfg(target_os = "linux")]
mod demo {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use gpui::{
        AnyElement, App, Bounds, Context, IntoElement, Render, Window, WindowBounds, WindowOptions,
        div, gpu_canvas, prelude::*, px, rgb, rgba, size,
    };
    use gpui_interop::{SurfaceEvent, SurfaceInput, SurfacePoint, surface_input};

    use crate::cube::Cube;

    /// The edge the producer renders at. The viewport is a square as tall as the window, so at a typical
    /// window height this is within a whisker of one texel a pixel.
    const TILE: u32 = 640;

    /// How far the cube turns per physical pixel dragged — about a third of a turn across the viewport.
    const SENSITIVITY: f32 = 0.006;

    pub fn run() {
        gpui::application().run(|cx: &mut App| {
            let bounds = Bounds::centered(None, size(px(900.), px(640.)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |_, cx| cx.new(|_| Showcase::new()),
            )
            .expect("a window to composite into");
            cx.activate(true);
        });
    }

    /// The window's view. The cube is built lazily at paint time — the renderer's device exists only
    /// then — so the view holds a shared slot the canvas fills on the first frame.
    struct Showcase {
        cube: Rc<RefCell<Option<Cube>>>,
        /// Why the cube could not be built, when it could not: shown in place of the device line.
        error: Rc<RefCell<Option<String>>>,
        /// The pointer's last surface-local position, so a move becomes the delta a drag turns by. Held
        /// across frames, since the element tree is rebuilt every render.
        last: Rc<Cell<Option<SurfacePoint>>>,
        /// Whether the cube turns on its own, toggled by the bar's button. Shared with the paint
        /// callback, which is what turns it.
        spinning: Rc<Cell<bool>>,
        /// How many frames have been requested: a couple to build the cube (or give up), and one per
        /// delivered input, which the gateway asks for itself.
        requested: u32,
    }

    impl Showcase {
        fn new() -> Self {
            Self {
                cube: Rc::new(RefCell::new(None)),
                error: Rc::new(RefCell::new(None)),
                last: Rc::new(Cell::new(None)),
                spinning: Rc::new(Cell::new(false)),
                requested: 0,
            }
        }
    }

    impl Render for Showcase {
        fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            // The viewport turns the cube only when the pointer or the Spin button turns it, and the
            // gateway asks for the frame each of those implies — but a spinning cube needs a frame
            // every tick, and a couple of frames up front build it (or give up).
            let built = self.cube.borrow().is_some();
            let spinning = self.spinning.get();
            if spinning || (!built && self.requested < 2) {
                if !built {
                    self.requested += 1;
                }
                window.request_animation_frame();
            }

            let status = match (self.cube.borrow().as_ref(), self.error.borrow().as_ref()) {
                (Some(cube), _) => format!(
                    "{} · {} frames rendered, {} dropped",
                    cube.gpu(),
                    cube.rendered(),
                    cube.dropped()
                ),
                (None, Some(error)) => error.clone(),
                (None, None) => "negotiating with the window's renderer…".to_owned(),
            };

            div()
                .relative()
                .size_full()
                .overflow_hidden()
                .bg(rgb(0x0b0b0f))
                // The viewport, as large as the window's shorter side allows and square, so the
                // producer's square buffer is not stretched. The chrome floats over the whole window,
                // not only over the square.
                .child(
                    div()
                        .size_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(div().h_full().aspect_square().child(canvas(
                            self.cube.clone(),
                            self.error.clone(),
                            self.last.clone(),
                            self.spinning.clone(),
                        ))),
                )
                .child(panel(status))
                .child(bar(cx, self.spinning.clone()))
        }
    }

    /// The viewport: the surface, wrapped in the input gateway. The first paint builds the cube, and
    /// every pointer event over the viewport turns it by the pointer's delta.
    fn canvas(
        cube: Rc<RefCell<Option<Cube>>>,
        error: Rc<RefCell<Option<String>>>,
        last: Rc<Cell<Option<SurfacePoint>>>,
        spinning: Rc<Cell<bool>>,
    ) -> AnyElement {
        let drawing = cube.clone();
        let surface = gpu_canvas(move |gpu| {
            let mut cube = drawing.borrow_mut();
            if cube.is_none() {
                match Cube::build(gpu, TILE) {
                    Ok(built) => *cube = Some(built),
                    Err(build) => {
                        log::error!("interop_cube_drag: cannot build the cube: {build:#}");
                        *error.borrow_mut() = Some(format!("{build}"));
                        return;
                    }
                }
            }
            let cube = cube.as_mut().expect("the cube was just built");
            if spinning.get() {
                cube.turn(0.013, 0.0078);
            }
            if let Some(handle) = cube.frame() {
                gpu.paint_surface(handle);
            }
        })
        .bg(rgb(0x121218))
        .size_full();

        let handler = move |input: SurfaceInput| {
            let mut cube = cube.borrow_mut();
            let Some(cube) = cube.as_mut() else {
                return;
            };
            match input.event {
                SurfaceEvent::PointerDown { position, .. } => last.set(Some(position)),
                SurfaceEvent::PointerMove {
                    position, pressed, ..
                } => {
                    let previous = last.replace(Some(position));
                    if pressed.is_some()
                        && let Some(previous) = previous
                    {
                        cube.turn(
                            (position.x - previous.x) * SENSITIVITY,
                            (position.y - previous.y) * SENSITIVITY,
                        );
                    }
                }
                SurfaceEvent::PointerUp { .. } => last.set(None),
            }
        };

        surface_input(surface, handler).into_any_element()
    }

    /// A translucent panel of text, top left. It is not interactive, so a drag over it still reaches the
    /// viewport underneath — only a control should take the pointer.
    fn panel(status: String) -> AnyElement {
        div()
            .absolute()
            .top(px(16.))
            .left(px(16.))
            .flex()
            .flex_col()
            .gap_1()
            .p_4()
            .rounded_lg()
            .bg(rgba(0x0a0a0fdd))
            .border_1()
            .border_color(rgba(0xffffff14))
            .child(
                div()
                    .text_lg()
                    .text_color(rgb(0xffffff))
                    .child("Foreign 3D viewport"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0xa8a8b2))
                    .max_w(px(300.))
                    .child(
                        "Drag the viewport to turn the cube — rendered by a producer that owns its \
                         device, handed over as a dma-buf, no CPU copy.",
                    ),
            )
            .child(div().text_xs().text_color(rgb(0x6f8fbf)).child(status))
            .into_any_element()
    }

    /// The bar of controls, bottom centre. Each button is a GPUI element in front of the viewport, so a
    /// click over one never reaches the cube.
    fn bar(cx: &mut Context<Showcase>, spinning: Rc<Cell<bool>>) -> AnyElement {
        let reset = div()
            .id("cube-reset")
            .occlude()
            .px_3()
            .py_1()
            .rounded_md()
            .bg(rgba(0xffffff1a))
            .text_sm()
            .text_color(rgb(0xececf2))
            .cursor_pointer()
            .hover(|style| style.bg(rgba(0xffffff2e)))
            .on_click(cx.listener(|this, _, _, cx| {
                if let Some(cube) = this.cube.borrow_mut().as_mut() {
                    cube.reset();
                }
                cx.notify();
            }))
            .child("Reset");

        let label = if spinning.get() { "Still" } else { "Spin" };
        let spin = div()
            .id("cube-spin")
            .occlude()
            .px_3()
            .py_1()
            .rounded_md()
            .bg(rgba(0xffffff1a))
            .text_sm()
            .text_color(rgb(0xececf2))
            .cursor_pointer()
            .hover(|style| style.bg(rgba(0xffffff2e)))
            .on_click(cx.listener(move |this, _, _, cx| {
                this.spinning.set(!this.spinning.get());
                cx.notify();
            }))
            .child(label);

        div()
            .absolute()
            .bottom(px(16.))
            .left_0()
            .right_0()
            .flex()
            .flex_row()
            .justify_center()
            .gap_2()
            .child(reset)
            .child(spin)
            .into_any_element()
    }
}
