#![cfg(feature = "input-test")]
//! The input gateway, end to end: pointer events a headless window dispatches, adapted for a producer.
//!
//! `surface_input` wraps a surface's element and hands a handler [`SurfaceInput`]s. This drives a real
//! window through the normal dispatch path — `Window::simulate_mouse_move`/`down`/`up` — and reads what
//! the handler saw, so the coordinate translation and the vocabulary mapping are checked rather than
//! assumed. The surface's own origin is not hardcoded: the first event pins it, and the rest are
//! checked against it.
//!
//! ```sh
//! cargo test -p gpui_interop --features input-test --test surface_input
//! ```
//!
//! No GPU and no platform backend: the "surface" is a plain `div` wrapped by the gateway, since the
//! gateway adapts input, not pixels — which is itself the point, a surface being a surface.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use gpui::{
    AnyWindowHandle, Context, IntoElement, Modifiers, MouseButton, Render, TestAppContext,
    VisualTestContext, Window, div, point, prelude::*, px,
};
use gpui_interop::{SurfaceButton, SurfaceEvent, SurfaceInput, surface_input};

/// The surface's edge, and the padding that insets it in the window.
const EDGE: f32 = 50.0;
const PAD: f32 = 10.0;

/// A view whose handler records everything the gateway hands the producer.
struct Recording {
    seen: Rc<RefCell<Vec<SurfaceInput>>>,
}

impl Render for Recording {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let seen = self.seen.clone();
        div()
            .size_full()
            .p(px(PAD))
            .child(surface_input(div().w(px(EDGE)).h(px(EDGE)), move |input| {
                seen.borrow_mut().push(input)
            }))
    }
}

#[gpui::test]
fn the_gateway_hands_a_producer_surface_local_physical_pixels(cx: &mut TestAppContext) {
    let seen = Rc::new(RefCell::new(Vec::new()));
    let recording = seen.clone();
    let window = AnyWindowHandle::from(cx.add_window(move |_, _| Recording { seen: recording }));
    let mut cx = VisualTestContext::from_window(window, cx);

    // One frame, so the surface's hitbox exists to be hit-tested against.
    cx.update(|window, cx| window.draw(cx).clear(cx));
    let scale = cx.update(|window, _| window.scale_factor());

    // A pointer at the surface's top-left-ish, then a drag down and to the right, all inside its
    // bounds so the flow is the ordinary one.
    let down = point(px(30.), px(30.));
    let dragged = point(px(45.), px(40.));
    cx.simulate_mouse_move(down, None::<MouseButton>, Modifiers::none());
    cx.simulate_mouse_down(down, MouseButton::Left, Modifiers::none());
    cx.simulate_mouse_move(dragged, MouseButton::Left, Modifiers::none());
    cx.simulate_mouse_up(dragged, MouseButton::Left, Modifiers::none());

    let seen = seen.borrow();
    let events: Vec<SurfaceEvent> = seen.iter().map(|input| input.event).collect();
    assert!(
        seen.iter().all(|input| input.scale == scale),
        "every event carries the window's scale factor ({scale}), got {:?}",
        seen.iter().map(|input| input.scale).collect::<Vec<_>>(),
    );

    let [
        SurfaceEvent::PointerMove {
            position: hover, ..
        },
        SurfaceEvent::PointerDown {
            position: pressed,
            button: SurfaceButton::Left,
            ..
        },
        SurfaceEvent::PointerMove {
            position: moved,
            pressed: drag_button,
            ..
        },
        SurfaceEvent::PointerUp {
            position: released,
            button: SurfaceButton::Left,
            ..
        },
    ] = events.as_slice()
    else {
        panic!("expected a hover, a down, a dragged move and an up, in order: {events:?}");
    };

    assert_eq!(hover, pressed, "the down lands where the pointer was");
    assert_eq!(
        drag_button,
        &Some(SurfaceButton::Left),
        "the dragged move reports the held button"
    );
    assert_eq!(moved, released, "the up lands where the pointer was");

    // The surface's origin, from the first event and the window position that produced it.
    let origin = [
        down.x.as_f32() - hover.x / scale,
        down.y.as_f32() - hover.y / scale,
    ];
    let expected = [
        (dragged.x.as_f32() - origin[0]) * scale,
        (dragged.y.as_f32() - origin[1]) * scale,
    ];
    assert!(
        (moved.x - expected[0]).abs() < 0.01 && (moved.y - expected[1]).abs() < 0.01,
        "the dragged position is surface-local: {moved:?}, expected {expected:?}",
    );
    assert!(
        origin[0] >= PAD && origin[1] >= PAD,
        "the surface sits where its padding puts it, not at the window origin: {origin:?}",
    );
}

/// A view with a full-window surface and two things over it: a text panel with nothing interactive in
/// it (top-left, `0..60` square), and a button (at `100..160` by `0..60`).
struct Layered {
    events: Rc<RefCell<Vec<SurfaceEvent>>>,
    clicks: Rc<Cell<usize>>,
}

impl Render for Layered {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let events = self.events.clone();
        let clicks = self.clicks.clone();
        div()
            .relative()
            .size_full()
            .child(surface_input(div().size_full(), move |input| {
                events.borrow_mut().push(input.event)
            }))
            // A plain text box: no id, no listeners, so no hitbox and no claim on the pointer.
            .child(
                div()
                    .absolute()
                    .left(px(0.))
                    .top(px(0.))
                    .w(px(60.))
                    .h(px(60.))
                    .child("a caption"),
            )
            // A control: an id and a click listener, so a hitbox in front of the surface — and it
            // occludes, so the surface under it is not hovered and does not see its events.
            .child(
                div()
                    .id("surface-overlay-button")
                    .occlude()
                    .absolute()
                    .left(px(100.))
                    .top(px(0.))
                    .w(px(60.))
                    .h(px(60.))
                    .on_click(move |_, _, _| clicks.set(clicks.get() + 1)),
            )
    }
}

/// A layered view drawn once, with the handle, the recorded events and the click count.
fn layered(
    cx: &mut TestAppContext,
) -> (
    VisualTestContext,
    Rc<RefCell<Vec<SurfaceEvent>>>,
    Rc<Cell<usize>>,
) {
    let events = Rc::new(RefCell::new(Vec::new()));
    let clicks = Rc::new(Cell::new(0));
    let (events_for_view, clicks_for_view) = (events.clone(), clicks.clone());
    let window = AnyWindowHandle::from(cx.add_window(move |_, _| Layered {
        events: events_for_view,
        clicks: clicks_for_view,
    }));
    let mut cx = VisualTestContext::from_window(window, cx);
    cx.update(|window, cx| window.draw(cx).clear(cx));
    (cx, events, clicks)
}

/// A point over the button.
fn over_button() -> gpui::Point<gpui::Pixels> {
    point(px(130.), px(30.))
}

/// A point inside the surface, clear of both overlays.
fn over_surface() -> gpui::Point<gpui::Pixels> {
    point(px(400.), px(300.))
}

/// A point over the caption, which is not a control.
fn over_caption() -> gpui::Point<gpui::Pixels> {
    point(px(30.), px(30.))
}

#[gpui::test]
fn a_caption_over_the_surface_does_not_claim_the_pointer(cx: &mut TestAppContext) {
    let (mut cx, events, _) = layered(cx);

    // A move over the caption reaches the surface: a box of text inserts no hitbox, so the surface
    // under it is still what the pointer is over.
    cx.simulate_mouse_move(over_caption(), None::<MouseButton>, Modifiers::none());
    assert!(
        matches!(
            events.borrow().as_slice(),
            [SurfaceEvent::PointerMove { .. }]
        ),
        "a move over the caption reaches the surface: {:?}",
        events.borrow(),
    );
}

#[gpui::test]
fn a_control_over_the_surface_takes_its_own_events(cx: &mut TestAppContext) {
    let (mut cx, events, clicks) = layered(cx);

    // A move over the button does not reach the surface — the button is in front of it.
    cx.simulate_mouse_move(over_button(), None::<MouseButton>, Modifiers::none());
    assert!(
        events.borrow().is_empty(),
        "the button is in front, so the surface sees nothing: {:?}",
        events.borrow(),
    );

    // And the click is the button's, not the surface's.
    cx.simulate_click(over_button(), Modifiers::none());
    assert_eq!(clicks.get(), 1, "the button got the click");
    assert!(
        !events
            .borrow()
            .iter()
            .any(|event| matches!(event, SurfaceEvent::PointerDown { .. })),
        "the surface saw no pointer-down: {:?}",
        events.borrow(),
    );
}

/// A view whose surface is smaller than the window, so a drag can leave it.
struct Bounded {
    events: Rc<RefCell<Vec<SurfaceEvent>>>,
}

impl Render for Bounded {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let events = self.events.clone();
        div().relative().size_full().child(surface_input(
            div()
                .absolute()
                .left(px(0.))
                .top(px(0.))
                .w(px(200.))
                .h(px(200.)),
            move |input| events.borrow_mut().push(input.event),
        ))
    }
}

#[gpui::test]
fn a_drag_that_leaves_the_surface_keeps_arriving(cx: &mut TestAppContext) {
    let events = Rc::new(RefCell::new(Vec::new()));
    let recorded = events.clone();
    let window = AnyWindowHandle::from(cx.add_window(move |_, _| Bounded { events: recorded }));
    let mut cx = VisualTestContext::from_window(window, cx);
    cx.update(|window, cx| window.draw(cx).clear(cx));
    let scale = cx.update(|window, _| window.scale_factor());

    // A drag that starts inside the surface and leaves it, still inside the window.
    let inside = point(px(100.), px(100.));
    let beyond = point(px(300.), px(250.));
    cx.simulate_mouse_move(inside, None::<MouseButton>, Modifiers::none());
    cx.simulate_mouse_down(inside, MouseButton::Left, Modifiers::none());
    cx.simulate_mouse_move(beyond, MouseButton::Left, Modifiers::none());

    let events = events.borrow();
    assert!(
        events.iter().any(|event| matches!(
            event,
            SurfaceEvent::PointerMove { pressed: Some(SurfaceButton::Left), position, .. }
                if (position.x - beyond.x.as_f32() * scale).abs() < 0.01
                    && (position.y - beyond.y.as_f32() * scale).abs() < 0.01
        )),
        "the captured drag keeps arriving past the surface's bounds: {events:?}",
    );
}

#[gpui::test]
fn a_drag_that_starts_on_a_control_is_not_the_surfaces(cx: &mut TestAppContext) {
    let (mut cx, events, clicks) = layered(cx);

    // The gesture begins on the button, then the pointer moves away and up over the surface.
    cx.simulate_mouse_move(over_button(), None::<MouseButton>, Modifiers::none());
    cx.simulate_mouse_down(over_button(), MouseButton::Left, Modifiers::none());
    cx.simulate_mouse_move(over_surface(), MouseButton::Left, Modifiers::none());
    cx.simulate_mouse_up(over_surface(), MouseButton::Left, Modifiers::none());

    assert!(
        !events
            .borrow()
            .iter()
            .any(|event| matches!(event, SurfaceEvent::PointerDown { .. })),
        "the button occludes, so a drag that starts on it is not the surface's: {:?}",
        events.borrow(),
    );
    assert_eq!(
        clicks.get(),
        0,
        "nor was it a click, the pointer having left the button before the up",
    );
}

#[gpui::test]
fn a_drag_from_the_surface_keeps_arriving_over_a_control(cx: &mut TestAppContext) {
    let (mut cx, events, clicks) = layered(cx);

    // Settle on mouse input, then drag from the surface onto the button and hold still.
    cx.simulate_mouse_move(over_surface(), None::<MouseButton>, Modifiers::none());
    cx.simulate_mouse_down(over_surface(), MouseButton::Left, Modifiers::none());
    cx.simulate_mouse_move(over_button(), MouseButton::Left, Modifiers::none());

    let events = events.borrow();
    assert!(
        events.iter().any(|event| matches!(
            event,
            SurfaceEvent::PointerMove {
                pressed: Some(SurfaceButton::Left),
                ..
            }
        )),
        "the captured drag keeps arriving where the surface would not be hovered: {events:?}",
    );
    assert_eq!(clicks.get(), 0, "and the button did not get the drag");
}
