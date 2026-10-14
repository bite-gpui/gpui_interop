//! The surface input gateway: GPUI's events, adapted for a producer.
//!
//! GPUI owns the window and its input. It hit-tests the scene front to back and dispatches to the
//! element under the pointer — a `gpu_canvas` is an ordinary element in that tree, so the events over
//! its bounds are the producer's. This module is how they reach a producer **without the producer
//! naming `gpui`**: the element side (which lives here, in the crate that already depends on the
//! facade) subscribes to GPUI's handlers and hands the producer a [`SurfaceInput`].
//!
//! ```text
//! surface_input(gpu_canvas(..), |input| { .. })   // the element, with a producer's handler
//!     SurfaceInput { event: PointerMove { position: .., .. }, scale }   // what the handler gets
//! ```
//!
//! What the gateway adapts:
//!
//! - **Coordinates.** GPUI's events carry a window-relative logical `Point<Pixels>`. A [`SurfaceEvent`]
//!   carries a [`SurfacePoint`] in **physical pixels from the surface's top-left corner**, and the
//!   [`SurfaceInput`] carries the `scale` it was computed with (physical pixels per logical pixel — the
//!   window's device-pixel ratio). A producer that wants logical surface-local coordinates divides the
//!   position by `scale`.
//! - **Vocabulary.** [`SurfaceEvent`] and its parts are the crate's, mapped from GPUI's, so a producer
//!   never names `MouseMoveEvent`.
//!
//! What it deliberately does **not** do:
//!
//! - **Route.** Hit-testing and dispatch are the engine's. The gateway only sees what GPUI already
//!   routed to the surface's element.
//! - **Decide pass-through.** Whether a producer wants an *event it can see* but not *handle* — so a
//!   GPUI element under the surface's rectangle still receives it — is the producer's to answer, and
//!   the engine has no "declines" signal on a hit target for the gateway to relay yet.
//!
//! **A gesture starts where it starts.** On pointer-down the gateway captures the pointer
//! ([`Window::capture_pointer`]) — the engine's mechanism for keeping a drag arriving once the pointer
//! leaves the element's bounds — and the engine releases the capture on the up. Two things about that
//! capture are the engine's and worth knowing before relying on it: it is keyed by a hitbox **id**, and
//! the engine numbers a frame's hitboxes afresh, so a capture does not survive a redraw; and an element
//! that **occludes** the surface ([`InteractiveElement::occlude`]) takes the events over its rectangle
//! whether or not the surface has captured. Over anything that does not occlude, the gesture arrives as
//! the rule says.

use std::cell::Cell;
use std::panic::Location;
use std::rc::Rc;

use gpui::{
    App, Bounds, Element, ElementId, GlobalElementId, Hitbox, HitboxId, InteractiveElement,
    IntoElement, LayoutId, Modifiers, MouseButton, Pixels, Point, Window,
};

/// Where a pointer is on the surface, in physical pixels from its top-left corner.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurfacePoint {
    /// Pixels to the right of the surface's left edge.
    pub x: f32,
    /// Pixels below the surface's top edge.
    pub y: f32,
}

/// A pointer button, as a producer sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceButton {
    /// The primary button.
    Left,
    /// The secondary button.
    Right,
    /// The middle button.
    Middle,
    /// A navigation button (back/forward), or one the gateway does not distinguish.
    Other,
}

/// The modifier keys held when an event arrived.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SurfaceModifiers {
    pub shift: bool,
    pub control: bool,
    pub alt: bool,
    /// The platform key — Command on macOS, the Super/Windows key elsewhere.
    pub platform: bool,
}

/// A producer-facing input event over a surface, in surface-local physical pixels.
///
/// The event's own position is the pointer's; only a [`Scroll`](Self::Scroll) event, or a gesture a
/// producer reads from absolute positions, needs it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SurfaceEvent {
    /// A pointer button went down over the surface.
    PointerDown {
        position: SurfacePoint,
        button: SurfaceButton,
        modifiers: SurfaceModifiers,
    },
    /// The pointer moved over the surface — while a button is held (`pressed`), while the surface has
    /// captured the pointer, or merely hovering.
    PointerMove {
        position: SurfacePoint,
        pressed: Option<SurfaceButton>,
        modifiers: SurfaceModifiers,
    },
    /// A pointer button went up.
    PointerUp {
        position: SurfacePoint,
        button: SurfaceButton,
        modifiers: SurfaceModifiers,
    },
}

/// A surface input, adapted from GPUI's, ready for a producer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceInput {
    /// What happened.
    pub event: SurfaceEvent,
    /// Physical pixels per logical pixel — the window's device-pixel ratio. Every
    /// [`SurfaceEvent`]'s position is already in these pixels; this is the factor they were computed
    /// with, for a producer that wants logical coordinates instead.
    pub scale: f32,
}

/// Wrap `element` so a producer receives its input through [`SurfaceInput`].
///
/// `element` is the surface's own element — a `gpu_canvas` — and `handler` is called with every pointer
/// event routed to it, in surface-local physical pixels. The window is marked for redraw after each
/// event, since a surface that handled input has almost certainly changed what it draws.
///
/// The element keeps the sizing, styling and painting it had; the gateway only adds the input
/// subscription, and reads the element's bounds from its own prepaint so the producer's coordinates are
/// surface-local without the caller plumbing them.
pub fn surface_input<E>(element: E, handler: impl Fn(SurfaceInput) + 'static) -> impl IntoElement
where
    E: InteractiveElement + Element<PrepaintState = Option<Hitbox>> + 'static,
{
    let surface = Rc::new(Surface {
        hitbox: Cell::new(None),
        handler: Rc::new(handler),
    });
    let inner = attach(element, &surface);
    SurfaceInputElement { inner, surface }
}

/// The state the listeners and the element share: the last hitbox, and the producer's handler.
struct Surface {
    /// The element's hitbox from the last prepaint. Its bounds are the surface-local origin, and its id
    /// is what a pointer-down captures.
    hitbox: Cell<Option<(Bounds<Pixels>, HitboxId)>>,
    handler: Rc<dyn Fn(SurfaceInput)>,
}

/// Subscribe GPUI's pointer handlers on `element`, translating each to a [`SurfaceInput`] through
/// `surface`.
fn attach<E: InteractiveElement>(element: E, surface: &Rc<Surface>) -> E {
    let down = surface.clone();
    let element = element.on_any_mouse_down(move |event, window, _cx| {
        // Keep the gesture with the surface, so its moves and up arrive even off its bounds.
        if let Some((_, id)) = down.hitbox.get() {
            window.capture_pointer(id);
        }
        let button = button(event.button);
        let modifiers = modifiers(event.modifiers);
        emit(&down, window, event.position, |position| {
            SurfaceEvent::PointerDown {
                position,
                button,
                modifiers,
            }
        });
    });

    let moved = surface.clone();
    let element = element.on_mouse_move(move |event, window, _cx| {
        let pressed = event.pressed_button.map(button);
        let modifiers = modifiers(event.modifiers);
        emit(&moved, window, event.position, |position| {
            SurfaceEvent::PointerMove {
                position,
                pressed,
                modifiers,
            }
        });
    });

    let up = surface.clone();
    element.capture_any_mouse_up(move |event, window, _cx| {
        let button = button(event.button);
        let modifiers = modifiers(event.modifiers);
        emit(&up, window, event.position, |position| {
            SurfaceEvent::PointerUp {
                position,
                button,
                modifiers,
            }
        });
    })
}

/// Translate `window_position` to surface-local physical pixels and hand the built event to the
/// producer, then ask for the redraw the handled input implies.
fn emit(
    surface: &Rc<Surface>,
    window: &mut Window,
    window_position: Point<Pixels>,
    event: impl FnOnce(SurfacePoint) -> SurfaceEvent,
) {
    let Some((bounds, _)) = surface.hitbox.get() else {
        // No prepaint yet — nothing has been laid out to translate against.
        return;
    };
    let scale = window.scale_factor();
    let local = window_position - bounds.origin;
    let position = SurfacePoint {
        x: local.x.as_f32() * scale,
        y: local.y.as_f32() * scale,
    };
    (surface.handler)(SurfaceInput {
        event: event(position),
        scale,
    });
    window.refresh();
}

/// The guest of a GPUI button, as a producer-facing one.
fn button(button: MouseButton) -> SurfaceButton {
    match button {
        MouseButton::Left => SurfaceButton::Left,
        MouseButton::Right => SurfaceButton::Right,
        MouseButton::Middle => SurfaceButton::Middle,
        MouseButton::Navigate(_) => SurfaceButton::Other,
    }
}

/// The guest of GPUI's modifiers, as a producer-facing set.
fn modifiers(modifiers: Modifiers) -> SurfaceModifiers {
    SurfaceModifiers {
        shift: modifiers.shift,
        control: modifiers.control,
        alt: modifiers.alt,
        platform: modifiers.platform,
    }
}

/// The element [`surface_input`] returns: it delegates everything to the surface's own element, and
/// only records the hitbox its prepaint yields.
struct SurfaceInputElement<E> {
    inner: E,
    surface: Rc<Surface>,
}

impl<E> Element for SurfaceInputElement<E>
where
    E: Element<PrepaintState = Option<Hitbox>>,
{
    type RequestLayoutState = E::RequestLayoutState;
    type PrepaintState = E::PrepaintState;

    fn id(&self) -> Option<ElementId> {
        self.inner.id()
    }

    fn source_location(&self) -> Option<&'static Location<'static>> {
        self.inner.source_location()
    }

    fn request_layout(
        &mut self,
        id: Option<&GlobalElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        self.inner.request_layout(id, window, cx)
    }

    fn prepaint(
        &mut self,
        id: Option<&GlobalElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let prepaint = self.inner.prepaint(id, bounds, request_layout, window, cx);
        if let Some(hitbox) = &prepaint {
            self.surface.hitbox.set(Some((hitbox.bounds, hitbox.id)));
        }
        prepaint
    }

    fn paint(
        &mut self,
        id: Option<&GlobalElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.inner
            .paint(id, bounds, request_layout, prepaint, window, cx);
    }
}

impl<E: Element<PrepaintState = Option<Hitbox>>> IntoElement for SurfaceInputElement<E> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}
