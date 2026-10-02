//! `gpui-interop`: the downstream bridge between a foreign producer and GPUI's renderer.
//!
//! A producer whose device is **not** the window renderer's — another API, another adapter, another
//! process — renders a frame and hands it to the scene through `surface()`, with no CPU copy. The
//! same-device case needs none of this; this crate is the "other device" row of
//! `bite-gpui-project/spi/rendering/surfaces.md` §2 — the OS-surface trinity, *match the adapter,
//! move a handle, order the queues*, in each platform's clothes.
//!
//! # Provisional, and deliberately incomplete
//!
//! The crate is being built **in the fork** so it can iterate quickly; it will move to its own
//! repository once its shape is settled (`bite-gpui-project/spi/rendering/interop-scaffold.md`). Only
//! what a *cleared* probe supports is written here.
//!
//! The shapes the unrun probes decide are deliberately absent, because a late answer is a redesign
//! (`interop-crate.md` §4):
//!
//! - **P6 (adapter matching)** decides the API. Windows LUID matching, a caller-supplied device, or
//!   neither — the answer fixes what [`Interop`] and [`Adapter`] must be, so the pool and its ring
//!   are not written yet.
//! - **P5 and P9** (the fence loop and device loss) gate the Windows module.
//! - **P2** gates the macOS module.
//! - **P3 is cleared**, so the Linux module is the one that can be built next.
//!
//! Each platform module lands with its probe. What is here is the part that is true today: reading
//! the window renderer's device, and naming why that can fail.
//!
//! [`interop-crate.md`]: https://github.com/bite-gpui/bite-gpui-project/blob/main/spi/rendering/interop-crate.md

mod adapter;
mod guest;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

pub use adapter::{Adapter, Unavailable};

use std::any::Any;
use std::rc::Rc;

/// Negotiate with `window`'s renderer.
///
/// Fails when there is nothing to bridge *to*: a renderer that lends no device — an offscreen or a
/// foreign one — leaves nothing for a producer to render on, and that is [`Unavailable::NoDevice`],
/// not a panic. This is the accessor the surface work adds
/// (`Window::device_any`), and it is the *only* thing this crate needs from core.
pub fn attach(window: &gpui::Window) -> Result<Interop, Unavailable> {
    let device = window.device_any().ok_or(Unavailable::NoDevice)?;
    Ok(Interop { device })
}

/// A window's renderer, as a producer sees it.
pub struct Interop {
    /// The window renderer's device, erased. The per-platform modules downcast it to the one type
    /// their transport needs — an `ID3D11Device`, a `WgpuContext` — which is what keeps the facade
    /// free of the platform graphics crates.
    #[allow(dead_code, reason = "read by the platform modules as they land")]
    device: Rc<dyn Any>,
}

impl Interop {
    /// The producer-side device, matched to the window's renderer.
    pub fn adapter(&self) -> Adapter<'_> {
        Adapter::new(&self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_unavailable_reasons_are_distinct() {
        assert_eq!(Unavailable::NoDevice, Unavailable::NoDevice);
        assert_ne!(Unavailable::NoAdapter, Unavailable::Unsupported);
    }
}
