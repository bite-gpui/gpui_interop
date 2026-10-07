//! `gpui-interop`: composite pixels produced outside GPUI's renderer into a GPUI scene.
//!
//! A producer whose graphics device is **not** the window renderer's — another API, another adapter,
//! another process — renders a frame and hands it to the scene through the `surface()` element,
//! without a CPU copy of the pixels. When the producer can render directly on the window's own
//! device, go through GPUI's built-in same-device path instead. Reach for this crate when the pixels
//! come from somewhere the window renderer cannot draw itself, and must still land in the scene
//! without a round trip through the CPU.
//!
//! # The shape of the bridge
//!
//! [`attach`] takes the device a `gpu_canvas` callback lends
//! ([`gpui::GpuCanvasContext::device`]) and yields an [`Interop`], or [`Unavailable`] when the
//! canvas lent no device — an offscreen or foreign renderer. It is called inside the canvas, not
//! before a frame is painted: the window's renderer, and so its device, does not exist until then.
//! From there:
//!
//! - [`Interop::adapter`] returns an [`Adapter`], the producer's view of the window's device.
//! - With the `wgpu` feature, `Adapter::wgpu` returns a wgpu `Device` and `Queue` on the adapter that
//!   matches the window's device, or `None` when none does. Rendering with those puts the producer's
//!   pixels on the same physical GPU as the window.
//! - On Windows, [`SharedSurface`] / [`Fence`] are the producer half of the Direct3D transport: a
//!   Direct3D 12 producer renders into a shared texture and signals a fence, and hands GPUI the
//!   `DirectXSource::Shared` handles — which the renderer opens on its own Direct3D 11 device, with
//!   the shared fence ordering the two queues.
//!
//! # Sketch
//!
//! ```rust
//! use gpui_interop::{attach, Unavailable};
//!
//! fn bridge(gpu: &mut gpui::GpuCanvasContext) -> Result<(), Unavailable> {
//!     // Negotiate with the window's renderer, inside the canvas that lends its device.
//!     let interop = attach(gpu.try_device::<gpui::DirectXRenderer>())?;
//!
//!     // With the `wgpu` feature, this gives a device and queue on a matching adapter.
//!     let (device, queue) = interop.adapter().wgpu().expect("no matching adapter");
//!
//!     // Render the frame on `device`, then hand its surface to the scene through `surface()`.
//!     Ok(())
//! }
//! ```
//!
//! The `wgpu` line requires the `wgpu` feature; without it, `Adapter::wgpu` is not compiled in, and
//! only the Windows handle transport is available.
//!
//! # Provisional
//!
//! The crate is provisional and deliberately incomplete, and its API may change. Only the platform
//! backends that are implemented today are exposed; the rest arrive as they are written.

mod adapter;
mod guest;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

pub use adapter::{Adapter, Unavailable};
#[cfg(target_os = "windows")]
pub use windows::{Fence, SharedSurface};

use std::any::Any;
use std::rc::Rc;

/// Negotiate with the window renderer a `gpu_canvas` callback reaches.
///
/// Called inside the canvas, with the device the callback lends — `gpu.device::<R>()`, or
/// `gpu.try_device::<R>()` for the fallible form. There is nothing to attach to before a frame is
/// painted, because the window's renderer, and its device, do not exist until then.
///
/// Returns [`Unavailable::NoDevice`] when the canvas lent no device — an offscreen or foreign
/// renderer — so this is a recoverable error rather than a panic.
pub fn attach<D: 'static>(device: Option<D>) -> Result<Interop, Unavailable> {
    let device = device.ok_or(Unavailable::NoDevice)?;
    Ok(Interop { device: Rc::new(device) })
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
