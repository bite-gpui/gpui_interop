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
//! (`GpuCanvasContext::device`) and yields an [`Interop`], or [`Unavailable`] when the
//! canvas lent no device — an offscreen or foreign renderer. It is called inside the canvas, not
//! before a frame is painted: the window's renderer, and so its device, does not exist until then.
//! From there:
//!
//! - [`Interop::adapter`] returns an [`Adapter`], the producer's view of the window's device.
//! - With the `wgpu` feature, `Adapter::wgpu` returns a wgpu `Device` and `Queue` on the adapter that
//!   matches the window's device (a LUID match on Windows), or `None` when none does. It is the
//!   adapter match, not a wgpu transport: the frame still leaves through the platform's native
//!   surface type, so a wgpu producer that can render on the window's own device should use GPUI's
//!   imported-texture path instead. On Linux the match is [`Adapter::gpu_device`], which makes a
//!   second logical device on the renderer's own `vk::PhysicalDevice`.
//! - On Windows, [`SharedSurface`] / [`Fence`] are the producer half of the Direct3D transport: a
//!   Direct3D 12 producer renders into a shared texture and signals a fence, and hands GPUI the
//!   `DirectXSource::Shared` handles — which the renderer opens on its own Direct3D 11 device, with
//!   the shared fence ordering the two queues.
//! - On Linux, [`SharedBuffer`] is the producer half of the dma-buf transport: a producer renders into
//!   a shareable image and exports it as a dma-buf, and hands GPUI the `gpui::DmaBufHandle` the
//!   renderer imports under its modifier. Single- or two-plane, linear or on a driver-tiled modifier
//!   the renderer can import, and ordered by a [`Fence`] that exports the acquire `sync_file`. It
//!   covers both sharing contracts: [`SharedBuffer::new`] lets the driver choose the image's layout and
//!   reports it back, while [`SharedBuffer::export_at`] pushes bytes at a plane layout the producer
//!   declares — the shape a decoder or capture pipeline whose layout comes from elsewhere needs.
//!
//! # The other direction
//!
//! With the `guest` feature, [`Guest`] runs the inverse: headless GPUI on a worker thread for a host
//! loop that owns the window and its cadence. Frames leave as shareable surfaces, so the host samples
//! them on the same device; work and input enter through [`Guest::update`]. Only the frame handle and
//! plain commands cross the thread; the `!Send` renderer stays pinned to the guest.
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
//!     // With the `wgpu` feature, this gives a device and queue on a matching adapter (Windows).
//!     let (device, queue) = interop.adapter().wgpu().expect("no matching adapter");
//!
//!     // The hand-off is native: a `SharedSurface` on Windows, a `SharedBuffer` on Linux.
//!     Ok(())
//! }
//! ```
//!
//! The `wgpu` line requires the `wgpu` feature; without it, `Adapter::wgpu` is not compiled in. The
//! Linux `SharedBuffer`/`Fence` transport and the Windows handles remain available without it, because
//! neither is a `wgpu` client.
//!
//! # Provisional
//!
//! The crate is provisional and deliberately incomplete, and its API may change. Only the platform
//! backends that are implemented today are exposed; the rest arrive as they are written.

mod adapter;
#[cfg(feature = "guest")]
mod guest;
#[cfg(feature = "gpui")]
mod input;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

pub use adapter::{Adapter, Unavailable};
#[cfg(feature = "guest")]
pub use guest::{Guest, GuestFrame, GuestOptions};
#[cfg(feature = "gpui")]
pub use input::{
    SurfaceButton, SurfaceEvent, SurfaceInput, SurfaceModifiers, SurfacePoint, surface_input,
};
#[cfg(target_os = "linux")]
pub use linux::{Fence, GpuDevice, GpuId, LinearLayout, PlaneLayout, SharedBuffer, Tiling, gpu_id, remember};
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
    Ok(Interop {
        device: Rc::new(device),
    })
}

/// A window's renderer, as a producer sees it.
pub struct Interop {
    /// The window renderer's device, erased. The per-platform modules downcast it to the one type
    /// their transport needs — an `ID3D11Device`, a `(wgpu::Device, wgpu::Queue)` — which is what
    /// keeps the facade free of the platform graphics crates.
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
