//! Matching a foreign device to the window's renderer.
//!
//! Cross-device sharing needs the **same physical adapter**, and that is not assertable from inside
//! either device — `bite-gpui-project/spi/rendering/probe-p6-adapter-luid.md` §1. `Adapter` is the
//! producer's view of the window's device; [`Adapter::wgpu`] returns a wgpu device and queue only
//! when one matches.
//!
//! **The match is not written yet.** P6 decides whether it can be found at all, or whether the
//! caller must supply the device — so claiming one now would freeze the answer the probe exists to
//! give. Off Windows the mechanism differs anyway: macOS has nothing to match (wgpu's adapter *is*
//! the `MetalRenderer`'s device), and Linux is device-node selection.

use std::any::Any;
use std::rc::Rc;
#[cfg(feature = "wgpu")]
use std::sync::Arc;

/// Why a window cannot be bridged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unavailable {
    /// The renderer lends no device — an offscreen or foreign renderer.
    NoDevice,
    /// No wgpu adapter matches the window's device (**P6**).
    NoAdapter,
    /// This platform's module is not built.
    Unsupported,
}

/// The producer-side device, matched to the window's renderer.
pub struct Adapter<'a> {
    #[allow(dead_code, reason = "read by the platform modules as they land")]
    device: &'a Rc<dyn Any>,
}

impl<'a> Adapter<'a> {
    pub(crate) fn new(device: &'a Rc<dyn Any>) -> Self {
        Self { device }
    }

    /// A wgpu device and queue the producer can render on, if one matches the window's.
    ///
    /// `None` until **P6** settles the rule — whether a match can be found, or the caller supplies
    /// the device (`bite-gpui-project/spi/rendering/interop-crate.md` §4).
    #[cfg(feature = "wgpu")]
    pub fn wgpu(&self) -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        None
    }
}
