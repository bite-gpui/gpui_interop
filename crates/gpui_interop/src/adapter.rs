//! Matching a foreign device to the window's renderer.
//!
//! Cross-device sharing needs the **same physical adapter**, and that is not assertable from inside
//! either device — a Direct3D 11 device and a wgpu adapter report their identity in different terms.
//! [`Adapter`] is the producer's view of the window's device; [`Adapter::wgpu`] returns a wgpu device
//! and queue only when one matches.
//!
//! On Windows the window's `ID3D11Device` names its adapter by LUID, and the same LUID identifies the
//! wgpu DX12 adapter, so a match can be found without the caller handing over its device. The LUID —
//! not the adapter name — is the identity, because DXGI and wgpu enumerate a *different number* of
//! `Microsoft Basic Render Driver` entries, so the names collide where the LUIDs do not. When no
//! adapter matches, [`Adapter::wgpu`] answers `None`; a caller-supplied device is the documented
//! escape for a driver that hides the LUID, but it is not part of this surface yet.
//!
//! The *interface* is universal though the *mechanism* is Windows-only: off Windows there is
//! nothing to LUID-match (macOS's wgpu adapter *is* the `MetalRenderer`'s device; Linux is
//! device-node selection), so [`Adapter::wgpu`] answers `None` there rather than failing a match.

use std::any::Any;
use std::rc::Rc;
#[cfg(feature = "wgpu")]
use std::sync::Arc;

/// Why a window cannot be bridged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unavailable {
    /// The renderer lends no device — an offscreen or foreign renderer.
    NoDevice,
    /// No wgpu adapter matches the window's device.
    NoAdapter,
    /// This platform's module is not built.
    Unsupported,
}

/// The producer-side device, matched to the window's renderer.
pub struct Adapter<'a> {
    #[allow(
        dead_code,
        reason = "held for the platform modules; without the `wgpu` feature none reads it"
    )]
    device: &'a Rc<dyn Any>,
}

impl<'a> Adapter<'a> {
    pub(crate) fn new(device: &'a Rc<dyn Any>) -> Self {
        Self { device }
    }

    /// A wgpu device and queue the producer can render on, if one matches the window's.
    ///
    /// On Windows this reads the window renderer's Direct3D 11 adapter LUID and finds the wgpu DX12
    /// adapter carrying the same LUID, then requests a device and queue on it; `None` when
    /// no adapter matches or the renderer lends no Direct3D device. A window whose renderer *is*
    /// wgpu reaches its device and queue without this crate.
    #[cfg(feature = "wgpu")]
    pub fn wgpu(&self) -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        #[cfg(target_os = "windows")]
        {
            matched::wgpu_device(self.device)
        }
        #[cfg(not(target_os = "windows"))]
        {
            // Nothing to match off Windows: each platform's module owns its own device story.
            None
        }
    }
}

/// The Windows mechanism: the window's `ID3D11Device` → its adapter LUID → the wgpu DX12 adapter with
/// the same LUID.
#[cfg(all(target_os = "windows", feature = "wgpu"))]
mod matched {
    use std::any::Any;
    use std::future::Future;
    use std::pin::pin;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    use windows::core::Interface;
    use windows::Win32::Foundation::LUID;
    use windows::Win32::Graphics::Direct3D11::ID3D11Device;
    use windows::Win32::Graphics::Dxgi::{IDXGIAdapter, IDXGIDevice};

    /// A wgpu device and queue on the adapter the window's Direct3D 11 device runs on, or `None`
    /// when the device lends no LUID or no DX12 adapter matches it.
    pub(super) fn wgpu_device(
        device: &Rc<dyn Any>,
    ) -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        let window_luid = device_adapter_luid(device.downcast_ref::<ID3D11Device>()?)?;

        let instance = wgpu::Instance::new(dx12_instance());
        let adapters = block_on(instance.enumerate_adapters(wgpu::Backends::DX12));
        let adapter = adapters
            .iter()
            .find(|adapter| wgpu_adapter_luid(adapter).is_some_and(|l| same_luid(l, window_luid)))?;

        let (device, queue) = block_on(adapter.request_device(&device_descriptor(adapter))).ok()?;
        Some((Arc::new(device), Arc::new(queue)))
    }

    /// The LUID of the adapter a Direct3D 11 device was created on: `IDXGIDevice::GetAdapter` then
    /// `IDXGIAdapter::GetDesc`.
    fn device_adapter_luid(device: &ID3D11Device) -> Option<LUID> {
        // SAFETY: the calls are the documented query path for a device's adapter; every value they
        // return is owned by this scope.
        unsafe {
            let dxgi_device: IDXGIDevice = device.cast().ok()?;
            let adapter: IDXGIAdapter = dxgi_device.GetAdapter().ok()?;
            Some(adapter.GetDesc().ok()?.AdapterLuid)
        }
    }

    /// The LUID of a wgpu DX12 adapter, read from the HAL adapter rather than the reported name.
    fn wgpu_adapter_luid(adapter: &wgpu::Adapter) -> Option<LUID> {
        // SAFETY: `raw_adapter` borrows the HAL adapter behind `as_hal`'s guard, which lives as long
        // as `adapter`, and `GetDesc` only reads.
        unsafe {
            let hal = adapter.as_hal::<wgpu::hal::dx12::Api>()?;
            Some(hal.raw_adapter().GetDesc().ok()?.AdapterLuid)
        }
    }

    /// Identity is the LUID, not the adapter name.
    fn same_luid(a: LUID, b: LUID) -> bool {
        a.LowPart == b.LowPart && a.HighPart == b.HighPart
    }

    /// A DX12-only instance: the window's renderer is Direct3D, so DX12 is the only backend that can
    /// share its adapter.
    fn dx12_instance() -> wgpu::InstanceDescriptor {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::DX12;
        descriptor
    }

    /// The producer's device, capped to the adapter's limits the way `gpui_wgpu` caps its own.
    fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
        wgpu::DeviceDescriptor {
            label: Some("gpui_interop_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults()
                .using_resolution(adapter.limits())
                .using_alignment(adapter.limits()),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }
    }

    /// `enumerate_adapters` is async but resolves without parking, so a no-op waker and a yield loop
    /// suffice — no async runtime is needed.
    fn block_on<F: Future>(future: F) -> F::Output {
        struct NoopWake;
        impl Wake for NoopWake {
            fn wake(self: Arc<Self>) {}
        }

        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        let mut future = pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }
}
