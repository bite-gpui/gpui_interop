//! The Windows module: a Direct3D 12 / `wgpu` producer's surface on GPUI's Direct3D 11 renderer.
//!
//! A producer whose device is not GPUI's renders into a **committed, shared** Direct3D 12 texture
//! and hands GPUI its NT handle; GPUI opens it on its Direct3D 11 device with `OpenSharedResource1`
//! and samples it through a shader resource view. Ordering is a shared `ID3D12Fence`, opened on the
//! Direct3D 11 side as an `ID3D11Fence` and waited **GPU-side** with `ID3D11DeviceContext4::Wait` —
//! never a CPU poll.
//!
//! Two facts shape this code:
//!
//! - **A placed resource cannot be shared.** `CreateSharedHandle` on a placed resource returns
//!   `E_INVALIDARG`; [`SharedSurface`] therefore allocates a **committed** resource on a
//!   `D3D12_HEAP_FLAG_SHARED` heap.
//! - **The fence is ordered GPU-side**, through `ID3D11Device5::OpenSharedFence` and
//!   `ID3D11DeviceContext4::Wait`, not `SetEventOnCompletion` plus a CPU wait.
//!
//! The types are the raw Direct3D transport; the renderer, the ring and device-loss recovery are not
//! here.

use windows::core::{Error, Interface, PCWSTR};
use windows::Win32::Foundation::{CloseHandle, E_POINTER, HANDLE};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11Device1, ID3D11Device5, ID3D11DeviceContext, ID3D11DeviceContext4,
    ID3D11Fence, ID3D11ShaderResourceView, ID3D11Texture2D,
};
use windows::Win32::Graphics::Direct3D12::{
    ID3D12CommandQueue, ID3D12Device, ID3D12Fence, ID3D12Resource, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    D3D12_FENCE_FLAG_SHARED, D3D12_HEAP_FLAG_SHARED, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON,
    D3D12_TEXTURE_LAYOUT_UNKNOWN,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};

/// `GENERIC_ALL`, the access `CreateSharedHandle` needs for a handle this process opens back.
///
/// Named here rather than imported; the value is the documented one (WinNT.h).
const GENERIC_ALL: u32 = 0x1000_0000;

/// A committed, shareable Direct3D 12 texture — the producer's render target.
///
/// The resource lives on a `D3D12_HEAP_FLAG_SHARED` heap and is committed, because a placed resource
/// cannot be shared: `CreateSharedHandle` rejects it with `E_INVALIDARG`. The NT handle outlives the
/// resource's use by GPUI and is closed when this value drops; other processes or devices keep their
/// own references.
pub struct SharedSurface {
    resource: ID3D12Resource,
    handle: HANDLE,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
}

impl SharedSurface {
    /// Allocate a shareable texture of `width` × `height`, render-target capable.
    ///
    /// The resource starts in `D3D12_RESOURCE_STATE_COMMON`, the state a shared texture must be in
    /// when it crosses a device boundary; the producer transitions it for its own passes.
    pub fn new(
        device: &ID3D12Device,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Self> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width as u64,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        };

        // SAFETY: every pointer is to a local that outlives the call, and the resource and handle
        // are owned by the returned value.
        unsafe {
            let mut resource: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )?;
            let resource = resource.ok_or_else(|| Error::from_hresult(E_POINTER))?;
            let handle = device.CreateSharedHandle(&resource, None, GENERIC_ALL, PCWSTR::null())?;
            Ok(Self {
                resource,
                handle,
                width,
                height,
                format,
            })
        }
    }

    /// The NT handle a Direct3D 11 device opens with [`OpenedSurface`]'s `OpenSharedResource1`.
    pub fn handle(&self) -> HANDLE {
        self.handle
    }

    /// The Direct3D 12 resource, for the producer's own render passes.
    pub fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    /// The texture's width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The texture's height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The texture's format.
    pub fn format(&self) -> DXGI_FORMAT {
        self.format
    }

    /// Open this surface on GPUI's Direct3D 11 device and make its shader resource view.
    ///
    /// The view is what the `surface()` element samples; the texture is kept beside it so the view's
    /// resource cannot be released underneath it.
    pub fn open(&self, device: &ID3D11Device) -> windows::core::Result<OpenedSurface> {
        // SAFETY: the handle names this surface's live resource, and the returned view and texture
        // are owned by `OpenedSurface`.
        unsafe {
            let device1: ID3D11Device1 = device.cast()?;
            let texture: ID3D11Texture2D = device1.OpenSharedResource1(self.handle)?;

            let mut view: Option<ID3D11ShaderResourceView> = None;
            device.CreateShaderResourceView(&texture, None, Some(&mut view))?;
            let view = view.ok_or_else(|| Error::from_hresult(E_POINTER))?;

            Ok(OpenedSurface { texture, view })
        }
    }
}

impl Drop for SharedSurface {
    fn drop(&mut self) {
        // SAFETY: `handle` was created by `CreateSharedHandle` and is closed exactly once, here.
        unsafe {
            let _ = CloseHandle(self.handle);
        }
    }
}

/// A [`SharedSurface`] opened on GPUI's Direct3D 11 device: the texture and the view of it.
pub struct OpenedSurface {
    texture: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
}

impl OpenedSurface {
    /// The imported Direct3D 11 texture.
    pub fn texture(&self) -> &ID3D11Texture2D {
        &self.texture
    }

    /// The view the `surface()` element takes.
    pub fn view(&self) -> &ID3D11ShaderResourceView {
        &self.view
    }
}

/// A shared `ID3D12Fence`, opened on a Direct3D 11 device and waited **GPU-side**.
///
/// The producer signals it on its queue after committing a frame; GPUI opens the NT handle as an
/// `ID3D11Fence` and inserts a GPU-side wait, so no CPU thread polls.
pub struct Fence {
    fence: ID3D12Fence,
    handle: HANDLE,
    value: u64,
}

impl Fence {
    /// Create a shareable fence starting at zero.
    pub fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        // SAFETY: the fence and handle are owned by the returned value; the name is null, so the
        // handle is unnamed and shared by value alone.
        unsafe {
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_SHARED)?;
            let handle = device.CreateSharedHandle(&fence, None, GENERIC_ALL, PCWSTR::null())?;
            Ok(Self {
                fence,
                handle,
                value: 0,
            })
        }
    }

    /// The NT handle a Direct3D 11 device opens with [`Fence::open`].
    pub fn handle(&self) -> HANDLE {
        self.handle
    }

    /// The last value signalled.
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Signal the next value on the producer's queue, returning it for the consumer's wait.
    pub fn signal(&mut self, queue: &ID3D12CommandQueue) -> windows::core::Result<u64> {
        self.value += 1;
        // SAFETY: `queue` and `fence` are live, and the value is this fence's next.
        unsafe {
            queue.Signal(&self.fence, self.value)?;
        }
        Ok(self.value)
    }

    /// Open this fence on GPUI's Direct3D 11 device, for a GPU-side wait.
    pub fn open(&self, device: &ID3D11Device) -> windows::core::Result<ID3D11Fence> {
        // SAFETY: the handle names this fence's live object; the opened fence is returned to the
        // caller.
        unsafe {
            let device5: ID3D11Device5 = device.cast()?;
            let mut fence: Option<ID3D11Fence> = None;
            device5.OpenSharedFence(self.handle, &mut fence)?;
            fence.ok_or_else(|| Error::from_hresult(E_POINTER))
        }
    }

    /// Insert a GPU-side wait on `context` for `fence` to reach `value`.
    ///
    /// This is `ID3D11DeviceContext4::Wait`; it does not block a CPU thread.
    pub fn wait_gpu(
        context: &ID3D11DeviceContext,
        fence: &ID3D11Fence,
        value: u64,
    ) -> windows::core::Result<()> {
        // SAFETY: `context` and `fence` are live; the wait is on the device's own queue.
        unsafe {
            let context4: ID3D11DeviceContext4 = context.cast()?;
            context4.Wait(fence, value)
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        // SAFETY: `handle` was created by `CreateSharedHandle` and is closed exactly once, here.
        unsafe {
            let _ = CloseHandle(self.handle);
        }
    }
}
