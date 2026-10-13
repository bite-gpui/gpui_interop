#![cfg(all(target_os = "linux", feature = "readback-test"))]
//! The hatch, end to end: the dma-buf a [`SharedBuffer`] exports, composited and read back.
//!
//! The crate's unit tests (`src/linux.rs`) prove the hatch produces the *descriptor* it claims — an
//! fd, a modifier, per-plane offsets and strides. They cannot prove the thing that actually matters:
//! that a renderer can import that image and sample the bytes the producer wrote. This test does, by
//! compositing the buffer through a headless window and reading the frame back:
//!
//! ```text
//! GpuDevice::open()                 // a dma-buf-capable Vulkan device, the producer's own
//! SharedBuffer::new(..).clear(..)   // a shareable image, written on the GPU
//! surface(buffer.handle())          // handed to the scene as the `surface()` element takes it
//! // ...the window composites it, and the pixel is read back and compared to the clear colour
//! ```
//!
//! The renderer is the headless `wgpu` one — the same renderer a window drives on Linux, offscreen —
//! so this needs no display.
//!
//! # The tiled arm is not here, and why
//!
//! A `Tiling::Linear` buffer round-trips byte for byte. A `Tiling::Tiled` one, added to the same
//! frame, does **not**: sampling it wedges the device. The read-back never completes —
//! `Failed to wait for headless rendering: The requested Wait timed out before the submission was
//! completed.`, followed by `GPU device was lost` — and the same happens with the tiled buffer on its
//! own, and on either the adapter's device or the producer's own. Two *linear* buffers in the same
//! frame are fine, so it is the modifier, not the second surface.
//!
//! That is a real bug and the reason this arm is worth having, but a test that hangs the GPU is not
//! one that can be committed. It is left out until the tiled export is fixed; when it is, add the
//! buffer here and this comment goes.
//!
//! # Why it is opt-in
//!
//! Behind the `readback-test` feature for the same reason the core's own dma-buf test is behind
//! `test-support`: driving a window pulls in the facade's platform bundle, and the test needs a real
//! Vulkan device. A machine without one skips rather than fails, but building it at all needs the
//! windowing stack's system libraries, which the crate's ordinary `cargo test` deliberately does not:
//!
//! ```sh
//! cargo test -p gpui_interop --features readback-test --test linux_dmabuf_readback
//! ```
//!
//! # What this test does not cover
//!
//! The adapter step — `attach(..)` → `Adapter::gpu_device()`, the second logical device on the
//! renderer's own GPU. It is exercised by the `interop_linux` example, not here, because reaching the
//! renderer's device needs a `gpu_canvas` callback, and the `surface()` element this test uses takes
//! its handle before any frame is painted.

use std::sync::Arc;

use gpui::{
    AnyWindowHandle, AppContext as _, Context, IntoElement, Render, SurfaceFormatKind, Window, div,
    prelude::*, px, size, surface,
};
use gpui_interop::{GpuDevice, SharedBuffer, Tiling};
use gpui_wgpu::CosmicTextSystem;

/// The edge of the produced surface, in pixels.
const EDGE: u32 = 64;

/// The colour the buffer is cleared to, as the RGBA `clear` takes. `Rgba8`, so the bytes stored are
/// these four in order; a round trip that scrambles a channel comes back a different colour.
const CLEAR: [f32; 4] = [128.0 / 255.0, 128.0 / 255.0, 0.0, 1.0];

/// [`CLEAR`] in bytes — what a read-back must return.
const EXPECTED: [u8; 4] = [128, 128, 0, 255];

/// The window's view. The handle is built before the window opens, because `surface()` takes it at
/// render time while the producer's device exists independently of the renderer's.
struct View {
    handle: gpui::DmaBufHandle,
}

impl Render for View {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(surface(self.handle.clone()).size_full())
    }
}

/// A linear buffer, exported as a dma-buf, composited by the renderer, and read back byte for byte.
#[test]
fn an_exported_buffer_composites_byte_for_byte() {
    if gpui::current_headless_renderer().is_none() {
        eprintln!("skipping: no headless renderer (no Vulkan adapter)");
        return;
    }
    let Some(device) = GpuDevice::open().ok() else {
        eprintln!("skipping: no Vulkan device exports a dma-buf");
        return;
    };

    // The device must outlive the buffer, and the buffer must outlive the frame that samples it.
    let buffer = SharedBuffer::new(&device, EDGE, EDGE, SurfaceFormatKind::rgba8(), Tiling::Linear)
        .expect("allocate a shareable buffer");
    buffer.clear(&device, CLEAR).expect("clear the buffer");
    let handle = buffer.handle();

    let text_system = Arc::new(CosmicTextSystem::new("fallback"));
    let mut cx = gpui::HeadlessAppContext::with_platform(text_system, Arc::new(()), || {
        Ok(gpui::current_headless_renderer())
    });

    let window = cx
        .open_window(size(px(EDGE as f32), px(EDGE as f32)), |_window, cx| {
            cx.new(|_| View { handle })
        })
        .expect("a window to composite into");
    let window: AnyWindowHandle = window.into();

    cx.run_until_parked();
    cx.update_window(window, |_, window, cx| {
        let _ = window.draw(cx);
    })
    .expect("draw a frame");

    let image = cx.capture_screenshot(window).expect("read the frame back");
    assert_eq!(
        image.get_pixel(EDGE / 2, EDGE / 2).0,
        EXPECTED,
        "the buffer did not round trip"
    );
}
