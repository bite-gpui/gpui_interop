#![cfg(all(target_os = "linux", feature = "readback-test"))]
//! The hatch, end to end: the dma-bufs a [`SharedBuffer`] exports, composited and read back.
//!
//! The crate's unit tests (`src/linux.rs`) prove the hatch produces the *descriptor* it claims — an
//! fd, a modifier, per-plane offsets and strides. They cannot prove the thing that actually matters:
//! that a renderer can import that image and sample the bytes the producer wrote. This test does, by
//! compositing a linear and a driver-tiled buffer side by side in one headless window and reading the
//! frame back.
//!
//! ```text
//! attach(gpu.try_device::<WgpuHeadlessRenderer>())  // the renderer's own device, as a producer sees
//!     .adapter().gpu_device()                       // it: a second dma-buf-capable device on that GPU
//! SharedBuffer::new(..).clear(..)                   // a shareable image, written on the GPU
//! gpu.paint_surface(buffer.handle())                // handed to the scene
//! // ...the window composites each, and the pixel is read back and compared to the clear colour
//! ```
//!
//! # Why the buffers are built through `attach`, not `GpuDevice::open`
//!
//! A dma-buf is importable on the GPU it was exported from, and a DRM modifier is meaningful only to
//! the vendor that defines it — the tiled one is picked from the *device's* own
//! `VkDrmFormatModifierProperties`. So the producer's device has to be the consumer's, or the two ends
//! disagree: `GpuDevice::open` picks its own GPU and a test built on it silently exports an Intel
//! `Y_TILED` buffer for an NVIDIA renderer to import. [`attach`] and
//! [`Adapter::gpu_device`](gpui_interop::Adapter::gpu_device) name the renderer's own
//! `vk::PhysicalDevice`, so the modifier is chosen against the GPU that will sample it — which is also
//! exactly the adapter step the crate exists for.
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

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use gpui::{
    AnyWindowHandle, AppContext as _, Context, GpuCanvasContext, IntoElement, Render,
    SurfaceFormatKind, Window, div, gpu_canvas, prelude::*, px, size,
};
use gpui_interop::{GpuDevice, SharedBuffer, Tiling, attach};
use gpui_wgpu::{CosmicTextSystem, WgpuHeadlessRenderer};

/// The edge of each produced surface, in pixels; the window holds two of them side by side.
const EDGE: u32 = 64;

/// The colour both buffers are cleared to, as the RGBA `clear` takes. `Rgba8`, so the bytes stored are
/// these four in order; a round trip that scrambles a channel comes back a different colour.
const CLEAR: [f32; 4] = [128.0 / 255.0, 128.0 / 255.0, 0.0, 1.0];

/// [`CLEAR`] in bytes — what a read-back must return.
const EXPECTED: [u8; 4] = [128, 128, 0, 255];

/// Which of the producer's two buffers a tile paints.
#[derive(Clone, Copy)]
enum Which {
    Linear,
    Tiled,
}

/// The producer's half: a dma-buf-capable device on the renderer's GPU and the two buffers it cleared.
///
/// Declaration order is load-bearing: a `SharedBuffer` frees its memory through the device it was made
/// on, so the device is declared *last* and Rust drops it after them.
struct Producer {
    linear: SharedBuffer,
    tiled: SharedBuffer,
    _device: GpuDevice,
}

impl Producer {
    /// Build on the device the canvas lends — the renderer's own GPU — or `None` when it lends none
    /// or cannot export a dma-buf there.
    fn build(gpu: &mut GpuCanvasContext) -> Option<Self> {
        let device = attach(gpu.try_device::<WgpuHeadlessRenderer>())
            .ok()?
            .adapter()
            .gpu_device()?;
        let linear = cleared(&device, Tiling::Linear)?;
        let tiled = cleared(&device, Tiling::Tiled)?;
        Some(Self {
            linear,
            tiled,
            _device: device,
        })
    }
}

/// A shareable buffer in `tiling`, cleared to [`CLEAR`]. `None` when the device cannot produce one.
fn cleared(device: &GpuDevice, tiling: Tiling) -> Option<SharedBuffer> {
    let buffer = SharedBuffer::new(device, EDGE, EDGE, SurfaceFormatKind::rgba8(), tiling).ok()?;
    buffer.clear(device, CLEAR).ok()?;
    Some(buffer)
}

/// The window's view: two tiles, one per tiling. The producer is built at paint time — the renderer's
/// device exists only then — so the slot is shared and filled on the first paint.
struct View {
    producer: Rc<RefCell<Option<Producer>>>,
    /// Whether the bridge built at all, so the test can tell a skip from a wrong pixel.
    built: Rc<Cell<bool>>,
}

impl Render for View {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let (left, right) = (self.producer.clone(), self.producer.clone());
        let (left_built, right_built) = (self.built.clone(), self.built.clone());
        div()
            .flex()
            .flex_row()
            .size_full()
            .child(tile(left, left_built, Which::Linear))
            .child(tile(right, right_built, Which::Tiled))
    }
}

/// One tile: build the producer on the first paint, then composite the buffer `which` names.
fn tile(slot: Rc<RefCell<Option<Producer>>>, built: Rc<Cell<bool>>, which: Which) -> impl IntoElement {
    gpu_canvas(move |gpu| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            match Producer::build(gpu) {
                Some(producer) => {
                    *slot = Some(producer);
                    built.set(true);
                }
                None => return,
            }
        }
        let handle = match which {
            Which::Linear => slot.as_ref().expect("just built").linear.handle(),
            Which::Tiled => slot.as_ref().expect("just built").tiled.handle(),
        };
        gpu.paint_surface(handle);
    })
    .w(px(EDGE as f32))
    .h(px(EDGE as f32))
}

/// A linear and a driver-tiled buffer, each exported as a dma-buf on the renderer's own GPU,
/// composited into one headless window, and read back byte for byte.
#[test]
fn an_exported_buffer_composites_byte_for_byte() {
    if gpui::current_headless_renderer().is_none() {
        eprintln!("skipping: no headless renderer (no Vulkan adapter)");
        return;
    }

    let text_system = Arc::new(CosmicTextSystem::new("fallback"));
    let mut cx = gpui::HeadlessAppContext::with_platform(text_system, Arc::new(()), || {
        Ok(gpui::current_headless_renderer())
    });

    let built = Rc::new(Cell::new(false));
    let window = cx
        .open_window(size(px((EDGE * 2) as f32), px(EDGE as f32)), |_window, cx| {
            cx.new(|_| View {
                producer: Rc::new(RefCell::new(None)),
                built: built.clone(),
            })
        })
        .expect("a window to composite into");
    let window: AnyWindowHandle = window.into();

    cx.run_until_parked();
    cx.update_window(window, |_, window, cx| {
        let _ = window.draw(cx);
    })
    .expect("draw a frame");

    if !built.get() {
        eprintln!("skipping: the renderer's device lent none that exports a dma-buf");
        return;
    }

    let image = cx.capture_screenshot(window).expect("read the frame back");
    // The left tile is the linear buffer, the right the tiled one; both were cleared to CLEAR.
    for (label, x) in [("linear", EDGE / 2), ("tiled", EDGE + EDGE / 2)] {
        assert_eq!(
            image.get_pixel(x, EDGE / 2).0,
            EXPECTED,
            "the {label} buffer did not round trip"
        );
    }
}
