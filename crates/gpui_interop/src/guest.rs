//! The guest runner: headless GPUI on a worker thread, handing a frame to a foreign loop.
//!
//! **Not written yet.** Gated on **P7**: the renderer factory is `!Send` by construction, and the
//! offscreen path must hand back a *shareable* surface rather than only bytes
//! (`bite-gpui-project/spi/rendering/surfaces.md` §4). It belongs beside the bridge because a host
//! loop needs both halves — the frame out, and translated input in.
