//! The guest runner: headless GPUI on a worker thread, handing a frame to a foreign loop.
//!
//! **Not yet implemented.** The renderer factory is `!Send` by construction, so a headless render on
//! a worker thread has to be arranged around that, and the offscreen path must hand back a
//! *shareable* surface rather than only bytes. It sits beside the bridge because a host loop needs
//! both halves — the frame out, and translated input in.
//!
//! P7 measured the threading half on Linux (`bite-gpui-project`'s `guest-thread-probe.md`): the
//! renderer stays pinned to its thread and the frame crosses as
//! `SceneRenderer::offscreen_surface_handle` (`WgpuOffscreenSurface`). The runner around it —
//! executors, cadence, input — is still to build.
