//! The macOS module: an `IOSurface` producer's surface, adopted into wgpu.
//!
//! **Not written yet.** Gated on **P2** (`IOSurface` adoption — building an `MTLTexture` over a
//! surface with `objc2-metal` and adopting it into wgpu, ordered by an `MTLSharedEvent`). The
//! matching half is already known to be trivial here: wgpu's adapter *is* the `MetalRenderer`'s
//! `MTLDevice`, so there is nothing to match
//! (`bite-gpui-project/spi/rendering/producer-reach.md` §7).
