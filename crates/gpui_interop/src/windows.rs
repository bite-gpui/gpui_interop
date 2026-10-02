//! The Windows module: a Direct3D 12 / `wgpu` producer's surface on GPUI's Direct3D 11 renderer.
//!
//! **Not written yet.** Gated on **P5** (the `ID3D12Fence` ↔ `ID3D11Fence` loop) and **P9** (device
//! loss and re-negotiation), and on **P6** for the adapter match. The shape is a shared NT handle:
//! allocate shareable on the producer's device, `CreateSharedHandle`, `OpenSharedResource1` on the
//! window's, then a shader resource view — the Windows row of the surface trinity
//! (`bite-gpui-project/spi/rendering/surfaces.md` §2).
