//! The Linux module: a dma-buf producer's surface, ordered by a dma-fence.
//!
//! **Not yet implemented.** This backend will accept a dma-buf from a foreign producer, which allocates
//! it on its own device, exports the fd (fourcc + modifier + plane offsets/strides) and a
//! `sync_file`, and hands both to `surface()` as a `gpui::DmaBufHandle`. What remains to build is the
//! producer-side allocation and the ownership the surface pool needs.
