//! The Linux module: a dma-buf producer's surface, ordered by a dma-fence.
//!
//! **Not written yet — and the next to write.** Its gate, **P3**, is cleared
//! (`bite-gpui-project/decisions/linux-dmabuf-probe.md`): a foreign producer allocates a dma-buf,
//! exports the fd (fourcc + modifier + plane offsets/strides) and a `sync_file`, and hands both to
//! `surface()` as a `gpui::DmaBufHandle`. What is left is the producer-side allocation on the
//! producer's own device, and the ownership the ring needs — the shapes **P6** and the pool decide.
