# Changelog

All notable changes to `gpui_interop` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor
version may break the API; a patch does not.

## [Unreleased]

### Changed

- Development and CI track `bite-gpui` **1.24.401** (was `1.24.400`). That release attaches
  `WindowsPlatform`'s Direct3D devices when `open_window` is called on a platform constructed for
  windows, so a direct caller no longer has to call `request_windowing` first
  (bite-gpui/bite-gpui#13).

### Removed

- The Windows cross-device test's `request_windowing` workaround and the `futures` dev-dependency
  it needed. The test opens its hidden window directly again, which exercises the core's lazy
  attach.

## [0.1.3] - 2026-10-22

A `bite-gpui` upgrade. No API changes.

### Changed

- The crate now builds against **`bite-gpui` 1.24.400** (was `1.23.305`). No
  source changes were needed: the offscreen contract, `Window::present`, the
  headless renderer factory, the dma-buf explicit-layout guard and the transfer
  function all keep the shapes the crate already targets.
- The 1.24 line carries the upstream platform/engine split and a leaner
  dependency set, so the lockfile now resolves fewer crates (the Bevy example is
  untouched).
- CI and the release workflow clone `bite_1.24.400`, and the publish staging
  script aliases the published packages at `1.24`.

## [0.1.2] - 2026-10-22

A producer-side ring that makes slow frames safe. No breaking API changes.

### Added

- `WgpuSharedBufferRing` (Linux, `wgpu` feature): a ring of exportable frames
  for a continuously rendering producer. `begin()`/`submit()` render into a
  free slot; `present()` returns only a frame whose acquire fence has already
  signalled, or repeats the last good frame while a new one is in flight, so
  GPUI's 16 ms poll never drops a slow frame. Slots are recycled after the
  consumer's release, with a timeout fallback. Frames overtaken by a newer one
  are dropped, so presentation never jumps backwards.
- `shared_buffer_ring` integration test: a deliberately long frame is never
  presented with a pending fence.

### Fixed

- The recommended dma-buf pattern no longer requires hand-rolling the ring,
  release descriptors and fence bookkeeping. A slow frame is repeated instead of
  being handed to GPUI with a pending fence, which is the cause of intermittent
  black viewport frames. GPUI's acquire-fence wait is CPU-side and unchanged;
  the producer now never makes it wait.

## [0.1.1] - 2026-10-21

A documentation and CI patch. No API changes.

### Added

- **The dma-buf lifetime contract**, in the repository README, the `wgpu_dmabuf`
  module docs and the crate-level docs:
  - `WgpuSharedBuffer::submit` starts the write; it does not wait for it. Keep a
    ring and present one frame behind the one being rendered, or attach
    `DmaBufHandle::release` and wait for it before writing a slot again.
  - A fixed "reuse after N submissions" rule is a heuristic; a deep consumer
    queue or a coalesced frame can outlast it.
  - GPUI polls a surface's acquire fence for 16 ms and drops the surface on
    timeout, which presents as a black frame. A frame that runs long is safer
    presented one frame late.
  - Same device does not mean same submission: gate on the release, not on
    device identity.
  - Measured cost of a full-screen overlay: ~2.4 ms at 1280x720 with 4x MSAA
    (~5-6 ms at 1080p) on an Intel HD 520.
- Issue #1 records the reproduction, the measurements and the release-semantics
  answers.

### Fixed

- CI and the release workflow check the `bite-gpui` core out beside the
  workspace, so the path dependencies resolve on a clean runner.
- The Windows cross-device test is gated on the `gpui` feature; a build without
  it no longer tries to compile that test.

### Changed

- Version bumped to 0.1.1.

## [0.1.0] - 2026-10-20

First public release.

- Zero-copy external surfaces in a GPUI window: `dma-buf` + `sync_file` on
  Linux, shared Direct3D 12 textures + fence on Windows, `IOSurface` planned.
- `SharedBuffer`/`Fence` (Linux) and `SharedSurface`/`Fence` (Windows) for a
  producer that owns its device.
- `WgpuSurface` (same-device `paint_texture`) and, on Linux, `WgpuSharedBuffer`
  (exported dma-buf `paint_surface`) for a wgpu producer.
- The `surface_input` input gateway and the `Guest` headless worker-thread
  runner.
- Examples: `interop_linux`, `interop_cube`, `interop_cube_drag`, plus the
  `interop_wgpu` and `interop_bevy` producer demos.

[Unreleased]: https://github.com/bite-gpui/gpui_interop/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/bite-gpui/gpui_interop/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/bite-gpui/gpui_interop/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/bite-gpui/gpui_interop/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bite-gpui/gpui_interop/releases/tag/v0.1.0
