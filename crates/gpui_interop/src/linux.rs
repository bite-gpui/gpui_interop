//! The Linux module: a dma-buf producer's surface, ordered by a dma-fence.
//!
//! This is the **producer half** of the Linux surface arm, the mirror of `windows.rs`: allocate a
//! shareable image on the producer's own device, export it as a **dma-buf** (one fd per plane, with
//! the DRM modifier and the plane offsets/strides) and hand GPUI the `gpui::DmaBufHandle` the
//! `surface()` element takes. The consumer half is core — `gpui_wgpu` imports under the modifier and
//! samples it — so nothing here opens or waits on a buffer.
//!
//! Today this covers `BGRA8`/`RGBA8` and two-plane `NV12`, either linear or on a driver-tiled
//! modifier the renderer can import ([`Tiling`]), exported with `vkGetMemoryFdKHR` as one dma-buf
//! with a plane per memory plane, and ordered by a [`Fence`] that exports the acquire `sync_file`
//! from `VK_KHR_external_fence_fd`. The payload is the same [`gpui_engine::DmaBufHandle`] either way,
//! so the consumer never changes.
//!
//! A VA-API producer does **not** use this module: `vaExportSurfaceHandle` hands it a dma-buf
//! directly. This is for a producer that has no driver-level export — a `wgpu`/3D engine or another
//! process — and must allocate and export its own.

use std::ffi::{CStr, CString};
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Result, anyhow};
use ash::vk;
use gpui_engine::{DmaBufHandle, DmaBufPlane, SurfaceFormatKind};

/// The device extensions an image needs to be exported as a dma-buf, tiled or not.
const DEVICE_EXTENSIONS: [&str; 4] = [
    "VK_KHR_external_memory_fd",
    "VK_EXT_external_memory_dma_buf",
    "VK_EXT_image_drm_format_modifier",
    "VK_KHR_external_fence_fd",
];

/// The instance extension that names a physical device's DRM nodes. Enabled on the instance
/// [`GpuDevice::open`] makes, and read through when the renderer's instance carries it too.
const DRM_EXTENSION: &str = "VK_EXT_physical_device_drm";

/// How the producer's image is laid out in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tiling {
    /// A linear buffer: uncompressed, always importable, and the slowest to sample.
    Linear,
    /// A driver-tiled modifier the renderer can import — when the device offers one **and** it is the
    /// consumer's own.
    ///
    /// A DRM modifier is meaningful only to the vendor that defines it: `Y_TILED` is Intel's, NVIDIA
    /// lists its own block, llvmpipe only `LINEAR`. The modifier is picked from the *device's* list,
    /// so tiling is sound only when that device is the one the renderer samples on — which is what
    /// `Adapter::gpu_device` lends, and what [`GpuDevice::open`] does not. On a device the producer
    /// opened itself the request is **refused and the buffer exported linear**, with a warning: the
    /// alternative is a buffer the consumer wedges on rather than refuses.
    Tiled,
}

/// The GPU a window draws on, as a producer that must render or decode on the **same** GPU names it.
///
/// A dma-buf is importable only on the GPU it was exported from, so a producer that owns its device —
/// a 3D engine, a VA decoder, a camera — has to land on the renderer's GPU. It cannot borrow the
/// renderer's device to do so (a foreign API or process has its own), so what it needs is the GPU's
/// *identity*, in terms it can match from its own instance:
///
/// - [`render_node`](Self::render_node), the DRM render node — what a non-Vulkan producer opens
///   (`vaGetDisplayDRM`, a camera, `gbm`);
/// - [`name`](Self::name), the device name Vulkan reports — what a Vulkan producer matches by.
///
/// Read it inside a `gpu_canvas` with `Adapter::gpu_id`, or, for a producer that runs off the paint,
/// after a canvas has remembered it: [`gpu_id`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuId {
    /// The device name Vulkan reports, e.g. `Intel(R) HD Graphics 520`.
    pub name: String,
    /// The DRM render node the GPU backs — `/dev/dri/renderD128` — when the driver reports one, which
    /// is what a non-Vulkan producer opens. `None` when `VK_EXT_physical_device_drm` is unavailable.
    pub render_node: Option<PathBuf>,
}

impl GpuId {
    /// Read the identity of `physical_device` on `instance`.
    pub(crate) fn of(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        // SAFETY: both calls only read the device; the structs are filled out and copied.
        unsafe {
            let properties = instance.get_physical_device_properties(physical_device);
            let name = extension_name(&properties.device_name).to_owned();

            let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
            instance.get_physical_device_properties2(physical_device, &mut properties);
            let render_node = (drm.has_render != 0)
                .then(|| PathBuf::from(format!("/dev/dri/renderD{}", drm.render_minor)));

            Self { name, render_node }
        }
    }
}

/// The renderer's GPU, remembered when a canvas last read it — for a producer that runs off the
/// paint, which is every decoder and capture loop.
static RENDERER_GPU: OnceLock<GpuId> = OnceLock::new();

/// Remember `id` as the GPU the window draws on, for [`gpu_id`] to hand a producer off the paint.
///
/// First write wins: a process is assumed to draw through one GPU, which is what the renderer's own
/// adapter steering assumes too.
pub fn remember(id: GpuId) {
    let _ = RENDERER_GPU.set(id);
}

/// The remembered GPU, or `None` until a `gpu_canvas` has read it with `Adapter::gpu_id`.
pub fn gpu_id() -> Option<GpuId> {
    RENDERER_GPU.get().cloned()
}

/// A Vulkan device a producer renders on and exports from.
///
/// [`GpuDevice::open`] opens its own instance and device on a node that can export a dma-buf. Where a
/// window's renderer is known, `Adapter::gpu_device` lends its `vk::PhysicalDevice` instead and
/// [`GpuDevice::on`] makes a second logical device on it — the Linux counterpart of a LUID match — so
/// the export lands on the GPU the window draws through; `open` stays the fallback.
pub struct GpuDevice {
    // Held so an instance this value made outlives the device; `None` when the instance is the
    // renderer's, lent through the `adapter`.
    _entry: Option<ash::Entry>,
    instance: ash::Instance,
    /// Whether the instance is this value's to destroy, or the renderer's.
    owns_instance: bool,
    /// Whether this device is the renderer's own (`GpuDevice::on`) rather than one the producer
    /// opened (`GpuDevice::open`). [`Tiling::Tiled`] is honoured only on the former.
    renderer_bound: bool,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    external_memory_fd: ash::khr::external_memory_fd::Device,
    drm_format_modifier: ash::ext::image_drm_format_modifier::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
}

impl GpuDevice {
    /// Open a device that can export a dma-buf, or an error if none on this machine can.
    ///
    /// It is **not** the renderer's device, and it does not pretend to be: [`Tiling::Tiled`] is
    /// refused on it and the buffer is exported linear, with a warning. Borrow the renderer's device
    /// with `Adapter::gpu_device` when there is one to borrow from.
    pub fn open() -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| anyhow!("load libvulkan: {e}"))?;
        let instance = create_instance(&entry)?;

        let (physical_device, queue_family_index) = pick_device(&instance).ok_or_else(|| {
            anyhow!("no Vulkan device exports a dma-buf (needs {DEVICE_EXTENSIONS:?})")
        })?;

        Self::assemble(
            Some(entry),
            instance,
            true,
            false,
            physical_device,
            queue_family_index,
        )
    }

    /// A device on `physical_device` of the renderer's `instance` — the `adapter`'s answer on Linux,
    /// which puts the export on the GPU the window draws on rather than whichever one `open` finds.
    pub fn on(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Result<Self> {
        ensure_extensions(instance, physical_device)?;
        let queue_family_index = render_family(instance, physical_device)
            .ok_or_else(|| anyhow!("the renderer's device has no graphics queue family"))?;
        Self::assemble(
            None,
            instance.clone(),
            false,
            true,
            physical_device,
            queue_family_index,
        )
    }

    /// Build on an already-chosen physical device and queue family.
    fn assemble(
        entry: Option<ash::Entry>,
        instance: ash::Instance,
        owns_instance: bool,
        renderer_bound: bool,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> Result<Self> {
        let device = create_device(&instance, physical_device, queue_family_index)?;

        let external_memory_fd = ash::khr::external_memory_fd::Device::new(&instance, &device);
        let drm_format_modifier =
            ash::ext::image_drm_format_modifier::Device::new(&instance, &device);

        // SAFETY: the device is live; the queue and pool belong to it and are destroyed with it.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|e| anyhow!("create the command pool: {e:?}"))?;

        Ok(Self {
            _entry: entry,
            instance,
            owns_instance,
            renderer_bound,
            physical_device,
            device,
            external_memory_fd,
            drm_format_modifier,
            queue,
            queue_family_index,
            command_pool,
        })
    }

    fn device(&self) -> &ash::Device {
        &self.device
    }

    fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    fn external_memory_fd(&self) -> &ash::khr::external_memory_fd::Device {
        &self.external_memory_fd
    }

    fn drm_format_modifier(&self) -> &ash::ext::image_drm_format_modifier::Device {
        &self.drm_format_modifier
    }

    fn queue(&self) -> vk::Queue {
        self.queue
    }

    fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    #[allow(dead_code, reason = "the fence path does not read it yet")]
    fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// A modifier of `format` the renderer can sample and render to — the tiled choice, or `None`
    /// where the device offers none.
    fn tiled_modifier(&self, format: vk::Format, required: vk::FormatFeatureFlags) -> Option<u64> {
        supported_modifiers(self, format)
            .into_iter()
            .filter(|p| p.drm_format_modifier != DmaBufHandle::LINEAR)
            .find(|p| p.drm_format_modifier_tiling_features.contains(required))
            .map(|p| p.drm_format_modifier)
    }

    /// The GPU this device is on — the identity a producer that must match the renderer's GPU reads.
    pub fn gpu_id(&self) -> GpuId {
        GpuId::of(&self.instance, self.physical_device)
    }

    /// The first memory type the image allows that carries every flag in `required`.
    fn memory_type(&self, bits: u32, required: vk::MemoryPropertyFlags) -> Result<u32> {
        // SAFETY: the call reads the physical device's memory properties.
        let properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        (0..properties.memory_type_count)
            .find(|i| {
                bits & (1u32 << i) != 0
                    && properties.memory_types[*i as usize]
                        .property_flags
                        .contains(required)
            })
            .ok_or_else(|| anyhow!("no memory type carrying {required:?} for the image"))
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        // SAFETY: the pool and device were made by this value; the instance only if it is ours.
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            if self.owns_instance {
                self.instance.destroy_instance(None);
            }
        }
    }
}

/// A committed, shareable image — the producer's render target, exported as a dma-buf.
///
/// The memory behind it is what the exported fd names, so this value must outlive every use of the
/// [`DmaBufHandle`] it handed out: dropping it frees the memory the renderer is still sampling. A
/// producer keeps it for as long as the surface is live — a pool keeps a ring of them.
///
/// It also frees through the [`GpuDevice`] it was made on, so that device must outlive it — and,
/// because Rust drops a struct's fields in declaration order, a type holding both must declare the
/// buffers *before* the device.
pub struct SharedBuffer {
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    handle: DmaBufHandle,
}

impl SharedBuffer {
    /// Allocate a shareable image of `width` × `height` in `format`, on `device`, with `tiling`.
    ///
    /// `format` may be a single-plane RGB format ([`SurfaceFormatKind::bgra8`] or
    /// [`SurfaceFormatKind::rgba8`]) or two-plane [`SurfaceFormatKind::nv12`]; a format with no
    /// Vulkan mapping, or one the device cannot make in the chosen tiling, is refused before
    /// `vkCreateImage` would answer a bare `VK_ERROR_FORMAT_NOT_SUPPORTED`.
    /// [`Tiling::Tiled`] asks for a driver modifier the renderer can import; it is honoured only on
    /// the renderer's **own** device and falls back to linear — with a warning — where the device
    /// offers no such modifier or is one the producer opened itself. `handle().modifier` is what was
    /// in fact chosen.
    pub fn new(
        device: &GpuDevice,
        width: u32,
        height: u32,
        format: SurfaceFormatKind,
        tiling: Tiling,
    ) -> Result<Self> {
        let vk_format = vk_format(format)?;
        let plane_count = format.plane_count();
        let usage = usage_for();
        let required = required_features(usage);
        let modifier = match tiling {
            Tiling::Linear => None,
            Tiling::Tiled if !device.renderer_bound => {
                // A modifier is a claim about the memory layout of a GPU the consumer shares, and only
                // the renderer's own device can make it truthfully — a different vendor's modifier is
                // not importable at all. Refusing here exports a buffer the consumer can sample, where
                // honouring it could wedge the device instead.
                log::warn!(
                    "gpui_interop: Tiling::Tiled needs the renderer's own device; this one was opened \
                     by the producer, so the modifier it would pick is not known to be importable. \
                     Exporting a linear buffer — borrow the renderer's device with \
                     `Adapter::gpu_device` to tile."
                );
                None
            }
            Tiling::Tiled => {
                let modifier = device.tiled_modifier(vk_format, required);
                if modifier.is_none() {
                    log::warn!(
                        "gpui_interop: this device offers no {format:?} modifier the renderer can \
                         import; exporting a linear buffer"
                    );
                }
                modifier
            }
        };

        ensure_image_supported(device, vk_format, width, height, usage, required, modifier)?;

        let image = create_image(device, vk_format, width, height, usage, modifier)?;

        let requirements = unsafe { device.device().get_image_memory_requirements(image) };
        let memory_type_index = device.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        // A dedicated allocation, as the two working dma-buf paths here make — `gpui_wgpu`'s importer
        // and `gpui_va`'s tiled fill. An image that is imported or exported wants one, and a driver is
        // allowed to require it.
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export)
            .push_next(&mut dedicated);

        // SAFETY: the allocation chains a local `export`, the size and type come from the image, and
        // the memory is freed when the returned value drops.
        let memory = unsafe { device.device().allocate_memory(&allocate_info, None) }
            .map_err(|e| anyhow!("allocate the image memory: {e:?}"))?;
        // SAFETY: `memory` was allocated for `image` (same requirements) and is bound once.
        if let Err(e) = unsafe { device.device().bind_image_memory(image, memory, 0) } {
            unsafe { device.device().free_memory(memory, None) };
            unsafe { device.device().destroy_image(image, None) };
            return Err(anyhow!("bind the image memory: {e:?}"));
        }

        // The modifier the driver actually chose, which is what the renderer imports under.
        let drm = modifier.is_some();
        let modifier = match modifier {
            Some(_) => {
                let mut properties = vk::ImageDrmFormatModifierPropertiesEXT::default();
                if let Err(e) = unsafe {
                    device
                        .drm_format_modifier()
                        .get_image_drm_format_modifier_properties(image, &mut properties)
                } {
                    unsafe { device.device().free_memory(memory, None) };
                    unsafe { device.device().destroy_image(image, None) };
                    return Err(anyhow!("read the image's modifier: {e:?}"));
                }
                properties.drm_format_modifier
            }
            None => DmaBufHandle::LINEAR,
        };

        // SAFETY: the fd is freshly exported for this memory, and takes ownership below.
        let fd = unsafe {
            device.external_memory_fd().get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
            )
        }
        .map_err(|e| anyhow!("export the dma-buf fd: {e:?}"))?;
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };

        let mut planes = Vec::with_capacity(plane_count);
        for plane in 0..plane_count {
            // SAFETY: `image` is live and the subresource names `plane`.
            let layout = unsafe {
                device.device().get_image_subresource_layout(
                    image,
                    vk::ImageSubresource {
                        aspect_mask: subresource_aspect(drm, plane, plane_count),
                        mip_level: 0,
                        array_layer: 0,
                    },
                )
            };
            let plane_fd = fd
                .try_clone()
                .map_err(|e| anyhow!("duplicate the dma-buf fd for plane {plane}: {e}"))?;
            planes.push(DmaBufPlane::new(
                plane_fd,
                layout.offset,
                layout.row_pitch as u32,
            ));
        }
        let handle = DmaBufHandle::new(width, height, format, modifier, planes, None);

        Ok(Self {
            device: device.device().clone(),
            image,
            memory,
            handle,
        })
    }

    /// The description the `surface()` element takes. Cheap to clone — the planes share their fds.
    pub fn handle(&self) -> DmaBufHandle {
        self.handle.clone()
    }

    /// The producer's own render target.
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// Clear the image to a solid `rgba`, ready to hand over — the producer's simplest render, and
    /// enough for a demo or a fixture. Single-plane RGB formats only: a multi-plane YCbCr buffer has
    /// no one clear colour, so [`fill`](Self::fill) is the way to write one.
    ///
    /// It waits for the clear on the CPU rather than handing back an acquire fence, so the buffer is
    /// ready when this returns. A producer with real work signals a [`Fence`] from its own submission
    /// instead.
    pub fn clear(&self, device: &GpuDevice, rgba: [f32; 4]) -> Result<()> {
        let image = self.image;
        submit_once(device, |buffer| {
            // SAFETY: `buffer` is recording, and the image belongs to `device` and this value.
            unsafe {
                let to_transfer = image_barrier(
                    image,
                    vk::ImageAspectFlags::COLOR,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ,
                );
                device.device().cmd_pipeline_barrier(
                    buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_transfer],
                );
                device.device().cmd_clear_color_image(
                    buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { float32: rgba },
                    &[vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }],
                );
                let to_general = image_barrier(
                    image,
                    vk::ImageAspectFlags::COLOR,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::GENERAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ,
                );
                device.device().cmd_pipeline_barrier(
                    buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
            }
        })
    }

    /// Fill the image from host `planes`, ready to hand over — the producer's render for a format
    /// [`clear`](Self::clear) cannot write, because it has no one clear colour.
    ///
    /// `planes` holds one slice per plane of the format, in plane order and tightly packed at each
    /// plane's own resolution: for [`SurfaceFormatKind::nv12`], the luma plane (`width` × `height`
    /// bytes) then the interleaved chroma (`width/2` × `height/2` × 2 bytes). A plane of the wrong
    /// length, or a count the format does not have, is refused. Like `clear` it waits on the CPU
    /// rather than handing back an acquire fence, so the buffer is ready when this returns.
    pub fn fill(&self, device: &GpuDevice, planes: &[&[u8]]) -> Result<()> {
        let geometry = plane_geometry(self.handle.format, self.handle.width, self.handle.height);
        if geometry.is_empty() {
            return Err(anyhow!(
                "{:?} has no plane geometry to fill",
                self.handle.format
            ));
        }
        if geometry.len() != planes.len() {
            return Err(anyhow!(
                "{:?} has {} planes, but {} were given",
                self.handle.format,
                geometry.len(),
                planes.len(),
            ));
        }
        let mut offsets = Vec::with_capacity(geometry.len());
        let mut total = 0u64;
        for (plane, (bytes, _, _)) in geometry.iter().enumerate() {
            if planes[plane].len() != *bytes {
                return Err(anyhow!(
                    "plane {plane} is {} bytes; {:?} at {}×{} wants {bytes}",
                    planes[plane].len(),
                    self.handle.format,
                    self.handle.width,
                    self.handle.height,
                ));
            }
            offsets.push(total);
            total += *bytes as u64;
        }

        let plane_count = geometry.len();
        // A DRM modifier image names its planes by memory plane in a *barrier* but by plane in a
        // *copy*; the two helpers carry that split.
        let drm = self.handle.modifier != DmaBufHandle::LINEAR;
        let regions: Vec<vk::BufferImageCopy> = geometry
            .iter()
            .enumerate()
            .map(|(plane, &(_, width, height))| {
                vk::BufferImageCopy::default()
                    .buffer_offset(offsets[plane])
                    .buffer_row_length(width)
                    .buffer_image_height(height)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: plane_aspect(plane, plane_count),
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
            })
            .collect();

        let staging = Staging::new(device, total)?;
        staging.write(planes)?;
        let image = self.image;
        submit_once(device, |buffer| {
            // SAFETY: `buffer` is recording; the image and the staging buffer are live, and the
            // copy's offsets and extents were checked against the image above.
            unsafe {
                let to_transfer: Vec<_> = (0..plane_count)
                    .map(|plane| {
                        image_barrier(
                            image,
                            subresource_aspect(drm, plane, plane_count),
                            vk::ImageLayout::UNDEFINED,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            vk::AccessFlags::empty(),
                            vk::AccessFlags::TRANSFER_WRITE,
                        )
                    })
                    .collect();
                device.device().cmd_pipeline_barrier(
                    buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &to_transfer,
                );
                device.device().cmd_copy_buffer_to_image(
                    buffer,
                    staging.buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                );
                let to_general: Vec<_> = (0..plane_count)
                    .map(|plane| {
                        image_barrier(
                            image,
                            subresource_aspect(drm, plane, plane_count),
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            vk::ImageLayout::GENERAL,
                            vk::AccessFlags::TRANSFER_WRITE,
                            vk::AccessFlags::SHADER_READ,
                        )
                    })
                    .collect();
                device.device().cmd_pipeline_barrier(
                    buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &to_general,
                );
            }
        })
    }
}

impl Drop for SharedBuffer {
    fn drop(&mut self) {
        // SAFETY: both were created by this value and are destroyed once, in the order Vulkan needs.
        unsafe {
            self.device.free_memory(self.memory, None);
            self.device.destroy_image(self.image, None);
        }
    }
}

/// A fence signalled when the producer's work on a buffer completes — the
/// [`DmaBufHandle::acquire_fence`](gpui_engine::DmaBufHandle) a recycling renderer waits on.
///
/// A producer that renders into a [`SharedBuffer`] signals this from its own submission and hands
/// over [`sync_file`](Self::sync_file); a producer with no queue of its own (a decoder handing over a
/// filled buffer) uses [`signal`](Self::signal), which submits an empty list.
pub struct Fence {
    device: ash::Device,
    external_fence_fd: ash::khr::external_fence_fd::Device,
    fence: vk::Fence,
}

impl Fence {
    /// Create an unsignalled fence.
    pub fn new(device: &GpuDevice) -> Result<Self> {
        let fence = unsafe {
            device
                .device()
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|e| anyhow!("create the fence: {e:?}"))?;
        Ok(Self {
            device: device.device().clone(),
            external_fence_fd: ash::khr::external_fence_fd::Device::new(
                device.instance(),
                device.device(),
            ),
            fence,
        })
    }

    /// Signal the fence from `device`'s queue with an empty command list — the "this buffer is ready"
    /// a producer that has no pass of its own hands over. A producer that renders signals the fence
    /// from the submission that did the rendering instead.
    pub fn signal(&self, device: &GpuDevice) -> Result<()> {
        // SAFETY: the pool, queue and fence belong to `device`; the buffer is freed on every path.
        unsafe {
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(device.command_pool())
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let buffers = device
                .device()
                .allocate_command_buffers(&allocate)
                .map_err(|e| anyhow!("allocate a command buffer: {e:?}"))?;
            let buffer = buffers[0];
            let result = (|| -> Result<()> {
                device
                    .device()
                    .begin_command_buffer(buffer, &vk::CommandBufferBeginInfo::default())
                    .map_err(|e| anyhow!("begin the command buffer: {e:?}"))?;
                device
                    .device()
                    .end_command_buffer(buffer)
                    .map_err(|e| anyhow!("end the command buffer: {e:?}"))?;
                let buffers = [buffer];
                let submit = vk::SubmitInfo::default().command_buffers(&buffers);
                device
                    .device()
                    .queue_submit(device.queue(), &[submit], self.fence)
                    .map_err(|e| anyhow!("submit: {e:?}"))?;
                Ok(())
            })();
            device
                .device()
                .free_command_buffers(device.command_pool(), &buffers);
            result
        }
    }

    /// The `sync_file` the renderer waits on, for `DmaBufHandle::acquire_fence`.
    ///
    /// Export it after the submission that signals the fence: the returned fd is the one the renderer
    /// waits on, and the fence may not be exported this way twice.
    pub fn sync_file(&self) -> Result<OwnedFd> {
        // SAFETY: `fence` is this value's and unsignalled; the fd it returns is owned here.
        let fd = unsafe {
            self.external_fence_fd.get_fence_fd(
                &vk::FenceGetFdInfoKHR::default()
                    .fence(self.fence)
                    .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD),
            )
        }
        .map_err(|e| anyhow!("export the fence fd: {e:?}"))?;
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// The fence itself, for a producer that submits its own work against it.
    pub fn fence(&self) -> vk::Fence {
        self.fence
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        // SAFETY: created by this value and destroyed once.
        unsafe { self.device.destroy_fence(self.fence, None) };
    }
}

/// Create the shareable image, on `modifier` when one was chosen and linear otherwise.
fn create_image(
    device: &GpuDevice,
    format: vk::Format,
    width: u32,
    height: u32,
    usage: vk::ImageUsageFlags,
    modifier: Option<u64>,
) -> Result<vk::Image> {
    let mut external = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external);

    // SAFETY: `info` chains locals that outlive the call, and the image is owned by the caller.
    let image = match modifier {
        Some(modifier) => {
            let modifiers = [modifier];
            let mut drm = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                .drm_format_modifiers(&modifiers);
            let info = info
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .push_next(&mut drm);
            unsafe { device.device().create_image(&info, None) }
        }
        None => {
            let info = info.tiling(vk::ImageTiling::LINEAR);
            unsafe { device.device().create_image(&info, None) }
        }
    };
    image.map_err(|e| anyhow!("create the image: {e:?}"))
}

/// Record `record` into a fresh command buffer, submit it on `device`'s queue, and wait for the GPU
/// to finish — the one-shot submission a producer renders a buffer with, so it hands over a finished
/// buffer rather than an acquire fence.
fn submit_once(device: &GpuDevice, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
    // SAFETY: the command buffer is allocated from the device's pool, recorded once and freed on
    // every path; the fence is waited on and destroyed.
    unsafe {
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(device.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let buffers = device
            .device()
            .allocate_command_buffers(&allocate)
            .map_err(|error| anyhow!("allocate a command buffer: {error:?}"))?;
        let buffer = buffers[0];
        let result = (|| -> Result<()> {
            device
                .device()
                .begin_command_buffer(buffer, &vk::CommandBufferBeginInfo::default())
                .map_err(|error| anyhow!("begin the command buffer: {error:?}"))?;
            record(buffer);
            device
                .device()
                .end_command_buffer(buffer)
                .map_err(|error| anyhow!("end the command buffer: {error:?}"))?;

            let buffers = [buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&buffers);
            let fence = device
                .device()
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|error| anyhow!("create the submission's fence: {error:?}"))?;
            device
                .device()
                .queue_submit(device.queue(), &[submit], fence)
                .map_err(|error| anyhow!("submit: {error:?}"))?;
            device
                .device()
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|error| anyhow!("wait for the submission: {error:?}"))?;
            device.device().destroy_fence(fence, None);
            Ok(())
        })();
        device
            .device()
            .free_command_buffers(device.command_pool(), &buffers);
        result
    }
}

/// A host-visible staging buffer and the memory behind it, freed together when it drops — what a
/// [`SharedBuffer::fill`] uploads host bytes through.
struct Staging {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

impl Staging {
    /// Allocate and bind a `size`-byte buffer the fill uploads through.
    fn new(device: &GpuDevice, size: u64) -> Result<Self> {
        // SAFETY: the buffer is created and bound once; every error path frees what it made.
        unsafe {
            let buffer = device
                .device()
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .map_err(|error| anyhow!("create the staging buffer: {error:?}"))?;
            let requirements = device.device().get_buffer_memory_requirements(buffer);
            let memory = match device.memory_type(
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ) {
                Ok(memory_type_index) => device
                    .device()
                    .allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(requirements.size)
                            .memory_type_index(memory_type_index),
                        None,
                    )
                    .map_err(|error| anyhow!("allocate the staging memory: {error:?}"))?,
                Err(error) => {
                    device.device().destroy_buffer(buffer, None);
                    return Err(error);
                }
            };
            if let Err(error) = device.device().bind_buffer_memory(buffer, memory, 0) {
                device.device().free_memory(memory, None);
                device.device().destroy_buffer(buffer, None);
                return Err(anyhow!("bind the staging memory: {error:?}"));
            }
            Ok(Self {
                device: device.device().clone(),
                buffer,
                memory,
                size,
            })
        }
    }

    /// Copy `planes` into the buffer, back to back.
    fn write(&self, planes: &[&[u8]]) -> Result<()> {
        // SAFETY: the mapping covers the whole allocation and is released before returning.
        unsafe {
            let mapped = self
                .device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
                .map_err(|error| anyhow!("map the staging memory: {error:?}"))?
                as *mut u8;
            let mut offset = 0usize;
            for plane in planes {
                std::ptr::copy_nonoverlapping(plane.as_ptr(), mapped.add(offset), plane.len());
                offset += plane.len();
            }
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        // SAFETY: both were made by this value and are freed once.
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// A full-image layout transition of `aspect`, from `from` to `to`, ordering `src_access` before
/// `dst_access`. `aspect` is one plane of a multi-plane image, or the colour aspect of a single one —
/// not a memory plane, which only the layout query names.
fn image_barrier(
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .old_layout(from)
        .new_layout(to)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
}

/// Create the instance [`GpuDevice::open`] runs on, carrying `VK_EXT_physical_device_drm` when the
/// loader offers it, so [`GpuId::of`] can name the GPU's DRM render node.
fn create_instance(entry: &ash::Entry) -> Result<ash::Instance> {
    let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
    // SAFETY: the call reads the loader's extension list.
    let available = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map_err(|e| anyhow!("enumerate instance extensions: {e:?}"))?;
    let drm = CString::new(DRM_EXTENSION).expect("a literal without a NUL");
    let names: Vec<*const c_char> = available
        .iter()
        .filter(|e| extension_name(&e.extension_name) == DRM_EXTENSION)
        .map(|_| drm.as_ptr())
        .collect();

    // SAFETY: `names` points to locals that outlive the call.
    unsafe {
        entry
            .create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_extension_names(&names),
                None,
            )
            .map_err(|e| anyhow!("create the Vulkan instance: {e:?}"))
    }
}

/// Create a device on `physical_device`'s `queue_family_index`, with the extensions the export needs.
fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<ash::Device> {
    let extensions: Vec<CString> = DEVICE_EXTENSIONS
        .iter()
        .map(|e| CString::new(*e).expect("a literal without a NUL"))
        .collect();
    let extension_ptrs: Vec<*const c_char> = extensions.iter().map(|e| e.as_ptr()).collect();

    let priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);
    // SAFETY: the pointers are to locals that outlive the call.
    unsafe {
        instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&extension_ptrs),
            None,
        )
    }
    .map_err(|e| anyhow!("create the Vulkan device: {e:?}"))
}

/// Whether `physical_device` supports every extension the export needs.
fn ensure_extensions(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Result<()> {
    // SAFETY: the call reads the device's extension list.
    let properties = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .map_err(|e| anyhow!("enumerate device extensions: {e:?}"))?;
    for wanted in DEVICE_EXTENSIONS {
        if !properties
            .iter()
            .any(|p| extension_name(&p.extension_name) == wanted)
        {
            return Err(anyhow!("the renderer's device does not support {wanted}"));
        }
    }
    Ok(())
}

/// The first graphics queue family of `physical_device`.
fn render_family(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<u32> {
    // SAFETY: the call reads the device's families.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|index| index as u32)
}

/// A device that can export a dma-buf, with a queue family that can render — or `None`.
fn pick_device(instance: &ash::Instance) -> Option<(vk::PhysicalDevice, u32)> {
    // SAFETY: the instance is live; the call only reads.
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.ok()?;
    for physical_device in physical_devices {
        // SAFETY: both calls read; the properties are copied out.
        let properties =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }.ok()?;
        let exports = DEVICE_EXTENSIONS.iter().all(|wanted| {
            properties
                .iter()
                .any(|p| extension_name(&p.extension_name) == *wanted)
        });
        if !exports {
            continue;
        }
        // SAFETY: the call reads the device's families.
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let Some(index) = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        else {
            continue;
        };
        return Some((physical_device, index as u32));
    }
    None
}

/// Every DRM format modifier `format` supports, with the features each carries.
fn supported_modifiers(
    device: &GpuDevice,
    format: vk::Format,
) -> Vec<vk::DrmFormatModifierPropertiesEXT> {
    // SAFETY: both calls read; the list is filled into a buffer sized from the first call's count.
    unsafe {
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut properties = vk::FormatProperties2::default().push_next(&mut list);
        device.instance().get_physical_device_format_properties2(
            device.physical_device(),
            format,
            &mut properties,
        );

        let count = list.drm_format_modifier_count as usize;
        if count == 0 {
            return Vec::new();
        }
        let mut modifiers = vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut modifiers);
        let mut properties = vk::FormatProperties2::default().push_next(&mut list);
        device.instance().get_physical_device_format_properties2(
            device.physical_device(),
            format,
            &mut properties,
        );
        modifiers
    }
}

/// The properties the device reports for `format` — the feature bits each tiling carries.
fn format_properties(device: &GpuDevice, format: vk::Format) -> vk::FormatProperties {
    // SAFETY: the call only reads; the properties are copied out.
    unsafe {
        device
            .instance()
            .get_physical_device_format_properties(device.physical_device(), format)
    }
}

/// Refuse, before `vkCreateImage` would, a format the device cannot make in the chosen tiling — the
/// feature bits that tiling carries, and the extent the device caps it at.
///
/// `vkCreateImage` answers an unsupported request with a bare `VK_ERROR_FORMAT_NOT_SUPPORTED`, which
/// names neither the format nor what is missing. This names both, so a caller can pick another format
/// or tiling instead of guessing.
fn ensure_image_supported(
    device: &GpuDevice,
    format: vk::Format,
    width: u32,
    height: u32,
    usage: vk::ImageUsageFlags,
    required: vk::FormatFeatureFlags,
    modifier: Option<u64>,
) -> Result<()> {
    let tiling = match modifier {
        Some(_) => vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
        None => vk::ImageTiling::LINEAR,
    };
    let features = match modifier {
        None => format_properties(device, format).linear_tiling_features,
        Some(modifier) => supported_modifiers(device, format)
            .into_iter()
            .find(|properties| properties.drm_format_modifier == modifier)
            .map(|properties| properties.drm_format_modifier_tiling_features)
            .ok_or_else(|| {
                anyhow!("the device lists no DRM modifier {modifier:#x} for {format:?}")
            })?,
    };
    if !features.contains(required) {
        return Err(anyhow!(
            "the device cannot make a {tiling:?} {format:?} image: it needs {required:?}, and the \
             format carries {features:?}"
        ));
    }

    // SAFETY: the calls only read; `info` chains a local that outlives the call.
    let max_extent = unsafe {
        let mut drm = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(modifier.unwrap_or(0));
        let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(tiling)
            .usage(usage);
        if modifier.is_some() {
            info = info.push_next(&mut drm);
        }
        let mut properties = vk::ImageFormatProperties2::default();
        device
            .instance()
            .get_physical_device_image_format_properties2(
                device.physical_device(),
                &info,
                &mut properties,
            )
            .map_err(|error| {
                anyhow!("ask the device for its {tiling:?} {format:?} limits: {error:?}")
            })?;
        properties.image_format_properties.max_extent
    };
    if width > max_extent.width || height > max_extent.height {
        return Err(anyhow!(
            "the device's {format:?} images cap at {}×{}, asked for {width}×{height}",
            max_extent.width,
            max_extent.height,
        ));
    }
    Ok(())
}

/// The Vulkan format a [`SurfaceFormatKind`] maps to.
fn vk_format(format: SurfaceFormatKind) -> Result<vk::Format> {
    match format {
        SurfaceFormatKind::Bgra8(_) => Ok(vk::Format::B8G8R8A8_UNORM),
        SurfaceFormatKind::Rgba8(_) => Ok(vk::Format::R8G8B8A8_UNORM),
        SurfaceFormatKind::Nv12(_) => Ok(vk::Format::G8_B8R8_2PLANE_420_UNORM),
        other => Err(anyhow!(
            "{other:?} has no Vulkan format in the dma-buf hatch yet"
        )),
    }
}

/// The usage the hatch makes of an image: the producer writes it with transfers ([`clear`] and
/// [`fill`]) and the consumer samples it. No format is rendered to, so none asks for
/// `COLOR_ATTACHMENT` — which linear tiling does not support, and which the crate offers no pipeline
/// to use anyway.
///
/// [`clear`]: SharedBuffer::clear
/// [`fill`]: SharedBuffer::fill
fn usage_for() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
}

/// The format feature bits `usage` needs, so a capability check and the image agree on what it is
/// made for. Transfer features are guaranteed for every format, so only sampling and colour attachment
/// carry one.
fn required_features(usage: vk::ImageUsageFlags) -> vk::FormatFeatureFlags {
    let mut required = vk::FormatFeatureFlags::empty();
    if usage.contains(vk::ImageUsageFlags::SAMPLED) {
        required |= vk::FormatFeatureFlags::SAMPLED_IMAGE;
    }
    if usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        required |= vk::FormatFeatureFlags::COLOR_ATTACHMENT;
    }
    required
}

/// Each plane of `format` at `width` × `height`, as `(byte length, extent width, extent height)` in
/// plane order — the geometry [`SharedBuffer::fill`] uploads against.
fn plane_geometry(format: SurfaceFormatKind, width: u32, height: u32) -> Vec<(usize, u32, u32)> {
    match format {
        SurfaceFormatKind::Bgra8(_) | SurfaceFormatKind::Rgba8(_) => {
            vec![((width as usize * height as usize) * 4, width, height)]
        }
        SurfaceFormatKind::Nv12(_) => vec![
            (width as usize * height as usize, width, height),
            (
                (width as usize / 2) * (height as usize / 2) * 2,
                width / 2,
                height / 2,
            ),
        ],
        _ => Vec::new(),
    }
}

/// The aspect a `fill`'s **copy** names for `plane`: the colour aspect for a one-plane image, else
/// that plane's own. A barrier names a DRM image's plane by memory plane instead — see
/// [`subresource_aspect`].
fn plane_aspect(plane: usize, plane_count: usize) -> vk::ImageAspectFlags {
    if plane_count == 1 {
        vk::ImageAspectFlags::COLOR
    } else if plane == 0 {
        vk::ImageAspectFlags::PLANE_0
    } else {
        vk::ImageAspectFlags::PLANE_1
    }
}

/// The aspect a subresource of `plane` is named by: a DRM-modifier image is asked about by *memory
/// plane*, a linear one by colour aspect (or by plane, if the format has more than one).
fn subresource_aspect(drm: bool, plane: usize, plane_count: usize) -> vk::ImageAspectFlags {
    if drm {
        if plane == 0 {
            vk::ImageAspectFlags::MEMORY_PLANE_0_EXT
        } else {
            vk::ImageAspectFlags::MEMORY_PLANE_1_EXT
        }
    } else if plane_count == 1 {
        vk::ImageAspectFlags::COLOR
    } else if plane == 0 {
        vk::ImageAspectFlags::PLANE_0
    } else {
        vk::ImageAspectFlags::PLANE_1
    }
}

/// A `[c_char; N]` extension name as a `&str`, stopping at the NUL the loader wrote.
fn extension_name(raw: &[c_char]) -> &str {
    // SAFETY: the loader NUL-terminates the name in the fixed array.
    unsafe { CStr::from_ptr(raw.as_ptr()) }
        .to_str()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_linear_buffer_exports_a_dma_buf() {
        let Ok(device) = GpuDevice::open() else {
            eprintln!("skipping: no Vulkan device exports a dma-buf");
            return;
        };
        let buffer = SharedBuffer::new(&device, 64, 64, SurfaceFormatKind::bgra8(), Tiling::Linear)
            .expect("allocate a shareable buffer");
        let handle = buffer.handle();
        assert_eq!(handle.modifier, DmaBufHandle::LINEAR, "asked for linear");
        assert_eq!(handle.planes.len(), 1, "a single-plane format");
        assert!(
            handle.planes[0].stride >= 64 * 4,
            "a BGRA8 row is at least four bytes a pixel"
        );
        assert_eq!(handle.width, 64);
        assert_eq!(handle.height, 64);
    }

    #[test]
    fn a_gpu_reports_the_render_node_it_backs() {
        let Ok(device) = GpuDevice::open() else {
            eprintln!("skipping: no Vulkan device exports a dma-buf");
            return;
        };
        let id = device.gpu_id();
        eprintln!("gpu: {} render node {:?}", id.name, id.render_node);
        assert!(!id.name.is_empty(), "a Vulkan device reports a name");
        if let Some(node) = &id.render_node {
            assert!(
                node.to_string_lossy().contains("renderD"),
                "a DRM render node names a render node: {node:?}"
            );
        }
    }

    /// A producer-opened device refuses `Tiling::Tiled`: the modifier it would pick is not known to be
    /// importable by the consumer, so the buffer is exported linear rather than risk a buffer the
    /// renderer wedges on. Real tiled coverage needs the renderer's own device —
    /// `tests/linux_dmabuf_readback.rs`, behind the `readback-test` feature.
    #[test]
    fn a_producers_own_device_refuses_tiling() {
        let Ok(device) = GpuDevice::open() else {
            eprintln!("skipping: no Vulkan device exports a dma-buf");
            return;
        };
        let buffer = SharedBuffer::new(&device, 64, 64, SurfaceFormatKind::bgra8(), Tiling::Tiled)
            .expect("allocate a shareable buffer");
        let handle = buffer.handle();
        assert_eq!(
            handle.modifier,
            DmaBufHandle::LINEAR,
            "a device the producer opened itself must not hand out a modifier"
        );
        assert_eq!(handle.planes.len(), 1, "a single-plane format");
        assert!(
            handle.planes[0].stride >= 64 * 4,
            "a linear row is at least four bytes a pixel"
        );
    }

    #[test]
    fn a_two_plane_nv12_buffer_exports_two_planes() {
        let Ok(device) = GpuDevice::open() else {
            eprintln!("skipping: no Vulkan device exports a dma-buf");
            return;
        };
        let buffer = SharedBuffer::new(&device, 64, 64, SurfaceFormatKind::nv12(), Tiling::Linear)
            .expect("allocate a shareable buffer");
        let handle = buffer.handle();
        assert_eq!(handle.planes.len(), 2, "NV12 is two planes");
        assert!(
            handle.planes[0].stride >= 64,
            "a luma row is at least a byte a pixel"
        );
        assert!(
            handle.planes[1].stride >= 32,
            "a chroma row is at least a byte every two pixels"
        );
        assert!(
            handle.planes[1].offset >= handle.planes[0].stride as u64 * 64,
            "the chroma plane follows the luma plane"
        );
        eprintln!(
            "nv12: luma offset {} stride {}, chroma offset {} stride {}",
            handle.planes[0].offset,
            handle.planes[0].stride,
            handle.planes[1].offset,
            handle.planes[1].stride,
        );
    }

    #[test]
    fn a_fence_exports_a_sync_file() {
        let Ok(device) = GpuDevice::open() else {
            eprintln!("skipping: no Vulkan device exports a dma-buf");
            return;
        };
        let fence = Fence::new(&device).expect("create a fence");
        fence.signal(&device).expect("signal the fence");
        let sync_file = fence.sync_file().expect("export a sync_file");
        assert!(
            std::os::fd::AsRawFd::as_raw_fd(&sync_file) >= 0,
            "a sync_file is a valid descriptor"
        );
    }

    #[test]
    fn a_cleared_buffer_is_ready_to_hand_over() {
        let Ok(device) = GpuDevice::open() else {
            eprintln!("skipping: no Vulkan device exports a dma-buf");
            return;
        };
        let buffer = SharedBuffer::new(&device, 64, 64, SurfaceFormatKind::bgra8(), Tiling::Linear)
            .expect("allocate a shareable buffer");
        buffer
            .clear(&device, [0.5, 0.25, 0.125, 1.0])
            .expect("clear the buffer");
        // The handle is unchanged — clearing writes through the same dma-buf.
        assert_eq!(buffer.handle().planes.len(), 1);
    }

    #[test]
    fn a_filled_nv12_buffer_is_ready_to_hand_over() {
        let Ok(device) = GpuDevice::open() else {
            eprintln!("skipping: no Vulkan device exports a dma-buf");
            return;
        };
        let buffer = SharedBuffer::new(&device, 64, 64, SurfaceFormatKind::nv12(), Tiling::Linear)
            .expect("allocate a shareable buffer");
        // A full-resolution luma plane, then a half-resolution interleaved chroma plane.
        let luma = vec![128u8; 64 * 64];
        let mut chroma = vec![0u8; 32 * 32 * 2];
        for pair in chroma.as_chunks_mut::<2>().0 {
            pair[0] = 90;
            pair[1] = 120;
        }
        buffer
            .fill(&device, &[&luma, &chroma])
            .expect("fill the two planes");
        assert_eq!(buffer.handle().planes.len(), 2, "NV12 stays two planes");
    }

    #[test]
    fn a_fill_of_the_wrong_size_is_refused() {
        let Ok(device) = GpuDevice::open() else {
            eprintln!("skipping: no Vulkan device exports a dma-buf");
            return;
        };
        let buffer = SharedBuffer::new(&device, 64, 64, SurfaceFormatKind::nv12(), Tiling::Linear)
            .expect("allocate a shareable buffer");
        let luma = vec![0u8; 64 * 64];
        let short_chroma = vec![0u8; 16];
        let error = buffer
            .fill(&device, &[&luma, &short_chroma])
            .expect_err("a chroma plane of the wrong length is refused");
        assert!(
            error.to_string().contains("plane 1"),
            "the error should name the plane: {error}"
        );
    }
}
