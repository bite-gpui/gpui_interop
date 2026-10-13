//! The Linux module: a dma-buf producer's surface, ordered by a dma-fence.
//!
//! This is the **producer half** of the Linux surface arm, the mirror of `windows.rs`: allocate a
//! shareable image on the producer's own device, export it as a **dma-buf** (one fd per plane, with
//! the DRM modifier and the plane offsets/strides) and hand GPUI the `gpui::DmaBufHandle` the
//! `surface()` element takes. The consumer half is core — `gpui_wgpu` imports under the modifier and
//! samples it — so nothing here opens or waits on a buffer.
//!
//! Today this is the **single-plane** slice: an uncompressed `BGRA8`/`RGBA8` image, either linear or
//! on a driver-tiled modifier the renderer can import (`[`Tiling`]`), exported with `vkGetMemoryFdKHR`.
//! Multi-plane YCbCr formats are the next slice, and the acquire fence a `sync_file` from
//! `VK_KHR_external_fence_fd` the one after. The payload all three converge on is the same
//! [`gpui::DmaBufHandle`], so the consumer never changes.
//!
//! A VA-API producer does **not** use this module: `vaExportSurfaceHandle` hands it a dma-buf
//! directly. This is for a producer that has no driver-level export — a `wgpu`/3D engine or another
//! process — and must allocate and export its own.

use std::ffi::{CStr, CString};
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::raw::c_char;

use anyhow::{Result, anyhow};
use ash::vk;
use gpui::{DmaBufHandle, DmaBufPlane, SurfaceFormatKind};

/// The device extensions an image needs to be exported as a dma-buf, tiled or not.
const DEVICE_EXTENSIONS: [&str; 3] = [
    "VK_KHR_external_memory_fd",
    "VK_EXT_external_memory_dma_buf",
    "VK_EXT_image_drm_format_modifier",
];

/// How the producer's image is laid out in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tiling {
    /// A linear buffer: uncompressed, always importable, and the slowest to sample.
    Linear,
    /// A driver-tiled modifier the renderer can import, when the device offers one; a linear buffer
    /// otherwise.
    Tiled,
}

/// A Vulkan device a producer renders on and exports from.
///
/// The hatch opens its own instance and device on a node that can export a dma-buf. Where a renderer's
/// device is already known, the `adapter` module will lend it instead — matching on Linux is
/// device-node selection, not a LUID — and this constructor stays the fallback.
pub struct GpuDevice {
    // Held so the instance outlives the device it made, and the device the memory it exported.
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    external_memory_fd: ash::khr::external_memory_fd::Device,
    drm_format_modifier: ash::ext::image_drm_format_modifier::Device,
}

impl GpuDevice {
    /// Open a device that can export a dma-buf, or an error if none on this machine can.
    pub fn open() -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| anyhow!("load libvulkan: {e}"))?;
        let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
        }
        .map_err(|e| anyhow!("create the Vulkan instance: {e:?}"))?;

        let (physical_device, queue_family_index) = pick_device(&instance)
            .ok_or_else(|| anyhow!("no Vulkan device exports a dma-buf (needs {DEVICE_EXTENSIONS:?})"))?;

        let extensions: Vec<CString> = DEVICE_EXTENSIONS
            .iter()
            .map(|e| CString::new(*e).expect("a literal without a NUL"))
            .collect();
        let extension_ptrs: Vec<*const c_char> = extensions.iter().map(|e| e.as_ptr()).collect();

        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);
        let device = unsafe {
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(std::slice::from_ref(&queue_info))
                    .enabled_extension_names(&extension_ptrs),
                None,
            )
        }
        .map_err(|e| anyhow!("create the Vulkan device: {e:?}"))?;

        let external_memory_fd = ash::khr::external_memory_fd::Device::new(&instance, &device);
        let drm_format_modifier =
            ash::ext::image_drm_format_modifier::Device::new(&instance, &device);

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            external_memory_fd,
            drm_format_modifier,
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

    /// A modifier of `format` the renderer can sample and render to — the tiled choice, or `None`
    /// where the device offers none.
    fn tiled_modifier(&self, format: vk::Format) -> Option<u64> {
        supported_modifiers(self, format)
            .into_iter()
            .filter(|p| p.drm_format_modifier != DmaBufHandle::LINEAR)
            .find(|p| {
                p.drm_format_modifier_tiling_features.contains(
                    vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE,
                )
            })
            .map(|p| p.drm_format_modifier)
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        // SAFETY: the device and instance are this value's, and are destroyed once, in order.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A committed, shareable image — the producer's render target, exported as a dma-buf.
///
/// The memory behind it is what the exported fd names, so this value must outlive every use of the
/// [`DmaBufHandle`] it handed out: dropping it frees the memory the renderer is still sampling. A
/// producer keeps it for as long as the surface is live — a pool keeps a ring of them.
pub struct SharedBuffer {
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    handle: DmaBufHandle,
}

impl SharedBuffer {
    /// Allocate a shareable image of `width` × `height` in `format`, on `device`, with `tiling`.
    ///
    /// `format` must be a single-plane RGB format ([`SurfaceFormatKind::bgra8`] or
    /// [`SurfaceFormatKind::rgba8`]) today; a multi-plane format is the next slice. [`Tiling::Tiled`]
    /// asks for a driver modifier the renderer can import and falls back to linear where the device
    /// offers none, so `handle().modifier` is what was in fact chosen.
    pub fn new(
        device: &GpuDevice,
        width: u32,
        height: u32,
        format: SurfaceFormatKind,
        tiling: Tiling,
    ) -> Result<Self> {
        if format.plane_count() != 1 {
            return Err(anyhow!(
                "{format:?} is not a single-plane format; the hatch exports BGRA8/RGBA8 today"
            ));
        }
        let vk_format = vk_format(format)?;
        let modifier = match tiling {
            Tiling::Linear => None,
            Tiling::Tiled => device.tiled_modifier(vk_format),
        };

        let image = create_image(device, vk_format, width, height, modifier)?;

        let requirements = unsafe { device.device().get_image_memory_requirements(image) };
        let memory_type_index = pick_memory_type(device, requirements.memory_type_bits)?;

        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export);

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

        // SAFETY: `image` is live and the subresource names its only plane. A DRM-modifier image is
        // asked about by *memory plane*, a linear one by colour aspect; anything else returns 0.
        let aspect_mask = if drm {
            vk::ImageAspectFlags::MEMORY_PLANE_0_EXT
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let layout = unsafe {
            device.device().get_image_subresource_layout(
                image,
                vk::ImageSubresource {
                    aspect_mask,
                    mip_level: 0,
                    array_layer: 0,
                },
            )
        };

        let plane_fd = fd
            .try_clone()
            .map_err(|e| anyhow!("duplicate the dma-buf fd for the plane: {e}"))?;
        let handle = DmaBufHandle::new(
            width,
            height,
            format,
            modifier,
            [DmaBufPlane::new(plane_fd, layout.offset, layout.row_pitch as u32)],
            None,
        );

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

/// Create the shareable image, on `modifier` when one was chosen and linear otherwise.
fn create_image(
    device: &GpuDevice,
    format: vk::Format,
    width: u32,
    height: u32,
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
        .usage(
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )
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
fn supported_modifiers(device: &GpuDevice, format: vk::Format) -> Vec<vk::DrmFormatModifierPropertiesEXT> {
    // SAFETY: both calls read; the list is filled into a buffer sized from the first call's count.
    unsafe {
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut properties =
            vk::FormatProperties2::default().push_next(&mut list);
        device
            .instance()
            .get_physical_device_format_properties2(device.physical_device(), format, &mut properties);

        let count = list.drm_format_modifier_count as usize;
        if count == 0 {
            return Vec::new();
        }
        let mut modifiers = vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut modifiers);
        let mut properties = vk::FormatProperties2::default().push_next(&mut list);
        device
            .instance()
            .get_physical_device_format_properties2(device.physical_device(), format, &mut properties);
        modifiers
    }
}

/// The first memory type the image allows that is device-local.
fn pick_memory_type(device: &GpuDevice, bits: u32) -> Result<u32> {
    // SAFETY: the call reads the physical device's memory properties.
    let properties = unsafe {
        device
            .instance()
            .get_physical_device_memory_properties(device.physical_device())
    };
    (0..properties.memory_type_count)
        .find(|i| {
            bits & (1u32 << i) != 0
                && properties.memory_types[*i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .ok_or_else(|| anyhow!("no device-local memory type for the image"))
}

/// The Vulkan format a single-plane [`SurfaceFormatKind`] maps to.
fn vk_format(format: SurfaceFormatKind) -> Result<vk::Format> {
    match format {
        SurfaceFormatKind::Bgra8(_) => Ok(vk::Format::B8G8R8A8_UNORM),
        SurfaceFormatKind::Rgba8(_) => Ok(vk::Format::R8G8B8A8_UNORM),
        other => Err(anyhow!("{other:?} has no Vulkan format in the dma-buf hatch yet")),
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
    fn a_tiled_buffer_exports_a_dma_buf_under_its_modifier() {
        let Ok(device) = GpuDevice::open() else {
            eprintln!("skipping: no Vulkan device exports a dma-buf");
            return;
        };
        let buffer = SharedBuffer::new(&device, 64, 64, SurfaceFormatKind::bgra8(), Tiling::Tiled)
            .expect("allocate a shareable buffer");
        let handle = buffer.handle();
        // A device that offers no importable modifier falls back to linear, so this asserts the
        // contract — the stride and planes are real whatever the modifier — and reports which.
        assert_eq!(handle.planes.len(), 1);
        assert!(
            handle.planes[0].stride >= 64 * 4,
            "a tiled row is at least four bytes a pixel"
        );
        if handle.modifier != DmaBufHandle::LINEAR {
            eprintln!("tiled with modifier {:#x}", handle.modifier);
        } else {
            eprintln!("no importable modifier offered; fell back to linear");
        }
    }
}
