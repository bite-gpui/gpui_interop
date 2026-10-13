//! The Linux module: a dma-buf producer's surface, ordered by a dma-fence.
//!
//! This is the **producer half** of the Linux surface arm, the mirror of `windows.rs`: allocate a
//! shareable image on the producer's own device, export it as a **dma-buf** (one fd per plane, with
//! the DRM modifier and the plane offsets/strides) and hand GPUI the `gpui::DmaBufHandle` the
//! `surface()` element takes. The consumer half is core — `gpui_wgpu` imports under the modifier and
//! samples it — so nothing here opens or waits on a buffer.
//!
//! Today this is the **linear, single-plane** slice: an uncompressed `BGRA8`/`RGBA8` image created
//! with `VK_EXT_external_memory_dma_buf` and exported with `vkGetMemoryFdKHR`. Tiled modifiers (a
//! `VkImageDrmFormatModifierExplicitCreateInfoEXT`) and multi-plane YCbCr formats are the next slice,
//! and the acquire fence a `sync_file` from `VK_KHR_external_fence_fd` the one after. The payload all
//! three converge on is the same [`gpui::DmaBufHandle`], so the consumer never changes.
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

/// The device extensions an image needs to be exported as a dma-buf.
const DEVICE_EXTENSIONS: [&str; 2] = [
    "VK_KHR_external_memory_fd",
    "VK_EXT_external_memory_dma_buf",
];

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

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            external_memory_fd,
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
    /// Allocate a shareable image of `width` × `height` in `format`, on `device`.
    ///
    /// Today `format` must be a single-plane RGB format ([`SurfaceFormatKind::bgra8`] or
    /// [`SurfaceFormatKind::rgba8`]); a multi-plane format is the next slice. The image is linear, so
    /// its modifier is [`DmaBufHandle::LINEAR`].
    pub fn new(
        device: &GpuDevice,
        width: u32,
        height: u32,
        format: SurfaceFormatKind,
    ) -> Result<Self> {
        if format.plane_count() != 1 {
            return Err(anyhow!(
                "{format:?} is not a single-plane format; the hatch exports BGRA8/RGBA8 today"
            ));
        }
        let vk_format = vk_format(format)?;

        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external);

        // SAFETY: `image_info` chains a local `external`, both live for the call, and the image is
        // owned by the returned value until it is dropped.
        let image = unsafe { device.device().create_image(&image_info, None) }
            .map_err(|e| anyhow!("create the image: {e:?}"))?;

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

        // SAFETY: `image` is live and the subresource names its only plane.
        let layout = unsafe {
            device.device().get_image_subresource_layout(
                image,
                vk::ImageSubresource {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
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
            DmaBufHandle::LINEAR,
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
    fn a_shareable_buffer_exports_a_dma_buf() {
        let Ok(device) = GpuDevice::open() else {
            eprintln!("skipping: no Vulkan device exports a dma-buf");
            return;
        };
        let buffer = SharedBuffer::new(&device, 64, 64, SurfaceFormatKind::bgra8())
            .expect("allocate a shareable buffer");
        let handle = buffer.handle();
        assert_eq!(handle.modifier, DmaBufHandle::LINEAR, "the linear slice");
        assert_eq!(handle.planes.len(), 1, "a single-plane format");
        assert!(
            handle.planes[0].stride >= 64 * 4,
            "a BGRA8 row is at least four bytes a pixel"
        );
        assert_eq!(handle.width, 64);
        assert_eq!(handle.height, 64);
    }
}
