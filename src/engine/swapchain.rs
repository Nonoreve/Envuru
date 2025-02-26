use crate::engine::Engine;
use ash::khr::{surface, swapchain};
use ash::{vk, Instance};
use winit::dpi::PhysicalSize;

pub struct Swapchain {
    swapchain_loader: swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,
    pub(crate) surface_resolution: vk::Extent2D
}

impl Swapchain {
    pub fn new(
        surface_capabilities: &vk::SurfaceCapabilitiesKHR,
        surface_loader: &surface::Instance,
        pdevice: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        instance: &Instance,
        device: &ash::Device,
        surface_format: &vk::SurfaceFormatKHR,
        dimensions: &PhysicalSize<u32>,
    ) -> Self {
        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => vk::Extent2D {
                width: dimensions.width,
                height: dimensions.height,
            },
            _ => surface_capabilities.current_extent,
        };
        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };
        unsafe {
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(pdevice, surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let swapchain_loader = swapchain::Device::new(instance, device);
            let create_info = vk::SwapchainCreateInfoKHR {
                flags: vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT,
                surface,
                min_image_count: desired_image_count,
                image_format: surface_format.format,
                image_color_space: surface_format.color_space,
                image_extent: surface_resolution,
                image_array_layers: 1,
                image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform,
                composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
                present_mode,
                clipped: vk::TRUE,
                ..Default::default()
            };
            let swapchain = swapchain_loader
                .create_swapchain(&create_info, None)
                .unwrap();
            let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let present_image_views = present_images
                .iter()
                .map(|&image| {
                    let create_info = vk::ImageViewCreateInfo {
                        flags: Default::default(), // TODO https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_fragment_density_map.html
                        image,
                        view_type: vk::ImageViewType::TYPE_2D,
                        format: surface_format.format,
                        components: crate::engine::DIRECT_MAPPING,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        ..Default::default()
                    };
                    device.create_image_view(&create_info, None).unwrap()
                })
                .collect();
            Self {
                swapchain_loader,
                swapchain,
                present_images,
                present_image_views,
                surface_resolution
            }
        }
    }

    pub fn next_image(&self, engine: &Engine, current_frame: usize) -> u32 {
        unsafe {
            let (present_index, _) = self
                .swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    engine.present_complete_semaphores[current_frame],
                    vk::Fence::null(),
                )
                .unwrap();
            present_index
        }
    }

    pub fn present(&self, engine: &Engine, current_frame: usize, present_index: u32) {
        let wait_semaphors = [engine.rendering_complete_semaphores[current_frame]];
        let swapchains = [self.swapchain];
        let image_indices = [present_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphors)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        unsafe {
            self.swapchain_loader
                .queue_present(engine.present_queue, &present_info)
                .unwrap();
        }
    }

    pub fn delete(&mut self, device: &ash::Device) {
        unsafe {
            for &image_view in self.present_image_views.iter() {
                device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}
