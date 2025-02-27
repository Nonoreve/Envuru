use crate::engine::memory::DepthImage;
use crate::engine::pipeline::Pipeline;
use crate::engine::swapchain::Swapchain;
use ash::ext::debug_utils;
use ash::khr::surface;
use ash::{khr, vk};
use std::{borrow, ffi, mem, process, sync};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{application, dpi, event, event_loop, window};

mod memory;
pub(crate) mod pipeline;
mod shader;
mod swapchain;
mod utils;

type DrawFrameFn = for<'a, 'b> fn(&'a Engine, &'b mut Pipeline);

pub(crate) struct EngineBuilder {
    event_loop: event_loop::EventLoop<()>,
    window_handler: WindowHandler,
}

struct WindowHandler {
    engine: Option<Engine>,
    pipeline: Option<Pipeline>,
    preferred_width: u32,
    preferred_height: u32,
    name: ffi::CString,
    draw_frame: DrawFrameFn,
}

impl application::ApplicationHandler for WindowHandler {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if self.engine.is_none() {
            let engine = Engine::new(
                self.preferred_width,
                self.preferred_height,
                self.name.clone(),
                event_loop,
            );
            self.pipeline = Some(Pipeline::new(&engine));
            self.engine = Some(engine);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        _id: window::WindowId,
        event: event::WindowEvent,
    ) {
        match event {
            event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            event::WindowEvent::RedrawRequested => (self.draw_frame)(
                self.engine.as_ref().unwrap(),
                self.pipeline.as_mut().unwrap(),
            ),
            event::WindowEvent::Resized(_new_size) => self
                .engine
                .as_mut()
                .unwrap()
                .on_window_resize(self.pipeline.as_mut().unwrap()),
            _ => (),
        }
    }

    // fn about_to_wait(&mut self, _event_loop: &event_loop::ActiveEventLoop) {
    // 	println!("abouttowait");
    // 	(self.draw_frame)(self.engine.as_ref().unwrap(), self.pipeline.as_mut().unwrap())
    // }

    fn exiting(&mut self, _event_loop: &event_loop::ActiveEventLoop) {
        self.pipeline
            .as_mut()
            .unwrap()
            .delete(self.engine.as_ref().unwrap());
    }
}

impl EngineBuilder {
    pub fn new(
        preferred_width: u32,
        preferred_height: u32,
        name: &str,
        draw_frame: DrawFrameFn,
    ) -> Self {
        let event_loop = event_loop::EventLoop::new().unwrap();
        event_loop.set_control_flow(event_loop::ControlFlow::Poll);
        let app_name = ffi::CString::new(name).unwrap();
        let mut window_handler = WindowHandler {
            engine: None,
            pipeline: None,
            preferred_width,
            preferred_height,
            name: app_name,
            draw_frame,
        };
        Self {
            event_loop,
            window_handler,
        }
    }

    pub fn start(mut self) {
        self.event_loop.run_app(&mut self.window_handler).unwrap();
    }
}

static ENTRY: sync::LazyLock<ash::Entry, fn() -> ash::Entry> =
    sync::LazyLock::new(|| ash::Entry::linked());
pub(crate) const DIRECT_MAPPING: vk::ComponentMapping = vk::ComponentMapping {
    r: vk::ComponentSwizzle::R,
    g: vk::ComponentSwizzle::G,
    b: vk::ComponentSwizzle::B,
    a: vk::ComponentSwizzle::A,
};
pub(crate) const MAX_FRAMES_IN_FLIGHT: u32 = 2;

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        borrow::Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        borrow::Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{message_severity:?}: {message_type:?} [{message_id_name} ({message_id_number})] : {message}",
    );

    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        process::exit(-1)
    }

    vk::FALSE
}

pub struct Engine {
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub surface_loader: surface::Instance,
    pub debug_utils_loader: debug_utils::Instance,
    pub window: mem::ManuallyDrop<window::Window>,
    pub debug_call_back: vk::DebugUtilsMessengerEXT,

    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub present_queue: vk::Queue,

    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,

    pub swapchain: Swapchain,

    pub pool: vk::CommandPool,
    pub draw_command_buffers: Vec<vk::CommandBuffer>,
    pub setup_command_buffer: vk::CommandBuffer,

    pub depth_image: DepthImage,

    pub present_complete_semaphores: Vec<vk::Semaphore>,
    pub rendering_complete_semaphores: Vec<vk::Semaphore>,

    pub draw_commands_reuse_fence: Vec<vk::Fence>,
    pub setup_commands_reuse_fence: vk::Fence,
    surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pdevice: vk::PhysicalDevice,
}

impl Engine {
    pub fn new(
        preferred_width: u32,
        preferred_height: u32,
        name: ffi::CString,
        event_loop: &event_loop::ActiveEventLoop,
    ) -> Self {
        let window = event_loop
            .create_window(window::Window::default_attributes().with_inner_size(
                dpi::LogicalSize::new(f64::from(preferred_width), f64::from(preferred_height)),
            ))
            .unwrap();
        let app_name = ffi::CString::new(name).unwrap();
        let appinfo = vk::ApplicationInfo::default()
            .application_name(app_name.as_c_str())
            .application_version(0)
            .engine_name(app_name.as_c_str())
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let validation_layer = ffi::CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let layer_names = [validation_layer.as_ptr()];
        let mut extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap()
                .to_vec();
        extension_names.push(debug_utils::NAME.as_ptr());
        let create_flags = vk::InstanceCreateFlags::default();
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&extension_names)
            .flags(create_flags);
        unsafe {
            let instance: ash::Instance = ENTRY.create_instance(&create_info, None).unwrap();
            let create_info = vk::DebugUtilsMessengerCreateInfoEXT {
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                pfn_user_callback: Some(vulkan_debug_callback),
                ..Default::default()
            };
            let debug_utils_loader = debug_utils::Instance::new(&ENTRY, &instance);
            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&create_info, None)
                .unwrap();
            let surface = ash_window::create_surface(
                &ENTRY,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .unwrap();
            let pdevices = instance.enumerate_physical_devices().unwrap();
            let surface_loader = surface::Instance::new(&ENTRY, &instance);
            let (pdevice, queue_family_index) = pdevices
                .iter()
                .find_map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            if supports_graphic_and_surface {
                                Some((*pdevice, index))
                            } else {
                                None
                            }
                        })
                })
                .expect("Couldn't find suitable device.");
            let queue_family_index = queue_family_index as u32;
            let device_extension_names = [khr::swapchain::NAME.as_ptr()];
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                ..Default::default()
            };
            let priorities = [1.0];
            let create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);
            let create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&create_info))
                .enabled_extension_names(&device_extension_names)
                .enabled_features(&features);
            let device: ash::Device = instance.create_device(pdevice, &create_info, None).unwrap();
            let present_queue = device.get_device_queue(queue_family_index, 0);
            let surface_format = surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()[0]; // TODO choose instead of first one ?
            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap();
            let swapchain = Swapchain::new(
                &surface_capabilities,
                &surface_loader,
                pdevice,
                surface,
                &instance,
                &device,
                &surface_format,
                &window.inner_size(),
            );
            let create_info = vk::CommandPoolCreateInfo {
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queue_family_index,
                ..Default::default()
            }; // TODO pool with transient flag for oneshot/out-of-drawloop commands ?
            let pool = device.create_command_pool(&create_info, None).unwrap();
            let allocate_info = vk::CommandBufferAllocateInfo {
                command_pool: pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: MAX_FRAMES_IN_FLIGHT + 1,
                ..Default::default()
            };
            let mut command_buffers = device.allocate_command_buffers(&allocate_info).unwrap();
            let setup_command_buffer = command_buffers.pop().unwrap();
            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);
            let create_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let draw_commands_reuse_fence = (1..=MAX_FRAMES_IN_FLIGHT)
                .map(|_| device.create_fence(&create_info, None).unwrap())
                .collect();
            let setup_commands_reuse_fence = device.create_fence(&create_info, None).unwrap();
            let depth_image = DepthImage::new(
                &swapchain,
                &device,
                &device_memory_properties,
                setup_command_buffer,
                setup_commands_reuse_fence,
                present_queue,
            );
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let present_complete_semaphore = (1..=MAX_FRAMES_IN_FLIGHT)
                .map(|_| {
                    device
                        .create_semaphore(&semaphore_create_info, None)
                        .unwrap()
                })
                .collect();
            let rendering_complete_semaphore = (1..=MAX_FRAMES_IN_FLIGHT)
                .map(|_| {
                    device
                        .create_semaphore(&semaphore_create_info, None)
                        .unwrap()
                })
                .collect();
            Self {
                instance,
                device,
                pdevice,
                device_memory_properties,
                window: mem::ManuallyDrop::new(window),
                surface_loader,
                surface_format,
                present_queue,
                swapchain,
                pool,
                draw_command_buffers: command_buffers,
                setup_command_buffer,
                depth_image,
                present_complete_semaphores: present_complete_semaphore,
                rendering_complete_semaphores: rendering_complete_semaphore,
                draw_commands_reuse_fence,
                setup_commands_reuse_fence,
                surface,
                surface_capabilities,
                debug_call_back,
                debug_utils_loader,
            }
        }
    }

    pub fn on_window_resize(&mut self, pipeline: &mut Pipeline) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.swapchain.delete(&self.device);
            pipeline.delete_framebuffers(self);
            self.surface_capabilities = self
                .surface_loader
                .get_physical_device_surface_capabilities(self.pdevice, self.surface)
                .unwrap();
            self.swapchain = Swapchain::new(
                &self.surface_capabilities,
                &self.surface_loader,
                self.pdevice,
                self.surface,
                &self.instance,
                &self.device,
                &self.surface_format,
                &self.window.inner_size(),
            );
            self.depth_image.delete(self);
            self.depth_image = DepthImage::new(
                &self.swapchain,
                &self.device,
                &self.device_memory_properties,
                self.setup_command_buffer,
                self.setup_commands_reuse_fence,
                self.present_queue,
            );
            pipeline.new_framebuffers(self);
            // TODO recreate renderpass if image format was changed
        }
    }

    pub fn find_memorytype_index(
        memory_req: &vk::MemoryRequirements,
        memory_prop: &vk::PhysicalDeviceMemoryProperties,
        flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        memory_prop.memory_types[..memory_prop.memory_type_count as _]
            .iter()
            .enumerate()
            .find(|(index, memory_type)| {
                (1 << index) as u32 & memory_req.memory_type_bits != 0
                    && memory_type.property_flags & flags == flags
            })
            .map(|(index, _memory_type)| index as _)
    }

    pub fn record_submit_commandbuffer<F: FnOnce(&ash::Device, vk::CommandBuffer)>(
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        command_buffer_reuse_fence: vk::Fence,
        submit_queue: vk::Queue,
        wait_mask: &[vk::PipelineStageFlags],
        wait_semaphores: &[vk::Semaphore],
        signal_semaphores: &[vk::Semaphore],
        f: F,
    ) {
        unsafe {
            device
                .wait_for_fences(&[command_buffer_reuse_fence], true, u64::MAX)
                .expect("Wait for fence failed.");

            device
                .reset_fences(&[command_buffer_reuse_fence])
                .expect("Reset fences failed.");

            device
                .reset_command_buffer(
                    command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Reset command buffer failed.");

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Begin commandbuffer");
            f(device, command_buffer);
            device
                .end_command_buffer(command_buffer)
                .expect("End commandbuffer");

            let command_buffers = vec![command_buffer];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(wait_mask)
                .command_buffers(&command_buffers)
                .signal_semaphores(signal_semaphores);

            device
                .queue_submit(submit_queue, &[submit_info], command_buffer_reuse_fence)
                .expect("queue submit failed.");
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            for &semaphore in self.present_complete_semaphores.iter() {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in self.rendering_complete_semaphores.iter() {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &fence in self.draw_commands_reuse_fence.iter() {
                self.device.destroy_fence(fence, None);
            }
            self.device
                .destroy_fence(self.setup_commands_reuse_fence, None);
            self.depth_image.delete(self);
            self.swapchain.delete(&self.device);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);
            self.instance.destroy_instance(None);
            let window = mem::ManuallyDrop::take(&mut self.window);
            drop(window);
        }
    }
}
