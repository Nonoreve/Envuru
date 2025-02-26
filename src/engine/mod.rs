use std::{ffi, sync, borrow, process, mem};
use ash::ext::debug_utils;
use ash::khr::surface;
use ash::{khr, vk};
use winit::{application, dpi, event, event_loop, window};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use crate::engine::pipeline::Pipeline;
use crate::engine::swapchain::Swapchain;

mod swapchain;
pub(crate) mod pipeline;
mod shader;
mod memory;
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
	draw_frame: DrawFrameFn
}

impl application::ApplicationHandler for WindowHandler {
	fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
		println!("resumed");
		if self.engine.is_none() {
			let engine = Engine::new(self.preferred_width, self.preferred_height, self.name.clone(), event_loop);
			self.pipeline = Some(Pipeline::new(&engine));
			self.engine = Some(engine);
		}
	}

	fn window_event(&mut self, event_loop: &event_loop::ActiveEventLoop, _id: window::WindowId, event: event::WindowEvent) {
		match event {
			event::WindowEvent::CloseRequested => {
				println!("The close button was pressed; stopping");
				event_loop.exit();
			},
			event::WindowEvent::RedrawRequested => {
				println!("RedrawRequested");
				(self.draw_frame)(self.engine.as_ref().unwrap(), self.pipeline.as_mut().unwrap())
			}
			_ => (),
		}
	}
	fn about_to_wait(&mut self, _event_loop: &event_loop::ActiveEventLoop) {
		println!("abouttowait");
		(self.draw_frame)(self.engine.as_ref().unwrap(), self.pipeline.as_mut().unwrap())
	}

	fn exiting(&mut self, _event_loop: &event_loop::ActiveEventLoop) {
		self.pipeline.as_mut().unwrap().delete(self.engine.as_ref().unwrap());
	}
}

impl EngineBuilder {
	pub fn new(preferred_width: u32, preferred_height: u32, name: &str, draw_frame: DrawFrameFn) -> Self {
		let event_loop = event_loop::EventLoop::new().unwrap();
		event_loop.set_control_flow(event_loop::ControlFlow::Poll);
		let app_name = ffi::CString::new(name).unwrap();
		let mut window_handler = WindowHandler { engine: None, pipeline: None, preferred_width, preferred_height, name: app_name, draw_frame };
		Self { event_loop, window_handler, }
	}

	pub fn start(mut self) {
		self.event_loop.run_app(&mut self.window_handler).unwrap();
	}
}

static ENTRY: sync::LazyLock<ash::Entry, fn() -> ash::Entry> = sync::LazyLock::new(|| ash::Entry::linked());
pub(crate) const DIRECT_MAPPING: vk::ComponentMapping = vk::ComponentMapping {
	r: vk::ComponentSwizzle::R,
	g: vk::ComponentSwizzle::G,
	b: vk::ComponentSwizzle::B,
	a: vk::ComponentSwizzle::A,
};
pub(crate) const MAX_FRAMES_IN_FLIGHT: u32 = 2;

/// Helper function for submitting command buffers. Immediately waits for the fence before the command buffer
/// is executed. That way we can delay the waiting for the fences by 1 frame which is good for performance.
/// Make sure to create the fence in a signaled state on the first use.
#[allow(clippy::too_many_arguments)]
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
		device.wait_for_fences(&[command_buffer_reuse_fence], true, u64::MAX).expect("Wait for fence failed.");

		device.reset_fences(&[command_buffer_reuse_fence]).expect("Reset fences failed.");

		device.reset_command_buffer(
			command_buffer,
			vk::CommandBufferResetFlags::RELEASE_RESOURCES,
		).expect("Reset command buffer failed.");

		let command_buffer_begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

		device.begin_command_buffer(command_buffer, &command_buffer_begin_info).expect("Begin commandbuffer");
		f(device, command_buffer);
		device.end_command_buffer(command_buffer).expect("End commandbuffer");

		let command_buffers = vec![command_buffer];

		let submit_info = vk::SubmitInfo::default().wait_semaphores(wait_semaphores).wait_dst_stage_mask(wait_mask).command_buffers(&command_buffers).signal_semaphores(signal_semaphores);

		device.queue_submit(submit_queue, &[submit_info], command_buffer_reuse_fence).expect("queue submit failed.");
	}
}

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

pub fn find_memorytype_index(
	memory_req: &vk::MemoryRequirements,
	memory_prop: &vk::PhysicalDeviceMemoryProperties,
	flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
	memory_prop.memory_types[..memory_prop.memory_type_count as _].iter().enumerate().find(|(index, memory_type)| {
		(1 << index) as u32 & memory_req.memory_type_bits != 0 && memory_type.property_flags & flags == flags
	}).map(|(index, _memory_type)| index as _)
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

	pub depth_image: vk::Image,
	pub depth_image_view: vk::ImageView,
	pub depth_image_memory: vk::DeviceMemory,

	pub present_complete_semaphores: Vec<vk::Semaphore>,
	pub rendering_complete_semaphores: Vec<vk::Semaphore>,

	pub draw_commands_reuse_fence: Vec<vk::Fence>,
	pub setup_commands_reuse_fence: vk::Fence,
	surface_capabilities: vk::SurfaceCapabilitiesKHR,
	pdevice: vk::PhysicalDevice,
}


impl Engine {
	pub fn new(preferred_width: u32, preferred_height: u32, name: ffi::CString, event_loop: &event_loop::ActiveEventLoop) -> Self {
		let window = event_loop.create_window(window::Window::default_attributes().with_inner_size(dpi::LogicalSize::new(
			f64::from(preferred_width),
			f64::from(preferred_height),
		))).unwrap();
		let app_name = ffi::CString::new(name).unwrap();
		let appinfo = vk::ApplicationInfo::default().application_name(app_name.as_c_str()).application_version(0).engine_name(app_name.as_c_str()).engine_version(0).api_version(vk::make_api_version(0, 1, 3, 0));
		let validation_layer = ffi::CString::new("VK_LAYER_KHRONOS_validation").unwrap();
		let layer_names = [validation_layer.as_ptr()];
		let mut extension_names = ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw()).unwrap().to_vec();
		extension_names.push(debug_utils::NAME.as_ptr());
		let create_flags = vk::InstanceCreateFlags::default();
		let create_info = vk::InstanceCreateInfo::default().application_info(&appinfo).enabled_layer_names(&layer_names).enabled_extension_names(&extension_names).flags(create_flags);
		unsafe {
			let instance: ash::Instance = ENTRY.create_instance(&create_info, None).unwrap();
			let create_info = vk::DebugUtilsMessengerCreateInfoEXT {
				message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
				message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
				pfn_user_callback: Some(vulkan_debug_callback),
				..Default::default()
			};
			let debug_utils_loader = debug_utils::Instance::new(&ENTRY, &instance);
			let debug_call_back = debug_utils_loader.create_debug_utils_messenger(&create_info, None).unwrap();
			let surface = ash_window::create_surface(
				&ENTRY,
				&instance,
				window.display_handle().unwrap().as_raw(),
				window.window_handle().unwrap().as_raw(),
				None,
			).unwrap();
			let pdevices = instance.enumerate_physical_devices().unwrap();
			let surface_loader = surface::Instance::new(&ENTRY, &instance);
			let (pdevice, queue_family_index) = pdevices.iter().find_map(|pdevice| {
				instance.get_physical_device_queue_family_properties(*pdevice).iter().enumerate().find_map(|(index, info)| {
					let supports_graphic_and_surface = info.queue_flags.contains(vk::QueueFlags::GRAPHICS) && surface_loader.get_physical_device_surface_support(
						*pdevice,
						index as u32,
						surface,
					).unwrap();
					if supports_graphic_and_surface {
						Some((*pdevice, index))
					} else {
						None
					}
				})
			}).expect("Couldn't find suitable device.");
			let queue_family_index = queue_family_index as u32;
			let device_extension_names = [khr::swapchain::NAME.as_ptr()];
			let features = vk::PhysicalDeviceFeatures {
				shader_clip_distance: 1,
				..Default::default()
			};
			let priorities = [1.0];
			let create_info = vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family_index).queue_priorities(&priorities);
			let create_info = vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&create_info)).enabled_extension_names(&device_extension_names).enabled_features(&features);
			let device: ash::Device = instance.create_device(pdevice, &create_info, None).unwrap();
			let present_queue = device.get_device_queue(queue_family_index, 0);
			let surface_format = surface_loader.get_physical_device_surface_formats(pdevice, surface).unwrap()[0]; // TODO choose instead of first one ?
			let surface_capabilities = surface_loader.get_physical_device_surface_capabilities(pdevice, surface).unwrap();
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
			let depth_image_create_info = vk::ImageCreateInfo {
				flags: Default::default(), // TODO check for optis
				image_type: vk::ImageType::TYPE_2D,
				format: vk::Format::D16_UNORM,
				extent: swapchain.surface_resolution.into(),
				mip_levels: 1,
				array_layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
				tiling: vk::ImageTiling::OPTIMAL,
				usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
				sharing_mode: vk::SharingMode::EXCLUSIVE,
				..Default::default()
			};
			let depth_image = device.create_image(&depth_image_create_info, None).unwrap();
			let depth_image_memory_req = device.get_image_memory_requirements(depth_image);
			let depth_image_memory_index = find_memorytype_index(
				&depth_image_memory_req,
				&device_memory_properties,
				vk::MemoryPropertyFlags::DEVICE_LOCAL,
			).expect("Unable to find suitable memory index for depth image.");
			let allocate_info = vk::MemoryAllocateInfo {
				allocation_size: depth_image_memory_req.size,
				memory_type_index: depth_image_memory_index,
				..Default::default()
			};
			let depth_image_memory = device.allocate_memory(&allocate_info, None).unwrap();
			device.bind_image_memory(depth_image, depth_image_memory, 0).unwrap();
			let create_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
			let draw_commands_reuse_fence = (1..=MAX_FRAMES_IN_FLIGHT).map(|_| device.create_fence(&create_info, None).unwrap()).collect();
			let setup_commands_reuse_fence = device.create_fence(&create_info, None).unwrap();
			let subresource_range = vk::ImageSubresourceRange {
				aspect_mask: vk::ImageAspectFlags::DEPTH,
				base_mip_level: 0,
				level_count: 1,
				base_array_layer: 0,
				layer_count: 1,
			};

			record_submit_commandbuffer(
				&device,
				setup_command_buffer,
				setup_commands_reuse_fence,
				present_queue,
				&[],
				&[],
				&[],
				|device, setup_command_buffer| {
					let layout_transition_barriers = vk::ImageMemoryBarrier {
						src_access_mask: Default::default(),
						dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
						old_layout: vk::ImageLayout::UNDEFINED,
						new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
						src_queue_family_index: 0, // TODO queue_family_index ?
						dst_queue_family_index: 0,
						image: depth_image,
						subresource_range,
						..Default::default()
					};
					device.cmd_pipeline_barrier(
						setup_command_buffer,
						vk::PipelineStageFlags::BOTTOM_OF_PIPE,
						vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
						vk::DependencyFlags::empty(),
						&[],
						&[],
						&[layout_transition_barriers],
					);
				},
			);
			let create_info = vk::ImageViewCreateInfo {
				flags: Default::default(), // TODO see first ImageViewCreateInfo and apply to other ones if necessary
				image: depth_image,
				view_type: vk::ImageViewType::TYPE_2D,
				format: depth_image_create_info.format,
				components: DIRECT_MAPPING,
				subresource_range,
				..Default::default()
			};
			let depth_image_view = device.create_image_view(&create_info, None).unwrap();
			let semaphore_create_info = vk::SemaphoreCreateInfo::default();
			let present_complete_semaphore = (1..=MAX_FRAMES_IN_FLIGHT).map(|_| {
				device.create_semaphore(&semaphore_create_info, None).unwrap()
			}).collect();
			let rendering_complete_semaphore = (1..=MAX_FRAMES_IN_FLIGHT).map(|_| {
				device.create_semaphore(&semaphore_create_info, None).unwrap()
			}).collect();
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
				depth_image_view,
				present_complete_semaphores: present_complete_semaphore,
				rendering_complete_semaphores: rendering_complete_semaphore,
				draw_commands_reuse_fence,
				setup_commands_reuse_fence,
				surface,
				surface_capabilities,
				debug_call_back,
				debug_utils_loader,
				depth_image_memory,
			}
		}
	}

	pub fn on_window_resize(&mut self, pipeline: &Pipeline) {
		unsafe {
			self.device.device_wait_idle().unwrap();
			self.swapchain.delete(&self.device);
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
			// TODO recreate renderpass if image format was changed
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
			self.device.destroy_fence(self.setup_commands_reuse_fence, None);
			self.device.free_memory(self.depth_image_memory, None);
			self.device.destroy_image_view(self.depth_image_view, None);
			self.device.destroy_image(self.depth_image, None);
			self.swapchain.delete(&self.device);
			self.device.destroy_command_pool(self.pool, None);
			self.device.destroy_device(None);
			self.surface_loader.destroy_surface(self.surface, None);
			self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_call_back, None);
			self.instance.destroy_instance(None);
			let window = mem::ManuallyDrop::take(&mut self.window);
			drop(window);
		}
	}
}
