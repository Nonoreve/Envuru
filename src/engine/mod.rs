#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;
use std::rc::Rc;
use std::{borrow, ffi, mem, process, sync};

use ash::ext::debug_utils;
use ash::khr::surface;
use ash::{khr, vk};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{application, dpi, event, event_loop, window};

use crate::engine::controller::Controller;
use crate::engine::memory::DepthImage;
use crate::engine::pipeline::Pipelines;
use crate::engine::scene::{MvpUbo, Scene, ShaderSet};
use crate::engine::swapchain::SurfaceData;
use crate::engine::swapchain::Swapchain;

pub mod api_resources;
pub mod controller;
mod memory;
pub mod pipeline;
pub mod scene;
pub(crate) mod shader;
mod swapchain;

type UpdateFn = for<'a, 'b, 'c, 'd> fn(
    &'b mut Scene,
    &'b mut Pipelines,
    &'c mut Controller,
    &'d cgmath::Vector2<f64>,
);

pub enum ShaderInterface {
    UniformBuffer,
    CombinedImageSampler,
}

pub enum MeshTopology {
    Triangles,
    Lines,
    Points,
}

pub struct EngineBuilder {
    event_loop: event_loop::EventLoop<()>,
    window_handler: WindowHandler,
}

struct WindowHandler {
    engine: Option<Engine>,
    pipeline: Option<Pipelines>,
    controller: Option<Controller>,
    preferred_width: u32,
    preferred_height: u32,
    name: String,
    update_function: UpdateFn,
    scenes: Vec<Scene>,
    starting_scene: usize,
}

impl application::ApplicationHandler for WindowHandler {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if self.engine.is_none() {
            let engine = Engine::new(
                self.preferred_width,
                self.preferred_height,
                &self.name,
                event_loop,
            );
            let scene = self.scenes.get_mut(self.starting_scene).unwrap();
            self.pipeline = Some(Pipelines::new(&engine, scene));
            self.engine = Some(engine);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        _id: window::WindowId,
        event: event::WindowEvent,
    ) {
        // TODO handle Options possibliy none in this context
        match event {
            event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            event::WindowEvent::RedrawRequested => {
                let window_position = self
                    .engine
                    .as_ref()
                    .unwrap()
                    .window
                    .inner_position()
                    .unwrap();
                (self.update_function)(
                    self.scenes.get_mut(self.starting_scene).unwrap(),
                    self.pipeline.as_mut().unwrap(),
                    self.controller.as_mut().unwrap(),
                    &cgmath::vec2(window_position.x as f64, window_position.y as f64),
                );
                self.engine.as_ref().unwrap().draw_frame(
                    self.pipeline.as_mut().unwrap(),
                    self.scenes.get(self.starting_scene).unwrap(),
                )
            }
            event::WindowEvent::Resized(_new_size) => {
                self.engine
                    .as_mut()
                    .unwrap()
                    .on_window_resize(self.pipeline.as_mut().unwrap());
                self.scenes
                    .get_mut(self.starting_scene)
                    .unwrap()
                    .camera
                    .projection
                    .aspect = self.pipeline.as_ref().unwrap().aspect_ratio
            }
            event::WindowEvent::ScaleFactorChanged { .. } => {}
            x => {
                let window_position = self
                    .engine
                    .as_ref()
                    .unwrap()
                    .window
                    .inner_position()
                    .unwrap();
                if self.controller.is_some() {
                    self.controller.as_mut().unwrap().handle_event(
                        x,
                        self.pipeline.as_ref().unwrap().frames,
                        &cgmath::vec2(window_position.x as f64, window_position.y as f64),
                    );
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &event_loop::ActiveEventLoop) {
        let window_position = self
            .engine
            .as_ref()
            .unwrap()
            .window
            .inner_position()
            .unwrap();
        (self.update_function)(
            self.scenes.get_mut(self.starting_scene).unwrap(),
            self.pipeline.as_mut().unwrap(),
            self.controller.as_mut().unwrap(),
            &cgmath::vec2(window_position.x as f64, window_position.y as f64),
        );
        self.engine.as_ref().unwrap().draw_frame(
            self.pipeline.as_mut().unwrap(),
            self.scenes.get(self.starting_scene).unwrap(),
        )
    }
}

impl WindowHandler {
    fn delete(&mut self) {
        for scene in self.scenes.iter_mut() {
            for object in scene.objects.iter_mut() {
                object.delete()
            }
            self.pipeline
                .as_mut()
                .unwrap()
                .delete(self.engine.as_ref().unwrap(), scene);
        }
    }
}

impl EngineBuilder {
    pub fn new(
        preferred_width: u32,
        preferred_height: u32,
        name: &str,
        update_function: UpdateFn,
        controller: Option<Controller>,
    ) -> Self {
        let event_loop = event_loop::EventLoop::new().unwrap();
        event_loop.set_control_flow(event_loop::ControlFlow::Poll);
        let window_handler = WindowHandler {
            engine: None,
            pipeline: None,
            controller,
            preferred_width,
            preferred_height,
            name: String::from(name),
            update_function,
            scenes: Vec::new(),
            starting_scene: 0,
        };
        Self {
            event_loop,
            window_handler,
        }
    }

    pub fn register_scene(&mut self, scene: Scene) -> usize {
        self.window_handler.scenes.push(scene);
        self.window_handler.scenes.len() - 1
    }

    pub fn start(mut self, scene_index: usize) {
        self.window_handler.starting_scene = scene_index;
        assert!(self.window_handler.scenes.len() > self.window_handler.starting_scene);
        self.event_loop.run_app(&mut self.window_handler).unwrap();
        self.window_handler.delete();
    }
}

static ENTRY: sync::LazyLock<ash::Entry, fn() -> ash::Entry> =
    sync::LazyLock::new(ash::Entry::linked);
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
    unsafe {
        let callback_data = *p_callback_data;
        // let message_id_number = callback_data.message_id_number;
        //
        // let message_id_name = if callback_data.p_message_id_name.is_null() {
        //     borrow::Cow::from("")
        // } else {
        //     ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
        // };

        let message = if callback_data.p_message.is_null() {
            borrow::Cow::from("")
        } else {
            ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
        };
        println!("{message_type:?} : {message}",);
    }

    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        process::exit(-1)
    }

    vk::FALSE
}

pub struct Engine {
    instance: ash::Instance,
    pub device: ash::Device,
    pub surface_data: SurfaceData,
    debug_utils_loader: debug_utils::Instance,
    window: mem::ManuallyDrop<window::Window>,
    debug_call_back: vk::DebugUtilsMessengerEXT,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub present_queue: vk::Queue,
    surface: vk::SurfaceKHR,
    swapchain: Swapchain,
    pool: vk::CommandPool,
    pub draw_command_buffers: Vec<vk::CommandBuffer>,
    setup_command_buffer: vk::CommandBuffer,
    depth_image: DepthImage,
    pub present_complete_semaphores: Vec<vk::Semaphore>,
    pub rendering_complete_semaphores: Vec<vk::Semaphore>,
    pub draw_commands_reuse_fence: Vec<vk::Fence>,
    setup_commands_reuse_fence: vk::Fence,
    pdevice: vk::PhysicalDevice,
    // pub limits: vk::PhysicalDeviceLimits,
}

impl Engine {
    pub fn new(
        preferred_width: u32,
        preferred_height: u32,
        name: &str,
        event_loop: &event_loop::ActiveEventLoop,
    ) -> Self {
        let window = event_loop
            .create_window(
                window::Window::default_attributes()
                    .with_inner_size(dpi::LogicalSize::new(
                        f64::from(preferred_width),
                        f64::from(preferred_height),
                    ))
                    .with_title(name),
            )
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
                sampler_anisotropy: vk::TRUE,
                fill_mode_non_solid: vk::TRUE,
                wide_lines: vk::TRUE,
                geometry_shader: vk::TRUE,
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
            let surface_data = SurfaceData {
                surface_capabilities,
                surface_loader,
                surface_format,
            };
            let swapchain = Swapchain::new(
                surface,
                &surface_data,
                pdevice,
                &instance,
                &device,
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
            let draw_commands_reuse_fence = (0..MAX_FRAMES_IN_FLIGHT)
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
            let present_complete_semaphore = (0..MAX_FRAMES_IN_FLIGHT)
                .map(|_| {
                    device
                        .create_semaphore(&semaphore_create_info, None)
                        .unwrap()
                })
                .collect();
            let rendering_complete_semaphore = (0..MAX_FRAMES_IN_FLIGHT)
                .map(|_| {
                    device
                        .create_semaphore(&semaphore_create_info, None)
                        .unwrap()
                })
                .collect();
            // let limits = instance.get_physical_device_properties(pdevice).limits;
            // println!("limits={:#?}", limits); // TODO get max number of images allowed on gpu
            Self {
                instance,
                device,
                pdevice,
                device_memory_properties,
                window: mem::ManuallyDrop::new(window),
                surface_data,
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
                debug_call_back,
                debug_utils_loader,
                // limits,
            }
        }
    }

    fn draw_frame(&self, pipeline: &mut Pipelines, scene: &Scene) {
        let current_frame = (pipeline.frames % MAX_FRAMES_IN_FLIGHT as u64) as usize;
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1.0, 1.0, 1.0, 0.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let viewports = [vk::Viewport {
            x: 0.0,
            y: self.swapchain.surface_resolution.height as f32,
            width: self.swapchain.surface_resolution.width as f32,
            height: -(self.swapchain.surface_resolution.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [self.swapchain.surface_resolution.into()];

        unsafe {
            self.device.device_wait_idle().unwrap();
            let present_index = self.swapchain.next_image(self, current_frame);
            let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                .render_pass(pipeline.renderpass)
                .framebuffer(pipeline.framebuffers[present_index as usize])
                .render_area(self.swapchain.surface_resolution.into())
                .clear_values(&clear_values);

            let mut mvps_per_shaderset: HashMap<Rc<ShaderSet>, Vec<MvpUbo>> = HashMap::new();

            for object in &scene.objects {
                let mvp = MvpUbo {
                    model: cgmath::Matrix4::from(object.model),
                    view: scene.camera.view_matrix(),
                    projection: cgmath::Matrix4::from(scene.camera.projection),
                };
                if let Some(mvps) = mvps_per_shaderset.get_mut(&object.shader_set) {
                    mvps.push(mvp);
                } else {
                    mvps_per_shaderset.insert(object.shader_set.clone(), vec![mvp]);
                }
            }
            for line in &scene.lines {
                let mvp = MvpUbo {
                    model: cgmath::Matrix4::from(line.model),
                    view: scene.camera.view_matrix(),
                    projection: cgmath::Matrix4::from(scene.camera.projection),
                };
                if let Some(mvps) = mvps_per_shaderset.get_mut(&line.shader_set) {
                    mvps.push(mvp);
                } else {
                    mvps_per_shaderset.insert(line.shader_set.clone(), vec![mvp]);
                }
            }
            for (shader_set, mvps) in mvps_per_shaderset {
                pipeline.update_uniforms(shader_set.clone(), mvps, current_frame);
            }

            Engine::record_submit_commandbuffer(
                &self.device,
                self.draw_command_buffers[current_frame],
                self.draw_commands_reuse_fence[current_frame],
                self.present_queue,
                &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                &[self.present_complete_semaphores[current_frame]],
                &[self.rendering_complete_semaphores[current_frame]],
                |device, draw_command_buffer| {
                    device.cmd_begin_render_pass(
                        draw_command_buffer,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    );
                    for (obj_index, object) in scene.objects.iter().enumerate() {
                        let mut shader_set_index = usize::MAX;
                        for (i, shader_set) in pipeline.shader_set_order.iter().enumerate() {
                            if &object.shader_set == shader_set {
                                shader_set_index = i
                            }
                        }
                        assert_ne!(shader_set_index, usize::MAX, "Shader set not found");
                        device.cmd_bind_pipeline(
                            draw_command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.graphics_pipelines[shader_set_index],
                        );
                        device.cmd_set_viewport(draw_command_buffer, 0, &viewports);
                        device.cmd_set_scissor(draw_command_buffer, 0, &scissors);
                        let sampler_index = object
                            .shader_set
                            .get_sampler_index(*object.material.global_index.borrow());
                        let sampler_bytes = sampler_index.to_le_bytes();

                        device.cmd_push_constants(
                            draw_command_buffer,
                            pipeline.pipeline_layouts[&object.shader_set],
                            vk::ShaderStageFlags::FRAGMENT,
                            0,
                            &sampler_bytes,
                        );

                        object.mesh.bind_buffers(self, current_frame);

                        device.cmd_bind_descriptor_sets(
                            draw_command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.pipeline_layouts[&object.shader_set],
                            0,
                            &[pipeline.descriptors_sets[&object.shader_set]],
                            &[object.shader_set.get_dynamic_offset(obj_index)],
                        );
                        device.cmd_draw_indexed(
                            draw_command_buffer,
                            object.mesh.get_index_count(),
                            1,
                            0,
                            0,
                            1,
                        );
                    }
                    for (line_index, line) in scene.lines.iter().enumerate() {
                        let mut shader_set_index = usize::MAX;
                        for (i, shader_set) in pipeline.shader_set_order.iter().enumerate() {
                            if &line.shader_set == shader_set {
                                shader_set_index = i
                            }
                        }
                        assert_ne!(shader_set_index, usize::MAX, "Shader set not found");
                        device.cmd_bind_pipeline(
                            draw_command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.graphics_pipelines[shader_set_index],
                        );
                        device.cmd_set_viewport(draw_command_buffer, 0, &viewports);
                        device.cmd_set_scissor(draw_command_buffer, 0, &scissors);
                        let sampler_index: u32 = 0;
                        let sampler_bytes = sampler_index.to_le_bytes();

                        device.cmd_push_constants(
                            draw_command_buffer,
                            pipeline.pipeline_layouts[&line.shader_set],
                            vk::ShaderStageFlags::FRAGMENT,
                            0,
                            &sampler_bytes,
                        );

                        line.mesh.bind_buffers(self, current_frame);
                        device.cmd_bind_descriptor_sets(
                            draw_command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.pipeline_layouts[&line.shader_set],
                            0,
                            &[pipeline.descriptors_sets[&line.shader_set]],
                            &[line.shader_set.get_dynamic_offset(line_index)],
                        );
                        device.cmd_set_line_width(draw_command_buffer, line.width);
                        device.cmd_draw_indexed(
                            draw_command_buffer,
                            line.mesh.get_index_count(),
                            1,
                            0,
                            0,
                            1,
                        );
                    }
                    device.cmd_end_render_pass(draw_command_buffer);
                },
            );
            self.swapchain.present(self, current_frame, present_index);
            pipeline.frames += 1 % u64::MAX;
        }
    }

    pub fn on_window_resize(&mut self, pipeline: &mut Pipelines) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.swapchain.delete(&self.device);
            pipeline.delete_framebuffers(self);
            self.surface_data.surface_capabilities = self
                .surface_data
                .surface_loader
                .get_physical_device_surface_capabilities(self.pdevice, self.surface)
                .unwrap();
            self.swapchain = Swapchain::new(
                self.surface,
                &self.surface_data,
                self.pdevice,
                &self.instance,
                &self.device,
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
            pipeline.aspect_ratio = self.swapchain.surface_resolution.width as f32
                / self.swapchain.surface_resolution.height as f32;
            pipeline.window_dimensions = cgmath::vec2(
                self.swapchain.surface_resolution.width as f64,
                self.swapchain.surface_resolution.height as f64,
            );
            // TODO recreate renderpass if image format was changed (like by changing monitor)
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
            let command_buffers = [command_buffer];
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
            self.surface_data
                .surface_loader
                .destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);
            self.instance.destroy_instance(None);
            let window = mem::ManuallyDrop::take(&mut self.window);
            drop(window);
        }
    }
}
