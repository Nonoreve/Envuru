use std::ffi::c_void;

use ash::{util, vk};

use crate::engine::shader::Vertex;
use crate::engine::swapchain::Swapchain;
use crate::engine::{DIRECT_MAPPING, Engine};

#[allow(dead_code)]
pub enum DataOrganization {
    ObjectMajor,
    FieldMajor,
    Smart,
}

pub struct GraphicsBuffer {
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
}

impl GraphicsBuffer {
    fn new(
        engine: &Engine,
        create_info: vk::BufferCreateInfo,
        device_local: bool,
    ) -> (Option<*mut c_void>, vk::MemoryRequirements, Self) {
        let flags = if device_local {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        } else {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        };
        unsafe {
            let vk_buffer = engine.device.create_buffer(&create_info, None).unwrap();
            let memory_requirements = engine.device.get_buffer_memory_requirements(vk_buffer);
            let memory_index = Engine::find_memorytype_index(
                &memory_requirements,
                &engine.device_memory_properties,
                flags,
            )
            .unwrap();
            let allocate_info = vk::MemoryAllocateInfo {
                allocation_size: memory_requirements.size,
                memory_type_index: memory_index,
                ..Default::default()
            };
            let device_memory = engine.device.allocate_memory(&allocate_info, None).unwrap();
            engine
                .device
                .bind_buffer_memory(vk_buffer, device_memory, 0)
                .unwrap();
            let data_ptr = if device_local {
                None
            } else {
                let ptr = engine.device.map_memory(
                    device_memory,
                    0,
                    memory_requirements.size,
                    vk::MemoryMapFlags::empty(),
                );
                Some(ptr.unwrap())
            };
            (
                data_ptr,
                memory_requirements,
                Self {
                    memory: device_memory,
                    buffer: vk_buffer,
                },
            )
        }
    }

    fn stage_up(
        &self,
        engine: &Engine,
        destination_buffer_info: vk::BufferCreateInfo,
        size: usize,
    ) -> Self {
        let (_, _, destination) = GraphicsBuffer::new(engine, destination_buffer_info, true);
        unsafe {
            Engine::record_submit_commandbuffer(
                &engine.device,
                engine.setup_command_buffer,
                engine.setup_commands_reuse_fence,
                engine.present_queue,
                &[],
                &[],
                &[],
                |device, command_buffer| {
                    let copy_region = vk::BufferCopy::default().size(size as vk::DeviceSize);
                    device.cmd_copy_buffer(
                        command_buffer,
                        self.buffer,
                        destination.buffer,
                        &[copy_region],
                    )
                },
            );
            destination
        }
    }

    fn delete(&self, engine: &Engine) {
        unsafe {
            engine.device.destroy_buffer(self.buffer, None);
            engine.device.free_memory(self.memory, None);
        }
    }
}

pub struct VertexBuffer(GraphicsBuffer);

impl VertexBuffer {
    pub fn new(engine: &Engine, vertices: Box<[Vertex]>, style: DataOrganization) -> Self {
        let source_size = size_of_val(&*vertices);
        let mut create_info = vk::BufferCreateInfo {
            size: source_size as vk::DeviceSize,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let (data_ptr, memory_requirements, staging_buffer) =
            GraphicsBuffer::new(engine, create_info, false);

        unsafe {
            let mut alignment = util::Align::new(
                data_ptr.unwrap(),
                align_of::<f32>() as u64,
                memory_requirements.size,
            );
            alignment.copy_from_slice(&*vertices);
            engine.device.unmap_memory(staging_buffer.memory);
            create_info = vk::BufferCreateInfo {
                size: source_size as vk::DeviceSize,
                usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
        }
        let destination = staging_buffer.stage_up(engine, create_info, source_size);
        staging_buffer.delete(engine);
        Self { 0: destination }
    }

    pub fn bind(&self, engine: &Engine, current_frame: usize) {
        unsafe {
            engine.device.cmd_bind_vertex_buffers(
                engine.draw_command_buffers[current_frame],
                0,
                &[self.0.buffer],
                &[0],
            );
        }
    }

    pub fn delete(&self, engine: &Engine) {
        self.0.delete(engine);
    }
}

pub struct IndexBuffer {
    graphics_buffer: GraphicsBuffer,
    pub index_count: u32,
}

impl IndexBuffer {
    pub fn new(engine: &Engine, indices: Box<[u32]>) -> Self {
        let source_size = size_of_val(&*indices);
        let mut create_info = vk::BufferCreateInfo {
            size: source_size as vk::DeviceSize,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let (data_ptr, memory_requirements, staging_buffer) =
            GraphicsBuffer::new(engine, create_info, false);
        unsafe {
            let mut alignment = util::Align::new(
                data_ptr.unwrap(),
                align_of::<u32>() as u64,
                memory_requirements.size,
            );
            alignment.copy_from_slice(&*indices);
            engine.device.unmap_memory(staging_buffer.memory);
            create_info = vk::BufferCreateInfo {
                size: source_size as vk::DeviceSize,
                usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
        }
        let destination = staging_buffer.stage_up(engine, create_info, source_size);
        staging_buffer.delete(engine);
        Self {
            graphics_buffer: destination,
            index_count: indices.len() as u32,
        }
    }

    pub fn bind(&self, engine: &Engine, current_frame: usize) {
        unsafe {
            engine.device.cmd_bind_index_buffer(
                engine.draw_command_buffers[current_frame],
                self.graphics_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );
        }
    }

    pub fn delete(&self, engine: &Engine) {
        self.graphics_buffer.delete(engine);
    }
}

pub struct UniformBuffer {
    graphics_buffer: GraphicsBuffer,
    pub descriptor: vk::DescriptorBufferInfo,
    pub data_ptr: Option<*mut c_void>,
    pub memory_requirements: vk::MemoryRequirements,
}

impl UniformBuffer {
    pub fn new<T>(engine: &Engine) -> Self {
        let size = size_of::<T>() as u64;
        let create_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let (data_ptr, memory_requirements, graphics_buffer) =
            GraphicsBuffer::new(engine, create_info, false);
        let descriptor_buffer = vk::DescriptorBufferInfo {
            buffer: graphics_buffer.buffer,
            offset: 0,
            range: size,
        };
        Self {
            data_ptr,
            memory_requirements,
            graphics_buffer,
            descriptor: descriptor_buffer,
        }
    }

    pub fn delete(&self, engine: &Engine) {
        self.graphics_buffer.delete(engine);
    }
}

struct ImageBuffer {
    image_view: vk::ImageView,
    memory: vk::DeviceMemory,
    image: vk::Image,
}

impl ImageBuffer {
    pub fn delete(&self, engine: &Engine) {
        unsafe {
            engine.device.free_memory(self.memory, None);
            engine.device.destroy_image_view(self.image_view, None);
            engine.device.destroy_image(self.image, None);
        }
    }
}

pub struct Texture {
    image_buffer: ImageBuffer,
}

impl Texture {
    pub fn new(engine: &Engine, image: image::RgbaImage) -> Self {
        let image_data = image.as_raw();
        let create_info = vk::BufferCreateInfo {
            size: (size_of::<u8>() * image_data.len()) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let (data_ptr, memory_requirements, staging_buffer) =
            GraphicsBuffer::new(engine, create_info, false);
        let (width, height) = image.dimensions();
        let image_extent = vk::Extent2D { width, height };
        let create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: image_extent.into(),
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        unsafe {
            let mut alignment = util::Align::new(
                data_ptr.unwrap(),
                align_of::<u8>() as u64,
                memory_requirements.size,
            );
            alignment.copy_from_slice(image_data);
            engine.device.unmap_memory(staging_buffer.memory);
            drop(image);
            let vk_image = engine.device.create_image(&create_info, None).unwrap();
            let memory_requirements = engine.device.get_image_memory_requirements(vk_image);
            let memory_index = Engine::find_memorytype_index(
                &memory_requirements,
                &engine.device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .unwrap();
            let allocate_info = vk::MemoryAllocateInfo {
                allocation_size: memory_requirements.size,
                memory_type_index: memory_index,
                ..Default::default()
            };
            let image_memory = engine.device.allocate_memory(&allocate_info, None).unwrap();
            engine
                .device
                .bind_image_memory(vk_image, image_memory, 0)
                .expect("Unable to bind depth image memory");

            Engine::record_submit_commandbuffer(
                &engine.device,
                engine.setup_command_buffer,
                engine.setup_commands_reuse_fence,
                engine.present_queue,
                &[],
                &[],
                &[],
                |device, texture_command_buffer| {
                    let texture_barrier = vk::ImageMemoryBarrier {
                        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        image: vk_image,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            level_count: 1,
                            layer_count: 1,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    device.cmd_pipeline_barrier(
                        texture_command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[texture_barrier],
                    );
                    let buffer_copy_regions = vk::BufferImageCopy::default()
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .image_extent(image_extent.into());

                    device.cmd_copy_buffer_to_image(
                        texture_command_buffer,
                        staging_buffer.buffer,
                        vk_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[buffer_copy_regions],
                    );
                    let texture_barrier_end = vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        image: vk_image,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            level_count: 1,
                            layer_count: 1,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    device.cmd_pipeline_barrier(
                        texture_command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[texture_barrier_end],
                    );
                },
            );
            engine.device.device_wait_idle().unwrap();
            staging_buffer.delete(engine);
            let create_info = vk::ImageViewCreateInfo {
                view_type: vk::ImageViewType::TYPE_2D,
                format: create_info.format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                image: vk_image,
                ..Default::default()
            };
            let image_view = engine.device.create_image_view(&create_info, None).unwrap();
            Self {
                image_buffer: ImageBuffer {
                    image_view,
                    memory: image_memory,
                    image: vk_image,
                },
            }
        }
    }

    pub fn get_image_view(&self) -> vk::ImageView {
        self.image_buffer.image_view
    }

    pub fn delete(&self, engine: &Engine) {
        self.image_buffer.delete(engine);
    }
}

pub struct DepthImage(ImageBuffer);

impl DepthImage {
    pub fn new(
        swapchain: &Swapchain,
        device: &ash::Device,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
        setup_command_buffer: vk::CommandBuffer,
        setup_commands_reuse_fence: vk::Fence,
        present_queue: vk::Queue,
    ) -> Self {
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
        unsafe {
            let vk_image = device.create_image(&depth_image_create_info, None).unwrap();
            let image_memory_requirements = device.get_image_memory_requirements(vk_image);
            let image_memory_index = Engine::find_memorytype_index(
                &image_memory_requirements,
                device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image.");
            let allocate_info = vk::MemoryAllocateInfo {
                allocation_size: image_memory_requirements.size,
                memory_type_index: image_memory_index,
                ..Default::default()
            };
            let device_memory = device.allocate_memory(&allocate_info, None).unwrap();
            device
                .bind_image_memory(vk_image, device_memory, 0)
                .unwrap();
            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            Engine::record_submit_commandbuffer(
                device,
                setup_command_buffer,
                setup_commands_reuse_fence,
                present_queue,
                &[],
                &[],
                &[],
                |device, setup_command_buffer| {
                    let layout_transition_barriers = vk::ImageMemoryBarrier {
                        src_access_mask: Default::default(),
                        dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        src_queue_family_index: 0, // TODO queue_family_index ?
                        dst_queue_family_index: 0,
                        image: vk_image,
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
                image: vk_image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: depth_image_create_info.format,
                components: DIRECT_MAPPING,
                subresource_range,
                ..Default::default()
            };
            let image_view = device.create_image_view(&create_info, None).unwrap();
            Self {
                0: ImageBuffer {
                    image_view,
                    memory: device_memory,
                    image: vk_image,
                },
            }
        }
    }

    pub fn get_image_view(&self) -> vk::ImageView {
        self.0.image_view
    }

    pub fn delete(&self, engine: &Engine) {
        self.0.delete(engine);
    }
}
