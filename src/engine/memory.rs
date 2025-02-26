use std::ffi::c_void;

use ash::util::Align;
use ash::vk;
use image::RgbaImage;

use crate::engine;
use crate::engine::Engine;
use crate::engine::utils::{Vector3, Vertex};

#[allow(dead_code)]
pub enum DataOrganization {
    ObjectMajor,
    FieldMajor,
    Smart,
}

pub struct GraphicsBuffer<T: ?Sized> {
    data: Box<T>,
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
}

impl<T: ?Sized> GraphicsBuffer<T> {
    fn prepare(
        engine: &Engine,
        create_info: vk::BufferCreateInfo,
        device_local: bool,
    ) -> (
        Option<*mut c_void>,
        vk::MemoryRequirements,
        vk::DeviceMemory,
        vk::Buffer,
    ) {
        let flags = if device_local {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        } else {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        };
        unsafe {
            let vk_buffer = engine.device.create_buffer(&create_info, None).unwrap();
            let memory_requirements = engine.device.get_buffer_memory_requirements(vk_buffer);
            let memory_index = engine::find_memorytype_index(
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
            (data_ptr, memory_requirements, device_memory, vk_buffer)
        }
    }

    fn delete(&self, engine: &Engine) {
        unsafe {
            engine.device.free_memory(self.memory, None);
            engine.device.destroy_buffer(self.buffer, None);
        }
    }
}

pub struct VertexBuffer(GraphicsBuffer<[Vertex]>);

impl VertexBuffer {
    pub fn new(engine: &Engine, vertices: Box<[Vertex]>, style: DataOrganization) -> Self {
        let source_size = size_of_val(&*vertices) as u64;
        let create_info = vk::BufferCreateInfo {
            size: source_size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let (data_ptr, memory_requirements, staging_memory, staging_buffer) =
            GraphicsBuffer::<Vertex>::prepare(engine, create_info, false);

        unsafe {
            let mut alignment = Align::new(
                data_ptr.unwrap(),
                align_of::<Vertex>() as u64,
                memory_requirements.size,
            );
            alignment.copy_from_slice(&*vertices);
            engine.device.unmap_memory(staging_memory);
            engine
                .device
                .bind_buffer_memory(staging_buffer, staging_memory, 0)
                .unwrap();

            let create_info = vk::BufferCreateInfo {
                size: source_size,
                usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let (_, _, device_memory, vk_buffer) =
                GraphicsBuffer::<Vertex>::prepare(engine, create_info, true);
            engine
                .device
                .bind_buffer_memory(vk_buffer, device_memory, 0)
                .unwrap();
            engine::record_submit_commandbuffer(
                &engine.device,
                engine.setup_command_buffer,
                engine.setup_commands_reuse_fence,
                engine.present_queue,
                &[],
                &[],
                &[],
                |device, command_buffer| {
                    let copy_region = vk::BufferCopy::default().size(source_size);
                    device.cmd_copy_buffer(
                        command_buffer,
                        staging_buffer,
                        vk_buffer,
                        &[copy_region],
                    )
                },
            );
            engine.device.free_memory(staging_memory, None);
            engine.device.destroy_buffer(staging_buffer, None);
            Self {
                0: GraphicsBuffer {
                    data: vertices,
                    memory: device_memory,
                    buffer: vk_buffer,
                },
            }
        }
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

pub struct IndexBuffer(GraphicsBuffer<[u32]>);

impl IndexBuffer {
    pub fn new(engine: &Engine, indices: Box<[u32]>) -> Self {
        let source_size = size_of_val(&*indices) as u64;
        let create_info = vk::BufferCreateInfo::default()
            .size(source_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let (data_ptr, memory_requirements, staging_memory, staging_buffer) =
            GraphicsBuffer::<u32>::prepare(engine, create_info, false);
        unsafe {
            let mut alignment = Align::new(
                data_ptr.unwrap(),
                align_of::<u32>() as u64,
                memory_requirements.size,
            );
            alignment.copy_from_slice(&*indices);
            engine.device.unmap_memory(staging_memory);
            engine
                .device
                .bind_buffer_memory(staging_buffer, staging_memory, 0)
                .unwrap();
            let create_info = vk::BufferCreateInfo::default()
                .size(source_size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let (_, _, device_memory, vk_buffer) =
                GraphicsBuffer::<u32>::prepare(engine, create_info, true);
            engine
                .device
                .bind_buffer_memory(vk_buffer, device_memory, 0)
                .unwrap();
            engine::record_submit_commandbuffer(
                &engine.device,
                engine.setup_command_buffer,
                engine.setup_commands_reuse_fence,
                engine.present_queue,
                &[],
                &[],
                &[],
                |device, command_buffer| {
                    let copy_region = vk::BufferCopy::default().size(source_size);
                    device.cmd_copy_buffer(
                        command_buffer,
                        staging_buffer,
                        vk_buffer,
                        &[copy_region],
                    )
                },
            );
            engine.device.free_memory(staging_memory, None);
            engine.device.destroy_buffer(staging_buffer, None);
            Self {
                0: GraphicsBuffer {
                    data: indices,
                    memory: device_memory,
                    buffer: vk_buffer,
                },
            }
        }
    }

    pub fn bind(&self, engine: &Engine, current_frame: usize) {
        unsafe {
            engine.device.cmd_bind_index_buffer(
                engine.draw_command_buffers[current_frame],
                self.0.buffer,
                0,
                vk::IndexType::UINT32,
            );
        }
    }

    pub fn index_count(&self) -> u32 {
        self.0.data.len() as u32
    }

    pub fn delete(&self, engine: &Engine) {
        self.0.delete(engine);
    }
}

pub struct ImageBuffer(GraphicsBuffer<RgbaImage>);

impl ImageBuffer {
    pub fn new(engine: &Engine, image: RgbaImage) -> Self {
        let image_data = image.as_raw();
        let create_info = vk::BufferCreateInfo {
            size: (size_of::<u8>() * image_data.len()) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let (data_ptr, memory_requirements, device_memory, vk_buffer) =
            GraphicsBuffer::<u8>::prepare(engine, create_info, false);
        unsafe {
            let mut alignment = Align::new(
                data_ptr.unwrap(),
                align_of::<u8>() as u64,
                memory_requirements.size,
            );
            alignment.copy_from_slice(image_data);
            engine.device.unmap_memory(device_memory);
            engine
                .device
                .bind_buffer_memory(vk_buffer, device_memory, 0)
                .unwrap();
        }
        Self {
            0: GraphicsBuffer {
                data: Box::new(image),
                memory: device_memory,
                buffer: vk_buffer,
            },
        }
    }

    pub fn get_buffer(&self) -> vk::Buffer {
        self.0.buffer
    }

    pub fn get_image(&self) -> &RgbaImage {
        &*self.0.data
    }

    pub fn delete(&self, engine: &Engine) {
        self.0.delete(engine);
    }
}

pub struct UniformBuffer {
    graphics_buffer: GraphicsBuffer<Vector3>,
    pub descriptor: vk::DescriptorBufferInfo,
}

impl UniformBuffer {
    pub fn new(engine: &Engine, uniform_color_buffer_data: Box<Vector3>) -> Self {
        let create_info = vk::BufferCreateInfo {
            size: size_of_val(&*uniform_color_buffer_data) as u64,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let (data_ptr, memory_requirements, device_memory, vk_buffer) =
            GraphicsBuffer::<Vector3>::prepare(engine, create_info, false);
        unsafe {
            let mut alignment = Align::new(
                data_ptr.unwrap(),
                align_of::<Vector3>() as u64,
                memory_requirements.size,
            );
            alignment.copy_from_slice(&[*uniform_color_buffer_data]);
            engine.device.unmap_memory(device_memory);
            engine
                .device
                .bind_buffer_memory(vk_buffer, device_memory, 0)
                .unwrap();
            let uniform_color_buffer_descriptor = vk::DescriptorBufferInfo {
                buffer: vk_buffer,
                offset: 0,
                range: size_of_val(&uniform_color_buffer_data) as u64,
            };
            Self {
                graphics_buffer: GraphicsBuffer {
                    data: uniform_color_buffer_data,
                    memory: device_memory,
                    buffer: vk_buffer,
                },
                descriptor: uniform_color_buffer_descriptor,
            }
        }
    }

    pub fn delete(&self, engine: &Engine) {
        self.graphics_buffer.delete(engine);
    }
}

pub struct Texture {
    pub image_view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub image: vk::Image,
    pub image_buffer: ImageBuffer,
}

impl Texture {
    pub fn new(engine: &Engine, image: RgbaImage) -> Self {
        let image_buffer = ImageBuffer::new(engine, image);
        let (width, height) = image_buffer.get_image().dimensions();
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
            let vk_image = engine.device.create_image(&create_info, None).unwrap();
            let memory_requirements = engine.device.get_image_memory_requirements(vk_image);
            let memory_index = engine::find_memorytype_index(
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
            let device_memory = engine.device.allocate_memory(&allocate_info, None).unwrap();
            engine
                .device
                .bind_image_memory(vk_image, device_memory, 0)
                .expect("Unable to bind depth image memory");

            engine::record_submit_commandbuffer(
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
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
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
                        image_buffer.get_buffer(),
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
                image_view,
                memory: device_memory,
                image: vk_image,
                image_buffer,
            }
        }
    }

    pub fn delete(&self, engine: &Engine) {
        self.image_buffer.delete(engine);
        unsafe {
            engine.device.free_memory(self.memory, None);
            engine.device.destroy_image_view(self.image_view, None);
            engine.device.destroy_image(self.image, None);
        }
    }
}
