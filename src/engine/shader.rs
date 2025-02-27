use std::io::Cursor;
use std::mem;

use ash::util::read_spv;
use ash::vk;

use crate::engine::Engine;
use crate::engine::memory::{DataOrganization, IndexBuffer, Texture, UniformBuffer, VertexBuffer};
use crate::engine::utils::{ Vertex};

#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = mem::zeroed();
            std::ptr::addr_of!(b.$field) as isize - std::ptr::addr_of!(b) as isize
        }
    }};
}

pub struct VertexShader {
    pub module: vk::ShaderModule,
    pub input_binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    pub input_attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
    pub vertex_buffer: VertexBuffer,
    pub index_buffer: IndexBuffer,
}

impl VertexShader {
    pub fn new(
        engine: &Engine,
        spv_data: &[u8],
        vertices: Box<[Vertex]>,
        indices: Box<[u32]>,
    ) -> Self {
        let mut spv_data = Cursor::new(spv_data);
        let spv_data = read_spv(&mut spv_data).unwrap();
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_data);
        let input_binding_descriptions = vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let input_attribute_descriptions = vec![
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Vertex, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Vertex, uv) as u32,
            },
        ];
        let vertex_buffer = VertexBuffer::new(engine, vertices, DataOrganization::ObjectMajor);
        let index_buffer = IndexBuffer::new(engine, indices);
        unsafe {
            let module = engine
                .device
                .create_shader_module(&vertex_shader_info, None)
                .unwrap();
            Self {
                module,
                input_binding_descriptions,
                input_attribute_descriptions,
                index_buffer,
                vertex_buffer,
            }
        }
    }

    pub fn delete(&mut self, engine: &Engine) {
        unsafe {
            engine.device.destroy_shader_module(self.module, None);
        }
        self.index_buffer.delete(engine);
        self.vertex_buffer.delete(engine);
    }
}

pub struct FragmentShader {
    pub module: vk::ShaderModule,
    pub desc_set_layouts: [vk::DescriptorSetLayout; 1],
    pub uniform_color_buffer: UniformBuffer,
    pub descriptor_pool: vk::DescriptorPool,
    pub sampler: vk::Sampler,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub texture: Texture,
}

impl FragmentShader {
    pub fn new(engine: &Engine, spv_data: &[u8], color_buffer_data: Box<cgmath::Vector3<f32>>) -> Self {
        let uniform_color_buffer = UniformBuffer::new(engine, color_buffer_data);
        let image =
            image::load_from_memory(include_bytes!("../../resources/textures/potoo_asks.jpg"))
                .unwrap()
                .to_rgba8();
        let texture = Texture::new(engine, image);
        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
            address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
            address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
            max_anisotropy: 1.0,
            border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
            compare_op: vk::CompareOp::NEVER,
            ..Default::default()
        };

        unsafe {
            let sampler = engine.device.create_sampler(&sampler_info, None).unwrap();
            let descriptor_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                },
            ];
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&descriptor_sizes)
                .max_sets(1);
            let descriptor_pool = engine
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap();
            let desc_layout_bindings = [
                vk::DescriptorSetLayoutBinding {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    ..Default::default()
                },
            ];
            let descriptor_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&desc_layout_bindings);
            let desc_set_layouts = [engine
                .device
                .create_descriptor_set_layout(&descriptor_info, None)
                .unwrap()];
            let desc_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&desc_set_layouts);
            let descriptor_sets = engine
                .device
                .allocate_descriptor_sets(&desc_alloc_info)
                .unwrap();
            let tex_descriptor = vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image_view: texture.image_view,
                sampler,
            };
            let write_desc_sets = [
                vk::WriteDescriptorSet {
                    dst_set: descriptor_sets[0],
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info: &uniform_color_buffer.descriptor,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: descriptor_sets[0],
                    dst_binding: 1,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: &tex_descriptor,
                    ..Default::default()
                },
            ];
            engine.device.update_descriptor_sets(&write_desc_sets, &[]);
            let mut frag_spv_file = Cursor::new(spv_data);
            let frag_code =
                read_spv(&mut frag_spv_file).expect("Failed to read fragment shader spv file");
            let frag_shader_info = vk::ShaderModuleCreateInfo::default().code(&frag_code);
            let fragment_shader_module = engine
                .device
                .create_shader_module(&frag_shader_info, None)
                .expect("Fragment shader module error");
            Self {
                desc_set_layouts,
                descriptor_sets,
                module: fragment_shader_module,
                uniform_color_buffer,
                descriptor_pool,
                sampler,
                texture,
            }
        }
    }

    pub unsafe fn delete(&mut self, engine: &Engine) {
        engine.device.destroy_shader_module(self.module, None);
        self.texture.delete(engine);
        self.uniform_color_buffer.delete(engine);
        for &descriptor_set_layout in self.desc_set_layouts.iter() {
            engine
                .device
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
        }
        engine
            .device
            .destroy_descriptor_pool(self.descriptor_pool, None);
        engine.device.destroy_sampler(self.sampler, None);
    }
}
