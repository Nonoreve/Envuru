use std::mem;

use ash::vk;

use crate::engine::Engine;
use crate::engine::memory::{DataOrganization, IndexBuffer, UniformBuffer, VertexBuffer};
use crate::engine::scene::Material;

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

#[derive(Copy, Clone)]
#[allow(dead_code)]
pub struct MvpUbo {
    pub model: cgmath::Matrix4<f32>,
    pub view: cgmath::Matrix4<f32>,
    pub projection: cgmath::Matrix4<f32>,
}

#[derive(Clone, Debug, Copy)]
pub struct Vertex {
    pub pos: cgmath::Vector4<f32>,
    pub uv: cgmath::Vector2<f32>,
}

#[derive(Debug)]
pub struct VertexShader {
    pub module: vk::ShaderModule,
    pub input_binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    pub input_attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
    pub vertex_buffer: VertexBuffer,
    pub index_buffer: IndexBuffer,
    pub uniform_mvp_buffers: Vec<UniformBuffer>,
}

impl VertexShader {
    pub fn new(
        engine: &Engine,
        spv_data: &Vec<u32>,
        vertices: &[Vertex],
        indices: &[u32],
        descriptor_sets: &Vec<vk::DescriptorSet>,
        style: DataOrganization,
    ) -> Self {
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(spv_data);
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
        let mut uniform_mvp_buffers = Vec::new();

        unsafe {
            for descriptor_set in descriptor_sets {
                let uniform_mvp_buffer = UniformBuffer::new::<MvpUbo>(engine);
                let write_desc_sets = [vk::WriteDescriptorSet {
                    dst_set: descriptor_set.clone(),
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info: &uniform_mvp_buffer.descriptor,
                    ..Default::default()
                }];
                engine.device.update_descriptor_sets(&write_desc_sets, &[]);
                uniform_mvp_buffers.push(uniform_mvp_buffer);
            }
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
                uniform_mvp_buffers,
            }
        }
    }

    pub fn get_vertex_input_state_info(&self) -> vk::PipelineVertexInputStateCreateInfo {
        vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&self.input_attribute_descriptions)
            .vertex_binding_descriptions(&self.input_binding_descriptions)
    }

    pub fn delete(&self, engine: &Engine) {
        unsafe {
            engine.device.destroy_shader_module(self.module, None);
            for uniform_mvp_buffer in self.uniform_mvp_buffers.iter() {
                uniform_mvp_buffer.delete(engine);
            }
        }
        self.index_buffer.delete(engine);
        self.vertex_buffer.delete(engine);
    }
}

#[derive(Debug)]
pub struct FragmentShader {
    pub module: vk::ShaderModule,
    sampler: vk::Sampler,
}

impl FragmentShader {
    pub fn new(
        engine: &Engine,
        spv_data: &Vec<u32>,
        material: &Material,
        descriptor_sets: &Vec<vk::DescriptorSet>,
    ) -> Self {
        let sampler_info = vk::SamplerCreateInfo {
            flags: Default::default(),
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
            address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
            address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
            mip_lod_bias: 0.0,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: 1.0,
            border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
            unnormalized_coordinates: 0,
            compare_op: vk::CompareOp::NEVER,
            min_lod: 0.0,
            compare_enable: 0,
            max_lod: 0.0,
            ..Default::default()
        };

        unsafe {
            let sampler = engine.device.create_sampler(&sampler_info, None).unwrap();
            let mut tex_descriptors = material.get_descriptor_image_infos(sampler, 0);
            let mut tex_descriptors_cycle = tex_descriptors.iter().cycle();

            for descriptor_set in descriptor_sets {
                let write_desc_sets = [vk::WriteDescriptorSet {
                    dst_set: descriptor_set.clone(),
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: tex_descriptors_cycle.next().unwrap(),
                    ..Default::default()
                }];
                engine.device.update_descriptor_sets(&write_desc_sets, &[]);
            }
            let frag_shader_info = vk::ShaderModuleCreateInfo::default().code(spv_data);
            let fragment_shader_module = engine
                .device
                .create_shader_module(&frag_shader_info, None)
                .expect("Fragment shader module error");
            Self {
                module: fragment_shader_module,
                sampler,
            }
        }
    }

    pub fn delete(&self, engine: &Engine) {
        unsafe {
            engine.device.destroy_shader_module(self.module, None);
            engine.device.destroy_sampler(self.sampler, None);
        }
    }
}
