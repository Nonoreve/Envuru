use std::mem;

use crate::engine::Engine;
use crate::engine::memory::{DataOrganization, UniformBuffer};
use crate::engine::scene::{MvpUbo, Object, Vertex};
use ash::vk;

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

#[derive(Debug)]
pub struct VertexShader {
    pub module: vk::ShaderModule,
    pub uniform_mvp_buffers: Vec<UniformBuffer>,
}

impl VertexShader {
    pub fn new(
        engine: &Engine,
        spv_data: &Vec<u32>,
        descriptor_sets: &Vec<vk::DescriptorSet>,
        style: DataOrganization,
    ) -> (
        VertexShader,
        Vec<vk::VertexInputAttributeDescription>,
        Vec<vk::VertexInputBindingDescription>,
    ) {
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
            (
                Self {
                    module,
                    uniform_mvp_buffers,
                },
                input_attribute_descriptions,
                input_binding_descriptions,
            )
        }
    }

    pub fn delete(&self, engine: &Engine) {
        unsafe {
            engine.device.destroy_shader_module(self.module, None);
            for uniform_mvp_buffer in self.uniform_mvp_buffers.iter() {
                uniform_mvp_buffer.delete(engine);
            }
        }
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
        objects: &Vec<Object>,
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

            for (i, descriptor_set) in descriptor_sets.iter().enumerate() {
                let tex_descriptor = objects
                    .get(i % objects.len())
                    .unwrap()
                    .material
                    .get_descriptor_image_info(sampler, 0);
                let write_desc_sets = [vk::WriteDescriptorSet {
                    dst_set: descriptor_set.clone(),
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: &tex_descriptor,
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
