use ash::vk;
use std::mem;
use std::mem::ManuallyDrop;
use std::rc::Rc;

use crate::engine::Engine;
use crate::engine::memory::{DataOrganization, UniformBuffer};
use crate::engine::scene::{Material, MvpUbo, Vertex};

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

pub trait Shader {
    fn delete(&mut self, engine: &Engine);
    fn get_uniform_data(&self, index: usize) -> Option<&UniformBuffer>;
}

pub struct VertexInputs {
    pub input_attribute_desc: Vec<vk::VertexInputAttributeDescription>,
    pub input_binding_desc: Vec<vk::VertexInputBindingDescription>,
}

#[derive(Debug)]
pub struct VertexShader {
    pub module: vk::ShaderModule,
    pub uniform_mvp_buffers: Vec<UniformBuffer>,
}

impl VertexShader {
    pub fn new(
        engine: &Engine,
        spv_data: &[u32],
        descriptor_set: &vk::DescriptorSet,
        vertex_descriptors: &[vk::DescriptorType],
        _style: DataOrganization, // TODO DataOrganization
        users: u64,
    ) -> (VertexShader, VertexInputs) {
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
            for descriptor in vertex_descriptors.iter() {
                // TODO make dynamically per FRAME_IN_FLIGHT
                let uniform_mvp_buffer_1 = UniformBuffer::new::<MvpUbo>(engine, users);
                let uniform_mvp_buffer_2 = UniformBuffer::new::<MvpUbo>(engine, users);
                let write_desc_sets = [
                    vk::WriteDescriptorSet {
                        dst_set: *descriptor_set,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: *descriptor,
                        p_buffer_info: &uniform_mvp_buffer_1.descriptor,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: *descriptor_set,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: *descriptor,
                        p_buffer_info: &uniform_mvp_buffer_2.descriptor,
                        ..Default::default()
                    },
                ];
                engine.device.update_descriptor_sets(&write_desc_sets, &[]);
                uniform_mvp_buffers.push(uniform_mvp_buffer_1);
                uniform_mvp_buffers.push(uniform_mvp_buffer_2);
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
                VertexInputs {
                    input_attribute_desc: input_attribute_descriptions,
                    input_binding_desc: input_binding_descriptions,
                },
            )
        }
    }
}

impl Shader for VertexShader {
    fn delete(&mut self, engine: &Engine) {
        unsafe {
            engine.device.destroy_shader_module(self.module, None);
            for uniform_mvp_buffer in self.uniform_mvp_buffers.iter() {
                uniform_mvp_buffer.delete(engine);
            }
        }
    }

    fn get_uniform_data(&self, index: usize) -> Option<&UniformBuffer>{
        Some(&self.uniform_mvp_buffers[index])
    }
}

#[derive(Debug)]
pub struct GeometryShader {
    pub module: vk::ShaderModule,
}

impl GeometryShader {
    pub fn new(
        engine: &Engine,
        spv_data: &[u32],
    ) -> GeometryShader {
        let geometry_shader_info = vk::ShaderModuleCreateInfo::default().code(spv_data);

        unsafe {
            let module = engine
                .device
                .create_shader_module(&geometry_shader_info, None)
                .unwrap();
            Self {
                module,
            }
        }
    }

    pub fn delete(&self, engine: &Engine) {
        unsafe {
            engine.device.destroy_shader_module(self.module, None);
        }
    }
}

#[derive(Debug)]
pub struct FragmentShader {
    pub module: vk::ShaderModule,
    samplers: ManuallyDrop<Vec<vk::Sampler>>,
}

impl FragmentShader {
    pub fn new(
        engine: &Engine,
        spv_data: &[u32],
        materials: &[&Rc<Material>],
        descriptor_sets: &[vk::DescriptorSet],
        descriptor_types: &[vk::DescriptorType],
        users: usize
    ) -> Self {
        let sampler_info = vk::SamplerCreateInfo {
            flags: Default::default(),
            mag_filter: vk::Filter::NEAREST,
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
            let samplers: Vec<vk::Sampler>;
            if descriptor_types.contains(&vk::DescriptorType::COMBINED_IMAGE_SAMPLER) {
                samplers = std::iter::repeat_n(
                    engine.device.create_sampler(&sampler_info, None).unwrap(),
                    users
                )
                .collect();
                let image_infos: Vec<vk::DescriptorImageInfo> = materials
                    .iter()
                    .enumerate()
                    .map(|(i, material)| material.get_descriptor_image_info(samplers[i], 0))
                    .collect();

                for descriptor_set in descriptor_sets.iter() {
                    let write_desc_sets = [vk::WriteDescriptorSet {
                        dst_set: *descriptor_set,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: users as u32,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: image_infos.as_ptr(),
                        ..Default::default()
                    }];
                    engine.device.update_descriptor_sets(&write_desc_sets, &[]);
                }
            } else {
                samplers = Vec::new()
            }
            let frag_shader_info = vk::ShaderModuleCreateInfo::default().code(spv_data);
            let fragment_shader_module = engine
                .device
                .create_shader_module(&frag_shader_info, None)
                .expect("Fragment shader module error");
            Self {
                module: fragment_shader_module,
                samplers: ManuallyDrop::new(samplers),
            }
        }
    }
}

impl Shader for FragmentShader {
    fn delete(&mut self, engine: &Engine) {
        unsafe {
            engine.device.destroy_shader_module(self.module, None);
            let samplers = ManuallyDrop::take(&mut self.samplers);
            for sampler in samplers {
                engine.device.destroy_sampler(sampler, None);
            }
        }
    }

    fn get_uniform_data(&self, _index: usize) -> Option<&UniformBuffer>{
        None
    }
}
