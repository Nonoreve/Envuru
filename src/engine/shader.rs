use std::io::Cursor;
use std::mem;

use crate::engine::memory::{DataOrganization, IndexBuffer, Texture, UniformBuffer, VertexBuffer};
use crate::engine::{Engine, MAX_FRAMES_IN_FLIGHT};
use ash::{util, vk};

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
pub struct MvpUbo {
    pub model: cgmath::Matrix4<f32>,
    pub view: cgmath::Matrix4<f32>,
    pub projection: cgmath::Matrix4<f32>,
}

pub struct VertexShaderInputs {
    pub vertices: Box<[Vertex]>,
    pub indices: Box<[u32]>,
}

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
        vertex_shader_bytes: &[u8],
        vertex_shader_inputs: VertexShaderInputs,
        descriptor_sets: &Vec<vk::DescriptorSet>,
    ) -> Self {
        let mut spv_data = Cursor::new(vertex_shader_bytes);
        let spv_data = util::read_spv(&mut spv_data).unwrap();
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
        let vertex_buffer = VertexBuffer::new(
            engine,
            vertex_shader_inputs.vertices,
            DataOrganization::ObjectMajor,
        );
        let index_buffer = IndexBuffer::new(engine, vertex_shader_inputs.indices);
        let uniform_mvp_buffers: Vec<UniformBuffer> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| UniformBuffer::new::<MvpUbo>(engine))
            .collect();

        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                let write_desc_sets = [vk::WriteDescriptorSet {
                    dst_set: descriptor_sets[i as usize],
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info: &uniform_mvp_buffers[i as usize].descriptor,
                    ..Default::default()
                }];
                engine.device.update_descriptor_sets(&write_desc_sets, &[]);
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

    pub fn delete(&mut self, engine: &Engine) {
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

struct ShaderImage();

pub struct FragmentShaderInputs {
    pub(crate) image: image::DynamicImage,
}

pub struct FragmentShader {
    pub module: vk::ShaderModule,
    sampler: vk::Sampler,
    texture: Texture,
}

impl FragmentShader {
    pub fn new(
        engine: &Engine,
        fragment_shader_bytes: &[u8],
        fragment_shader_inputs: FragmentShaderInputs,
        descriptor_sets: &Vec<vk::DescriptorSet>,
    ) -> Self {
        let image = fragment_shader_inputs.image.to_rgba8();
        let texture = Texture::new(engine, image);
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
            let tex_descriptors: Vec<vk::DescriptorImageInfo> = (0..MAX_FRAMES_IN_FLIGHT)
                .map(|_| vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image_view: texture.get_image_view(),
                    sampler,
                })
                .collect();

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                let write_desc_sets = [vk::WriteDescriptorSet {
                    dst_set: descriptor_sets[i as usize],
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: &tex_descriptors[i as usize],
                    ..Default::default()
                }];
                engine.device.update_descriptor_sets(&write_desc_sets, &[]);
            }
            let mut frag_spv_file = Cursor::new(fragment_shader_bytes);
            let frag_code = util::read_spv(&mut frag_spv_file)
                .expect("Failed to read fragment shader spv file");
            let frag_shader_info = vk::ShaderModuleCreateInfo::default().code(&frag_code);
            let fragment_shader_module = engine
                .device
                .create_shader_module(&frag_shader_info, None)
                .expect("Fragment shader module error");
            Self {
                module: fragment_shader_module,
                sampler,
                texture,
            }
        }
    }

    pub unsafe fn delete(&mut self, engine: &Engine) {
        engine.device.destroy_shader_module(self.module, None);
        self.texture.delete(engine);
        engine.device.destroy_sampler(self.sampler, None);
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Vertex {
    pub pos: cgmath::Vector4<f32>,
    pub uv: cgmath::Vector2<f32>,
}
