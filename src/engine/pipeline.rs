use std::{ffi, mem};

use ash::vk;

use crate::engine::shader::{
    FragmentShader, FragmentShaderInputs, VertexShader, VertexShaderInputs,
};
use crate::engine::{Engine, MAX_FRAMES_IN_FLIGHT};

pub(crate) struct Pipeline {
    pub renderpass: vk::RenderPass,
    pub framebuffers: mem::ManuallyDrop<Vec<vk::Framebuffer>>,
    pub graphics_pipelines: mem::ManuallyDrop<Vec<vk::Pipeline>>,
    pub vertex_shader: VertexShader,
    pub pipeline_layout: vk::PipelineLayout,
    pub fragment_shader: FragmentShader,
    pub frames: u64,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Pipeline {
    pub fn new(
        engine: &Engine,
        vertex_shader_bytes: &[u8],
        vertex_shader_inputs: VertexShaderInputs,
        fragment_shader_bytes: &[u8],
        fragment_shader_inputs: FragmentShaderInputs,
    ) -> Self {
        let renderpass_attachments = [
            vk::AttachmentDescription {
                format: engine.surface_format.format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                ..Default::default()
            },
            vk::AttachmentDescription {
                format: vk::Format::D16_UNORM,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ..Default::default()
            },
        ];
        let color_attachment_refs = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        }];
        let subpass = vk::SubpassDescription::default()
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);
        let renderpass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&renderpass_attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);
        unsafe {
            let renderpass = engine
                .device
                .create_render_pass(&renderpass_create_info, None)
                .unwrap();
            let desc_layout_bindings = [
                vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
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
            let descriptor_set_layout = engine
                .device
                .create_descriptor_set_layout(&descriptor_info, None)
                .unwrap();
            let framebuffers: Vec<vk::Framebuffer> = engine
                .swapchain
                .present_image_views
                .iter()
                .map(|&present_image_view| {
                    let framebuffer_attachments =
                        [present_image_view, engine.depth_image.get_image_view()];
                    let frame_buffer_create_info = vk::FramebufferCreateInfo::default()
                        .render_pass(renderpass)
                        .attachments(&framebuffer_attachments)
                        .width(engine.swapchain.surface_resolution.width)
                        .height(engine.swapchain.surface_resolution.height)
                        .layers(1);
                    engine
                        .device
                        .create_framebuffer(&frame_buffer_create_info, None)
                        .unwrap()
                })
                .collect();
            let descriptor_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: MAX_FRAMES_IN_FLIGHT,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: MAX_FRAMES_IN_FLIGHT,
                },
            ];
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&descriptor_sizes)
                .max_sets(MAX_FRAMES_IN_FLIGHT);
            let descriptor_pool = engine
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap();
            let desc_set_layouts: Vec<vk::DescriptorSetLayout> = (0..MAX_FRAMES_IN_FLIGHT)
                .map(|_| descriptor_set_layout.clone())
                .collect();
            let desc_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&desc_set_layouts);
            let descriptor_sets = engine
                .device
                .allocate_descriptor_sets(&desc_alloc_info)
                .unwrap();
            let fragment_shader = FragmentShader::new(
                &engine,
                fragment_shader_bytes,
                fragment_shader_inputs,
                &descriptor_sets,
            );
            let vertex_shader = VertexShader::new(
                &engine,
                vertex_shader_bytes,
                vertex_shader_inputs,
                &descriptor_sets,
            );
            let layout_create_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&desc_set_layouts[..1]);
            let pipeline_layout = engine
                .device
                .create_pipeline_layout(&layout_create_info, None)
                .unwrap();
            let shader_entry_name = ffi::CStr::from_bytes_with_nul_unchecked(b"main\0");
            let shader_stage_create_infos = [
                vk::PipelineShaderStageCreateInfo {
                    module: vertex_shader.module,
                    p_name: shader_entry_name.as_ptr(),
                    stage: vk::ShaderStageFlags::VERTEX,
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo {
                    s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                    module: fragment_shader.module,
                    p_name: shader_entry_name.as_ptr(),
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    ..Default::default()
                },
            ];
            let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_attribute_descriptions(&vertex_shader.input_attribute_descriptions)
                .vertex_binding_descriptions(&vertex_shader.input_binding_descriptions);
            let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };
            let viewport_state_info = vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            };
            let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                polygon_mode: vk::PolygonMode::FILL,
                ..Default::default()
            };
            let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };
            let noop_stencil_state = vk::StencilOpState {
                fail_op: vk::StencilOp::KEEP,
                pass_op: vk::StencilOp::KEEP,
                depth_fail_op: vk::StencilOp::KEEP,
                compare_op: vk::CompareOp::ALWAYS,
                ..Default::default()
            };
            let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: 1,
                depth_write_enable: 1,
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                front: noop_stencil_state,
                back: noop_stencil_state,
                max_depth_bounds: 1.0,
                ..Default::default()
            };
            let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
                blend_enable: 0,
                src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ZERO,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
            }];
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op(vk::LogicOp::CLEAR)
                .attachments(&color_blend_attachment_states);
            let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);
            let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stage_create_infos)
                .vertex_input_state(&vertex_input_state_info)
                .input_assembly_state(&vertex_input_assembly_state_info)
                .viewport_state(&viewport_state_info)
                .rasterization_state(&rasterization_info)
                .multisample_state(&multisample_state_info)
                .depth_stencil_state(&depth_state_info)
                .color_blend_state(&color_blend_state)
                .dynamic_state(&dynamic_state_info)
                .layout(pipeline_layout)
                .render_pass(renderpass);
            let graphics_pipelines = engine
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[graphic_pipeline_info],
                    None,
                )
                .unwrap();
            Pipeline {
                renderpass,
                framebuffers: mem::ManuallyDrop::new(framebuffers),
                graphics_pipelines: mem::ManuallyDrop::new(graphics_pipelines),
                vertex_shader,
                fragment_shader,
                pipeline_layout,
                frames: 0,
                descriptor_set_layout,
                descriptor_pool,
                descriptor_sets,
            }
        }
    }

    pub fn new_framebuffers(&mut self, engine: &Engine) {
        unsafe {
            let framebuffers: Vec<vk::Framebuffer> = engine
                .swapchain
                .present_image_views
                .iter()
                .map(|&present_image_view| {
                    let framebuffer_attachments =
                        [present_image_view, engine.depth_image.get_image_view()];
                    let frame_buffer_create_info = vk::FramebufferCreateInfo::default()
                        .render_pass(self.renderpass)
                        .attachments(&framebuffer_attachments)
                        .width(engine.swapchain.surface_resolution.width)
                        .height(engine.swapchain.surface_resolution.height)
                        .layers(1);
                    engine
                        .device
                        .create_framebuffer(&frame_buffer_create_info, None)
                        .unwrap()
                })
                .collect();
            self.framebuffers = mem::ManuallyDrop::new(framebuffers)
        }
    }

    pub fn delete_framebuffers(&mut self, engine: &Engine) {
        unsafe {
            let framebuffers = mem::ManuallyDrop::take(&mut self.framebuffers);
            for framebuffer in framebuffers {
                engine.device.destroy_framebuffer(framebuffer, None);
            }
        }
    }
    pub fn delete(&mut self, engine: &Engine) {
        unsafe {
            engine.device.device_wait_idle().unwrap();
            let graphics_pipelines = mem::ManuallyDrop::take(&mut self.graphics_pipelines);
            for pipeline in graphics_pipelines {
                engine.device.destroy_pipeline(pipeline, None);
            }
            engine
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.fragment_shader.delete(&engine);

            self.vertex_shader.delete(&engine);
            engine
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            engine
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            let framebuffers = mem::ManuallyDrop::take(&mut self.framebuffers);
            for framebuffer in framebuffers {
                engine.device.destroy_framebuffer(framebuffer, None);
            }
            engine.device.destroy_render_pass(self.renderpass, None);
        }
    }
}
