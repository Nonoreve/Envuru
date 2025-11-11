use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::rc::Rc;

use ash::{util, vk};

use crate::engine::Engine;
use crate::engine::scene::{MvpUbo, Scene, ShaderSet};
use crate::engine::shader::Shader;

type ShaderMap = HashMap<Rc<ShaderSet>, Vec<Box<dyn Shader>>>;

pub struct Pipelines {
    pub renderpass: vk::RenderPass,
    pub framebuffers: ManuallyDrop<Vec<vk::Framebuffer>>,
    pub graphics_pipelines: ManuallyDrop<Vec<vk::Pipeline>>,
    pub pipeline_layouts: ManuallyDrop<HashMap<Rc<ShaderSet>, vk::PipelineLayout>>,
    pub frames: u64,
    descriptor_set_layouts: ManuallyDrop<Vec<vk::DescriptorSetLayout>>,
    descriptor_pool: vk::DescriptorPool,
    pub descriptors_sets: HashMap<Rc<ShaderSet>, vk::DescriptorSet>,
    pub aspect_ratio: f32,
    all_shaders: ManuallyDrop<ShaderMap>,
    pub shader_set_order: Vec<Rc<ShaderSet>>,
    pub window_dimensions: cgmath::Vector2<f64>,
}

impl Pipelines {
    pub fn new(engine: &Engine, scene: &mut Scene) -> Self {
        let renderpass_attachments = [
            vk::AttachmentDescription {
                format: engine.surface_data.surface_format.format,
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
            let descriptor_pool = scene.get_pool(engine);
            let descriptor_set_layouts = scene.get_descriptor_set_layouts(engine);
            let mut duplicated_set_layouts = HashMap::new();
            for (shader_set, descriptor_set_layout) in descriptor_set_layouts.iter() {
                let desc_set_layouts = [*descriptor_set_layout];
                duplicated_set_layouts.insert(shader_set.clone(), desc_set_layouts);
            }
            let mut desc_alloc_infos = HashMap::new();
            for (shader_set, duplicated_set_layout) in duplicated_set_layouts.iter() {
                let desc_alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(duplicated_set_layout);
                desc_alloc_infos.insert(shader_set.clone(), desc_alloc_info);
            }
            let push_constants = &[vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                size: size_of::<u32>() as u32,
                offset: 0,
            }];
            let mut pipeline_layouts = HashMap::new();
            for (shader_set, duplicated_set_layout) in duplicated_set_layouts.iter() {
                let layout_create_info = vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(duplicated_set_layout)
                    .push_constant_ranges(push_constants);
                let pipeline_layout = engine
                    .device
                    .create_pipeline_layout(&layout_create_info, None)
                    .unwrap();
                pipeline_layouts.insert(shader_set.clone(), pipeline_layout);
            }
            let viewport_state_info = vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
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
            let dynamic_state = [
                vk::DynamicState::VIEWPORT,
                vk::DynamicState::SCISSOR,
                vk::DynamicState::LINE_WIDTH,
            ];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);
            let mut all_shaders = HashMap::new();
            let mut all_vertex_inputs = HashMap::new();
            let mut descriptors_sets = HashMap::new();
            let mut shader_stages_create_infos = HashMap::new();
            let mut vertex_input_assembly_state_infos = HashMap::new();
            let mut rasterization_infos = HashMap::new();
            for (shader_set, tuple) in scene.load_resources(engine, &desc_alloc_infos) {
                let (
                    shaders,
                    vertex_inputs,
                    descriptor_set,
                    vertex_input_assembly_state_info,
                    rasterization_info,
                    shader_stage_create_infos,
                ) = tuple;
                all_vertex_inputs.insert(shader_set.clone(), vertex_inputs);
                descriptors_sets.insert(shader_set.clone(), descriptor_set);
                all_shaders.insert(shader_set.clone(), shaders);
                shader_stages_create_infos.insert(shader_set.clone(), shader_stage_create_infos);
                vertex_input_assembly_state_infos
                    .insert(shader_set.clone(), vertex_input_assembly_state_info);
                rasterization_infos.insert(shader_set.clone(), rasterization_info);
            }
            let mut vertex_input_state_infos = HashMap::new();
            for (shader_set, vertex_inputs) in all_vertex_inputs.iter() {
                let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_attribute_descriptions(vertex_inputs.input_attribute_desc.as_slice())
                    .vertex_binding_descriptions(vertex_inputs.input_binding_desc.as_slice());
                vertex_input_state_infos.insert(shader_set.clone(), vertex_input_state_info);
            }
            let mut graphic_pipeline_infos = Vec::new();
            let mut shader_set_order = Vec::new();
            for (shader_set, shader_stages_create_info) in shader_stages_create_infos.iter() {
                let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                    .stages(shader_stages_create_info)
                    .vertex_input_state(vertex_input_state_infos.get(shader_set).unwrap())
                    .input_assembly_state(
                        vertex_input_assembly_state_infos.get(shader_set).unwrap(),
                    )
                    .viewport_state(&viewport_state_info)
                    .rasterization_state(rasterization_infos.get(shader_set).unwrap())
                    .multisample_state(&multisample_state_info)
                    .depth_stencil_state(&depth_state_info)
                    .color_blend_state(&color_blend_state)
                    .dynamic_state(&dynamic_state_info)
                    .layout(*pipeline_layouts.get(shader_set).unwrap())
                    .render_pass(renderpass);
                graphic_pipeline_infos.push(graphic_pipeline_info);
                shader_set_order.push(shader_set.clone());
            }
            let graphics_pipelines = engine
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(), // TODO pipeline cache
                    graphic_pipeline_infos.as_slice(),
                    None,
                )
                .unwrap();
            let aspect_ratio = engine.swapchain.surface_resolution.width as f32
                / engine.swapchain.surface_resolution.height as f32;
            Pipelines {
                renderpass,
                framebuffers: ManuallyDrop::new(framebuffers),
                graphics_pipelines: ManuallyDrop::new(graphics_pipelines),
                pipeline_layouts: ManuallyDrop::new(pipeline_layouts),
                frames: 0,
                descriptor_set_layouts: ManuallyDrop::new(
                    descriptor_set_layouts.into_values().collect(),
                ),
                descriptor_pool,
                descriptors_sets,
                aspect_ratio,
                all_shaders: ManuallyDrop::new(all_shaders),
                shader_set_order,
                window_dimensions: cgmath::vec2(
                    engine.swapchain.surface_resolution.width as f64,
                    engine.swapchain.surface_resolution.height as f64,
                ),
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
            self.framebuffers = ManuallyDrop::new(framebuffers)
        }
    }

    pub fn update_uniforms(
        &self,
        pipeline_index: Rc<ShaderSet>,
        mvps: Vec<MvpUbo>,
        current_frame: usize,
    ) {
        unsafe {
            let shaders = &self.all_shaders[&pipeline_index];
            for shader in shaders {
                if let Some(uniform_mvp_buffer) = shader.get_uniform_data(current_frame) {
                    let mut alignment = util::Align::new(
                        uniform_mvp_buffer.data_ptr.unwrap(),
                        align_of::<f32>() as u64,
                        uniform_mvp_buffer.memory_requirements.size,
                    );
                    alignment.copy_from_slice(mvps.as_slice());
                    return;
                }
            }
        }
    }

    pub fn delete_framebuffers(&mut self, engine: &Engine) {
        unsafe {
            let framebuffers = ManuallyDrop::take(&mut self.framebuffers);
            for framebuffer in framebuffers {
                engine.device.destroy_framebuffer(framebuffer, None);
            }
        }
    }

    pub fn delete(&mut self, engine: &Engine, scene: &mut Scene) {
        unsafe {
            engine.device.device_wait_idle().unwrap();
            let graphics_pipelines = ManuallyDrop::take(&mut self.graphics_pipelines);
            for pipeline in graphics_pipelines {
                engine.device.destroy_pipeline(pipeline, None);
            }
            let pipeline_layouts = ManuallyDrop::take(&mut self.pipeline_layouts);
            for pipeline_layout in pipeline_layouts {
                engine
                    .device
                    .destroy_pipeline_layout(pipeline_layout.1, None);
            }
            let mut all_shaders = ManuallyDrop::take(&mut self.all_shaders);
            for shaders_tuple in all_shaders.iter_mut() {
                for shader in shaders_tuple.1.iter_mut() {
                    shader.delete(engine);
                }
            }
            for mesh in scene.meshes.iter() {
                mesh.delete(engine);
            }
            for material in scene.materials.iter() {
                material.delete(engine);
            }
            let descriptor_set_layouts = ManuallyDrop::take(&mut self.descriptor_set_layouts);
            for descriptor_set_layout in descriptor_set_layouts {
                engine
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            engine
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            let framebuffers = ManuallyDrop::take(&mut self.framebuffers);
            for framebuffer in framebuffers {
                engine.device.destroy_framebuffer(framebuffer, None);
            }
            engine.device.destroy_render_pass(self.renderpass, None);
        }
    }
}
