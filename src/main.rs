use ash::vk;

use crate::engine::{Engine, pipeline::Pipeline};

mod engine;

fn draw_frame(engine: &Engine, runtime_data: &mut Pipeline) {
    unsafe {
        let current_frame = (runtime_data.frames % engine::MAX_FRAMES_IN_FLIGHT as u64) as usize;
        engine.device.device_wait_idle().unwrap();
        let present_index = engine.swapchain.next_image(engine, current_frame);
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(runtime_data.renderpass)
            .framebuffer(runtime_data.framebuffers[present_index as usize])
            .render_area(engine.swapchain.surface_resolution.into())
            .clear_values(&clear_values);

        Engine::record_submit_commandbuffer(
            &engine.device,
            engine.draw_command_buffers[current_frame],
            engine.draw_commands_reuse_fence[current_frame],
            engine.present_queue,
            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            &[engine.present_complete_semaphores[current_frame]],
            &[engine.rendering_complete_semaphores[current_frame]],
            |device, draw_command_buffer| {
                device.cmd_begin_render_pass(
                    draw_command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_descriptor_sets(
                    draw_command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    runtime_data.pipeline_layout,
                    0,
                    &runtime_data.fragment_shader.descriptor_sets[..],
                    &[],
                );
                device.cmd_bind_pipeline(
                    draw_command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    runtime_data.graphics_pipelines[0], // TODO choose instead of asert first one
                );
                device.cmd_set_viewport(draw_command_buffer, 0, &runtime_data.viewports);
                device.cmd_set_scissor(draw_command_buffer, 0, &runtime_data.scissors);
                runtime_data
                    .vertex_shader
                    .vertex_buffer
                    .bind(engine, current_frame);
                runtime_data
                    .vertex_shader
                    .index_buffer
                    .bind(engine, current_frame);
                device.cmd_draw_indexed(
                    draw_command_buffer,
                    runtime_data.vertex_shader.index_buffer.index_count(),
                    1,
                    0,
                    0,
                    1,
                );
                device.cmd_end_render_pass(draw_command_buffer);
            },
        );
        engine
            .swapchain
            .present(engine, current_frame, present_index);
        runtime_data.frames += 1 % u64::MAX;
    }
}

fn main() {
    let mut engine_builder = engine::EngineBuilder::new(960, 540, "Envuru", draw_frame);
    engine_builder.start();
}
