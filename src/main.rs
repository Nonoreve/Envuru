use std::rc::Rc;

use cgmath::Rotation3;
use winit::{event, keyboard};

use engine::scene::Vertex;

use crate::engine::ShaderInterface;
use crate::engine::controller::{Controller, KeyBind, MouseOrKey};
use crate::engine::pipeline::Pipelines;
use crate::engine::scene::{Camera, Line, Material, Mesh, Object, Scene, ShaderSet};

mod engine;

enum KeyActions {
    Forward,
    OrientCamera,
}

fn update(
    scene: &mut Scene,
    runtime_data: &mut Pipelines,
    controller: &mut Controller,
    window_position: &cgmath::Vector2<f64>,
) {
    scene.objects[0].model.rot =
        cgmath::Quaternion::from_angle_z(cgmath::Deg(runtime_data.frames as f32 * 0.1));
    if controller.is_hold(KeyActions::Forward as usize) {
        let offset = cgmath::vec3(1.0, 1.0, 1.0);
        scene.camera.move_offset(&offset)
    }
    if controller.is_hold(KeyActions::OrientCamera as usize) {
        let direction = controller.mouse_direction(runtime_data.frames);
        scene.camera.rotate(cgmath::vec3(
            direction.y as f32 * 0.0002,
            direction.x as f32 * 0.0002,
            0.0,
        ));
        println!(
            "mouse_direction={direction:?} \ncamera.direction={:?}",
            scene.camera.view_matrix()
        );
    }
}

fn main() {
    let width = 69.0 * 15.0;
    let height = 42.0 * 15.0;
    let mut controller = Some(Controller::new(width, height));
    controller.as_mut().unwrap().register_bind_action(
        KeyActions::Forward as usize,
        KeyBind::new(MouseOrKey::Key(keyboard::PhysicalKey::Code(
            keyboard::KeyCode::KeyQ,
        ))),
    );
    controller.as_mut().unwrap().register_bind_action(
        KeyActions::OrientCamera as usize,
        KeyBind::new(MouseOrKey::Mouse(event::MouseButton::Left)),
    );
    let mut engine_builder =
        engine::EngineBuilder::new(width as u32, height as u32, "Envuru", update, controller);
    let projection = cgmath::PerspectiveFov {
        fovy: cgmath::Rad::from(cgmath::Deg(45.0)),
        aspect: 1.0,
        near: 0.1,
        far: 10.0,
    };
    let camera = Camera::new(
        cgmath::point3(1.0, 1.0, 1.0),
        cgmath::vec3(0.0, 0.0, 0.0),
        projection,
    );
    let rectangle_vertices = [
        Vertex {
            pos: cgmath::vec4(-1.0, -1.0, 0.0, 1.0),
            uv: cgmath::vec2(0.0, 0.0),
        },
        Vertex {
            pos: cgmath::vec4(-1.0, 1.0, 0.0, 1.0),
            uv: cgmath::vec2(0.0, 1.0),
        },
        Vertex {
            pos: cgmath::vec4(1.0, 1.0, 0.0, 1.0),
            uv: cgmath::vec2(1.0, 1.0),
        },
        Vertex {
            pos: cgmath::vec4(1.0, -1.0, 0.0, 1.0),
            uv: cgmath::vec2(1.0, 0.0),
        },
    ];
    let rectangle_vertices2 = [
        Vertex {
            pos: cgmath::vec4(-1.0, -1.0, 0.0, 1.0),
            uv: cgmath::vec2(0.0, 0.0),
        },
        Vertex {
            pos: cgmath::vec4(-1.0, 1.0, 0.0, 1.0),
            uv: cgmath::vec2(0.0, 2.0),
        },
        Vertex {
            pos: cgmath::vec4(1.0, 1.0, 0.0, 1.0),
            uv: cgmath::vec2(2.0, 2.0),
        },
        Vertex {
            pos: cgmath::vec4(1.0, -1.0, 0.0, 1.0),
            uv: cgmath::vec2(2.0, 0.0),
        },
    ];
    let rectangle_indices = [0u32, 1, 2, 2, 3, 0];
    let rectangle_mesh = Rc::new(Mesh::new(
        rectangle_vertices.into(),
        rectangle_indices.into(),
    ));
    let rectangle_mesh2 = Rc::new(Mesh::new(
        rectangle_vertices2.into(),
        rectangle_indices.into(),
    ));
    let charlie = Rc::new(Material::new(vec![
        image::load_from_memory(include_bytes!("../resources/textures/charlie.jpg")).unwrap(),
    ]));
    let potoo = Rc::new(Material::new(vec![
        image::load_from_memory(include_bytes!("../resources/textures/potoo_asks.jpg")).unwrap(),
    ]));
    let object_shader_set = Rc::new(ShaderSet::new(
        include_bytes!("../target/object_vert.spv"),
        vec![ShaderInterface::UniformBuffer],
        include_bytes!("../target/object_frag.spv"),
        vec![ShaderInterface::CombinedImageSampler],
    ));
    let line_shader_set = Rc::new(ShaderSet::new(
        include_bytes!("../target/line_vert.spv"),
        vec![ShaderInterface::UniformBuffer],
        include_bytes!("../target/line_frag.spv"),
        Vec::default(),
    ));
    let demo_model = cgmath::Decomposed {
        scale: 1.0,
        rot: cgmath::Quaternion::from_angle_z(cgmath::Deg(0.1)),
        disp: cgmath::vec3(0.0, 0.0, 0.0),
    };
    let demo = Object {
        mesh: Rc::clone(&rectangle_mesh),
        model: demo_model,
        material: Rc::clone(&charlie),
        shader_set: Rc::clone(&object_shader_set),
    };
    let demo_model2 = cgmath::Decomposed {
        scale: 1.0,
        rot: cgmath::Quaternion::from_angle_z(cgmath::Deg(0.1)),
        disp: cgmath::vec3(1.0, 0.0, 0.0),
    };
    let demo2 = Object {
        mesh: Rc::clone(&rectangle_mesh2),
        model: demo_model2,
        material: Rc::clone(&charlie),
        shader_set: Rc::clone(&object_shader_set),
    };
    let demo_model3 = cgmath::Decomposed {
        scale: 1.0,
        rot: cgmath::Quaternion::from_angle_z(cgmath::Deg(45.0)),
        disp: cgmath::vec3(0.0, 1.0, 0.0),
    };
    let demo3 = Object {
        mesh: Rc::clone(&rectangle_mesh),
        model: demo_model3,
        material: Rc::clone(&potoo),
        shader_set: Rc::clone(&object_shader_set),
    };
    let line_vertices = [
        Vertex {
            pos: cgmath::vec4(-1.0, -1.0, 0.0, 1.0),
            uv: cgmath::vec2(0.0, 0.0),
        },
        Vertex {
            pos: cgmath::vec4(-1.0, 1.0, 0.0, 1.0),
            uv: cgmath::vec2(0.0, 1.0),
        },
    ];
    let line_indices = [0u32, 1];
    let line_mesh = Rc::new(Mesh::new(line_vertices.into(), line_indices.into()));
    let line_model = cgmath::Decomposed {
        scale: 1.0,
        rot: cgmath::Quaternion::from_angle_z(cgmath::Deg(78.0)),
        disp: cgmath::vec3(0.0, 0.0, 0.0),
    };
    let line = Line {
        mesh: Rc::clone(&line_mesh),
        model: line_model,
        width: 4.0,
        shader_set: Rc::clone(&line_shader_set),
    };
    let start_scene = Scene::new(
        camera,
        vec![line],
        vec![demo, demo2, demo3],
        vec![rectangle_mesh, rectangle_mesh2, line_mesh],
        vec![charlie, potoo],
        vec![object_shader_set, line_shader_set],
    );
    let scene_handle = engine_builder.register_scene(start_scene);
    engine_builder.start(scene_handle);
}
