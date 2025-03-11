use std::rc::Rc;

use cgmath::Rotation3;

use crate::engine::pipeline::Pipeline;
use crate::engine::scene::{Camera, Material, Mesh, Object, Scene};
use crate::engine::shader::Vertex;

mod engine;

fn update(scene: &mut Scene, runtime_data: &mut Pipeline) {
    // let model = cgmath::Decomposed {
    //     scale: 1.0,
    //     rot: cgmath::Quaternion::from_angle_z(cgmath::Deg(runtime_data.frames as f32 * 0.1)),
    //     disp: cgmath::vec3(0.0, 0.0, 0.0),
    // };
    // scene.objects[0].model = model;
    scene.objects[0].model.rot =
        cgmath::Quaternion::from_angle_z(cgmath::Deg(runtime_data.frames as f32 * 0.1));
}

fn main() {
    let mut engine_builder = engine::EngineBuilder::new(960, 540, "Envuru", update);
    let view = cgmath::Matrix4::look_at_rh(
        cgmath::point3(1.0, 1.0, 1.0),
        cgmath::point3(0.0, 0.0, 0.0),
        cgmath::vec3(0.0, 0.0, 1.0),
    );
    let projection = cgmath::PerspectiveFov {
        fovy: cgmath::Rad::from(cgmath::Deg(45.0)),
        aspect: 1.0,
        near: 0.1,
        far: 10.0,
    };
    let camera = Camera { view, projection };
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
    let rectangle_indices = [0u32, 1, 2, 2, 3, 0];
    let rectangle_mesh = Rc::new(Mesh::new(
        rectangle_vertices.into(),
        rectangle_indices.into(),
    ));
    let charlie = Rc::new(Material::new(vec![
        image::load_from_memory(include_bytes!("../resources/textures/charlie.jpg")).unwrap(),
    ]));
    let potoo = Rc::new(Material::new(vec![
        image::load_from_memory(include_bytes!("../resources/textures/potoo_asks.jpg")).unwrap(),
    ]));
    let demo_model = cgmath::Decomposed {
        scale: 0.2,
        rot: cgmath::Quaternion::from_angle_z(cgmath::Deg(0.1)),
        disp: cgmath::vec3(-0.5, 0.0, 0.0),
    };
    let demo = Object {
        mesh: Rc::clone(&rectangle_mesh),
        model: demo_model,
        material: Rc::clone(&charlie),
    };
    let demo_model2 = cgmath::Decomposed {
        scale: 0.4,
        rot: cgmath::Quaternion::from_angle_z(cgmath::Deg(0.1)),
        disp: cgmath::vec3(-0.2, 0.2, -0.2),
    };
    let demo2 = Object {
        mesh: Rc::clone(&rectangle_mesh),
        model: demo_model2,
        material: Rc::clone(&charlie),
    };
    let demo_model3 = cgmath::Decomposed {
        scale: 0.6,
        rot: cgmath::Quaternion::from_angle_z(cgmath::Deg(45.0)),
        disp: cgmath::vec3(0.2, 0.4, -0.4),
    };
    let demo3 = Object {
        mesh: Rc::clone(&rectangle_mesh),
        model: demo_model3,
        material: Rc::clone(&potoo),
    };
    let demo_model4 = cgmath::Decomposed {
        scale: 0.8,
        rot: cgmath::Quaternion::from_angle_z(cgmath::Deg(78.0)),
        disp: cgmath::vec3(0.5, 0.6, -0.6),
    };
    let demo4 = Object {
        mesh: Rc::clone(&rectangle_mesh),
        model: demo_model4,
        material: Rc::clone(&charlie),
    };
    let meshes = vec![rectangle_mesh];
    let start_scene = Scene::new(
        include_bytes!("../target/vert.spv"),
        include_bytes!("../target/frag.spv"),
        camera,
        meshes,
        vec![charlie, potoo],
        vec![demo, demo2, demo3, demo4],
    );
    let scene_handle = engine_builder.register_scene(start_scene);
    engine_builder.start(scene_handle);
}
