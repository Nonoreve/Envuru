use cgmath::{One, Rotation3};

use crate::engine::shader::MvpUbo;
use crate::engine::{Engine, pipeline::Pipeline};

mod engine;

fn update(engine: &Engine, runtime_data: &mut Pipeline) -> MvpUbo {
    let model = cgmath::Decomposed {
        scale: 1.0,
        rot: cgmath::Quaternion::from_angle_z(cgmath::Deg(runtime_data.frames as f32 * 0.1)),
        disp: cgmath::vec3(0.0, 0.0, 0.0),
    };
    // let model = cgmath::Matrix4::one();
    let view = cgmath::Matrix4::look_at_rh(
        cgmath::point3(1.0, 1.0, 1.0),
        cgmath::point3(0.0, 0.0, 0.0),
        cgmath::vec3(0.0, 0.0, 1.0),
    );
    let projection = cgmath::PerspectiveFov {
        fovy: cgmath::Rad::from(cgmath::Deg(45.0)),
        aspect: runtime_data.aspect_ratio,
        near: 0.1,
        far: 10.0,
    };
    MvpUbo {
        model: cgmath::Matrix4::from(model),
        view,
        projection: cgmath::Matrix4::from(projection),
    }
}

fn main() {
    let engine_builder = engine::EngineBuilder::new(960, 540, "Envuru", update);
    engine_builder.start();
}
