use crate::engine::shader::{FragmentShader, Vertex, VertexShader};
use std::cell::OnceCell;
use std::rc::Rc;

pub struct Camera {
    pub view: cgmath::Matrix4<f32>,
    pub projection: cgmath::PerspectiveFov<f32>,
}

pub struct Mesh {
    vertex_shader: OnceCell<VertexShader>,
    fragment_shader: OnceCell<FragmentShader>,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self {
            vertex_shader: OnceCell::new(),
            fragment_shader: OnceCell::new(),
            vertices,
            indices,
        }
    }
}

pub struct Material {}

pub struct Object {
    pub mesh: Rc<Mesh>,
    pub model: cgmath::Decomposed<cgmath::Vector3<f32>, cgmath::Quaternion<f32>>,
    pub material: Material,
}

pub struct Scene {
    pub camera: Camera,
    pub meshes: Vec<Rc<Mesh>>,
    pub objects: Vec<Object>,
}
