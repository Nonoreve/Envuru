use std::cell::{OnceCell, RefCell};
use std::f32::consts::PI;
use std::rc::Rc;

use ash::vk;
use cgmath::{Angle, Rotation3};

use crate::engine::Engine;
use crate::engine::memory::{DataOrganization, IndexBuffer, Texture, VertexBuffer};
use crate::engine::scene::{ShaderSet, Vertex};

const RIGHT_ANGLE: f32 = PI / 2.0;

pub struct Camera {
    position: cgmath::Point3<f32>,
    rotation: cgmath::Vector3<f32>,
    pub projection: cgmath::PerspectiveFov<f32>,
}

impl Camera {
    pub fn new(
        position: cgmath::Point3<f32>,
        rotation: cgmath::Vector3<f32>,
        projection: cgmath::PerspectiveFov<f32>,
    ) -> Self {
        let fliped = cgmath::point3(position.x, -position.y, position.z);
        Self {
            position: fliped,
            rotation,
            projection,
        }
    }

    #[allow(clippy::neg_multiply)]
    pub fn move_offset(&mut self, offset: &cgmath::Vector3<f32>) {
        if offset.z < 0.0002 || offset.z > 0.00021 {
            let rad = cgmath::Rad(self.rotation.y);
            self.position.x += rad.sin() * -1.0 * offset.z;
            self.position.z += rad.cos() * offset.z;
        }
        if offset.x < 0.0002 || offset.x > 0.00021 {
            let rad = cgmath::Rad(self.rotation.y - RIGHT_ANGLE);
            self.position.x += rad.sin() * -1.0 * offset.x;
            self.position.z += rad.cos() * offset.x;
        } // TODO special case for both axis
        self.position.y -= offset.y;
    }

    pub fn view_matrix(&self) -> cgmath::Matrix4<f32> {
        let rad = cgmath::Rad(self.rotation.x);
        let mut result = cgmath::Basis3::from_angle_x(rad);
        let rad = cgmath::Rad(self.rotation.y);
        result = result * cgmath::Basis3::from_angle_y(rad);
        let a: cgmath::Matrix3<f32> = result.into();
        let mut b: cgmath::Matrix4<f32> = a.into();
        b = b * cgmath::Matrix4::from_translation(cgmath::vec3(
            self.position.x,
            self.position.y,
            self.position.z,
        ));
        b
    }

    pub fn rotate(&mut self, offset: cgmath::Vector3<f32>) {
        self.rotation += offset
    }

    pub fn position(&self) -> cgmath::Point3<f32> {
        self.position
    }
}

pub struct Mesh {
    vertices: RefCell<Vec<Vertex>>,
    indices: RefCell<Vec<u32>>,
    vertex_buffer: OnceCell<VertexBuffer>,
    index_buffer: OnceCell<IndexBuffer>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        let vertices_cell = RefCell::new(vertices);
        let indices_cell = RefCell::new(indices);
        Self {
            vertices: vertices_cell,
            indices: indices_cell,
            vertex_buffer: OnceCell::new(),
            index_buffer: OnceCell::new(),
        }
    }

    pub fn load_mesh(&self, engine: &Engine) {
        let vertex_buffer = VertexBuffer::new(
            engine,
            &self.vertices.borrow(),
            DataOrganization::ObjectMajor,
        );
        let index_buffer = IndexBuffer::new(engine, &self.indices.borrow());
        self.vertex_buffer.set(vertex_buffer).unwrap();
        self.index_buffer.set(index_buffer).unwrap();
        self.vertices.borrow_mut().clear();
        self.indices.borrow_mut().clear();
    }

    pub fn bind_buffers(&self, engine: &Engine, current_frame: usize) {
        self.vertex_buffer
            .get()
            .unwrap()
            .bind(engine, current_frame);
        self.index_buffer.get().unwrap().bind(engine, current_frame);
    }

    pub fn get_index_count(&self) -> u32 {
        self.index_buffer.get().unwrap().index_count
    }

    pub fn delete(&self, engine: &Engine) {
        self.index_buffer.get().unwrap().delete(engine);
        self.vertex_buffer.get().unwrap().delete(engine);
    }
}

pub struct Material {
    textures: RefCell<Vec<Texture>>,
    images: Vec<image::DynamicImage>,
    pub global_index: RefCell<u32>,
}

impl Material {
    pub fn new(images: Vec<image::DynamicImage>) -> Self {
        Self {
            textures: RefCell::new(Vec::new()),
            images,
            global_index: RefCell::new(0),
        }
    }

    pub fn load_textures(&self, engine: &Engine) {
        for image in self.images.iter() {
            let image = image.to_rgba8();
            let texture = Texture::new(engine, &image);
            drop(image);
            self.textures.borrow_mut().push(texture)
        }
    }

    pub fn get_descriptor_image_info(
        &self,
        sampler: vk::Sampler,
        index: usize,
    ) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image_view: self.textures.borrow().get(index).unwrap().get_image_view(),
            sampler,
        }
    }

    pub fn delete(&self, engine: &Engine) {
        for texture in self.textures.borrow().iter() {
            texture.delete(engine);
        }
    }
}

pub struct Object {
    pub mesh: Rc<Mesh>,
    pub model: cgmath::Decomposed<cgmath::Vector3<f32>, cgmath::Quaternion<f32>>,
    pub material: Rc<Material>,
    pub shader_set: Rc<ShaderSet>,
}

impl Object {
    pub fn delete(&mut self) {
        let _ = &mut self.mesh;
        let _ = &mut self.material;
    }
}

pub struct Line {
    pub mesh: Rc<Mesh>,
    pub model: cgmath::Decomposed<cgmath::Vector3<f32>, cgmath::Quaternion<f32>>,
    pub width: f32,
    pub shader_set: Rc<ShaderSet>,
}
