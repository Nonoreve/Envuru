use std::cell::{OnceCell, RefCell};
use std::io::Cursor;
use std::rc::Rc;

use ash::vk::DescriptorSet;
use ash::{util, vk};

use crate::engine::Engine;
use crate::engine::memory::{DataOrganization, IndexBuffer, Texture, VertexBuffer};
use crate::engine::shader::{FragmentShader, Vertex, VertexShader};

pub struct Camera {
    pub view: cgmath::Matrix4<f32>,
    pub projection: cgmath::PerspectiveFov<f32>,
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
}

impl Material {
    pub fn new(images: Vec<image::DynamicImage>) -> Self {
        Self {
            textures: RefCell::new(Vec::new()),
            images,
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
}

impl Object {
    pub fn delete(&mut self) {
        let _ = &mut self.mesh;
        let _ = &mut self.material;
    }
}

pub struct Scene {
    pub camera: Camera,
    pub meshes: Vec<Rc<Mesh>>,
    pub materials: Vec<Rc<Material>>,
    pub objects: Vec<Object>,
    vertex_spv: RefCell<Vec<u32>>,
    fragment_spv: RefCell<Vec<u32>>,
}

impl Scene {
    pub fn new(
        vertex_shader_bytes: &[u8],
        framgent_shader_bytes: &[u8],
        camera: Camera,
        meshes: Vec<Rc<Mesh>>,
        materials: Vec<Rc<Material>>,
        objects: Vec<Object>,
    ) -> Self {
        // TODO support on-the-fly shader compil
        let mut spv_data = Cursor::new(vertex_shader_bytes);
        let vertex_spv = RefCell::new(util::read_spv(&mut spv_data).unwrap());
        let mut spv_data = Cursor::new(framgent_shader_bytes);
        let fragment_spv = RefCell::new(util::read_spv(&mut spv_data).unwrap());
        Self {
            camera,
            meshes,
            materials,
            objects,
            vertex_spv,
            fragment_spv,
        }
    }

    pub fn load_resources(
        &self,
        engine: &Engine,
        descriptor_sets: &Vec<DescriptorSet>,
    ) -> (VertexShader, FragmentShader) {
        for mesh in self.meshes.iter() {
            mesh.load_mesh(engine)
        }
        for material in self.materials.iter() {
            material.load_textures(engine);
        }
        let vertex_shader = VertexShader::new(
            &engine,
            &self.vertex_spv.borrow(),
            descriptor_sets,
            DataOrganization::ObjectMajor,
        );
        let fragment_shader = FragmentShader::new(
            &engine,
            &self.fragment_spv.borrow(),
            &self.objects,
            descriptor_sets,
        );
        self.vertex_spv.borrow_mut().clear();
        self.fragment_spv.borrow_mut().clear();
        (vertex_shader, fragment_shader)
    }
}
