use std::cell::{OnceCell, RefCell};
use std::io::Cursor;
use std::rc::Rc;

use ash::vk::DescriptorSetAllocateInfo;
use ash::{util, vk};

use crate::engine::Engine;
use crate::engine::memory::{DataOrganization, IndexBuffer, Texture, VertexBuffer};
use crate::engine::shader::{FragmentShader, VertexShader};

pub struct Camera {
    pub position: cgmath::Point3<f32>,
    pub direction: cgmath::Point3<f32>,
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
    vertex_spvs: Vec<RefCell<Vec<u32>>>,
    fragment_spvs: Vec<RefCell<Vec<u32>>>,
    pub shader_sets: usize,
}

impl Scene {
    pub fn new(
        vertex_shader_bytes: Vec<&[u8]>,
        fragment_shader_bytes: Vec<&[u8]>,
        camera: Camera,
        meshes: Vec<Rc<Mesh>>,
        materials: Vec<Rc<Material>>,
        objects: Vec<Object>,
    ) -> Self {
        // TODO support on-the-fly shader compil
        let mut vertex_spvs = Vec::new();
        let mut fragment_spvs = Vec::new();
        let shader_sets = vertex_shader_bytes.len();
        for i in 0..shader_sets {
            let mut spv_data = Cursor::new(vertex_shader_bytes[i]);
            let vertex_spv = RefCell::new(util::read_spv(&mut spv_data).unwrap());
            let mut spv_data = Cursor::new(fragment_shader_bytes[i]);
            let fragment_spv = RefCell::new(util::read_spv(&mut spv_data).unwrap());
            vertex_spvs.push(vertex_spv);
            fragment_spvs.push(fragment_spv);
        }
        Self {
            camera,
            meshes,
            materials,
            objects,
            vertex_spvs,
            fragment_spvs,
            shader_sets,
        }
    }

    pub fn load_resources(
        &self,
        engine: &Engine,
        desc_alloc_info: &DescriptorSetAllocateInfo,
    ) -> Vec<(
        VertexShader,
        Vec<vk::VertexInputAttributeDescription>,
        Vec<vk::VertexInputBindingDescription>,
        FragmentShader,
        Vec<vk::DescriptorSet>,
    )> {
        for mesh in self.meshes.iter() {
            mesh.load_mesh(engine)
        }
        for material in self.materials.iter() {
            material.load_textures(engine);
        }
        let mut result = Vec::new();
        for i in 0..self.shader_sets {
            unsafe {
                let descriptor_sets = engine
                    .device
                    .allocate_descriptor_sets(desc_alloc_info)
                    .unwrap();
                let (vertex_shader, input_attribute_descriptions, input_binding_descriptions) =
                    VertexShader::new(
                        &engine,
                        &self.vertex_spvs[i].borrow(),
                        &descriptor_sets,
                        DataOrganization::ObjectMajor,
                    );
                let fragment_shader = FragmentShader::new(
                    &engine,
                    &self.fragment_spvs[i].borrow(),
                    &self.objects,
                    &descriptor_sets,
                );
                self.vertex_spvs[i].borrow_mut().clear();
                self.fragment_spvs[i].borrow_mut().clear();
                result.push((
                    vertex_shader,
                    input_attribute_descriptions,
                    input_binding_descriptions,
                    fragment_shader,
                    descriptor_sets,
                ))
            }
        }
        result
    }
}

#[derive(Copy, Clone)]
#[allow(dead_code)]
pub struct MvpUbo {
    pub model: cgmath::Matrix4<f32>,
    pub view: cgmath::Matrix4<f32>,
    pub projection: cgmath::Matrix4<f32>,
}

#[derive(Clone, Debug, Copy)]
pub struct Vertex {
    pub pos: cgmath::Vector4<f32>,
    pub uv: cgmath::Vector2<f32>,
}
