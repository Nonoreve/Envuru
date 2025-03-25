use std::cell::{OnceCell, RefCell};
use std::io::Cursor;
use std::rc::Rc;

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
    pub shader_set: Rc<ShaderSet>,
}

pub struct ShaderSet {
    vertex_spv: RefCell<Vec<u32>>,
    fragment_spv: RefCell<Vec<u32>>,
    topology: OnceCell<vk::PrimitiveTopology>,
    index: OnceCell<usize>,
}

impl ShaderSet {
    pub fn new(vertex_shader_bytes: &[u8], fragment_shader_bytes: &[u8]) -> Self {
        // TODO support on-the-fly shader compil
        let mut spv_data = Cursor::new(vertex_shader_bytes);
        let vertex_spv = RefCell::new(util::read_spv(&mut spv_data).unwrap());
        let mut spv_data = Cursor::new(fragment_shader_bytes);
        let fragment_spv = RefCell::new(util::read_spv(&mut spv_data).unwrap());
        Self {
            vertex_spv,
            fragment_spv,
            topology: OnceCell::new(),
            index: OnceCell::new(),
        }
    }

    pub fn get_index(&self) -> usize {
        self.index.get().unwrap().clone()
    }
}

pub struct Scene {
    pub camera: Camera,
    pub lines: Vec<Line>,
    pub objects: Vec<Object>,
    pub meshes: Vec<Rc<Mesh>>,
    pub materials: Vec<Rc<Material>>,
    shader_sets: Vec<Rc<ShaderSet>>,
}

impl Scene {
    pub fn new(
        camera: Camera,
        mut lines: Vec<Line>,
        mut objects: Vec<Object>,
        meshes: Vec<Rc<Mesh>>,
        materials: Vec<Rc<Material>>,
        shader_sets: Vec<Rc<ShaderSet>>,
    ) -> Self {// TODO check all meshes and materials used in lines and objects are given in the dedicated array
        // or even better, create the dedicated arrays by collecting them from objs and lines
        for object in objects.iter_mut() {
            if object.shader_set.topology.get().is_none() {
                object
                    .shader_set
                    .topology
                    .set(vk::PrimitiveTopology::TRIANGLE_LIST)
                    .unwrap()
            }
        }
        for line in lines.iter_mut() {
            if line.shader_set.topology.get().is_none() {
                line.shader_set
                    .topology
                    .set(vk::PrimitiveTopology::LINE_LIST)
                    .unwrap();
            }
        }
        let mut used_shader_sets = Vec::new();
        for (i, shader_set) in shader_sets.into_iter().enumerate() {
            if shader_set.topology.get().is_some() {
                used_shader_sets.push(shader_set);
            } else {
                println!("Shader set {i} not used");
            }
        }
        for (i, shader_set) in used_shader_sets.iter().enumerate() {
            shader_set.index.set(i).unwrap();
        }
        Self {
            camera,
            lines,
            objects,
            meshes,
            materials,
            shader_sets: used_shader_sets,
        }
    }

    pub fn load_resources(
        &mut self,
        engine: &Engine,
        object_desc_alloc_info: &vk::DescriptorSetAllocateInfo,
        line_desc_alloc_info: &vk::DescriptorSetAllocateInfo,
    ) -> Vec<(
        VertexShader,
        Vec<vk::VertexInputAttributeDescription>,
        Vec<vk::VertexInputBindingDescription>,
        FragmentShader,
        Vec<vk::DescriptorSet>,
        vk::PipelineInputAssemblyStateCreateInfo,
    )> {
        for mesh in self.meshes.iter() {
            mesh.load_mesh(engine)
        }
        for material in self.materials.iter() {
            material.load_textures(engine);
        }
        let mut result = Vec::new();
        for shader_set in self.shader_sets.iter() {
            let topology = *shader_set.topology.get().unwrap();
            unsafe {
                let descriptor_sets = if topology == vk::PrimitiveTopology::TRIANGLE_LIST {
                    engine
                        .device
                        .allocate_descriptor_sets(object_desc_alloc_info)
                        .unwrap()
                } else {
                    assert_eq!(topology, vk::PrimitiveTopology::LINE_LIST);
                    engine
                        .device
                        .allocate_descriptor_sets(line_desc_alloc_info)
                        .unwrap()
                };
                let (vertex_shader, input_attribute_descriptions, input_binding_descriptions) =
                    VertexShader::new(
                        &engine,
                        &shader_set.vertex_spv.borrow(),
                        &descriptor_sets,
                        DataOrganization::ObjectMajor,
                    );
                let fragment_shader = FragmentShader::new(
                    &engine,
                    &shader_set.fragment_spv.borrow(),
                    &self.objects,
                    &descriptor_sets,
                    topology,
                );
                shader_set.vertex_spv.borrow_mut().clear();
                shader_set.fragment_spv.borrow_mut().clear();
                let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
                    topology,
                    ..Default::default()
                };
                result.push((
                    vertex_shader,
                    input_attribute_descriptions,
                    input_binding_descriptions,
                    fragment_shader,
                    descriptor_sets,
                    vertex_input_assembly_state_info,
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
