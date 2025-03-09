use std::cell::{OnceCell, RefCell};
use std::io::Cursor;
use std::rc::Rc;

use ash::{util, vk};

use crate::engine::memory::{DataOrganization, Texture};
use crate::engine::shader::{FragmentShader, MvpUbo, Vertex, VertexShader};
use crate::engine::{Engine, MAX_FRAMES_IN_FLIGHT};

pub struct Camera {
    pub view: cgmath::Matrix4<f32>,
    pub projection: cgmath::PerspectiveFov<f32>,
}

pub struct Mesh {
    vertices: RefCell<Vec<Vertex>>,
    indices: RefCell<Vec<u32>>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        let vertices_cell = RefCell::new(vertices);
        let indices_cell = RefCell::new(indices);
        Self {
            vertices: vertices_cell,
            indices: indices_cell,
        }
    }
}

pub struct Material {
    vertex_shader: OnceCell<VertexShader>,
    fragment_shader: OnceCell<FragmentShader>,
    vertex_spv: RefCell<Vec<u32>>,
    fragment_spv: RefCell<Vec<u32>>,
    textures: RefCell<Vec<Texture>>,
    images: Vec<image::DynamicImage>,
}

impl Material {
    pub fn new(
        vertex_shader_bytes: &[u8],
        framgent_shader_bytes: &[u8],
        images: Vec<image::DynamicImage>,
    ) -> Self {
        // TODO support on-the-fly shader compil
        let mut spv_data = Cursor::new(vertex_shader_bytes);
        let vertex_spv = RefCell::new(util::read_spv(&mut spv_data).unwrap());
        let mut spv_data = Cursor::new(framgent_shader_bytes);
        let fragment_spv = RefCell::new(util::read_spv(&mut spv_data).unwrap());
        Self {
            vertex_shader: OnceCell::new(),
            fragment_shader: OnceCell::new(),
            vertex_spv,
            fragment_spv,
            textures: RefCell::new(Vec::new()),
            images,
        }
    }

    pub fn load_shaders(
        &self,
        engine: &Engine,
        mesh: &Mesh,
        descriptor_sets: &Vec<vk::DescriptorSet>,
    ) {
        assert!(
            mesh.vertices.borrow().len() > 0,
            "Shaders already loaded for this mesh (vertices empty)"
        );
        let vertex_shader = VertexShader::new(
            &engine,
            &self.vertex_spv.borrow(),
            mesh.vertices.borrow().as_slice(),
            mesh.indices.borrow().as_slice(),
            descriptor_sets,
            DataOrganization::ObjectMajor,
        );
        let fragment_shader =
            FragmentShader::new(&engine, &self.fragment_spv.borrow(), self, descriptor_sets);
        self.vertex_shader.set(vertex_shader).unwrap();
        self.fragment_shader.set(fragment_shader).unwrap();
        mesh.vertices.borrow_mut().clear();
        mesh.indices.borrow_mut().clear();
        self.vertex_spv.borrow_mut().clear();
        self.fragment_spv.borrow_mut().clear();
    }

    pub fn load_textures(&self, engine: &Engine) {
        for image in self.images.iter() {
            let image = image.to_rgba8();
            let texture = Texture::new(engine, &image);
            drop(image);
            self.textures.borrow_mut().push(texture)
        }
    }

    pub fn get_descriptor_image_infos(
        &self,
        sampler: vk::Sampler,
        index: usize,
    ) -> Vec<vk::DescriptorImageInfo> {
        (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image_view: self.textures.borrow().get(index).unwrap().get_image_view(),
                sampler,
            })
            .collect()
    }

    pub fn update_uniforms(&self, mvp: MvpUbo, current_frame: usize) {
        unsafe {
            let mut alignment = util::Align::new(
                self.vertex_shader.get().unwrap().uniform_mvp_buffers[current_frame]
                    .data_ptr
                    .unwrap(),
                align_of::<f32>() as u64,
                self.vertex_shader.get().unwrap().uniform_mvp_buffers[current_frame]
                    .memory_requirements
                    .size,
            );
            alignment.copy_from_slice(&[mvp]);
        }
    }

    pub fn bind_buffers(&self, engine: &Engine, current_frame: usize) {
        self.vertex_shader
            .get()
            .unwrap()
            .vertex_buffer
            .bind(engine, current_frame);
        self.vertex_shader
            .get()
            .unwrap()
            .index_buffer
            .bind(engine, current_frame);
    }

    pub fn get_modules(&self) -> (vk::ShaderModule, vk::ShaderModule) {
        (
            self.vertex_shader.get().unwrap().module,
            self.fragment_shader.get().unwrap().module,
        )
    }

    pub fn get_vertex_input_state_info(&self) -> vk::PipelineVertexInputStateCreateInfo {
        self.vertex_shader
            .get()
            .unwrap()
            .get_vertex_input_state_info()
    }

    pub fn get_index_count(&self) -> u32 {
        self.vertex_shader.get().unwrap().index_buffer.index_count
    }

    pub fn delete(&self, engine: &Engine) {
        self.vertex_shader.get().unwrap().delete(&engine);
        self.fragment_shader.get().unwrap().delete(&engine);
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
}
