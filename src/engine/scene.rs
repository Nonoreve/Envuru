use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::rc::Rc;

use ash::vk::DescriptorSetAllocateInfo;
use ash::{util, vk};
use cgmath::{Angle, Rotation3};

use crate::engine::memory::{DataOrganization, IndexBuffer, Texture, VertexBuffer};
use crate::engine::shader::{FragmentShader, VertexShader};
use crate::engine::{Engine, ShaderInterface};

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
        Self {
            position,
            rotation,
            projection,
        }
    }

    pub fn move_offset(&mut self, offset: &cgmath::Vector3<f32>) {
        if offset.z > 0.0 {
            let rad = cgmath::Rad(self.rotation.y);
            self.position.x += rad.sin() * -1.0 * offset.z;
            self.position.z += rad.cos() * offset.z;
        }
        if offset.x > 0.0 {
            let rad = cgmath::Rad(self.rotation.y - 90.0);
            self.position.x += rad.sin() * -1.0 * offset.x;
            self.position.z += rad.cos() * offset.x;
        }
        self.position.y += offset.y;
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
    pub width: f32,
    pub shader_set: Rc<ShaderSet>,
}

#[derive(Eq, PartialEq)]
pub struct ShaderSet {
    vertex_spv: RefCell<Vec<u32>>,
    fragment_spv: RefCell<Vec<u32>>,
    topology: OnceCell<vk::PrimitiveTopology>,
    index: OnceCell<usize>,
    vertex_descriptors: Vec<vk::DescriptorType>,
    fragment_descriptors: Vec<vk::DescriptorType>,
}

impl ShaderSet {
    pub fn new(
        vertex_shader_bytes: &[u8],
        vertex_shader_interface: Vec<ShaderInterface>,
        fragment_shader_bytes: &[u8],
        fragment_shader_interface: Vec<ShaderInterface>,
    ) -> Self {
        // TODO support on-the-fly shader compil and read shader contents to fill descriptors
        let mut spv_data = Cursor::new(vertex_shader_bytes);
        let vertex_spv = RefCell::new(util::read_spv(&mut spv_data).unwrap());
        let mut spv_data = Cursor::new(fragment_shader_bytes);
        let fragment_spv = RefCell::new(util::read_spv(&mut spv_data).unwrap());
        let vertex_descriptors = vertex_shader_interface
            .iter()
            .map(|shader_interface| match shader_interface {
                ShaderInterface::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                ShaderInterface::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            })
            .collect();
        let fragment_descriptors = fragment_shader_interface
            .iter()
            .map(|shader_interface| match shader_interface {
                ShaderInterface::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                ShaderInterface::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            })
            .collect();
        Self {
            vertex_spv,
            fragment_spv,
            topology: OnceCell::new(),
            index: OnceCell::new(),
            vertex_descriptors,
            fragment_descriptors,
        }
    }

    pub fn get_descriptor_set_layout(&self, engine: &Engine) -> vk::DescriptorSetLayout {
        let mut desc_layout_bindings = Vec::new();
        for (i, vertex_descriptor) in self.vertex_descriptors.iter().enumerate() {
            desc_layout_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: i as u32,
                descriptor_type: vertex_descriptor.clone(),
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            })
        }
        for (i, fragment_descriptor) in self.fragment_descriptors.iter().enumerate() {
            desc_layout_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: (self.vertex_descriptors.len() + i) as u32,
                descriptor_type: fragment_descriptor.clone(),
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            })
        }
        let descriptor_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&desc_layout_bindings);
        unsafe {
            engine
                .device
                .create_descriptor_set_layout(&descriptor_info, None)
                .unwrap()
        }
    }

    pub fn get_index(&self) -> usize {
        self.index.get().unwrap().clone()
    }
}

impl Hash for ShaderSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex_descriptors.hash(state);
        self.fragment_descriptors.hash(state);
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
    ) -> Self {
        // TODO check all meshes and materials used in lines and objects are given in the dedicated array
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
        desc_alloc_infos: &HashMap<Rc<ShaderSet>, DescriptorSetAllocateInfo>,
    ) -> HashMap<
        Rc<ShaderSet>,
        (
            VertexShader,
            Vec<vk::VertexInputAttributeDescription>,
            Vec<vk::VertexInputBindingDescription>,
            FragmentShader,
            Vec<vk::DescriptorSet>,
            vk::PipelineInputAssemblyStateCreateInfo,
            vk::PipelineRasterizationStateCreateInfo,
        ),
    > {
        for mesh in self.meshes.iter() {
            mesh.load_mesh(engine)
        }
        for material in self.materials.iter() {
            material.load_textures(engine);
        }
        let mut result = HashMap::new();
        for shader_set in self.shader_sets.iter() {
            let topology = shader_set.topology.get().unwrap();
            unsafe {
                let descriptor_sets = engine
                    .device
                    .allocate_descriptor_sets(desc_alloc_infos.get(shader_set).unwrap())
                    .unwrap();
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
                    &shader_set.fragment_descriptors,
                );
                shader_set.vertex_spv.borrow_mut().clear();
                shader_set.fragment_spv.borrow_mut().clear();
                let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
                    topology: topology.clone(),
                    ..Default::default()
                };
                let rasterization_info = match topology {
                    &vk::PrimitiveTopology::LINE_LIST => vk::PipelineRasterizationStateCreateInfo {
                        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                        line_width: 10.0,
                        polygon_mode: vk::PolygonMode::LINE,
                        ..Default::default()
                    },
                    &vk::PrimitiveTopology::POINT_LIST => {
                        vk::PipelineRasterizationStateCreateInfo {
                            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                            line_width: 2.0,
                            polygon_mode: vk::PolygonMode::POINT,
                            ..Default::default()
                        }
                    }
                    _ => vk::PipelineRasterizationStateCreateInfo {
                        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                        line_width: 1.0,
                        polygon_mode: vk::PolygonMode::FILL,
                        ..Default::default()
                    },
                };
                result.insert(
                    shader_set.clone(),
                    (
                        vertex_shader,
                        input_attribute_descriptions,
                        input_binding_descriptions,
                        fragment_shader,
                        descriptor_sets,
                        vertex_input_assembly_state_info,
                        rasterization_info,
                    ),
                );
            }
        }
        result
    }

    pub fn get_descriptor_set_layouts(
        &self,
        engine: &Engine,
    ) -> HashMap<Rc<ShaderSet>, vk::DescriptorSetLayout> {
        let mut descriptor_set_layouts = HashMap::new();
        for shader_set in self.shader_sets.iter() {
            let result = descriptor_set_layouts.insert(
                shader_set.clone(),
                shader_set.get_descriptor_set_layout(engine),
            );
            match result {
                Some(_) => panic!("Duplicate entry inserted into hash"),
                None => (),
            }
        }
        descriptor_set_layouts
    }

    pub fn get_shader_set_users(&self, shader_set: &Rc<ShaderSet>) -> u32 {
        let mut result = 0;
        for object in self.objects.iter() {
            if &object.shader_set.index.get().unwrap() == &shader_set.index.get().unwrap() {
                result += 1
            }
        }
        for line in self.lines.iter() {
            if &line.shader_set.index.get().unwrap() == &shader_set.index.get().unwrap() {
                result += 1
            }
        }
        result
    }

    pub fn get_pool_sizes(&self) -> Vec<vk::DescriptorPoolSize> {
        let mut types_map: HashMap<vk::DescriptorType, vk::DescriptorPoolSize> = HashMap::new();
        for shader_set in self
            .objects
            .iter()
            .map(|x| x.shader_set.clone())
            .chain(self.lines.iter().map(|x1| x1.shader_set.clone()))
        {
            for descriptor in shader_set
                .clone()
                .vertex_descriptors
                .iter()
                .chain(shader_set.clone().fragment_descriptors.iter())
            {
                if types_map.contains_key(descriptor) {
                    types_map.get_mut(descriptor).unwrap().descriptor_count += 1;
                } else {
                    types_map.insert(
                        descriptor.clone(),
                        vk::DescriptorPoolSize {
                            ty: descriptor.clone(),
                            descriptor_count: 1,
                        },
                    );
                }
            }
        }
        types_map.into_values().collect()
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
