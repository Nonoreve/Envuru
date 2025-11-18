#![allow(clippy::mutable_key_type)]

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::HashMap;
use std::ffi::CStr;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::rc::Rc;

use ash::vk::DescriptorSetAllocateInfo;
use ash::{util, vk};

use crate::engine::api_resources::{Camera, Line, Material, Mesh, Object};
use crate::engine::memory::DataOrganization;
use crate::engine::shader::Shader;
use crate::engine::shader::{FragmentShader, GeometryShader, VertexInputs, VertexShader};
use crate::engine::{Engine, MAX_FRAMES_IN_FLIGHT, MeshTopology, ShaderInterface};

const SHADER_ENTRY_NAME: &CStr = c"main";

#[derive(Eq, PartialEq, Debug)]
pub struct ShaderSet {
    vertex_spv: RefCell<Vec<u32>>,
    fragment_spv: RefCell<Vec<u32>>,
    pub topology: vk::PrimitiveTopology, // TODO remove pub
    index: OnceCell<usize>,
    users: Cell<usize>,
    vertex_descriptors: Vec<vk::DescriptorType>,
    fragment_descriptors: Vec<vk::DescriptorType>,
    dynamic_offset_size: usize,
    sampler_indices: OnceCell<HashMap<u32, u32>>,
}

impl ShaderSet {
    pub fn new(
        vertex_shader_bytes: &[u8],
        vertex_shader_interface: Vec<ShaderInterface>,
        fragment_shader_bytes: &[u8],
        fragment_shader_interface: Vec<ShaderInterface>,
        topology: MeshTopology,
    ) -> Self {
        // TODO support on-the-fly shader compil and read shader contents to fill descriptors
        let mut spv_data = Cursor::new(vertex_shader_bytes);
        let vertex_spv = RefCell::new(util::read_spv(&mut spv_data).unwrap());
        let mut spv_data = Cursor::new(fragment_shader_bytes);
        let fragment_spv = RefCell::new(util::read_spv(&mut spv_data).unwrap());
        let vertex_descriptors = vertex_shader_interface
            .iter()
            .map(|shader_interface| match shader_interface {
                ShaderInterface::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
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
        let (dynamic_offset_size, vk_topology) = match topology {
            MeshTopology::Triangles => (size_of::<MvpUbo>(), vk::PrimitiveTopology::TRIANGLE_LIST),
            MeshTopology::Lines => (size_of::<MvpUbo>(), vk::PrimitiveTopology::LINE_LIST),
            MeshTopology::Points => (0, vk::PrimitiveTopology::POINT_LIST),
            #[allow(unreachable_patterns)]
            _ => unimplemented!(),
        };
        Self {
            vertex_spv,
            fragment_spv,
            topology: vk_topology,
            index: OnceCell::new(),
            users: Cell::new(0),
            vertex_descriptors,
            fragment_descriptors,
            dynamic_offset_size,
            sampler_indices: OnceCell::new(),
        }
    }

    pub fn get_descriptor_set_layout(
        &self,
        engine: &Engine,
        fragment_descriptors_count: u32,
        material_indices: &[u32],
    ) -> vk::DescriptorSetLayout {
        let mut desc_layout_bindings = Vec::new();
        for (i, vertex_descriptor) in self.vertex_descriptors.iter().enumerate() {
            desc_layout_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: i as u32,
                descriptor_type: *vertex_descriptor,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            })
        }
        for (i, fragment_descriptor) in self.fragment_descriptors.iter().enumerate() {
            desc_layout_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: (self.vertex_descriptors.len() + i) as u32,
                descriptor_type: *fragment_descriptor,
                descriptor_count: fragment_descriptors_count,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            })
        }
        let descriptor_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&desc_layout_bindings);
        let mut sampler_indices = HashMap::new();
        for (i, material_index) in material_indices.iter().enumerate() {
            sampler_indices.insert(*material_index, i as u32);
        }
        self.sampler_indices.set(sampler_indices).unwrap();
        unsafe {
            engine
                .device
                .create_descriptor_set_layout(&descriptor_info, None)
                .unwrap()
        }
    }

    pub fn get_dynamic_offset(&self, index: usize) -> u32 {
        (self.dynamic_offset_size * index) as u32
    }

    pub fn get_sampler_index(&self, index: u32) -> &u32 {
        self.sampler_indices.get().unwrap().get(&index).unwrap()
    }

    pub fn inc_users(&self) {
        self.users.update(|x| x + 1);
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
        mut shader_sets: Vec<Rc<ShaderSet>>,
    ) -> Self {
        // TODO check all meshes and materials used in lines and objects are given in the dedicated array
        // or even better, create the dedicated arrays by collecting them from objs and lines
        for object in objects.iter_mut() {
            object.shader_set.inc_users()
        }
        for line in lines.iter_mut() {
            line.shader_set.inc_users()
        }
        for (i, shader_set) in shader_sets.iter_mut().enumerate() {
            shader_set.index.set(i).unwrap();
        }
        Self {
            camera,
            lines,
            objects,
            meshes,
            materials,
            shader_sets,
        }
    }

    pub fn load_resources(
        &mut self,
        engine: &Engine,
        desc_alloc_infos: &HashMap<Rc<ShaderSet>, DescriptorSetAllocateInfo>,
    ) -> HashMap<
        Rc<ShaderSet>,
        (
            Vec<Box<dyn Shader>>,
            VertexInputs,
            vk::DescriptorSet,
            vk::PipelineInputAssemblyStateCreateInfo<'_>,
            vk::PipelineRasterizationStateCreateInfo<'_>,
            Vec<vk::PipelineShaderStageCreateInfo<'_>>,
        ),
    > {
        for mesh in self.meshes.iter() {
            mesh.load_mesh(engine)
        }
        for material in &self.materials {
            material.load_textures(engine);
        }
        let mut result = HashMap::new();
        for shader_set in self.shader_sets.iter() {
            // println!(
            //     "shaders set {i:?}: {st:?}",
            //     i = shader_set.index.get().unwrap(),
            //     st = shader_set.topology
            // );
            unsafe {
                let descriptor_sets = engine
                    .device
                    .allocate_descriptor_sets(desc_alloc_infos.get(shader_set).unwrap())
                    .unwrap();

                let mut materials = Vec::new();
                for object in &self.objects {
                    if &object.shader_set == shader_set {
                        materials.push(&object.material)
                    }
                }

                let fragment_shader = FragmentShader::new(
                    engine,
                    &shader_set.fragment_spv.borrow(),
                    &materials,
                    &descriptor_sets,
                    &shader_set.fragment_descriptors,
                    shader_set.users.get(),
                );
                let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
                    topology: shader_set.topology,
                    ..Default::default()
                };
                let mut shader_stage_create_infos = vec![vk::PipelineShaderStageCreateInfo {
                    s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                    module: fragment_shader.module,
                    p_name: SHADER_ENTRY_NAME.as_ptr(),
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    ..Default::default()
                }];
                let rasterization_info;
                let (vertex_shader, vertex_inputs);
                match shader_set.topology {
                    vk::PrimitiveTopology::LINE_LIST => {
                        (vertex_shader, vertex_inputs) = VertexShader::new(
                            engine,
                            &shader_set.vertex_spv.borrow(),
                            &descriptor_sets[0],
                            &shader_set.vertex_descriptors,
                            DataOrganization::ObjectMajor,
                            shader_set.users.get() as u64,
                        );
                        rasterization_info = vk::PipelineRasterizationStateCreateInfo {
                            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                            line_width: 10.0,
                            polygon_mode: vk::PolygonMode::LINE,
                            ..Default::default()
                        };
                    }
                    vk::PrimitiveTopology::TRIANGLE_LIST => {
                        (vertex_shader, vertex_inputs) = VertexShader::new(
                            engine,
                            &shader_set.vertex_spv.borrow(),
                            &descriptor_sets[0],
                            &shader_set.vertex_descriptors,
                            DataOrganization::ObjectMajor,
                            shader_set.users.get() as u64,
                        );
                        rasterization_info = vk::PipelineRasterizationStateCreateInfo {
                            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                            line_width: 2.0,
                            polygon_mode: vk::PolygonMode::FILL,
                            cull_mode: vk::CullModeFlags::FRONT,
                            ..Default::default()
                        };
                    }
                    vk::PrimitiveTopology::POINT_LIST => {
                        (vertex_shader, vertex_inputs) = VertexShader::new(
                            engine,
                            &shader_set.vertex_spv.borrow(),
                            &descriptor_sets[0],
                            &shader_set.vertex_descriptors,
                            DataOrganization::ObjectMajor,
                            shader_set.users.get() as u64,
                        );
                        let mut spv_data =
                            Cursor::new(include_bytes!("../../target/voxel_geom.spv"));
                        let geometry_spv = util::read_spv(&mut spv_data).unwrap();
                        let geometry_shader = GeometryShader::new(engine, geometry_spv.as_slice());
                        rasterization_info = vk::PipelineRasterizationStateCreateInfo {
                            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                            line_width: 2.0,
                            polygon_mode: vk::PolygonMode::FILL,
                            cull_mode: vk::CullModeFlags::FRONT,
                            ..Default::default()
                        };
                        shader_stage_create_infos.push(vk::PipelineShaderStageCreateInfo {
                            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                            module: geometry_shader.module,
                            p_name: SHADER_ENTRY_NAME.as_ptr(),
                            stage: vk::ShaderStageFlags::GEOMETRY,
                            ..Default::default()
                        })
                    }
                    _ => unimplemented!(),
                };

                shader_stage_create_infos.push(vk::PipelineShaderStageCreateInfo {
                    s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                    module: vertex_shader.module,
                    p_name: SHADER_ENTRY_NAME.as_ptr(),
                    stage: vk::ShaderStageFlags::VERTEX,
                    ..Default::default()
                });
                shader_set.vertex_spv.borrow_mut().clear();
                shader_set.fragment_spv.borrow_mut().clear();
                let shaders: Vec<Box<dyn Shader>> =
                    vec![Box::new(vertex_shader), Box::new(fragment_shader)];
                result.insert(
                    shader_set.clone(),
                    (
                        shaders,
                        vertex_inputs,
                        descriptor_sets[0],
                        vertex_input_assembly_state_info,
                        rasterization_info,
                        shader_stage_create_infos,
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

        for (i, material) in self.materials.iter().enumerate() {
            material.global_index.replace(i as u32);
        }
        for shader_set in &self.shader_sets {
            let mut material_indices = Vec::new();
            for object in &self.objects {
                if &object.shader_set == shader_set {
                    material_indices.push(*object.material.global_index.borrow())
                }
            }
            let result = descriptor_set_layouts.insert(
                shader_set.clone(),
                shader_set.get_descriptor_set_layout(
                    engine,
                    shader_set.users.get() as u32,
                    &material_indices,
                ),
            );
            if result.is_some() {
                panic!("Duplicate entry inserted into hash")
            }
        }
        descriptor_set_layouts
    }

    pub fn get_pool(&self, engine: &Engine) -> vk::DescriptorPool {
        let mut types_map: HashMap<vk::DescriptorType, vk::DescriptorPoolSize> = HashMap::new();
        for shader_set in self.shader_sets.iter() {
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
                        *descriptor,
                        vk::DescriptorPoolSize {
                            ty: *descriptor,
                            descriptor_count: 1,
                        },
                    );
                }
            }
        }

        let mut descriptor_sizes: Vec<vk::DescriptorPoolSize> = types_map.into_values().collect();
        descriptor_sizes.iter_mut().for_each(|e| {
            e.descriptor_count *= MAX_FRAMES_IN_FLIGHT;
        });
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_sizes)
            .max_sets(self.shader_sets.len() as u32);
        unsafe {
            engine
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap()
        }
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
