use ash::vk;
use gpu_allocator::vulkan::{AllocationScheme, Allocator};

use crate::GpuBuffer;

pub struct Model<V, I> {
    vertex_data: Vec<V>,
    index_data: Vec<u32>,
    handle_to_index: std::collections::HashMap<usize, usize>,
    handles: Vec<usize>,
    instances: Vec<I>,
    first_invisible: usize,
    next_handle: usize,
    pub vertex_buffer: Option<GpuBuffer>,
    pub index_buffer: Option<GpuBuffer>,
    pub instance_buffer: Option<GpuBuffer>,
    logical_device: ash::Device,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct VertexData {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl VertexData {
    fn midpoint(a: &VertexData, b: &VertexData) -> VertexData {
        VertexData {
            position: [
                0.5 * (a.position[0] + b.position[0]),
                0.5 * (a.position[1] + b.position[1]),
                0.5 * (a.position[2] + b.position[2]),
            ],
            normal: normalize([
                0.5 * (a.normal[0] + b.normal[0]),
                0.5 * (a.normal[1] + b.normal[1]),
                0.5 * (a.normal[2] + b.normal[2]),
            ]),
        }
    }
}
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

#[repr(C)]
pub struct InstanceData {
    pub model_matrix: [[f32; 4]; 4],
    pub inverse_model_matrix: [[f32; 4]; 4],
    pub colour: [f32; 3],
}
impl InstanceData {
    pub fn from_matrix_and_colour(
        model_matrix: na::Matrix4<f32>,
        colour: [f32; 3],
    ) -> InstanceData {
        InstanceData {
            model_matrix: model_matrix.into(),
            inverse_model_matrix: model_matrix.try_inverse().unwrap().into(),
            colour,
        }
    }
}

#[derive(Debug, Clone)]
struct InvalidHandle;
impl std::fmt::Display for InvalidHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "invalid handle")
    }
}
impl std::error::Error for InvalidHandle {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl<V, I> Model<V, I> {
    fn get(&self, handle: usize) -> Option<&I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.instances.get(index)
        } else {
            None
        }
    }
    pub fn get_mut(&mut self, handle: usize) -> Option<&mut I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.instances.get_mut(index)
        } else {
            None
        }
    }
    fn swap_by_handle(&mut self, handle1: usize, handle2: usize) -> Result<(), InvalidHandle> {
        if handle1 == handle2 {
            return Ok(());
        }
        if let (Some(&index1), Some(&index2)) = (
            self.handle_to_index.get(&handle1),
            self.handle_to_index.get(&handle2),
        ) {
            self.handles.swap(index1, index2);
            self.instances.swap(index1, index2);
            self.handle_to_index.insert(index1, handle2);
            self.handle_to_index.insert(index2, handle1);
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }
    fn swap_by_index(&mut self, index1: usize, index2: usize) {
        if index1 == index2 {
            return;
        }
        let handle1 = self.handles[index1];
        let handle2 = self.handles[index2];
        self.handles.swap(index1, index2);
        self.instances.swap(index1, index2);
        self.handle_to_index.insert(index1, handle2);
        self.handle_to_index.insert(index2, handle1);
    }
    fn is_visible(&self, handle: usize) -> Result<bool, InvalidHandle> {
        if let Some(index) = self.handle_to_index.get(&handle) {
            Ok(index < &self.first_invisible)
        } else {
            Err(InvalidHandle)
        }
    }

    fn make_visible(&mut self, handle: usize) -> Result<(), InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index < self.first_invisible {
                return Ok(());
            }
            self.swap_by_index(index, self.first_invisible);
            self.first_invisible += 1;
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    fn make_invisible(&mut self, handle: usize) -> Result<(), InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index >= self.first_invisible {
                return Ok(());
            }
            self.swap_by_index(index, self.first_invisible - 1);
            self.first_invisible -= 1;
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    fn insert(&mut self, element: I) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;
        let index = self.instances.len();
        self.instances.push(element);
        self.handles.push(handle);
        self.handle_to_index.insert(handle, index);
        handle
    }
    pub fn insert_visibly(&mut self, element: I) -> usize {
        let new_handle = self.insert(element);
        self.make_visible(new_handle).ok(); //can't go wrong, see previous line
        new_handle
    }
    fn remove(&mut self, handle: usize) -> Result<I, InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index < self.first_invisible {
                self.swap_by_index(index, self.first_invisible - 1);
                self.first_invisible -= 1;
            }
            self.swap_by_index(self.first_invisible, self.instances.len() - 1);
            self.handles.pop();
            self.handle_to_index.remove(&handle);
            //must be Some(), otherwise we couldn't have found an index
            Ok(self.instances.pop().unwrap())
        } else {
            Err(InvalidHandle)
        }
    }

    pub fn update_vertex_buffer(&mut self, allocator: &mut Allocator) -> Result<(), vk::Result> {
        if let Some(buffer) = &mut self.vertex_buffer {
            buffer.write_to_memory(allocator, &self.vertex_data)?;
            Ok(())
        } else {
            let bytes = (self.vertex_data.len() * std::mem::size_of::<V>()) as u64;
            let mut buffer = GpuBuffer::new(
                allocator,
                bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                self.logical_device.clone(),
                "VertexBuffer".to_string(),
                gpu_allocator::MemoryLocation::CpuToGpu,
                false,
                AllocationScheme::GpuAllocatorManaged,
            )?;

            buffer.write_to_memory(allocator, &self.vertex_data)?;
            self.vertex_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn update_index_buffer(&mut self, allocator: &mut Allocator) -> Result<(), vk::Result> {
        if let Some(buffer) = &mut self.index_buffer {
            buffer.write_to_memory(allocator, &self.index_data)?;
            Ok(())
        } else {
            let bytes = (self.index_data.len() * std::mem::size_of::<u32>()) as u64;
            let mut buffer = GpuBuffer::new(
                allocator,
                bytes,
                vk::BufferUsageFlags::INDEX_BUFFER,
                self.logical_device.clone(),
                "IndexBuffer".to_string(),
                gpu_allocator::MemoryLocation::CpuToGpu,
                false,
                AllocationScheme::GpuAllocatorManaged,
            )?;

            buffer.write_to_memory(allocator, &self.index_data)?;
            self.index_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn update_instance_buffer(&mut self, allocator: &mut Allocator) -> Result<(), vk::Result> {
        if let Some(buffer) = &mut self.instance_buffer {
            buffer.write_to_memory(allocator, &self.instances[0..self.first_invisible])?;
            Ok(())
        } else {
            let bytes = (self.first_invisible * std::mem::size_of::<I>()) as u64;
            let mut buffer = GpuBuffer::new(
                allocator,
                bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                self.logical_device.clone(),
                "VertexBuffer".to_string(),
                gpu_allocator::MemoryLocation::CpuToGpu,
                true,
                AllocationScheme::GpuAllocatorManaged,
            )?;
            buffer.write_to_memory(allocator, &self.instances[0..self.first_invisible])?;
            self.instance_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn draw(&self, command_buffer: vk::CommandBuffer) {
        if let Some(vertex_buffer) = &self.vertex_buffer {
            if let Some(index_buffer) = &self.index_buffer {
                if let Some(instance_buffer) = &self.instance_buffer {
                    if self.first_invisible > 0 {
                        unsafe {
                            self.logical_device.cmd_bind_vertex_buffers(
                                command_buffer,
                                0,
                                &[vertex_buffer.buffer],
                                &[0],
                            );
                            self.logical_device.cmd_bind_vertex_buffers(
                                command_buffer,
                                1,
                                &[instance_buffer.buffer],
                                &[0],
                            );
                            self.logical_device.cmd_bind_index_buffer(
                                command_buffer,
                                index_buffer.buffer,
                                0,
                                vk::IndexType::UINT32,
                            );
                            self.logical_device.cmd_draw_indexed(
                                command_buffer,
                                self.index_data.len() as u32,
                                self.first_invisible as u32,
                                0,
                                0,
                                0,
                            );
                        }
                    }
                }
            }
        }
    }
}

impl Model<[f32; 3], InstanceData> {
    pub fn cube(logical_device: &ash::Device) -> Model<[f32; 3], InstanceData> {
        let lbf = [-1.0, 1.0, -1.0]; //lbf: left-bottom-front
        let lbb = [-1.0, 1.0, 1.0];
        let ltf = [-1.0, -1.0, -1.0];
        let ltb = [-1.0, -1.0, 1.0];
        let rbf = [1.0, 1.0, -1.0];
        let rbb = [1.0, 1.0, 1.0];
        let rtf = [1.0, -1.0, -1.0];
        let rtb = [1.0, -1.0, 1.0];
        Model {
            vertex_data: vec![lbf, lbb, ltf, ltb, rbf, rbb, rtf, rtb],
            index_data: vec![
                0, 1, 5, 0, 5, 4, //bottom
                2, 7, 3, 2, 6, 7, //top
                0, 6, 2, 0, 4, 6, //front
                1, 3, 7, 1, 7, 5, //back
                0, 2, 1, 1, 2, 3, //left
                4, 5, 6, 5, 7, 6, //right
            ],
            handle_to_index: std::collections::HashMap::new(),
            handles: Vec::new(),
            instances: Vec::new(),
            first_invisible: 0,
            next_handle: 0,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
            logical_device: logical_device.clone(),
        }
    }
}

impl Model<VertexData, InstanceData> {
    pub fn icosahedron(logical_device: &ash::Device) -> Model<VertexData, InstanceData> {
        let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let darkgreen_front_top = VertexData {
            position: [phi, -1.0, 0.0],
            normal: normalize([phi, -1.0, 0.0]),
        }; //0
        let darkgreen_front_bottom = VertexData {
            position: [phi, 1.0, 0.0],
            normal: normalize([phi, 1.0, 0.0]),
        }; //1
        let darkgreen_back_top = VertexData {
            position: [-phi, -1.0, 0.0],
            normal: normalize([-phi, -1.0, 0.0]),
        }; //2
        let darkgreen_back_bottom = VertexData {
            position: [-phi, 1.0, 0.0],
            normal: normalize([-phi, 1.0, 0.0]),
        }; //3
        let lightgreen_front_right = VertexData {
            position: [1.0, 0.0, -phi],
            normal: normalize([1.0, 0.0, -phi]),
        }; //4
        let lightgreen_front_left = VertexData {
            position: [-1.0, 0.0, -phi],
            normal: normalize([-1.0, 0.0, -phi]),
        }; //5
        let lightgreen_back_right = VertexData {
            position: [1.0, 0.0, phi],
            normal: normalize([1.0, 0.0, phi]),
        }; //6
        let lightgreen_back_left = VertexData {
            position: [-1.0, 0.0, phi],
            normal: normalize([-1.0, 0.0, phi]),
        }; //7
        let purple_top_left = VertexData {
            position: [0.0, -phi, -1.0],
            normal: normalize([0.0, -phi, -1.0]),
        }; //8
        let purple_top_right = VertexData {
            position: [0.0, -phi, 1.0],
            normal: normalize([0.0, -phi, 1.0]),
        }; //9
        let purple_bottom_left = VertexData {
            position: [0.0, phi, -1.0],
            normal: normalize([0.0, phi, -1.0]),
        }; //10
        let purple_bottom_right = VertexData {
            position: [0.0, phi, 1.0],
            normal: normalize([0.0, phi, 1.0]),
        }; //11
        Model {
            vertex_data: vec![
                darkgreen_front_top,
                darkgreen_front_bottom,
                darkgreen_back_top,
                darkgreen_back_bottom,
                lightgreen_front_right,
                lightgreen_front_left,
                lightgreen_back_right,
                lightgreen_back_left,
                purple_top_left,
                purple_top_right,
                purple_bottom_left,
                purple_bottom_right,
            ],
            index_data: vec![
                0, 9, 8, //
                0, 8, 4, //
                0, 4, 1, //
                0, 1, 6, //
                0, 6, 9, //
                8, 9, 2, //
                8, 2, 5, //
                8, 5, 4, //
                4, 5, 10, //
                4, 10, 1, //
                1, 10, 11, //
                1, 11, 6, //
                2, 3, 5, //
                2, 7, 3, //
                2, 9, 7, //
                5, 3, 10, //
                3, 11, 10, //
                3, 7, 11, //
                6, 7, 9, //
                6, 11, 7, //
            ],
            handle_to_index: std::collections::HashMap::new(),
            handles: Vec::new(),
            instances: Vec::new(),
            first_invisible: 0,
            next_handle: 0,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
            logical_device: logical_device.clone(),
        }
    }
    pub fn sphere(
        refinements: u32,
        logical_device: &ash::Device,
    ) -> Model<VertexData, InstanceData> {
        let mut model = Model::icosahedron(logical_device);
        for _ in 0..refinements {
            model.refine();
        }
        for v in &mut model.vertex_data {
            v.position = normalize(v.position);
        }
        model
    }
    pub fn refine(&mut self) {
        let mut new_indices = vec![];
        let mut midpoints = std::collections::HashMap::<(u32, u32), u32>::new();
        for triangle in self.index_data.chunks(3) {
            let a = triangle[0];
            let b = triangle[1];
            let c = triangle[2];
            let vertex_a = self.vertex_data[a as usize];
            let vertex_b = self.vertex_data[b as usize];
            let vertex_c = self.vertex_data[c as usize];
            let mab = if let Some(ab) = midpoints.get(&(a, b)) {
                *ab
            } else {
                let vertex_ab = VertexData::midpoint(&vertex_a, &vertex_b);
                let mab = self.vertex_data.len() as u32;
                self.vertex_data.push(vertex_ab);
                midpoints.insert((a, b), mab);
                midpoints.insert((b, a), mab);
                mab
            };
            let mbc = if let Some(bc) = midpoints.get(&(b, c)) {
                *bc
            } else {
                let vertex_bc = VertexData::midpoint(&vertex_b, &vertex_c);
                let mbc = self.vertex_data.len() as u32;
                midpoints.insert((b, c), mbc);
                midpoints.insert((c, b), mbc);
                self.vertex_data.push(vertex_bc);
                mbc
            };
            let mca = if let Some(ca) = midpoints.get(&(c, a)) {
                *ca
            } else {
                let vertex_ca = VertexData::midpoint(&vertex_c, &vertex_a);
                let mca = self.vertex_data.len() as u32;
                midpoints.insert((c, a), mca);
                midpoints.insert((a, c), mca);
                self.vertex_data.push(vertex_ca);
                mca
            };
            new_indices.extend_from_slice(&[mca, a, mab, mab, b, mbc, mbc, c, mca, mab, mbc, mca]);
        }
        self.index_data = new_indices;
    }
}
