use ash::{
    vk::{self, DeviceMemory},
    Instance,
};
use std::{ffi::c_void, mem, ptr::copy_nonoverlapping as memcpy};

pub struct GpuBuffer {
    instance: &'static Instance,
    logical_device: ash::Device,
    physical_device: vk::PhysicalDevice,
    pub buffer: vk::Buffer,
    pub allocation: DeviceMemory,
    size_in_bytes: u64,
    buffer_usage: vk::BufferUsageFlags,
    allocation_name: String,
    allocation_location: gpu_allocator::MemoryLocation,
    linear: bool,
}

impl GpuBuffer {
    pub fn new(
        instance: &ash::Instance,
        bytes: u64,
        buffer_usage: vk::BufferUsageFlags,
        logical_device: ash::Device,
        physical_device: vk::PhysicalDevice,
        allocation_name: String,
        allocation_location: gpu_allocator::MemoryLocation,
        linear: bool,
    ) -> Result<GpuBuffer, vk::Result> {
        let size_in_bytes = bytes;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size_in_bytes)
            .usage(buffer_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { logical_device.create_buffer(&buffer_info, None) }?;

        // Memory

        let requirements = unsafe { logical_device.get_buffer_memory_requirements(buffer) };

        let memory_type = unsafe {
            get_memory_type_index(
                instance,
                physical_device,
                vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::DEVICE_LOCAL,
                requirements,
            )
            .expect("Failed to find suitable memory type index")
        };

        let memory_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);

        let allocation = unsafe { logical_device.allocate_memory(&memory_info, None) }?;

        unsafe { logical_device.bind_buffer_memory(buffer, allocation, 0) }?;

        Ok(GpuBuffer {
            instance: unsafe { std::mem::transmute(instance) },
            logical_device,
            physical_device,
            buffer,
            allocation,
            size_in_bytes,
            buffer_usage,
            allocation_name: allocation_name.to_string(),
            allocation_location,
            linear,
        })
    }

    pub fn write_to_memory<T: Sized>(&mut self, data: &[T]) -> Result<(), vk::Result> {
        let bytes_to_write = std::mem::size_of_val(data) as u64;
        if bytes_to_write > self.size_in_bytes {
            let new_buffer = GpuBuffer::new(
                self.instance,
                bytes_to_write,
                self.buffer_usage,
                self.logical_device.clone(),
                self.physical_device,
                self.allocation_name.clone(),
                self.allocation_location,
                self.linear,
            )?;
            *self = new_buffer;
        };
        let memory = unsafe {
            self.logical_device.map_memory(
                self.allocation,
                0,
                bytes_to_write,
                vk::MemoryMapFlags::empty(),
            )
        }?;
        //unsafe { memory.copy_from(data.as_ptr() as *const c_void, bytes_to_write as usize) };
        //unsafe { memcpy(data.as_ptr(), memory.cast(), bytes_to_write as usize) };
        unsafe {
            memory.copy_from_nonoverlapping(data.as_ptr() as *const c_void, bytes_to_write as usize)
        };

        unsafe { self.logical_device.unmap_memory(self.allocation) };
        Ok(())
    }
}

pub unsafe fn get_memory_type_index(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Option<u32> {
    let memory = instance.get_physical_device_memory_properties(physical_device);
    (0..memory.memory_type_count).find(|i| {
        let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
        let memory_type = memory.memory_types[*i as usize];
        suitable && memory_type.property_flags.contains(properties)
    })
}

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}
