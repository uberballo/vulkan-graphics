use ash::{
    ext::metal_objects::Device,
    vk::{self, DeviceMemory},
    Instance,
};
use std::error::Error;
use std::ptr::copy_nonoverlapping as memcpy;

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: DeviceMemory,
    logical_device: ash::Device,
    physical_device: vk::PhysicalDevice,
    size_in_bytes: u64,
    buffer_usage: vk::BufferUsageFlags,
    allocation_name: String,
    allocation_location: gpu_allocator::MemoryLocation,
    linear: bool,
}

impl GpuBuffer {
    pub fn new(
        instance: &Instance,
        size_in_bytes: u64,
        buffer_usage: vk::BufferUsageFlags,
        logical_device: ash::Device,
        physical_device: vk::PhysicalDevice,
        allocation_name: String,
        allocation_location: gpu_allocator::MemoryLocation,
        linear: bool,
    ) -> Result<GpuBuffer, vk::Result> {
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
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                requirements,
            )
            .unwrap()
        };
        let memory_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size_in_bytes)
            .memory_type_index(memory_type);

        let allocation = unsafe { logical_device.allocate_memory(&memory_info, None) }?;

        unsafe { logical_device.bind_buffer_memory(buffer, allocation, 0) }?;

        Ok(GpuBuffer {
            buffer,
            allocation,
            logical_device,
            physical_device,
            size_in_bytes,
            buffer_usage,
            allocation_name: allocation_name.to_string(),
            allocation_location,
            linear,
        })

        //let buffer_create_info = vk::BufferCreateInfo::default()
        //    .size(size_in_bytes)
        //    .usage(buffer_usage);

        //let buffer = unsafe { logical_device.create_buffer(&buffer_create_info, None)? };
        //let requirements = unsafe { logical_device.get_buffer_memory_requirements(buffer) };

        //let allocation_info = gpu_allocator::vulkan::AllocationCreateDesc {
        //    name: &allocation_name,
        //    requirements,
        //    location: allocation_location,
        //    linear,
        //    allocation_scheme,
        //};

        //let allocation = allocator.allocate(&allocation_info).unwrap();
        //unsafe {
        //    logical_device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?
        //}
        //Ok(GpuBuffer {
        //    buffer,
        //    allocation,
        //    logical_device,
        //    size_in_bytes,
        //    buffer_usage,
        //    allocation_name: allocation_name.to_string(),
        //    allocation_location,
        //    linear,
        //    allocation_scheme,
        //})
    }

    // pub fn new_write_to_memory() {
    //     let memory = device.map_memory(
    //         data.vertex_buffer_memory,
    //         0,
    //         buffer_info.size,
    //         vk::MemoryMapFlags::empty(),
    //     )?;
    // }

    pub fn write_to_memory<T: Sized>(
        &mut self,
        instance: &Instance,
        data: &[T],
    ) -> Result<(), vk::Result> {
        let bytes_to_write = std::mem::size_of_val(data) as u64;
        if bytes_to_write > self.size_in_bytes {
            let new_buffer = GpuBuffer::new(
                instance,
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

        unsafe { memcpy(data.as_ptr(), memory.cast(), bytes_to_write as usize) };

        unsafe { self.logical_device.unmap_memory(self.allocation) };
        //let data_ptr = self.allocation.mapped_ptr().unwrap().as_ptr() as *mut T;
        //unsafe { data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()) };
        Ok(())
    }

    //pub fn write_to_memory<T: Sized>(
    //    &mut self,
    //    allocator: &mut Allocator,
    //    data: &[T],
    //) -> Result<(), vk::Result> {
    //    let bytes_to_write = std::mem::size_of_val(data) as u64;
    //    if bytes_to_write > self.size_in_bytes {
    //        allocator
    //            .free(std::mem::take(&mut self.allocation))
    //            .expect("Error freeing model buffer");
    //        unsafe { self.logical_device.destroy_buffer(self.buffer, None) };

    //        let new_buffer = GpuBuffer::new(
    //            allocator,
    //            bytes_to_write,
    //            self.buffer_usage,
    //            self.logical_device.clone(),
    //            self.allocation_name.clone(),
    //            self.allocation_location,
    //            self.linear,
    //            self.allocation_scheme,
    //        )?;
    //        *self = new_buffer;
    //    };

    //    let data_ptr = self.allocation.mapped_ptr().unwrap().as_ptr() as *mut T;
    //    unsafe { data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()) };
    //    Ok(())
    //}
}

unsafe fn get_memory_type_index(
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
