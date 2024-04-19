use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationScheme, Allocator};

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    logical_device: ash::Device,
    size_in_bytes: u64,
    buffer_usage: vk::BufferUsageFlags,
    allocation_name: String,
    allocation_location: gpu_allocator::MemoryLocation,
    linear: bool,
    allocation_scheme: AllocationScheme,
}

impl GpuBuffer {
    pub fn new(
        allocator: &mut Allocator,
        size_in_bytes: u64,
        buffer_usage: vk::BufferUsageFlags,
        logical_device: ash::Device,
        allocation_name: String,
        allocation_location: gpu_allocator::MemoryLocation,
        linear: bool,
        allocation_scheme: AllocationScheme,
    ) -> Result<GpuBuffer, vk::Result> {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size_in_bytes)
            .usage(buffer_usage);

        let buffer = unsafe { logical_device.create_buffer(&buffer_create_info, None)? };
        let requirements = unsafe { logical_device.get_buffer_memory_requirements(buffer) };

        let allocation_info = gpu_allocator::vulkan::AllocationCreateDesc {
            name: &allocation_name,
            requirements,
            location: allocation_location,
            linear,
            allocation_scheme,
        };

        let allocation = allocator.allocate(&allocation_info).unwrap();
        unsafe {
            logical_device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?
        }
        Ok(GpuBuffer {
            buffer,
            allocation,
            logical_device,
            size_in_bytes,
            buffer_usage,
            allocation_name: allocation_name.to_string(),
            allocation_location,
            linear,
            allocation_scheme,
        })
    }

    pub fn write_to_memory<T: Sized>(
        &mut self,
        allocator: &mut Allocator,
        data: &[T],
    ) -> Result<(), vk::Result> {
        let bytes_to_write = std::mem::size_of_val(data) as u64;
        if bytes_to_write > self.size_in_bytes {
            allocator
                .free(std::mem::take(&mut self.allocation))
                .expect("Error freeing model buffer");
            unsafe { self.logical_device.destroy_buffer(self.buffer, None) };

            let new_buffer = GpuBuffer::new(
                allocator,
                bytes_to_write,
                self.buffer_usage,
                self.logical_device.clone(),
                self.allocation_name.clone(),
                self.allocation_location,
                self.linear,
                self.allocation_scheme,
            )?;
            *self = new_buffer;
        };

        let data_ptr = self.allocation.mapped_ptr().unwrap().as_ptr() as *mut T;
        unsafe { data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()) };
        Ok(())
    }
}
