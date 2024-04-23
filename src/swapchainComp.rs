use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::{
    gpuBuffer::get_memory_type_index, queueFamilies::QueueFamilies, queues::Queues,
    surface::Surface,
};

pub struct SwapchainComp {
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub imageviews: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub surface_format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub image_available: Vec<vk::Semaphore>,
    pub rendering_finished: Vec<vk::Semaphore>,
    pub amount_of_images: u32,
    pub current_image: usize,
    pub may_begin_drawing: Vec<vk::Fence>,
    depth_image: vk::Image,
    depth_image_allocation: vk::DeviceMemory,
    depth_imageview: vk::ImageView,
}

impl SwapchainComp {
    pub fn init(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        logical_device: &ash::Device,
        surfaces: &Surface,
        queue_families: &QueueFamilies,
        _queues: &Queues,
    ) -> Result<SwapchainComp, vk::Result> {
        let surface_capabilities = surfaces.get_capabilities(physical_device)?;
        let extent = surface_capabilities.current_extent;
        let _surface_present_modes = surfaces.get_present_modes(physical_device)?;
        let surface_format = *surfaces.get_formats(physical_device)?.first().unwrap();
        let queue_families_indices = [queue_families.graphics_q_index.unwrap()];

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surfaces.surface)
            .min_image_count(
                3.max(surface_capabilities.min_image_count)
                    .min(surface_capabilities.max_image_count),
            )
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families_indices)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);

        let swapchain_loader = ash::khr::swapchain::Device::new(instance, logical_device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let amount_of_images = swapchain_images.len() as u32;
        let mut swapchain_imageviews = Vec::with_capacity(swapchain_images.len());

        for image in &swapchain_images {
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let imageview_create_info = vk::ImageViewCreateInfo::default()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .subresource_range(subresource_range);

            let imageview =
                unsafe { logical_device.create_image_view(&imageview_create_info, None) }?;
            swapchain_imageviews.push(imageview);
        }
        let mut image_available = vec![];
        let mut rendering_finished = vec![];
        let mut may_begin_drawing: Vec<vk::Fence> = vec![];
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..amount_of_images {
            let semaphore_available =
                unsafe { logical_device.create_semaphore(&semaphore_info, None) }?;
            let semaphore_finished =
                unsafe { logical_device.create_semaphore(&semaphore_info, None) }?;
            image_available.push(semaphore_available);
            rendering_finished.push(semaphore_finished);
            let fence = unsafe { logical_device.create_fence(&fence_create_info, None)? };
            may_begin_drawing.push(fence);
        }

        let extent3d = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        };

        let depth_image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(extent3d)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families_indices);

        let (depth_image, depth_image_allocation) = unsafe {
            Self::create_image(
                instance,
                logical_device,
                physical_device,
                depth_image_info,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?
        };
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let imageview_create_info = vk::ImageViewCreateInfo::default()
            .image(depth_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(subresource_range);
        let depth_imageview =
            unsafe { logical_device.create_image_view(&imageview_create_info, None) }?;

        Ok(SwapchainComp {
            swapchain_loader,
            swapchain,
            images: swapchain_images,
            imageviews: swapchain_imageviews,
            framebuffers: vec![],
            surface_format,
            extent,
            amount_of_images,
            current_image: 0,
            image_available,
            rendering_finished,
            may_begin_drawing,
            depth_image,
            depth_image_allocation: depth_image_allocation,
            depth_imageview,
        })
    }

    unsafe fn create_image(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        image_create_info: vk::ImageCreateInfo,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory), vk::Result> {
        let image = device.create_image(&image_create_info, None)?;

        // Memory

        let requirements = device.get_image_memory_requirements(image);

        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(
                get_memory_type_index(instance, physical_device, properties, requirements).unwrap(),
            );

        let image_memory = device.allocate_memory(&info, None)?;

        device.bind_image_memory(image, image_memory, 0)?;

        Ok((image, image_memory))
    }

    pub fn create_framebuffers(
        &mut self,
        logical_device: &ash::Device,
        renderpass: vk::RenderPass,
    ) -> Result<(), vk::Result> {
        for iv in &self.imageviews {
            let iview = [*iv, self.depth_imageview];
            let frame_buffer_info = vk::FramebufferCreateInfo::default()
                .render_pass(renderpass)
                .attachments(&iview)
                .width(self.extent.width)
                .height(self.extent.height)
                .layers(1);
            let fb = unsafe { logical_device.create_framebuffer(&frame_buffer_info, None) }?;
            self.framebuffers.push(fb);
        }
        Ok(())
    }

    pub unsafe fn cleanup(&mut self, logical_device: &ash::Device) {
        logical_device.destroy_image(self.depth_image, None);
        logical_device.free_memory(self.depth_image_allocation, None);
        logical_device.destroy_image_view(self.depth_imageview, None);
        for fence in &self.may_begin_drawing {
            logical_device.destroy_fence(*fence, None);
        }
        for semaphore in &self.image_available {
            logical_device.destroy_semaphore(*semaphore, None);
        }
        for semaphore in &self.rendering_finished {
            logical_device.destroy_semaphore(*semaphore, None);
        }
        for fb in &self.framebuffers {
            logical_device.destroy_framebuffer(*fb, None);
        }
        for iv in &self.imageviews {
            logical_device.destroy_image_view(*iv, None);
        }
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None)
    }
}

impl Drop for SwapchainComp {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None)
        };
    }
}
