use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::io::Cursor;
use std::os::raw::c_void;

use ash::extensions::khr::Swapchain;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::CommandBuffer;
use ash::vk::PhysicalDevice;
use ash::vk::RenderPass;
use ash::Entry;
use ash::Instance;
use gpu_allocator::vulkan::*;
use winit::event::Event;
use winit::event::WindowEvent;
use winit::platform::windows::DeviceIdExtWindows;
use winit::platform::windows::WindowExtWindows;

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    println!("[Debug][{}][{}] {:?}", severity, ty, message);
    vk::FALSE
}

fn init_instance(entry: &ash::Entry, layer_names: &[&str]) -> VkResult<Instance> {
    let engine_name = std::ffi::CString::new("UnknownGameEngine").unwrap();
    let app_name = std::ffi::CString::new("Tittle!").unwrap();
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&app_name)
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 40, 0))
        .application_version(vk::make_api_version(0, 0, 0, 1))
        .api_version(vk::make_api_version(0, 1, 0, 0));
    let layer_names_c: Vec<std::ffi::CString> = layer_names
        .iter()
        .map(|&ln| std::ffi::CString::new(ln).unwrap())
        .collect();
    let layer_name_pointers: Vec<*const i8> = layer_names_c.iter().map(|x| x.as_ptr()).collect();
    let extension_name_pointers: Vec<*const i8> = vec![
        ash::extensions::ext::DebugUtils::name().as_ptr(),
        ash::extensions::khr::Surface::name().as_ptr(),
        ash::extensions::khr::Win32Surface::name().as_ptr(),
    ];

    let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(vulkan_debug_utils_callback));

    let instance_create_info = vk::InstanceCreateInfo::builder()
        .push_next(&mut debug_create_info)
        .application_info(&app_info)
        .enabled_layer_names(&layer_name_pointers)
        .enabled_extension_names(&extension_name_pointers);

    unsafe { entry.create_instance(&instance_create_info, None) }
}

fn init_physical_device_and_properties(
    instance: &ash::Instance,
) -> Result<(PhysicalDevice, vk::PhysicalDeviceProperties), vk::Result> {
    let mut chosen = None;
    let phys_devs = unsafe { instance.enumerate_physical_devices()? };
    for p in phys_devs {
        let props = unsafe { instance.get_physical_device_properties(p) };
        // If we want to use a certain GPU
        //let name = String::from(
        //    unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
        //        .to_str()
        //        .unwrap(),
        //);
        //if name == "NVIDIA GeForce RTX 2060 SUPER" {
        //    chosen = Some((p, props));
        //}
        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            chosen = Some((p, props));
        }
    }
    Ok(chosen.unwrap())
}

struct Debug {
    loader: ash::extensions::ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl Debug {
    fn init(entry: &ash::Entry, instance: &ash::Instance) -> Result<Debug, vk::Result> {
        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback));

        let loader = ash::extensions::ext::DebugUtils::new(entry, instance);
        let messenger = unsafe { loader.create_debug_utils_messenger(&debug_create_info, None)? };

        Ok(Debug { loader, messenger })
    }
}

impl Drop for Debug {
    fn drop(&mut self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None)
        };
    }
}

struct Surface {
    win_surface_loader: ash::extensions::khr::Win32Surface,
    surface: vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
}

impl Surface {
    fn init(
        window: &winit::window::Window,
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<Surface, vk::Result> {
        let hwnd = window.hwnd();
        let create_info = vk::Win32SurfaceCreateInfoKHR::builder().hwnd(hwnd as *const c_void);

        let win_surface_loader = ash::extensions::khr::Win32Surface::new(&entry, &instance);
        let surface = unsafe { win_surface_loader.create_win32_surface(&create_info, None) }?;
        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
        Ok(Surface {
            win_surface_loader,
            surface,
            surface_loader,
        })
    }

    fn get_capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<vk::SurfaceCapabilitiesKHR, vk::Result> {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(physical_device, self.surface)
        }
    }

    fn get_present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::PresentModeKHR>, vk::Result> {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(physical_device, self.surface)
        }
    }
    fn get_formats(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::SurfaceFormatKHR>, vk::Result> {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(physical_device, self.surface)
        }
    }
    fn get_physical_device_surface_support(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: usize,
    ) -> Result<bool, vk::Result> {
        unsafe {
            self.surface_loader.get_physical_device_surface_support(
                physical_device,
                queue_family_index as u32,
                self.surface,
            )
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

struct QueueFamilies {
    graphics_q_index: Option<u32>,
    transfer_q_index: Option<u32>,
}

impl QueueFamilies {
    fn init(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surfaces: &Surface,
    ) -> Result<QueueFamilies, vk::Result> {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let mut found_graphics_q_index = None;
        let mut found_transfer_q_index = None;
        for (index, qfam) in queue_family_properties.iter().enumerate() {
            if qfam.queue_count > 0
                && qfam.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && surfaces.get_physical_device_surface_support(physical_device, index)?
            {
                found_graphics_q_index = Some(index as u32);
            }
            if qfam.queue_count > 0 && qfam.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                if found_transfer_q_index.is_none()
                    || !qfam.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    found_transfer_q_index = Some(index as u32);
                }
            }
        }
        Ok(QueueFamilies {
            graphics_q_index: found_graphics_q_index,
            transfer_q_index: found_transfer_q_index,
        })
    }
}

struct Queues {
    graphics_queue: vk::Queue,
    transfer_queue: vk::Queue,
}

fn init_device_and_queues(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_families: &QueueFamilies,
    layer_names: &[&str],
) -> Result<(ash::Device, Queues), vk::Result> {
    let layer_names_c: Vec<std::ffi::CString> = layer_names
        .iter()
        .map(|&ln| std::ffi::CString::new(ln).unwrap())
        .collect();
    let layer_name_pointers: Vec<*const i8> = layer_names_c
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();

    let priorities = [1.0f32];
    let queue_infos = [
        vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_families.graphics_q_index.unwrap())
            .queue_priorities(&priorities)
            .build(),
        vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_families.transfer_q_index.unwrap())
            .queue_priorities(&priorities)
            .build(),
    ];
    let device_extension_name_pointers: Vec<*const i8> =
        vec![ash::extensions::khr::Swapchain::name().as_ptr()];

    let features = vk::PhysicalDeviceFeatures::builder().fill_mode_non_solid(true);
    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&device_extension_name_pointers)
        .enabled_layer_names(&layer_name_pointers)
        .enabled_features(&features);
    let logical_device =
        unsafe { instance.create_device(physical_device, &device_create_info, None)? };
    let graphics_queue =
        unsafe { logical_device.get_device_queue(queue_families.graphics_q_index.unwrap(), 0) };
    let transfer_queue =
        unsafe { logical_device.get_device_queue(queue_families.transfer_q_index.unwrap(), 0) };
    Ok((
        logical_device,
        Queues {
            graphics_queue,
            transfer_queue,
        },
    ))
}

struct SwapchainComp {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    imageviews: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    surface_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    image_available: Vec<vk::Semaphore>,
    rendering_finished: Vec<vk::Semaphore>,
    amount_of_images: u32,
    current_image: usize,
    may_begin_drawing: Vec<vk::Fence>,
}

impl SwapchainComp {
    fn init(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        logical_device: &ash::Device,
        surfaces: &Surface,
        queue_families: &QueueFamilies,
        queues: &Queues,
    ) -> Result<SwapchainComp, vk::Result> {
        let surface_capabilities = surfaces.get_capabilities(physical_device)?;
        let extent = surface_capabilities.current_extent;
        let surface_present_modes = surfaces.get_present_modes(physical_device)?;
        let surface_format = *surfaces.get_formats(physical_device)?.first().unwrap();
        let queue_families_indices = [queue_families.graphics_q_index.unwrap()];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
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
        let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, logical_device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let amount_of_images = swapchain_images.len() as u32;
        let mut swapchain_imageviews = Vec::with_capacity(swapchain_images.len());
        for image in &swapchain_images {
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let imageview_create_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .subresource_range(*subresource_range);
            let imageview =
                unsafe { logical_device.create_image_view(&imageview_create_info, None) }?;
            swapchain_imageviews.push(imageview);
        }
        let mut image_available = vec![];
        let mut rendering_finished = vec![];
        let mut may_begin_drawing: Vec<vk::Fence> = vec![];
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_create_info =
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

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
        })
    }

    fn create_framebuffers(
        &mut self,
        logical_device: &ash::Device,
        renderpass: vk::RenderPass,
    ) -> Result<(), vk::Result> {
        for iv in &self.imageviews {
            let iview = [*iv];
            let frame_buffer_info = vk::FramebufferCreateInfo::builder()
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

    unsafe fn cleanup(&mut self, logical_device: &ash::Device) {
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
struct Pipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

impl Pipeline {
    fn cleanup(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.destroy_pipeline(self.pipeline, None);
            logical_device.destroy_pipeline_layout(self.layout, None);
        }
    }
    fn init(
        logical_device: &ash::Device,
        swapchain: &SwapchainComp,
        renderpass: &vk::RenderPass,
    ) -> Result<Pipeline, vk::Result> {
        let mut vert_spv_file = Cursor::new(&include_bytes!("../shaders/vert.spv")[..]);
        let mut frag_spv_file = Cursor::new(&include_bytes!("../shaders/frag.spv")[..]);
        let vertex_code =
            ash::util::read_spv(&mut vert_spv_file).expect("Failed to read vertex shader spv file");
        let frag_code =
            ash::util::read_spv(&mut frag_spv_file).expect("Failed to read vertex shader spv file");

        let vertex_shader_create_info = vk::ShaderModuleCreateInfo::builder().code(&vertex_code);
        let vertex_shader_module =
            unsafe { logical_device.create_shader_module(&vertex_shader_create_info, None)? };
        let fragment_shader_create_info = vk::ShaderModuleCreateInfo::builder().code(&frag_code);
        let fragment_shader_module =
            unsafe { logical_device.create_shader_module(&fragment_shader_create_info, None)? };
        let main_function_name = std::ffi::CString::new("main").unwrap();
        let vertex_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&main_function_name);
        let fragment_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&main_function_name);
        let shader_stages = vec![vertex_shader_stage.build(), fragment_shader_stage.build()];

        let vertex_attrib_descs = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                offset: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 1,
                offset: 0,
                format: vk::Format::R32_SFLOAT,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 2,
                offset: 4,
                format: vk::Format::R32G32B32A32_SFLOAT,
            },
        ];

        let vertex_binding_descs = [
            vk::VertexInputBindingDescription {
                binding: 0,
                stride: 16,
                input_rate: vk::VertexInputRate::VERTEX,
            },
            vk::VertexInputBindingDescription {
                binding: 1,
                stride: 20,
                input_rate: vk::VertexInputRate::VERTEX,
            },
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_attrib_descs)
            .vertex_binding_descriptions(&vertex_binding_descs);
        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewports = [vk::Viewport {
            x: 0.,
            y: 0.,
            width: swapchain.extent.width as f32,
            height: swapchain.extent.height as f32,
            min_depth: 0.,
            max_depth: 1.,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
        }];

        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::NONE)
            .polygon_mode(vk::PolygonMode::FILL);
        let multisampler_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let colourblend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build()];
        let colourblend_info =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&colourblend_attachments);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder();
        let pipeline_layout =
            unsafe { logical_device.create_pipeline_layout(&pipeline_layout_info, None) }?;
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampler_info)
            .color_blend_state(&colourblend_info)
            .layout(pipeline_layout)
            .render_pass(*renderpass)
            .subpass(0);
        let graphics_pipeline = unsafe {
            logical_device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info.build()],
                    None,
                )
                .expect("A problem with the pipeline creation")
        }[0];
        unsafe {
            logical_device.destroy_shader_module(fragment_shader_module, None);
            logical_device.destroy_shader_module(vertex_shader_module, None);
        }
        Ok(Pipeline {
            pipeline: graphics_pipeline,
            layout: pipeline_layout,
        })
    }
}

struct Pools {
    command_pool_graphics: vk::CommandPool,
    command_pool_transfer: vk::CommandPool,
}

impl Pools {
    fn init(
        logical_device: &ash::Device,
        queue_families: &QueueFamilies,
    ) -> Result<Pools, vk::Result> {
        let graphics_command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_families.graphics_q_index.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool_graphics =
            unsafe { logical_device.create_command_pool(&graphics_command_pool_info, None)? };
        let transfer_command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_families.transfer_q_index.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool_transfer =
            unsafe { logical_device.create_command_pool(&transfer_command_pool_info, None) }?;

        Ok(Pools {
            command_pool_graphics,
            command_pool_transfer,
        })
    }

    fn cleanup(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.destroy_command_pool(self.command_pool_graphics, None);
            logical_device.destroy_command_pool(self.command_pool_transfer, None);
        }
    }
}

fn init_renderpass(
    physical_device: vk::PhysicalDevice,
    logical_device: &ash::Device,
    format: vk::Format,
) -> Result<RenderPass, vk::Result> {
    let attachments = [vk::AttachmentDescription::builder()
        .format(format)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .samples(vk::SampleCountFlags::TYPE_1)
        .build()];
    let color_attachment_references = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpasses = [vk::SubpassDescription::builder()
        .color_attachments(&color_attachment_references)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .build()];
    let subpass_dependencies = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_subpass(0)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .build()];
    let renderpass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&subpass_dependencies);
    let renderpass = unsafe { logical_device.create_render_pass(&renderpass_info, None)? };
    Ok(renderpass)
}

fn create_command_buffers(
    logical_device: &ash::Device,
    pools: &Pools,
    amount: usize,
) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(pools.command_pool_graphics)
        .command_buffer_count(amount as u32);
    unsafe { logical_device.allocate_command_buffers(&command_buffer_allocate_info) }
}
fn fill_commandbuffers(
    commandbuffers: &[vk::CommandBuffer],
    logical_device: &ash::Device,
    renderpass: &vk::RenderPass,
    swapchain: &SwapchainComp,
    pipeline: &Pipeline,
    vb: &vk::Buffer,
    vb2: &vk::Buffer,
) -> Result<(), vk::Result> {
    for (i, &commandbuffer) in commandbuffers.iter().enumerate() {
        let commandbuffer_begininfo = vk::CommandBufferBeginInfo::builder();
        unsafe {
            logical_device.begin_command_buffer(commandbuffer, &commandbuffer_begininfo)?;
        }
        let clearvalues = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.08, 1.0],
            },
        }];
        let renderpass_begininfo = vk::RenderPassBeginInfo::builder()
            .render_pass(*renderpass)
            .framebuffer(swapchain.framebuffers[i])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.extent,
            })
            .clear_values(&clearvalues);
        unsafe {
            logical_device.cmd_begin_render_pass(
                commandbuffer,
                &renderpass_begininfo,
                vk::SubpassContents::INLINE,
            );
            logical_device.cmd_bind_pipeline(
                commandbuffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );
            logical_device.cmd_bind_vertex_buffers(commandbuffer, 0, &[*vb], &[0]);
            logical_device.cmd_bind_vertex_buffers(commandbuffer, 1, &[*vb2], &[0]);
            logical_device.cmd_draw(commandbuffer, 6, 1, 0, 0);
            logical_device.cmd_end_render_pass(commandbuffer);
            logical_device.end_command_buffer(commandbuffer)?;
        }
    }
    Ok(())
}

struct GpuBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
    logical_device: ash::Device,
    size_in_bytes: u64,
    buffer_usage: vk::BufferUsageFlags,
    allocation_name: String,
    allocation_location: gpu_allocator::MemoryLocation,
    linear: bool,
    allocation_scheme: AllocationScheme,
}

impl GpuBuffer {
    fn new(
        allocator: &mut Allocator,
        size_in_bytes: u64,
        buffer_usage: vk::BufferUsageFlags,
        logical_device: ash::Device,
        allocation_name: String,
        allocation_location: gpu_allocator::MemoryLocation,
        linear: bool,
        allocation_scheme: AllocationScheme,
    ) -> Result<GpuBuffer, vk::Result> {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size_in_bytes)
            .usage(buffer_usage);
        let buffer = unsafe { logical_device.create_buffer(&buffer_create_info, None)? };
        let requirements = unsafe { logical_device.get_buffer_memory_requirements(buffer) };
        let allocation_info = AllocationCreateDesc {
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
            logical_device: logical_device,
            size_in_bytes,
            buffer_usage,
            allocation_name: allocation_name.to_string(),
            allocation_location,
            linear,
            allocation_scheme,
        })
    }

    fn write_to_memory<T: Sized>(
        &mut self,
        allocator: &mut Allocator,
        data: &[T],
    ) -> Result<(), vk::Result> {
        let bytes_to_write = (data.len() * std::mem::size_of::<T>()) as u64;
        if bytes_to_write > self.size_in_bytes {
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
        println!("Pituus: {:?}", data.len());
        Ok(())
    }
}

struct Kompura {
    window: winit::window::Window,
    entry: ash::Entry,
    instance: ash::Instance,
    debug: std::mem::ManuallyDrop<Debug>,
    surfaces: std::mem::ManuallyDrop<Surface>,
    physical_device: vk::PhysicalDevice,
    physical_device_properties: vk::PhysicalDeviceProperties,
    queue_families: QueueFamilies,
    queues: Queues,
    device: ash::Device,
    swapchain: SwapchainComp,
    renderpass: vk::RenderPass,
    pipeline: Pipeline,
    pools: Pools,
    command_buffers: Vec<CommandBuffer>,
    allocator: std::mem::ManuallyDrop<Allocator>,
    buffers: Vec<GpuBuffer>,
}

impl Kompura {
    fn init(window: winit::window::Window) -> Result<Kompura, Box<dyn std::error::Error>> {
        let entry = unsafe { Entry::load() }?;

        let layer_names = vec!["VK_LAYER_KHRONOS_validation"];
        let instance = init_instance(&entry, &layer_names)?;
        let debug = Debug::init(&entry, &instance)?;
        let surfaces = Surface::init(&window, &entry, &instance)?;

        let (physical_device, physical_device_properties) =
            init_physical_device_and_properties(&instance)?;

        let queue_families = QueueFamilies::init(&instance, physical_device, &surfaces)?;

        let (logical_device, queues) =
            init_device_and_queues(&instance, physical_device, &queue_families, &layer_names)?;
        let mut swapchain = SwapchainComp::init(
            &instance,
            physical_device,
            &logical_device,
            &surfaces,
            &queue_families,
            &queues,
        )?;

        let renderpass = init_renderpass(
            physical_device,
            &logical_device,
            swapchain.surface_format.format,
        )?;
        swapchain.create_framebuffers(&logical_device, renderpass)?;
        let pipeline = Pipeline::init(&logical_device, &swapchain, &renderpass)?;
        let pools = Pools::init(&logical_device, &queue_families)?;

        let allocator_create_info = AllocatorCreateDesc {
            instance: instance.clone(),
            device: logical_device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false, // Ideally, check the BufferDeviceAddressFeatures struct.
            allocation_sizes: Default::default(),
        };

        let mut allocator = Allocator::new(&allocator_create_info)?;

        let mut gpu_buffer1 = GpuBuffer::new(
            &mut allocator,
            96,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            logical_device.clone(),
            "buffer1".to_string(),
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            AllocationScheme::GpuAllocatorManaged,
        )?;

        let data = [
            0.5f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.2f32, 0.0f32, 1.0f32, -0.5f32, 0.0f32,
            0.0f32, 1.0f32, -0.9f32, -0.9f32, 0.0f32, 1.0f32, 0.3f32, -0.8f32, 0.0f32, 1.0f32,
            0.0f32, -0.6f32, 0.0f32, 1.0f32,
        ];

        gpu_buffer1.write_to_memory(&mut allocator, &data)?;

        let mut gpu_buffer2 = GpuBuffer::new(
            &mut allocator,
            120,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            logical_device.clone(),
            "buffer2".to_string(),
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            AllocationScheme::GpuAllocatorManaged,
        )?;

        let data2 = [
            15.0f32, 0.0f32, 1.0f32, 0.0f32, 1.0f32, 15.0f32, 0.0f32, 1.0f32, 0.0f32, 1.0f32,
            15.0f32, 0.0f32, 1.0f32, 0.0f32, 1.0f32, 1.0f32, 0.8f32, 0.7f32, 0.0f32, 1.0f32,
            1.0f32, 0.8f32, 0.7f32, 0.0f32, 1.0f32, 1.0f32, 0.0f32, 0.0f32, 1.0f32, 1.0f32,
        ];

        gpu_buffer2.write_to_memory(&mut allocator, &data2)?;

        let command_buffers =
            create_command_buffers(&logical_device, &pools, swapchain.framebuffers.len())?;

        fill_commandbuffers(
            &command_buffers,
            &logical_device,
            &renderpass,
            &swapchain,
            &pipeline,
            &gpu_buffer1.buffer,
            &gpu_buffer2.buffer,
        )?;

        Ok(Kompura {
            window,
            entry,
            instance,
            debug: std::mem::ManuallyDrop::new(debug),
            surfaces: std::mem::ManuallyDrop::new(surfaces),
            physical_device,
            physical_device_properties,
            queue_families,
            queues,
            device: logical_device,
            swapchain,
            renderpass,
            pipeline,
            pools,
            command_buffers,
            allocator: std::mem::ManuallyDrop::new(allocator),
            buffers: vec![gpu_buffer1, gpu_buffer2],
        })
    }
}

impl Drop for Kompura {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("something wrong while waiting");

            for b in &mut self.buffers {
                self.allocator
                    .free(std::mem::take(&mut b.allocation))
                    .expect("problem with buffer destruction");
                self.device.destroy_buffer(b.buffer, None);
            }

            std::mem::ManuallyDrop::drop(&mut self.allocator);
            self.pools.cleanup(&self.device);
            self.pipeline.cleanup(&self.device);
            self.device.destroy_render_pass(self.renderpass, None);
            self.swapchain.cleanup(&self.device);
            self.device.destroy_device(None);
            std::mem::ManuallyDrop::drop(&mut self.surfaces);
            std::mem::ManuallyDrop::drop(&mut self.debug);
            self.instance.destroy_instance(None)
        };
    }
}

struct Model<V, I> {
    vertex_data: Vec<V>,
    handle_to_index: std::collections::HashMap<usize, usize>,
    handles: Vec<usize>,
    instances: Vec<I>,
    first_invisible: usize,
    next_handle: usize,
    vertex_buffer: Option<GpuBuffer>,
    instance_buffer: Option<GpuBuffer>,
    logical_device: ash::Device,
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
    fn get_mut(&mut self, handle: usize) -> Option<&mut I> {
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
    fn insert_visibly(&mut self, element: I) -> usize {
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

    fn update_vertexbuffer(&mut self, allocator: &mut Allocator) -> Result<(), vk::Result> {
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
                true,
                AllocationScheme::GpuAllocatorManaged,
            )?;
            buffer.write_to_memory(allocator, &self.vertex_data)?;
            self.vertex_buffer = Some(buffer);
            Ok(())
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let eventloop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&eventloop)?;
    let mut kompura = Kompura::init(window)?;

    eventloop.run(move |event, _, controlflow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *controlflow = winit::event_loop::ControlFlow::Exit;
        }
        Event::MainEventsCleared => {
            kompura.window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            let (image_index, _) = unsafe {
                kompura
                    .swapchain
                    .swapchain_loader
                    .acquire_next_image(
                        kompura.swapchain.swapchain,
                        std::u64::MAX,
                        kompura.swapchain.image_available[kompura.swapchain.current_image],
                        vk::Fence::null(),
                    )
                    .expect("image acquisition trouble")
            };
            unsafe {
                kompura
                    .device
                    .wait_for_fences(
                        &[kompura.swapchain.may_begin_drawing[kompura.swapchain.current_image]],
                        true,
                        std::u64::MAX,
                    )
                    .expect("fence-waiting");
                kompura
                    .device
                    .reset_fences(&[
                        kompura.swapchain.may_begin_drawing[kompura.swapchain.current_image]
                    ])
                    .expect("resetting fences");
            }
            let semaphores_available =
                [kompura.swapchain.image_available[kompura.swapchain.current_image]];
            let waiting_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let semaphores_finished =
                [kompura.swapchain.rendering_finished[kompura.swapchain.current_image]];
            let commandbuffers = [kompura.command_buffers[image_index as usize]];
            let submit_info = [vk::SubmitInfo::builder()
                .wait_semaphores(&semaphores_available)
                .wait_dst_stage_mask(&waiting_stages)
                .command_buffers(&commandbuffers)
                .signal_semaphores(&semaphores_finished)
                .build()];
            unsafe {
                kompura
                    .device
                    .queue_submit(
                        kompura.queues.graphics_queue,
                        &submit_info,
                        kompura.swapchain.may_begin_drawing[kompura.swapchain.current_image],
                    )
                    .expect("queue submission");
            };
            let swapchains = [kompura.swapchain.swapchain];
            let indices = [image_index];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&semaphores_finished)
                .swapchains(&swapchains)
                .image_indices(&indices);
            unsafe {
                kompura
                    .swapchain
                    .swapchain_loader
                    .queue_present(kompura.queues.graphics_queue, &present_info)
                    .expect("queue presentation");
            };
            kompura.swapchain.current_image =
                (kompura.swapchain.current_image + 1) % kompura.swapchain.amount_of_images as usize;
        }
        _ => {}
    });

    Ok(())
}
