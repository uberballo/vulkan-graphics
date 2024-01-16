use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::io::Cursor;
use std::os::raw::c_void;

use ash::extensions::khr::Swapchain;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::CommandBuffer;
use ash::vk::DescriptorPool;
use ash::vk::DescriptorSet;
use ash::vk::PhysicalDevice;
use ash::vk::Pipeline;
use ash::vk::RenderPass;
use ash::Entry;
use ash::Instance;
use gpuBuffer::GpuBuffer;
use gpu_allocator::vulkan::*;
use model::InstanceData;
use pools::Pools;
use queues::init_device_and_queues;
use queues::Queues;
use swapchainComp::SwapchainComp;
use winit::event::WindowEvent;
use winit::platform::windows::DeviceIdExtWindows;
use winit::platform::windows::WindowExtWindows;
mod model;
use model::Model;
mod camera;
use camera::Camera;
mod debug;
use debug::Debug;
mod surface;
use surface::Surface;
mod queueFamilies;
use queueFamilies::QueueFamilies;
mod pipelineComp;
use pipelineComp::PipelineComp;
mod gpuBuffer;
mod pools;
mod queues;
mod swapchainComp;

extern crate nalgebra as na;

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
        .pfn_user_callback(Some(debug::vulkan_debug_utils_callback));

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

fn init_renderpass(
    logical_device: &ash::Device,
    format: vk::Format,
) -> Result<RenderPass, vk::Result> {
    let attachments = [
        vk::AttachmentDescription::builder()
            .format(format)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .samples(vk::SampleCountFlags::TYPE_1)
            .build(),
        vk::AttachmentDescription::builder()
            .format(vk::Format::D32_SFLOAT)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .samples(vk::SampleCountFlags::TYPE_1)
            .build(),
    ];
    let color_attachment_references = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_attachment_reference = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let subpasses = [vk::SubpassDescription::builder()
        .color_attachments(&color_attachment_references)
        .depth_stencil_attachment(&depth_attachment_reference)
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
    pipeline: PipelineComp,
    pools: Pools,
    command_buffers: Vec<CommandBuffer>,
    allocator: std::mem::ManuallyDrop<Allocator>,
    buffers: Vec<GpuBuffer>,
    models: Vec<Model<[f32; 3], InstanceData>>,
    uniform_buffer: GpuBuffer,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<DescriptorSet>,
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

        let allocator_create_info = AllocatorCreateDesc {
            instance: instance.clone(),
            device: logical_device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false, // Ideally, check the BufferDeviceAddressFeatures struct.
            allocation_sizes: Default::default(),
        };

        let mut allocator = Allocator::new(&allocator_create_info)?;
        let mut swapchain = SwapchainComp::init(
            &instance,
            physical_device,
            &logical_device,
            &surfaces,
            &queue_families,
            &queues,
            &mut allocator,
        )?;

        let renderpass = init_renderpass(&logical_device, swapchain.surface_format.format)?;
        swapchain.create_framebuffers(&logical_device, renderpass)?;
        let pipeline = PipelineComp::init(&logical_device, &swapchain, &renderpass)?;
        let pools = Pools::init(&logical_device, &queue_families)?;

        let gpu_buffer1 = GpuBuffer::new(
            &mut allocator,
            96,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            logical_device.clone(),
            "buffer1".to_string(),
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            AllocationScheme::GpuAllocatorManaged,
        )?;

        let gpu_buffer2 = GpuBuffer::new(
            &mut allocator,
            120,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            logical_device.clone(),
            "buffer2".to_string(),
            gpu_allocator::MemoryLocation::CpuToGpu,
            true,
            AllocationScheme::GpuAllocatorManaged,
        )?;

        let command_buffers =
            create_command_buffers(&logical_device, &pools, swapchain.framebuffers.len())?;

        let mut uniform_buffer = GpuBuffer::new(
            &mut allocator,
            64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            logical_device.clone(),
            "".to_string(),
            gpu_allocator::MemoryLocation::CpuToGpu,
            false,
            AllocationScheme::GpuAllocatorManaged,
        )?;

        let cameratransform: [[f32; 4]; 4] = na::Matrix4::identity().into();
        uniform_buffer
            .write_to_memory(&mut allocator, &cameratransform)
            .unwrap();

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: swapchain.amount_of_images,
        }];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(swapchain.amount_of_images)
            .pool_sizes(&pool_sizes);
        let descriptor_pool =
            unsafe { logical_device.create_descriptor_pool(&descriptor_pool_info, None) }?;

        let desc_layouts =
            vec![pipeline.descriptor_set_layouts[0]; swapchain.amount_of_images as usize];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&desc_layouts);
        let descriptor_sets =
            unsafe { logical_device.allocate_descriptor_sets(&descriptor_set_allocate_info) }?;

        for (_, descset) in descriptor_sets.iter().enumerate() {
            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: uniform_buffer.buffer,
                offset: 0,
                range: 64,
            }];
            let desc_sets_write = [vk::WriteDescriptorSet::builder()
                .dst_set(*descset)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos)
                .build()];

            unsafe { logical_device.update_descriptor_sets(&desc_sets_write, &[]) };
        }

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
            models: vec![],
            uniform_buffer,
            descriptor_pool,
            descriptor_sets,
        })
    }

    fn update_command_buffer(&mut self, index: usize) -> Result<(), vk::Result> {
        let command_buffer = self.command_buffers[index];
        let commandbuffer_begininfo = vk::CommandBufferBeginInfo::builder();
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &commandbuffer_begininfo)?;
        }
        let clearvalues = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.08, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let renderpass_begininfo = vk::RenderPassBeginInfo::builder()
            .render_pass(self.renderpass)
            .framebuffer(self.swapchain.framebuffers[index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent,
            })
            .clear_values(&clearvalues);
        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &renderpass_begininfo,
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &[self.descriptor_sets[index]],
                &[],
            );
            for m in &self.models {
                m.draw(command_buffer);
            }
            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)?;
        }
        Ok(())
    }
}

impl Drop for Kompura {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("something wrong while waiting");

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.allocator
                .free(std::mem::take(&mut self.uniform_buffer.allocation))
                .expect("problem with buffer destruction");
            self.device.destroy_buffer(self.uniform_buffer.buffer, None);
            for b in &mut self.buffers {
                self.allocator
                    .free(std::mem::take(&mut b.allocation))
                    .expect("problem with buffer destruction");
                self.device.destroy_buffer(b.buffer, None);
            }
            for m in &mut self.models {
                if let Some(vb) = &mut m.vertex_buffer {
                    self.allocator
                        .free(std::mem::take(&mut vb.allocation))
                        .expect("problem with buffer destruction");
                    self.device.destroy_buffer(vb.buffer, None);
                };
                if let Some(ib) = &mut m.instance_buffer {
                    self.allocator
                        .free(std::mem::take(&mut ib.allocation))
                        .expect("problem with buffer destruction");
                    self.device.destroy_buffer(ib.buffer, None);
                };
            }

            std::mem::ManuallyDrop::drop(&mut self.allocator);
            self.pools.cleanup(&self.device);
            self.pipeline.cleanup(&self.device);
            self.device.destroy_render_pass(self.renderpass, None);
            self.swapchain.cleanup(&self.device, &mut self.allocator);
            self.device.destroy_device(None);
            std::mem::ManuallyDrop::drop(&mut self.surfaces);
            std::mem::ManuallyDrop::drop(&mut self.debug);
            self.instance.destroy_instance(None)
        };
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let eventloop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&eventloop)?;
    let mut kompura = Kompura::init(window)?;
    let mut camera = Camera::default();
    let mut cube = Model::cube(&kompura.device);
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, 0.5))
            * na::Matrix4::new_scaling(0.5))
        .into(),
        colour: [0.2, 0.4, 1.0],
    });
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(0.05, 0.05, 0.0))
            * na::Matrix4::new_scaling(0.5))
        .into(),
        colour: [1.0, 1.0, 0.2],
    });
    for i in 0..10 {
        for j in 0..10 {
            cube.insert_visibly(InstanceData {
                model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(
                    i as f32 * 0.2 - 1.0,
                    j as f32 * 0.2 - 1.0,
                    0.5,
                )) * na::Matrix4::new_scaling(0.03))
                .into(),
                colour: [1.0, i as f32 * 0.07, j as f32 * 0.07],
            });
            cube.insert_visibly(InstanceData {
                model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(
                    i as f32 * 0.2 - 1.0,
                    0.0,
                    j as f32 * 0.2 - 1.0,
                )) * na::Matrix4::new_scaling(0.02))
                .into(),
                colour: [i as f32 * 0.07, j as f32 * 0.07, 1.0],
            });
        }
    }
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Matrix4::from_scaled_axis(na::Vector3::new(0.0, 0.0, 1.4))
            * na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.5, 0.0))
            * na::Matrix4::new_scaling(0.1))
        .into(),
        colour: [0.0, 0.5, 0.0],
    });
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(0.5, 0.0, 0.0))
            * na::Matrix4::new_nonuniform_scaling(&na::Vector3::new(0.5, 0.01, 0.01)))
        .into(),
        colour: [1.0, 0.5, 0.5],
    });
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.5, 0.0))
            * na::Matrix4::new_nonuniform_scaling(&na::Vector3::new(0.01, 0.5, 0.01)))
        .into(),
        colour: [0.5, 1.0, 0.5],
    });
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, 0.0))
            * na::Matrix4::new_nonuniform_scaling(&na::Vector3::new(0.01, 0.01, 0.5)))
        .into(),
        colour: [0.5, 0.5, 1.0],
    });
    cube.update_vertex_buffer(&mut kompura.allocator);
    cube.update_instance_buffer(&mut kompura.allocator);
    kompura.models = vec![cube];
    use winit::event::{Event, WindowEvent};
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
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput { input, .. },
            ..
        } => match input {
            winit::event::KeyboardInput {
                state: winit::event::ElementState::Pressed,
                virtual_keycode: Some(keycode),
                ..
            } => match keycode {
                winit::event::VirtualKeyCode::Right => {
                    camera.turn_right(0.1);
                }
                winit::event::VirtualKeyCode::Left => {
                    camera.turn_left(0.1);
                }
                winit::event::VirtualKeyCode::Up => {
                    camera.move_forward(0.05);
                }
                winit::event::VirtualKeyCode::Down => {
                    camera.move_backward(0.05);
                }
                winit::event::VirtualKeyCode::PageUp => {
                    camera.turn_up(0.02);
                }
                winit::event::VirtualKeyCode::PageDown => {
                    camera.turn_down(0.02);
                }
                _ => {}
            },
            _ => {}
        },
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
            camera.update_buffer(&mut kompura.allocator, &mut kompura.uniform_buffer);
            for m in &mut kompura.models {
                m.update_instance_buffer(&mut kompura.allocator);
            }
            kompura
                .update_command_buffer(image_index as usize)
                .expect("updating the command buffer");
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
}
