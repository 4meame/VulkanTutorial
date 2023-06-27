mod platform;
mod debug;
mod tools;
mod structures;
use ash::{vk, Device};
use ash::{Entry, Instance};
use debug::ValidationInfo;
use structures::{QueueFamilyIndices, SurfaceStuff, DeviceExtension, SwapchainStuff, SwapchainSupportDetail};
use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use std::collections::HashSet;
use std::os::raw::{c_char, c_void};
use std::ffi::{CString};
use std::path::Path;
use std::ptr;

const WINDOW_TITLE: &'static str = "Vulkan Renderer By Rust";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

const VALIDATION: ValidationInfo = ValidationInfo {
    is_enable: cfg!(debug_assertions),
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};

const DEVICE_EXTENSIONS: DeviceExtension = DeviceExtension {
    names: ["VK_KHR_swapchain"],
};

struct VulkanApplication {
    entry: Entry,
    // root: represent a vulkan api global context
    instance: Instance,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    // handle of reference to the actual GPU
    physical_device: vk::PhysicalDevice,
    // actual driver on the GPU
    logical_device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_loader: ash::extensions::khr::Swapchain,
    // perform rendering into the screen
    swapchain: vk::SwapchainKHR,
    // actual image object
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    // wrapper for image
    swapchain_imageviews: Vec<vk::ImageView>,

    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline
}

impl VulkanApplication {
    pub fn new(window: &winit::window::Window) -> VulkanApplication {
        let entry = unsafe { Entry::load() }.unwrap();
        let instance = Self::create_instance(&entry);
        let (debug_utils_loader, debug_messenger) = debug::setup_debug_utils(VALIDATION.is_enable, &entry, &instance);
        let surface_stuff = VulkanApplication::create_surface(&entry, &instance, window);
        let physical_device = VulkanApplication::pick_physical_device(&instance, &surface_stuff);
        let (logical_device, queue_family_indices) = VulkanApplication::create_logical_device(&instance, physical_device, &surface_stuff, &VALIDATION);
        let graphics_queue = unsafe {
            logical_device
                .get_device_queue(queue_family_indices.graphics_family.unwrap(), 0)
        };
        let present_queue = unsafe {
            logical_device
                .get_device_queue(queue_family_indices.present_family.unwrap(), 0)
        };
        
        let swapchain_stuff = VulkanApplication::create_swapchain(&instance, &logical_device, physical_device, &surface_stuff, &queue_family_indices);
        let swapchain_imageviews = VulkanApplication::create_swapchain_imageviews(&logical_device, &swapchain_stuff.swapchain_images, swapchain_stuff.swapchain_format);

        let render_pass = VulkanApplication::create_render_pass(&logical_device, swapchain_stuff.swapchain_format);
        let (pipeline_layout, graphics_pipeline) = VulkanApplication::create_pipeline(&logical_device, render_pass, swapchain_stuff.swapchain_extent);

        VulkanApplication {
            entry,
            instance,
            debug_utils_loader,
            debug_messenger,
            surface_loader: surface_stuff.surface_loader,
            surface: surface_stuff.surface,
            physical_device,
            logical_device,
            graphics_queue,
            present_queue,
            swapchain_loader: swapchain_stuff.swapchain_loader,
            swapchain: swapchain_stuff.swapchain,
            swapchain_images: swapchain_stuff.swapchain_images,
            swapchain_format: swapchain_stuff.swapchain_format,
            swapchain_extent: swapchain_stuff.swapchain_extent,
            swapchain_imageviews,
            render_pass,
            pipeline_layout,
            graphics_pipeline
        }
    }

    fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
        .with_title(WINDOW_TITLE)
        .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build(event_loop)
        .expect("Failed to create window!")
    }

    fn create_instance(entry: &Entry) -> Instance {
        let app_name = CString::new("Vulkan Renderer").expect("CString(app_name)::new failed");
        let engine_name = CString::new("Tusk Engine").expect("CString(engine_name)::new failed");
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            application_version: vk::make_api_version(0, 0, 1, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 0, 1, 0),
            api_version: vk::make_api_version(0, 1, 0, 0)
        };

        let debug_utils_create_info = debug::populate_debug_messenger_create_info();

        let extensions_names = platform::required_extension_names();

        let required_validation_layer_raw_names: Vec<CString> = VALIDATION
            .required_validation_layers
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();

        let enabled_layer_names: Vec<*const i8> = required_validation_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: if VALIDATION.is_enable {
                &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
                    as *const c_void
            } else {
                ptr::null()
            },
            flags: vk::InstanceCreateFlags::empty(),
            p_application_info: &app_info,
            enabled_layer_count: if VALIDATION.is_enable {
                enabled_layer_names.len()
            } else {
                0
            } as u32,
            pp_enabled_layer_names: if VALIDATION.is_enable {
                enabled_layer_names.as_ptr()
            } else {
                ptr::null()
            },
            enabled_extension_count: extensions_names.len() as u32,
            pp_enabled_extension_names: extensions_names.as_ptr()
        };

        let instance: ash::Instance = unsafe {
            entry.create_instance(&create_info, None)
            .expect("Failed to create Instance!")
        };
        
        instance
    }

    fn create_surface(entry: &Entry, instance: &Instance, window: &winit::window::Window) -> SurfaceStuff {
        let surface = unsafe {
            platform::create_surface(entry, instance, window)
            .expect("Failed to create Surface!")
        };
        let surface_loader = ash::extensions::khr::Surface::new(entry, instance);

        SurfaceStuff {
            surface_loader,
            surface
        }
    }

    fn pick_physical_device(instance: &Instance, surface_stuff: &SurfaceStuff) -> vk::PhysicalDevice {
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate Physical Devices!")
        };
        println!(
            "{} devices (GPU) found with vulkan support",
            physical_devices.len()
        );
        let mut result = None;
        for &physical_device in physical_devices.iter() {
            if VulkanApplication::is_physical_device_suitable(&instance, &surface_stuff, physical_device) {
                if result.is_none() {
                    result = Some(physical_device);
                }
            }
        }
        match result {
            None => panic!("Failed to find a suitable GPU!"),
            Some(physical_device) => physical_device,
        }
    }

    fn is_physical_device_suitable(instance: &Instance, surface_stuff: &SurfaceStuff, physical_device: vk::PhysicalDevice) -> bool {
        let device_properties = unsafe {
            instance.get_physical_device_properties(physical_device)
        };
        let device_features = unsafe {
            instance.get_physical_device_features(physical_device)
        };
        let device_queue_families = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };

        let device_type = match device_properties.device_type {
            vk::PhysicalDeviceType::CPU => "CPU",
            vk::PhysicalDeviceType::DISCRETE_GPU => "DISCRETE GPU",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "INTEGRATED_GPU",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "VIRTUAL_GPU",
            vk::PhysicalDeviceType::OTHER => "OTHER",
            _ => panic!(),
        };

        let device_name = tools::vk_str_to_string(&device_properties.device_name);
        println!(
            "\tDevice Name: {}, id: {}, type: {}",
            device_name, device_properties.device_id, device_type
        );

        let major_version = vk::api_version_major(device_properties.api_version);
        let minor_version = vk::api_version_minor(device_properties.api_version);
        let patch_version = vk::api_version_patch(device_properties.api_version);
        println!(
            "\tAPI Version: {}.{}.{}",
            major_version, minor_version, patch_version
        );

        println!("\tSupport Queue Family: {}", device_queue_families.len());
        println!("\t\tQueue Count | Graphics, Compute, Transfer, Sparse Binding");
        for queue_family in device_queue_families.iter() {
            let is_graphics_support = if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                "support"
            } else {
                "unsupport"
            };
            let is_compute_support = if queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                "support"
            } else {
                "unsupport"
            };
            let is_transfer_support = if queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                "support"
            } else {
                "unsupport"
            };
            let is_sparse_support = if queue_family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                "support"
            } else {
                "unsupport"
            };

            println!(
                "\t\t{}\t    | {},  {},  {},  {}",
                queue_family.queue_count,
                is_graphics_support,
                is_compute_support,
                is_transfer_support,
                is_sparse_support
            );
        }

        // there are plenty of features
        println!(
            "\tGeometry Shader support: {}",
            if device_features.geometry_shader == 1 {
                "support"
            } else {
                "unsupport"
            }
        );

        let indices: QueueFamilyIndices = VulkanApplication::find_queue_family(&instance, &surface_stuff, physical_device);

        let is_queue_family_supported = indices.is_complete();
        let is_device_extension_supported = VulkanApplication::check_device_extension_supported(instance, physical_device);
        let is_swapchain_supported = if is_device_extension_supported {
            let swapchain_support = VulkanApplication::find_swapchain_support(physical_device, surface_stuff);
            !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
        } else {
            false
        };

        return is_queue_family_supported
            && is_device_extension_supported
            && is_swapchain_supported;
    }

    fn find_queue_family(instance: &Instance, surface_stuff: &SurfaceStuff, physical_device: vk::PhysicalDevice) -> QueueFamilyIndices {
        let queue_families = unsafe { 
            instance.get_physical_device_queue_family_properties(physical_device) 
        };

        let mut queue_family_indices = QueueFamilyIndices::new();

        let mut index = 0;
        for queue_family in queue_families.iter() {
            if queue_family.queue_count > 0 && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                queue_family_indices.graphics_family = Some(index);
            }

            let is_present_support = unsafe {
                surface_stuff
                    .surface_loader
                    .get_physical_device_surface_support(physical_device, index as u32, surface_stuff.surface)
                    .expect("Failed to get physical deviece surface support!")
            };
            if queue_family.queue_count > 0 && is_present_support {
                queue_family_indices.present_family = Some(index)
            }

            if queue_family_indices.is_complete() {
                break;
            }

            index += 1;
        }

        queue_family_indices
    }

    fn create_logical_device(instance: &Instance, physical_device: vk::PhysicalDevice, surface_stuff: &SurfaceStuff, validation: &ValidationInfo) -> (Device, QueueFamilyIndices) {
        let indices = VulkanApplication::find_queue_family(instance, &surface_stuff, physical_device);
        // right now we are only interested in graphics queue
        let queue_priorities = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: indices.graphics_family.unwrap(),
            p_queue_priorities: queue_priorities.as_ptr(),
            queue_count: queue_priorities.len() as u32,
        };

        let physical_device_features = vk::PhysicalDeviceFeatures {
            // default no feature
            ..Default::default()
        };
        let required_validation_layer_raw_names: Vec<CString> = validation
            .required_validation_layers
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        let enabled_layer_names: Vec<*const c_char> = required_validation_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        // currently just enable the Swapchain extension.
        let enable_extension_names = [
            ash::extensions::khr::Swapchain::name().as_ptr(),
        ];

        let device_create_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceCreateFlags::empty(),
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create_info,
            enabled_layer_count: if validation.is_enable {
                enabled_layer_names.len()
            } else {
                0
            } as u32,
            pp_enabled_layer_names: if validation.is_enable {
                enabled_layer_names.as_ptr()
            } else {
                ptr::null()
            },
            enabled_extension_count: enable_extension_names.len() as u32,
            pp_enabled_extension_names: enable_extension_names.as_ptr(),
            p_enabled_features: &physical_device_features,
        };

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create Logical Device!")
        };

        (device, indices)
    }

    fn check_device_extension_supported(instance: &Instance, physical_device: vk::PhysicalDevice) -> bool {
        let available_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .expect("Failed to get device extension properties!")
        };
        let mut available_extension_names = vec![];
        println!("\tAvailable Device Extensions: ");
        for extension in available_extensions.iter() {
            let extension_name = tools::vk_str_to_string(&extension.extension_name);
            println!(
                "\t\tName: {}, Version: {}",
                extension_name, extension.spec_version
            );

            available_extension_names.push(extension_name);
        }
        let mut required_extensions = HashSet::new();
        for extension in DEVICE_EXTENSIONS.names.iter() {
            required_extensions.insert(extension.to_string());
        }
        for extension_name in available_extension_names.iter() {
            required_extensions.remove(extension_name);
        }
        return required_extensions.is_empty();
    }

    fn find_swapchain_support(physical_device: vk::PhysicalDevice, surface_stuff: &SurfaceStuff) -> SwapchainSupportDetail {
        unsafe {
            let capabilities = surface_stuff
                .surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface_stuff.surface)
                .expect("Failed to find surface capabilities!");
            let formats = surface_stuff
                .surface_loader
                .get_physical_device_surface_formats(physical_device, surface_stuff.surface)
                .expect("Failed to find surface formats!");
            let present_modes = surface_stuff
                .surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface_stuff.surface)
                .expect("Failed to find surface present modes!");

            SwapchainSupportDetail {
                capabilities,
                formats,
                present_modes
            }
        }
    }

    fn create_swapchain(instance: &Instance, device: &Device, physical_device: vk::PhysicalDevice, surface_stuff: &SurfaceStuff, queue_family_indices: &QueueFamilyIndices) -> SwapchainStuff {
        let swapchain_support = VulkanApplication::find_swapchain_support(physical_device, surface_stuff);
        let surface_format = VulkanApplication::choose_swapchain_format(&swapchain_support.formats);
        let present_mode = VulkanApplication::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = VulkanApplication::choose_swapchain_extent(&swapchain_support.capabilities);
        // it is recommended to request at least one more image than the minimum, sometimes have to wait on the driver to complete internal operations before we can acquire another image to render to
        let image_count = swapchain_support.capabilities.min_image_count + 1;
        // 0 is a special value that means that there is no maximum
        let image_count = if swapchain_support.capabilities.max_image_count > 0 {
            image_count.min(swapchain_support.capabilities.max_image_count)
        } else {
            image_count
        };

        // vk::SharingMode::EXCLUSIVE – An image is owned by one queue family at a time and ownership must be explicitly transferred before using it in another queue family. This option offers the best performance.
        // vk::SharingMode::CONCURRENT – Images can be used across multiple queue families without explicit ownership transfers.
        let (image_sharing_mode, queue_family_index_count, queue_family_indices) = {
            if queue_family_indices.graphics_family != queue_family_indices.present_family {
                (vk::SharingMode::CONCURRENT, 2, vec![queue_family_indices.graphics_family.unwrap(), queue_family_indices.present_family.unwrap()])
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, vec![])
            }
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: surface_stuff.surface,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            queue_family_index_count,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
        };

        let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, device);
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create Swapchain!")
        };
        let swapchain_images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get Swapchain Images!")
        };

        SwapchainStuff {
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_format: surface_format.format,
            swapchain_extent: extent
        }
    }

    fn choose_swapchain_format(available_formats: &Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
        // check if list contains the most widely used R8G8B8A8 format with nolinear color space
        for available_format in available_formats {
            if available_format.format == vk::Format::B8G8R8A8_SRGB && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
                return available_format.clone();
            }
        }

        // return the first format from the list
        return available_formats.first().unwrap().clone();
    }

    fn choose_swapchain_present_mode(available_present_modes: &Vec<vk::PresentModeKHR>) -> vk::PresentModeKHR {
        for available_present_mode in available_present_modes {
            if *available_present_mode == vk::PresentModeKHR::MAILBOX {
                return available_present_mode.clone();
            }
        }

        return vk::PresentModeKHR::FIFO;
    }

    fn choose_swapchain_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));

            vk::Extent2D {
                width: clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width, WINDOW_WIDTH),
                height: clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height, WINDOW_HEIGHT)
            }
        }
    }

    fn create_swapchain_imageviews(device: &Device, swapchain_images: &Vec<vk::Image>, swapchain_format: vk::Format) -> Vec<vk::ImageView> {
        let mut swapchain_imageviews = vec![];
        for swapchain_image in swapchain_images.into_iter() {
            let imageview_create_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::ImageViewCreateFlags::empty(),
                image: *swapchain_image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: swapchain_format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1
                },
            };
            let swapchain_imageview = unsafe {
                device
                    .create_image_view(&imageview_create_info, None)
                    .expect("Failed to create Imageview!")
            };
            swapchain_imageviews.push(swapchain_imageview);
        }
        swapchain_imageviews
    }

    fn create_render_pass(device: &Device, swapchain_format: vk::Format) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: swapchain_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };

        let subpass = vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_resolve_attachments: ptr::null(),
            p_depth_stencil_attachment: ptr::null(),
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null()
        };

        let render_pass_attachments = [color_attachment];

        let render_pass_create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: render_pass_attachments.len() as u32,
            p_attachments: render_pass_attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 0,
            p_dependencies: ptr::null()
        };

        unsafe {
            device
                .create_render_pass(&render_pass_create_info, None)
                .expect("Failed to create Render Pass!")
        }
    }

    fn create_pipeline(device: &Device, render_pass: vk::RenderPass, swapchain_extent: vk::Extent2D) -> (vk::PipelineLayout, vk::Pipeline) {
        let vert_shader_code = VulkanApplication::read_shader_code(Path::new("shaders/spv/vert.spv"));
        let frag_shader_code = VulkanApplication::read_shader_code(Path::new("shaders/spv/frag.spv"));
        let vert_shader_module = VulkanApplication::create_shader_module(device, vert_shader_code);
        let frag_shader_module = VulkanApplication::create_shader_module(device, frag_shader_code);
        let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

        let shader_stages_create_info = [
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                module: vert_shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: ptr::null(),
                stage: vk::ShaderStageFlags::VERTEX
            },

            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                module: frag_shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: ptr::null(),
                stage: vk::ShaderStageFlags::FRAGMENT
            }
        ];

        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: ptr::null(),
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: ptr::null()
        };

        let vertex_input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
            primitive_restart_enable: vk::FALSE,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST

        };

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D {x: 0, y: 0},
            extent:swapchain_extent
        }];

        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineViewportStateCreateFlags::empty(),
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr()
        };

        let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0
        };

        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: vk::FALSE,
            alpha_to_one_enable: vk::FALSE
        };

        let stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            compare_mask: 0,
            write_mask: 0,
            reference: 0
        };

        let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType:: PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
            depth_test_enable: vk::FALSE,
            depth_write_enable: vk::FALSE,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            front: stencil_state,
            back: stencil_state,
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0
        };

        let color_blend_attachment_state = [vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::RGBA,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
        }];

        let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: color_blend_attachment_state.len() as u32,
            p_attachments: color_blend_attachment_state.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0]
        };

        //ignore dynamic state for now

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: 0,
            p_set_layouts: ptr::null(),
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null()
        };

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create Pipeline Layout!")
        };

        let graphics_pipeline_create_info = [vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage_count: shader_stages_create_info.len() as u32,
            p_stages: shader_stages_create_info.as_ptr(),
            p_vertex_input_state: &vertex_input_state_create_info,
            p_input_assembly_state: &vertex_input_assembly_state_create_info,
            p_tessellation_state: ptr::null(),
            p_viewport_state: &viewport_state_create_info,
            p_rasterization_state: &rasterization_state_create_info,
            p_multisample_state: &multisample_state_create_info,
            p_depth_stencil_state: &depth_state_create_info,
            p_color_blend_state: &color_blend_state_create_info,
            p_dynamic_state: ptr::null(),
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
        }];

        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &graphics_pipeline_create_info, None)
                .expect("Failed to create Graphics Pipeline!")
        };

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        (pipeline_layout, graphics_pipeline[0])
    }

    fn create_shader_module(device: &Device, code: Vec<u8>) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: code.len(),
            p_code: code.as_ptr() as *const u32
        };

        unsafe {
            device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create Shader Module!")
        }
    }

    fn read_shader_code(shader_path: &Path) -> Vec<u8> {
        use std::fs::File;
        use std::io::Read;

        let spv_file = File::open(shader_path)
            .expect(&format!("Failed to find spv file at {:?}", shader_path));
        let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

        bytes_code
    }

    fn draw_frame(&mut self) {

    }

    fn main_loop(mut self, event_loop: EventLoop<()>, window: winit::window::Window) {
        event_loop.run(move |event, _, control_flow| {
            match event {
                | Event::WindowEvent { event, .. } => {
                    match event {
                        | WindowEvent::CloseRequested => {
                            //'*'defer
                            *control_flow = ControlFlow::Exit
                        },
                        | WindowEvent::KeyboardInput { input, .. } => {
                            match input {
                                | KeyboardInput { virtual_keycode, state, ..} => {
                                    match (virtual_keycode, state) {
                                        | (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                                            dbg!();
                                            *control_flow = ControlFlow::Exit
                                        },
                                        | _ => {},
                                    }
                                },
                            }
                        },
                        | _ => {},
                    }
                },
                | Event::MainEventsCleared => {
                    window.request_redraw();
                },
                | Event::RedrawRequested(window_id) => {
                    self.draw_frame();
                },
                _ => (),
            }
        })
    }
}

// clean up, order is opposite with init
impl Drop for VulkanApplication {
    fn drop(&mut self) {
        unsafe{
            self.logical_device.destroy_pipeline(self.graphics_pipeline, None);;

            self.logical_device.destroy_pipeline_layout(self.pipeline_layout, None);

            self.logical_device.destroy_render_pass(self.render_pass, None);

            for imageview in self.swapchain_imageviews.iter() {
                self.logical_device.destroy_image_view(*imageview, None);
            }

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);

            self.logical_device.destroy_device(None);
            
            self.surface_loader.destroy_surface(self.surface, None);

            if VALIDATION.is_enable {
                self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_messenger, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = VulkanApplication::init_window(&event_loop);
    let vulkan_app = VulkanApplication::new(&window);

    VulkanApplication::main_loop(vulkan_app, event_loop, window);
}
