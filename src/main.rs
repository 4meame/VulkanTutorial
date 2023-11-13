mod platform;
mod debug;
mod tools;
mod structures;
use ash::{vk, Device};
use ash::{Entry, Instance};
use debug::ValidationInfo;
use image::GenericImageView;
use structures::{QueueFamilyIndices, SurfaceStuff, DeviceExtension, SwapchainStuff, SwapchainSupportDetail, SyncObjects, Vertex, UniformBufferObject};
use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use cgmath::{Deg, Point3, Matrix4, Vector3, perspective};
use std::cmp::max;
use std::collections::HashSet;
use std::os::raw::{c_char, c_void};
use std::ffi::CString;
use std::path::Path;
use std::ptr;
use std::time::Instant;
use std::mem::size_of;

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

const MAX_FRAMES_IN_FLIGHTS: usize = 2;

const VERTICES_DATA: [Vertex; 8] = [
    Vertex {
        pos: [-0.5, -0.5, 0.0],
        color: [1.0, 0.0, 0.0],
        texcoord: [0.0, 0.0]
    },
    Vertex {
        pos: [0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
        texcoord: [1.0, 0.0]
    },
    Vertex {
        pos: [0.5, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
        texcoord: [1.0, 1.0]
    },
    Vertex {
        pos: [-0.5, 0.5, 0.0],
        color: [1.0, 1.0, 1.0],
        texcoord: [0.0, 1.0]
    },

    Vertex {
        pos: [-0.7, -0.3, 0.5],
        color: [1.0, 0.0, 0.0],
        texcoord: [0.0, 0.0]
    },
    Vertex {
        pos: [0.3, -0.3, 0.5],
        color: [0.0, 1.0, 0.0],
        texcoord: [1.0, 0.0]
    },
    Vertex {
        pos: [0.3, 0.7, 0.5],
        color: [0.0, 0.0, 1.0],
        texcoord: [1.0, 1.0]
    },
    Vertex {
        pos: [-0.7, 0.7, 0.5],
        color: [1.0, 1.0, 1.0],
        texcoord: [0.0, 1.0]
    }
];

const INDICES_DATA: [u32; 12] = [4, 5, 6, 6, 7, 4, 0, 1, 2, 2, 3, 0];

const TEXTURE_PATH: &'static str = "resources/texture.png";

const MODEL_PATH: &'static str = "resources/model.obj";

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
    queue_family_indices: QueueFamilyIndices,
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
    ubo_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,

    command_pool: vk::CommandPool,

    msaa_sample_count: vk::SampleCountFlags,

    color_image: vk::Image,
    color_image_memory: vk::DeviceMemory,
    color_imageview: vk::ImageView,

    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_imageview: vk::ImageView,

    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_imageview: vk::ImageView,
    texture_sampler: vk::Sampler,
    texture_mip_levels: u32,

    vertices: Vec<Vertex>,
    indices: Vec<u32>,

    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    is_framebuffer_resized: bool,

    start: Instant
}

impl VulkanApplication {
    pub fn new(window: &winit::window::Window) -> VulkanApplication {
        let entry = unsafe { Entry::load() }.unwrap();
        let instance = Self::create_instance(&entry);
        let (debug_utils_loader, debug_messenger) = debug::setup_debug_utils(VALIDATION.is_enable, &entry, &instance);
        let surface_stuff = VulkanApplication::create_surface(&entry, &instance, window);
        let physical_device = VulkanApplication::pick_physical_device(&instance, &surface_stuff);
        
        let device_properties = unsafe {
            instance.get_physical_device_properties(physical_device)
        };

        let device_memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        let msaa_sample_count = VulkanApplication::get_max_msaa_sample_count(device_properties);

        let (logical_device, queue_family_indices) = VulkanApplication::create_logical_device(&instance, physical_device, &surface_stuff, &VALIDATION);
        let graphics_queue = unsafe {
            logical_device
                .get_device_queue(queue_family_indices.graphics_family.unwrap(), 0)
        };
        let present_queue = unsafe {
            logical_device
                .get_device_queue(queue_family_indices.present_family.unwrap(), 0)
        };
        
        let swapchain_stuff = VulkanApplication::create_swapchain(&instance, physical_device, &logical_device, window, &surface_stuff, &queue_family_indices);
        let swapchain_imageviews = VulkanApplication::create_swapchain_imageviews(&logical_device, &swapchain_stuff.swapchain_images, swapchain_stuff.swapchain_format);

        let render_pass = VulkanApplication::create_render_pass(&instance, physical_device, &logical_device, swapchain_stuff.swapchain_format, msaa_sample_count);
        let ubo_layout = VulkanApplication::create_descriptor_set_layout(&logical_device);
        let (pipeline_layout, graphics_pipeline) = VulkanApplication::create_pipeline(&logical_device, render_pass, swapchain_stuff.swapchain_extent, ubo_layout, msaa_sample_count);
        let command_pool = VulkanApplication::create_command_pool(&logical_device, &queue_family_indices);

        let (color_image, color_image_memory, color_imageview) = VulkanApplication::create_color_resources(&logical_device, swapchain_stuff.swapchain_format, swapchain_stuff.swapchain_extent, device_memory_properties, msaa_sample_count);
        let (depth_image, depth_image_memory, depth_imageview) = VulkanApplication::create_depth_resources(&instance, physical_device, &logical_device,graphics_queue, command_pool, swapchain_stuff.swapchain_extent, device_memory_properties, msaa_sample_count);

        let framebuffers = VulkanApplication::create_framebuffers(&logical_device, swapchain_stuff.swapchain_extent, &swapchain_imageviews, color_imageview, depth_imageview, render_pass);

        let image_path = Path::new(TEXTURE_PATH);
        VulkanApplication::check_mipmap_support(&instance, physical_device, vk::Format::R8G8B8A8_UNORM);
        let (texture_image, texture_image_memory, texture_mip_levels) = VulkanApplication::create_texture_image(&logical_device, graphics_queue, command_pool, device_memory_properties, image_path);
        let texture_imageview = VulkanApplication::create_texture_imageview(&logical_device, texture_image, texture_mip_levels);
        let texture_sampler = VulkanApplication::create_texture_sampler(&logical_device, texture_mip_levels);
        
        let model_path = Path::new(MODEL_PATH);
        let (vertices, indices) = VulkanApplication::load_model(model_path);

        let (vertex_buffer, vertex_buffer_memory) = VulkanApplication::create_vertex_buffer(&logical_device, device_memory_properties, graphics_queue, command_pool, &vertices);
        let (index_buffer, index_buffer_memory) = VulkanApplication::create_index_buffer(&logical_device, device_memory_properties, graphics_queue, command_pool, &indices);
        let (uniform_buffers, uniform_buffers_memory) = VulkanApplication::create_uniform_buffer(&logical_device, device_memory_properties, swapchain_stuff.swapchain_images.len());
        
        let descriptor_pool = VulkanApplication::create_descriptor_pool(&logical_device, swapchain_stuff.swapchain_images.len());
        let descriptor_sets = VulkanApplication::create_descriptor_sets(&logical_device, swapchain_stuff.swapchain_images.len(), ubo_layout, descriptor_pool, &uniform_buffers, texture_sampler, texture_imageview);
        
        let command_buffers = VulkanApplication::create_command_buffers(&logical_device, command_pool, graphics_pipeline, &framebuffers, render_pass, swapchain_stuff.swapchain_extent, vertex_buffer, index_buffer, pipeline_layout, &descriptor_sets, indices.len() as u32);

        let sync_objects = VulkanApplication::create_sync_objects(&logical_device);

        let start: Instant = Instant::now();
        VulkanApplication {
            entry,
            instance,
            debug_utils_loader,
            debug_messenger,
            surface_loader: surface_stuff.surface_loader,
            surface: surface_stuff.surface,
            physical_device,
            logical_device,
            queue_family_indices,
            graphics_queue,
            present_queue,
            swapchain_loader: swapchain_stuff.swapchain_loader,
            swapchain: swapchain_stuff.swapchain,
            swapchain_images: swapchain_stuff.swapchain_images,
            swapchain_format: swapchain_stuff.swapchain_format,
            swapchain_extent: swapchain_stuff.swapchain_extent,
            swapchain_imageviews,
            render_pass,
            ubo_layout,
            pipeline_layout,
            graphics_pipeline,
            framebuffers,
            command_pool,
            msaa_sample_count,
            color_image,
            color_image_memory,
            color_imageview,
            depth_image,
            depth_image_memory,
            depth_imageview,
            texture_image,
            texture_image_memory,
            texture_imageview,
            texture_sampler,
            texture_mip_levels,
            vertices,
            indices,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            uniform_buffers,
            uniform_buffers_memory,
            descriptor_pool,
            descriptor_sets,
            command_buffers,
            image_available_semaphores: sync_objects.image_available_semaphores,
            render_finished_semaphores: sync_objects.render_finished_semaphores,
            in_flight_fences: sync_objects.in_flight_fences,
            current_frame: 0,
            is_framebuffer_resized: false,
            start
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

    fn get_max_msaa_sample_count(physical_device_properties: vk::PhysicalDeviceProperties) -> vk::SampleCountFlags {
        let count = std::cmp::min(
            physical_device_properties
                .limits
                .framebuffer_color_sample_counts,
            physical_device_properties
                .limits
                .framebuffer_depth_sample_counts,
        );

        if count.contains(vk::SampleCountFlags::TYPE_64) {
            return vk::SampleCountFlags::TYPE_64;
        }
        if count.contains(vk::SampleCountFlags::TYPE_32) {
            return vk::SampleCountFlags::TYPE_32;
        }
        if count.contains(vk::SampleCountFlags::TYPE_16) {
            return vk::SampleCountFlags::TYPE_16;
        }
        if count.contains(vk::SampleCountFlags::TYPE_8) {
            return vk::SampleCountFlags::TYPE_8;
        }
        if count.contains(vk::SampleCountFlags::TYPE_4) {
            return vk::SampleCountFlags::TYPE_4;
        }
        if count.contains(vk::SampleCountFlags::TYPE_2) {
            return vk::SampleCountFlags::TYPE_2;
        }

        vk::SampleCountFlags::TYPE_1
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
            // enable anisotropy device feature
            sampler_anisotropy: vk::TRUE,
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

    fn create_swapchain(instance: &Instance, physical_device: vk::PhysicalDevice, device: &Device, window: &winit::window::Window, surface_stuff: &SurfaceStuff, queue_family_indices: &QueueFamilyIndices) -> SwapchainStuff {
        let swapchain_support = VulkanApplication::find_swapchain_support(physical_device, surface_stuff);
        let surface_format = VulkanApplication::choose_swapchain_format(&swapchain_support.formats);
        let present_mode = VulkanApplication::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = VulkanApplication::choose_swapchain_extent(&swapchain_support.capabilities, window);
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

    fn recreate_swapchain(&mut self, window: &winit::window::Window) {
        unsafe {
            self.logical_device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
        self.destroy_swapchain();

        let surface_stuff = SurfaceStuff { 
            surface_loader: self.surface_loader.clone(), 
            surface: self.surface
        };
        let swapchain_stuff = VulkanApplication::create_swapchain(&self.instance, self.physical_device, &self.logical_device, window, &surface_stuff, &self.queue_family_indices);
        self.swapchain_loader = swapchain_stuff.swapchain_loader;
        self.swapchain = swapchain_stuff.swapchain;
        self.swapchain_images = swapchain_stuff.swapchain_images;
        self.swapchain_format = swapchain_stuff.swapchain_format;
        self.swapchain_extent = swapchain_stuff.swapchain_extent;

        self.swapchain_imageviews = VulkanApplication::create_swapchain_imageviews(&self.logical_device, &self.swapchain_images, self.swapchain_format);
        
        self.render_pass = VulkanApplication::create_render_pass(&self.instance, self.physical_device, &self.logical_device, self.swapchain_format, self.msaa_sample_count);

        let (pipeline_layout, graphics_pipeline) = VulkanApplication::create_pipeline(&self.logical_device, self.render_pass, self.swapchain_extent, self.ubo_layout, self.msaa_sample_count);
        self.pipeline_layout = pipeline_layout;
        self.graphics_pipeline = graphics_pipeline;

        let device_memory_properties = unsafe {
            self.instance.get_physical_device_memory_properties(self.physical_device)
        };

        (self.color_image, self.color_image_memory, self.color_imageview) = VulkanApplication::create_color_resources(&self.logical_device, self.swapchain_format,self.swapchain_extent, device_memory_properties, self.msaa_sample_count);
        (self.depth_image, self.depth_image_memory, self.depth_imageview) = VulkanApplication::create_depth_resources(&self.instance, self.physical_device, &self.logical_device, self.graphics_queue, self.command_pool, self.swapchain_extent, device_memory_properties, self.msaa_sample_count);

        self.framebuffers = VulkanApplication::create_framebuffers(&self.logical_device, self.swapchain_extent, &self.swapchain_imageviews, self.color_imageview, self.depth_imageview, self.render_pass);

        (self.uniform_buffers, self.uniform_buffers_memory) = VulkanApplication::create_uniform_buffer(&self.logical_device, device_memory_properties, self.swapchain_images.len());

        self.descriptor_pool = VulkanApplication::create_descriptor_pool(&self.logical_device, self.swapchain_images.len());
        self.descriptor_sets = VulkanApplication::create_descriptor_sets(&self.logical_device, self.swapchain_images.len(), self.ubo_layout, self.descriptor_pool, &self.uniform_buffers, self.texture_sampler, self.texture_imageview);

        self.command_buffers = VulkanApplication::create_command_buffers(&self.logical_device, self.command_pool, self.graphics_pipeline, &self.framebuffers, self.render_pass, self.swapchain_extent, self.vertex_buffer, self.index_buffer, self.pipeline_layout, &self.descriptor_sets, self.indices.len() as u32,);
    }

    fn destroy_swapchain(&mut self) {
        unsafe {       
            self.logical_device.destroy_descriptor_pool(self.descriptor_pool, None);

            self.logical_device.destroy_image_view(self.color_imageview, None);
            self.logical_device.free_memory(self.color_image_memory, None);
            self.logical_device.destroy_image(self.color_image, None);

            self.logical_device.destroy_image_view(self.depth_imageview, None);
            self.logical_device.free_memory(self.depth_image_memory, None);
            self.logical_device.destroy_image(self.depth_image, None);

            self.logical_device.free_command_buffers(self.command_pool, &self.command_buffers);

            for uniform_buffer in self.uniform_buffers.iter() {
                self.logical_device.destroy_buffer(*uniform_buffer, None);
            }

            for uniform_buffer_memory in self.uniform_buffers_memory.iter() {
                self.logical_device.free_memory(*uniform_buffer_memory, None);
            }

            for framebuffer in self.framebuffers.iter() {
                self.logical_device.destroy_framebuffer(*framebuffer, None);
            }

            self.logical_device.destroy_pipeline(self.graphics_pipeline, None);

            self.logical_device.destroy_pipeline_layout(self.pipeline_layout, None);

            self.logical_device.destroy_render_pass(self.render_pass, None);

            for imageview in self.swapchain_imageviews.iter() {
                self.logical_device.destroy_image_view(*imageview, None);
            }

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
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

    fn choose_swapchain_extent(capabilities: &vk::SurfaceCapabilitiesKHR, window: &winit::window::Window) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));

            let window_size = window.inner_size();
            println!(
                "\t\tInner Window Size: ({}, {})",
                window_size.width, window_size.height
            );            

            vk::Extent2D {
                width: clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width, window_size.width),
                height: clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height, window_size.height)
            }
        }
    }

    fn create_swapchain_imageviews(device: &Device, swapchain_images: &Vec<vk::Image>, swapchain_format: vk::Format) -> Vec<vk::ImageView> {
        let swapchain_imageviews: Vec<vk::ImageView> = swapchain_images
        .iter()
        .map(|&image| {
            VulkanApplication::create_image_view(
                device,
                image,
                swapchain_format,
                vk::ImageAspectFlags::COLOR,
                1,
            )
        })
        .collect();
        swapchain_imageviews
    }

    fn create_texture_imageview(device: &Device, texture_image: vk::Image, mip_levels: u32) -> vk::ImageView {
        VulkanApplication::create_image_view(
            device, 
            texture_image, 
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            mip_levels
        )
    }

    fn create_image_view(device: &Device, image: vk::Image, format: vk::Format, aspect_flags:vk::ImageAspectFlags, mip_levels: u32) -> vk::ImageView {
        let image_view_create_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageViewCreateFlags::empty(),
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1
            },
        };
        unsafe {
            device
                .create_image_view(&image_view_create_info, None)
                .expect("Failed to create Image View!")
        }
    }

    fn create_color_resources(device: &Device, swapchain_format: vk::Format, swapchain_extent: vk::Extent2D, memory_properties: vk::PhysicalDeviceMemoryProperties, sample_count: vk::SampleCountFlags) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let color_format = swapchain_format;
        let (color_image, color_image_memory) = VulkanApplication::create_image(
            device, 
            swapchain_extent.width, 
            swapchain_extent.height, 
            color_format, 
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            memory_properties, 
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            1,
            sample_count
        );

        let color_imageview = VulkanApplication::create_image_view(
            device, 
            color_image, 
            color_format, 
            vk::ImageAspectFlags::COLOR,
            1
        );

        (color_image, color_image_memory, color_imageview)
    }

    fn create_depth_resources(instance: &Instance, physical_device: vk::PhysicalDevice, device: &Device, queue: vk::Queue, command_pool: vk::CommandPool, swapchain_extent: vk::Extent2D, memory_properties: vk::PhysicalDeviceMemoryProperties, sample_count: vk::SampleCountFlags) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let depth_format = VulkanApplication::find_depth_format(instance, physical_device);
        let (depth_image, depth_image_memory) = VulkanApplication::create_image(
            device, 
            swapchain_extent.width, 
            swapchain_extent.height, 
            depth_format, 
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            memory_properties, 
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            1,
            sample_count
        );

        let depth_imageview = VulkanApplication::create_image_view(
            device, 
            depth_image, 
            depth_format, 
            vk::ImageAspectFlags::DEPTH, 
            1
        );

        VulkanApplication::transition_image_layout(
            device, queue, 
            command_pool, 
            depth_image, 
            depth_format, 
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            1
        );

        (depth_image, depth_image_memory, depth_imageview)
    }

    fn find_depth_format(instance: &Instance, physical_device: vk::PhysicalDevice) -> vk::Format {
        let candidate_formats = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        VulkanApplication::find_supported_format(
            instance, 
            physical_device, 
            &candidate_formats, 
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
        )
    }

    fn find_supported_format(instance: &Instance, physical_device: vk::PhysicalDevice, candidate_formats: &[vk::Format], tiling: vk::ImageTiling, features: vk::FormatFeatureFlags) -> vk::Format {
        for &format in candidate_formats.iter() {
            let format_properties = unsafe { 
                instance.get_physical_device_format_properties(physical_device, format) 
            };
            if tiling == vk::ImageTiling::LINEAR
                && format_properties.linear_tiling_features.contains(features) {
                return format.clone();
            } else if tiling == vk::ImageTiling::OPTIMAL
                && format_properties.optimal_tiling_features.contains(features) {
                return format.clone();
            }
        }

        panic!("Failed to find supported format!")
    }

    fn create_render_pass(instance: &Instance, physical_device: vk::PhysicalDevice, device: &Device, swapchain_format: vk::Format, sample_count: vk::SampleCountFlags) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: swapchain_format,
            samples: sample_count,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };

        let depth_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: VulkanApplication::find_depth_format(instance, physical_device),
            samples: sample_count,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        };

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        };

        let color_resolve_attachment = vk::AttachmentDescription {
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

        let color_resolve_attachment_ref = vk::AttachmentReference {
            attachment: 2,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };

        let subpass = [vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_resolve_attachments:&color_resolve_attachment_ref,
            p_depth_stencil_attachment: &depth_attachment_ref,
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null()
        }];

        let subpass_dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            src_access_mask: vk::AccessFlags::empty(),
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty()
        }];

        let render_pass_attachments = [color_attachment, depth_attachment, color_resolve_attachment];

        let render_pass_create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: render_pass_attachments.len() as u32,
            p_attachments: render_pass_attachments.as_ptr(),
            subpass_count: subpass.len() as u32,
            p_subpasses: subpass.as_ptr(),
            dependency_count: subpass_dependencies.len() as u32,
            p_dependencies: subpass_dependencies.as_ptr()
        };

        unsafe {
            device
                .create_render_pass(&render_pass_create_info, None)
                .expect("Failed to create Render Pass!")
        }
    }

    fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
        let ubo_binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: ptr::null()
        };

        let sampler_binding = vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: ptr::null()
        };

        let descriptor_set_layout_binds = [ubo_binding, sampler_binding];

        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            p_bindings: descriptor_set_layout_binds.as_ptr(),
            binding_count: descriptor_set_layout_binds.len() as u32
        };

        unsafe {
            device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .expect("Failed to create Descriptor Set Layout!")
        }
    }

    fn create_descriptor_pool(device: &Device, swapchain_images_count: usize) -> vk::DescriptorPool {
        let ubo_size = vk:: DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: swapchain_images_count as u32
        };
        
        let sampler_size = vk:: DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: swapchain_images_count as u32
        };

        let pool_size = [ubo_size, sampler_size];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
            max_sets: swapchain_images_count as u32,
            p_pool_sizes: pool_size.as_ptr(),
            pool_size_count: pool_size.len() as u32
        };

        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create Descriptor Pool!")
        }
    }

    fn create_descriptor_sets(device: &Device, swapchain_images_count: usize, descriptor_set_layout: vk::DescriptorSetLayout, descriptor_pool: vk::DescriptorPool, uniform_buffers: &Vec<vk::Buffer>, texture_sampler:vk::Sampler, texture_imageview: vk::ImageView) -> Vec<vk::DescriptorSet> {
        let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        for _ in 0..swapchain_images_count {
            layouts.push(descriptor_set_layout);
        }

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool,
            descriptor_set_count: swapchain_images_count as u32,
            p_set_layouts: layouts.as_ptr(),
        };

        let descriptor_sets = unsafe {
             device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate Descriptor Sets!")
        };

        for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
            let descriptor_buffer_info = [vk::DescriptorBufferInfo {
                buffer: uniform_buffers[i],
                offset: 0,
                range: std::mem::size_of::<UniformBufferObject>() as u64,
            }];

            let descriptor_image_infos = [vk::DescriptorImageInfo {
                sampler: texture_sampler,
                image_view: texture_imageview,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];

            let buffer_write_set = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descriptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: descriptor_buffer_info.as_ptr(),
                p_texel_buffer_view: ptr::null()
            };

            let sampler_write_set = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descriptor_set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: descriptor_image_infos.as_ptr(),
                p_buffer_info: ptr::null(),
                p_texel_buffer_view: ptr::null()
            };

            let descriptor_write_sets = [buffer_write_set, sampler_write_set];

            unsafe {
                device.update_descriptor_sets(&descriptor_write_sets, &[] as &[vk::CopyDescriptorSet]);
            }
        }

        descriptor_sets
    }

    fn create_pipeline(device: &Device, render_pass: vk::RenderPass, swapchain_extent: vk::Extent2D, ubo_set_layout: vk::DescriptorSetLayout, sample_count: vk::SampleCountFlags) -> (vk::PipelineLayout, vk::Pipeline) {
        let vert_shader_code = VulkanApplication::read_shader_code(Path::new("shaders/spv/vert.spv"));
        let frag_shader_code = VulkanApplication::read_shader_code(Path::new("shaders/spv/frag.spv"));
        let vert_shader_module = VulkanApplication::create_shader_module(device, vert_shader_code);
        let frag_shader_module = VulkanApplication::create_shader_module(device, frag_shader_code);
        let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

        let binding_description = Vertex::get_binding_description();
        let attribute_description = Vertex::get_attribute_description();

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
            vertex_binding_description_count: binding_description.len() as u32,
            p_vertex_binding_descriptions: binding_description.as_ptr(),
            vertex_attribute_description_count: attribute_description.len() as u32,
            p_vertex_attribute_descriptions: attribute_description.as_ptr()
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
            cull_mode: vk::CullModeFlags::NONE,
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
            rasterization_samples: sample_count,
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
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS,
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

        let set_layouts = [ubo_set_layout];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
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
            base_pipeline_index: -1
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

    fn create_framebuffers(device: &Device, swapchain_extent: vk::Extent2D, image_views: &Vec<vk::ImageView>, color_imageview: vk::ImageView, depth_imageview: vk::ImageView, render_pass: vk::RenderPass) -> Vec<vk::Framebuffer> {
        let mut framebuffers = vec![];
        for image_view in image_views.iter() {
            // order attachments #0,1,2
            let attachments = [color_imageview, depth_imageview, *image_view];
            let framebuffer_create_info = vk::FramebufferCreateInfo {
                s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::FramebufferCreateFlags::empty(),
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: swapchain_extent.width,
                height: swapchain_extent.height,
                layers: 1
            };
            let framebuffer = unsafe {
                device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .expect("Failed to create Framebuffer!")
            };
            framebuffers.push(framebuffer);
        }
        framebuffers
    }

    fn create_command_pool(device: &Device, queue_family_indices: &QueueFamilyIndices) -> vk::CommandPool {
        let command_pool_create_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::CommandPoolCreateFlags::empty(),
            queue_family_index: queue_family_indices.graphics_family.unwrap()
        };
        unsafe {
            device
            .create_command_pool(&command_pool_create_info, None)
            .expect("Failed to create Command Pool!")
        }
    }

    fn create_texture_sampler(device: &Device, mip_levels: u32) -> vk::Sampler {
        let sampler_create_info = vk::SamplerCreateInfo {
            s_type: vk::StructureType:: SAMPLER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SamplerCreateFlags::empty(),
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            mip_lod_bias: 0.0,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: 16.0,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            min_lod: 0.0,
            max_lod: mip_levels as f32,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
        };

        unsafe {
            device
                .create_sampler(&sampler_create_info, None)
                .expect("Failed to create Sampler!")
        }
    }

    fn create_texture_image(device: &Device, queue: vk::Queue, command_pool: vk::CommandPool, device_memory_properties: vk::PhysicalDeviceMemoryProperties, image_path: &Path) -> (vk::Image, vk::DeviceMemory, u32) {
        let mut image_object = image::open(image_path).unwrap();
        image_object = image_object.flipv();
        let (image_width, image_height) = (image_object.width(), image_object.height());
        let image_size =
            (size_of::<u8>() as u32 * image_width * image_height * 4) as vk::DeviceSize;
        
        if image_size <= 0 {
            panic!("Failed to load texture image!")
        }

        let mip_levels = ((std::cmp::max(image_width, image_height) as f32).log2().floor() as u32) + 1;

        let image_data = match &image_object {
            image::DynamicImage::ImageLuma8(_)
            | image::DynamicImage::ImageBgr8(_)
            | image::DynamicImage::ImageRgb8(_) => image_object.to_rgba().into_raw(),
            image::DynamicImage::ImageLumaA8(_)
            | image::DynamicImage::ImageBgra8(_)
            | image::DynamicImage::ImageRgba8(_) => image_object.raw_pixels(),
        };

        let staging_buffer_usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let staging_buffer_memory_required_property = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;     
        let (staging_buffer, staging_buffer_memory) = VulkanApplication::create_buffer(device, image_size, staging_buffer_usage, device_memory_properties, staging_buffer_memory_required_property);

        unsafe {
            let data_ptr = device
                .map_memory(staging_buffer_memory, 0, image_size, vk::MemoryMapFlags::empty()).expect("Failed to map memory!") as *mut u8;

            data_ptr.copy_from_nonoverlapping(image_data.as_ptr(), image_data.len());

            device.unmap_memory(staging_buffer_memory);
        }

        let format = vk::Format::R8G8B8A8_SRGB;
        let tiling = vk::ImageTiling::OPTIMAL;
        let image_texture_usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
        let image_texture_required_memory_properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let (texture_image, texture_image_memory) = VulkanApplication::create_image(device, image_width, image_height, format, tiling, image_texture_usage, device_memory_properties, image_texture_required_memory_properties, mip_levels, vk::SampleCountFlags::TYPE_1);

        VulkanApplication::transition_image_layout(device, queue, command_pool, texture_image, format, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, mip_levels);
        VulkanApplication::copy_buffer_to_image(device, queue, command_pool, staging_buffer, texture_image, image_width, image_height);
        VulkanApplication::generate_mipmaps(device, queue, command_pool, texture_image, image_width, image_height, mip_levels);

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (texture_image, texture_image_memory, mip_levels)
    }

    fn transition_image_layout(device: &Device, queue: vk::Queue, command_pool: vk::CommandPool, image: vk::Image, format: vk::Format, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, mip_levels: u32) {
        let command_buffer = VulkanApplication::begin_single_time_command(device, command_pool);

        let src_access_mask;
        let dst_access_mask;
        let src_stage_mask;
        let dst_stage_mask;

        if old_layout == vk::ImageLayout::UNDEFINED && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
            //transfer writes that don't need to wait on anything
            src_access_mask = vk::AccessFlags::empty();
            dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            src_stage_mask = vk::PipelineStageFlags::TOP_OF_PIPE;
            dst_stage_mask = vk::PipelineStageFlags::TRANSFER;
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            //shader reads should wait on transfer write
            src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            dst_access_mask = vk::AccessFlags::SHADER_READ;
            src_stage_mask = vk::PipelineStageFlags::TRANSFER;
            dst_stage_mask = vk::PipelineStageFlags::FRAGMENT_SHADER;
        }  else if old_layout ==  vk::ImageLayout::UNDEFINED && new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            src_access_mask = vk::AccessFlags::empty();
            dst_access_mask = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
            src_stage_mask = vk::PipelineStageFlags::TOP_OF_PIPE;
            dst_stage_mask = vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS;
        }
        else {
            panic!("Unsupported layout transition!")
        }

        let aspect_mask;
        if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            aspect_mask = match format {
                vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                _ => vk::ImageAspectFlags::DEPTH
            }
        } else {
            aspect_mask = vk::ImageAspectFlags::COLOR;
        }

        let barriers = [vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask,
            dst_access_mask,
            old_layout,
            new_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: mip_levels,
                layer_count: 1,
                base_array_layer: 0
            }
        }];

        unsafe {
            device.cmd_pipeline_barrier(
                    command_buffer, 
                    src_stage_mask, 
                    dst_stage_mask, 
                    vk::DependencyFlags::empty(), 
                    &[] as &[vk::MemoryBarrier], 
                    &[]as &[vk::BufferMemoryBarrier], 
                    &barriers
                );
        }

        VulkanApplication::end_single_time_command(device, queue, command_pool, command_buffer);
    }

    fn copy_buffer_to_image(device: &Device, queue: vk::Queue, command_pool: vk::CommandPool, buffer: vk::Buffer, image: vk::Image, width: u32, height: u32) {
        let command_buffer = VulkanApplication::begin_single_time_command(device, command_pool);

        let buffer_image_regions = [vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            buffer_offset: 0,
            buffer_image_height: 0,
            buffer_row_length: 0,
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        }];

        unsafe {
            device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &buffer_image_regions,
            );
        }

        VulkanApplication::end_single_time_command(device, queue, command_pool, command_buffer);
    }

    fn generate_mipmaps(device: &Device, queue: vk::Queue, command_pool: vk::CommandPool, image: vk::Image, texture_width: u32, texture_height: u32, mip_levels: u32) {
        let command_buffer = VulkanApplication::begin_single_time_command(device, command_pool);

        let mut image_barrier= vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::empty(),
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::UNDEFINED,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }, 
        };

        let mut mip_width = texture_width as i32;
        let mut mip_height = texture_height as i32;

        for i in 1..mip_levels {
            image_barrier.subresource_range.base_mip_level = i - 1;
            image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            image_barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            image_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier.clone()],
                );
            }

            let blits = [vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: max(mip_width / 2, 1),
                        y: max(mip_height / 2, 1),
                        z: 1,
                    },
                ],
            }];

            unsafe {
                device.cmd_blit_image(
                    command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &blits,
                    vk::Filter::LINEAR,
                );
            }

            image_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier.clone()],
                );
            }

            mip_width = max(mip_width / 2, 1);
            mip_height = max(mip_height / 2, 1);
        }

        image_barrier.subresource_range.base_mip_level = mip_levels - 1;
        image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier.clone()],
            );
        }

        VulkanApplication::end_single_time_command(device, queue, command_pool, command_buffer);
    }

    fn check_mipmap_support(instance: &ash::Instance, physical_device: vk::PhysicalDevice, image_format: vk::Format) {
        let format_properties = unsafe {
            instance.get_physical_device_format_properties(physical_device, image_format)
        };

        let is_sample_image_filter_linear_support = format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);

        if is_sample_image_filter_linear_support == false {
            panic!("Texture Image format does not support linear blitting!")
        }
    }

    fn create_image(device: &Device, width: u32, height: u32, format: vk::Format, tiling: vk::ImageTiling, usage: vk::ImageUsageFlags, memory_properties: vk::PhysicalDeviceMemoryProperties, required_memory_properties: vk::MemoryPropertyFlags, mip_levels: u32, sample_count: vk::SampleCountFlags) -> (vk::Image, vk::DeviceMemory) {
        let image_create_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageCreateFlags::empty(),
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D{
                width,
                height,
                depth: 1
            },
            mip_levels,
            array_layers: 1,
            samples: sample_count,
            tiling,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: vk::ImageLayout::UNDEFINED
        };

        let texture_image = unsafe {
            device
                .create_image(&image_create_info, None)
                .expect("Failed to create Texture Image!")
        };

        let memory_requirements = unsafe {
            device.get_image_memory_requirements(texture_image)
        };

        let memory_type = VulkanApplication::find_memory_type(memory_properties, required_memory_properties, memory_requirements.memory_type_bits);

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: memory_requirements.size,
            memory_type_index: memory_type
        };

        let texture_image_memeory = unsafe {
            device
                .allocate_memory(&allocate_info, None)
                .expect("Failed to allocate Texture Image memory!")
        };

        unsafe {
            device
                .bind_image_memory(texture_image, texture_image_memeory, 0)
                .expect("Failed to bind Texture Image memory!");
        }

        (texture_image, texture_image_memeory)
    }

    fn create_vertex_buffer<T>(device: &Device, device_memory_properties: vk::PhysicalDeviceMemoryProperties, queue: vk::Queue, command_pool: vk::CommandPool, data: &[T]) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;
        let staging_buffer_usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let staging_buffer_memory_required_property = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;     
        let (staging_buffer, staging_buffer_memory) = VulkanApplication::create_buffer(device, buffer_size, staging_buffer_usage, device_memory_properties, staging_buffer_memory_required_property);

        unsafe {
            let data_ptr = device
                .map_memory(staging_buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty()).expect("Failed to map memory!") as *mut T;

            data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());

            device.unmap_memory(staging_buffer_memory);
        }

        let vertex_buffer_usage = vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER;
        let vertex_buffer_memory_required_property = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let (vertex_buffer, vertex_buffer_memory) = VulkanApplication::create_buffer(device, buffer_size, vertex_buffer_usage, device_memory_properties, vertex_buffer_memory_required_property);

        VulkanApplication::copy_buffer(device, queue, command_pool, staging_buffer, vertex_buffer, buffer_size);

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (vertex_buffer, vertex_buffer_memory)
    }

    fn create_index_buffer(device: &Device, device_memory_properties: vk::PhysicalDeviceMemoryProperties, queue: vk::Queue, command_pool: vk::CommandPool, data: &[u32]) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;
        let staging_buffer_usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let staging_buffer_memory_required_property = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;     
        let (staging_buffer, staging_buffer_memory) = VulkanApplication::create_buffer(device, buffer_size, staging_buffer_usage, device_memory_properties, staging_buffer_memory_required_property);

        unsafe {
            let data_ptr = device
                .map_memory(staging_buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty()).expect("Failed to map memory!") as *mut u32;

            data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());

            device.unmap_memory(staging_buffer_memory);
        }

        let index_buffer_usage = vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER;
        let index_buffer_memory_required_property = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let (index_buffer, index_buffer_memory) = VulkanApplication::create_buffer(device, buffer_size, index_buffer_usage, device_memory_properties, index_buffer_memory_required_property);

        VulkanApplication::copy_buffer(device, queue, command_pool, staging_buffer, index_buffer, buffer_size);

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (index_buffer, index_buffer_memory)
    }

    fn create_uniform_buffer(device: &Device, device_memory_properties: vk::PhysicalDeviceMemoryProperties, swapchain_image_count: usize) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
        let size = std::mem::size_of::<UniformBufferObject>();  
        let usage = vk::BufferUsageFlags::UNIFORM_BUFFER;
        let required_memory_properties = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mut uniform_buffers = vec![];
        let mut uniform_buffers_memory = vec![];

        for _ in 0..swapchain_image_count {
            let (uniform_buffer, uniform_buffer_memory) = VulkanApplication::create_buffer(device, size as u64, usage, device_memory_properties, required_memory_properties);
            uniform_buffers.push(uniform_buffer);
            uniform_buffers_memory.push(uniform_buffer_memory);
        }

        (uniform_buffers, uniform_buffers_memory)
    }

    fn update_uniform_buffer(&self, current_image: usize) {
        let time = self.start.elapsed().as_secs_f32();
        let model = Matrix4::from_angle_z(Deg(90.0 * time * 0.1));
        let view = Matrix4::look_at_rh(Point3 { x: 2.0, y: 2.0, z: 2.0 }, Point3 { x: 0.0, y: 0.0, z: 0.0 }, Vector3{ x: 0.0, y: 0.0, z: 1.0 });
        let correction = Matrix4::new(
            1.0,  0.0,       0.0, 0.0,
            // The perspective projection matrix generated by cgmath::perspective uses the OpenGL depth range of -1.0 to 1.0
            // We want to use the Vulkan range of 0.0 to 1.0
            // We're also flipping the Y-axis with this line's `-1.0`
            0.0, -1.0,       0.0, 0.0,
            0.0,  0.0, 1.0 / 2.0, 0.0,
            0.0,  0.0, 1.0 / 2.0, 1.0,
        );
        let proj = correction * perspective(Deg(45.0), self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32, 0.1, 20.0);
        // proj[1][1] *= -1.0;
        let ubos = [
            UniformBufferObject {
                model,
                view,
                proj
            }
        ];
        let size = (std::mem::size_of::<UniformBufferObject>() * ubos.len()) as u64;
        unsafe {
            let data_ptr =
                self.logical_device
                    .map_memory(
                        self.uniform_buffers_memory[current_image],
                        0,
                        size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to map memory!") as *mut UniformBufferObject;

            data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

            self.logical_device
                .unmap_memory(self.uniform_buffers_memory[current_image]);
        }
    }

    fn create_buffer(device: &Device, size: vk::DeviceSize, usage: vk::BufferUsageFlags, memory_properties: vk::PhysicalDeviceMemoryProperties, required_memory_properties: vk::MemoryPropertyFlags) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_create_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::BufferCreateFlags::empty(),
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null()
        };

        let buffer = unsafe {
            device
                .create_buffer(&buffer_create_info, None)
                .expect("Failed to create Buffer!")
        };

        let memory_requirements = unsafe {
            device.get_buffer_memory_requirements(buffer)
        };

        let memory_type = VulkanApplication::find_memory_type(memory_properties, required_memory_properties, memory_requirements.memory_type_bits);

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: memory_requirements.size,
            memory_type_index: memory_type
        };

        let buffer_memory = unsafe {
            device
                .allocate_memory(&allocate_info, None)
                .expect("Failed to allocate Buffer memory!")
        };

        unsafe {
            device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind Buffer memory!");
        }

        (buffer, buffer_memory)
    }

    fn copy_buffer(device: &Device, queue: vk::Queue, command_pool: vk::CommandPool, src_buffer: vk::Buffer, dst_buffer: vk::Buffer, size: vk::DeviceSize) -> () {     
        let command_buffer = VulkanApplication::begin_single_time_command(device, command_pool);
        
        unsafe {
            let copy_regions = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }];

            device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &copy_regions);
        }

        VulkanApplication::end_single_time_command(device, queue, command_pool, command_buffer);
    }

    fn begin_single_time_command(device: &Device, command_pool: vk::CommandPool) -> vk::CommandBuffer {
        let allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool,
            command_buffer_count: 1,
            level: vk::CommandBufferLevel::PRIMARY
        };

        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate Command Buffer!")
        }[0];

        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            p_inheritance_info: ptr::null()
        };

        unsafe {
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin Command Buffer!");
        }

        command_buffer
    }

    fn end_single_time_command(device: &Device, queue: vk::Queue, command_pool: vk::CommandPool, command_buffer: vk::CommandBuffer) {
        unsafe {
            device
            .end_command_buffer(command_buffer)
            .expect("Failed to end Command Buffer!");
        }

        let buffers_to_submit = [command_buffer];

        let submit_info = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        }];

        unsafe {
            device
                .queue_submit(queue, &submit_info, vk::Fence::null())
                .expect("Failed to execute queue submit!");
            device
                .queue_wait_idle(queue)
                .expect("Failed to wait queue idle!");
            device.free_command_buffers(command_pool, &buffers_to_submit);
        }

    }

    fn find_memory_type(memory_properties: vk::PhysicalDeviceMemoryProperties, required_properties: vk::MemoryPropertyFlags, type_filter: u32) -> u32 {
        for (i, memory_type) in memory_properties.memory_types.iter().enumerate() {
            if (type_filter & (1 << i)) > 0 && memory_type.property_flags.contains(required_properties) {
                return i as u32
            }
        }

        panic!("Failed to find suitable memory type!")
    }

    fn create_command_buffers(device: &Device, command_pool: vk::CommandPool, graphics_pipeline: vk::Pipeline, framebuffers: &Vec<vk::Framebuffer>, render_pass: vk::RenderPass, swapchain_extent: vk::Extent2D, vertex_buffer: vk::Buffer, index_buffer: vk::Buffer, pipeline_layout: vk::PipelineLayout, descriptor_sets: &Vec<vk::DescriptorSet>, index_count: u32) -> Vec<vk::CommandBuffer> {
        let command_buffer_allocation_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_buffer_count: framebuffers.len() as u32,
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY
        };
        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocation_info)
                .expect("Failed to allocate Command Buffer!")
        };
        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
                p_inheritance_info: ptr::null()
            };

            unsafe {
                device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin recording Command Buffer at beginning!");
            }

            let color_clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],

                }
            };

            let depth_clear_value = vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue{
                    depth: 1.0,
                    stencil: 0
                }
            };

            let clear_values = [color_clear_value, depth_clear_value];

            let render_pass_begin_info = vk::RenderPassBeginInfo {
                s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
                p_next: ptr::null(),
                render_pass,
                framebuffer: framebuffers[i],
                render_area: vk::Rect2D {
                    offset: vk::Offset2D {x: 0, y: 0},
                    extent: swapchain_extent
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr()
            };

            //  All of the functions that record commands can be recognized by their cmd_ prefix. They all return ()
            unsafe {
                device.cmd_begin_render_pass(
                    command_buffer, 
                    &render_pass_begin_info, 
                    vk::SubpassContents::INLINE
                );
                device.cmd_bind_pipeline(
                    command_buffer, 
                    vk::PipelineBindPoint::GRAPHICS, 
                    graphics_pipeline
                );

                let vertex_buffers = [vertex_buffer];
                let offsets = [0_u64];
                let descriptor_sets_to_bind = [descriptor_sets[i]];
                device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(
                    command_buffer, 
                    index_buffer, 
                    0, 
                    vk::IndexType::UINT32
                );
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );
                device.cmd_draw_indexed(command_buffer, index_count, 1, 0, 0, 0);
                device.cmd_end_render_pass(command_buffer);
                device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to record Command Buffer at ending!");
            }
        }

        command_buffers
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

    fn create_sync_objects(device: &Device) -> SyncObjects {
        let mut sync_objects = SyncObjects {
            image_available_semaphores: vec![],
            render_finished_semaphores: vec![],
            in_flight_fences:vec![]
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty()
        };

        let fence_create_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            // wait_for_fences will wait forever if we haven't used the fence before
            // change the fence creation to initialize it in the signaled state as if we had rendered an initial frame that finished
            flags: vk::FenceCreateFlags::SIGNALED
        };

        for _ in 0..MAX_FRAMES_IN_FLIGHTS {
            unsafe {
                let image_available_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");
                let render_finished_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");
                let in_flight_fence = device
                    .create_fence(&fence_create_info, None)
                    .expect("Failed to create Fence Object!");

                sync_objects
                    .image_available_semaphores
                    .push(image_available_semaphore);
                sync_objects
                    .render_finished_semaphores
                    .push(render_finished_semaphore);
                sync_objects.in_flight_fences.push(in_flight_fence);
            }
        }

        sync_objects
    }

    fn load_model(model_path: &Path) -> (Vec<Vertex>, Vec<u32>) {
        let model_obj = tobj::load_obj(model_path, &tobj::LoadOptions::default()).expect("Failed to load model object!");
        let mut vertices = vec![];
        let mut indices = vec![];
        let (models, _) = model_obj;
        for model in models.iter() {
            let mesh = &model.mesh;
            let total_vertices_count = mesh.positions.len() / 3;
            let total_indices_count = mesh.indices.len();
            for i in 0..total_vertices_count {
                let vertex = Vertex {
                    pos: [
                        mesh.positions[i * 3],
                        mesh.positions[i * 3 + 1],
                        mesh.positions[i * 3 + 2]
                    ],
                    color: [1.0, 1.0, 1.0],
                    texcoord: [mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1]]
                };
                vertices.push(vertex);
            }
            for j in 0..total_indices_count {
                let index = mesh.indices[j];
                indices.push(index);    
            }
        }
        (vertices, indices)
    }

    fn draw_frame(&mut self, window: &winit::window::Window) {
        let wait_fences = [self.in_flight_fences[self.current_frame]];

        unsafe {
            self.logical_device
            .wait_for_fences(&wait_fences, true, std::u64::MAX)
            .expect("Failed to wait for Fences!");
        }
        
        // acquire an image from the swapchain
        let image_index = unsafe {
            let result = self.swapchain_loader.acquire_next_image(self.swapchain, std::u64::MAX, self.image_available_semaphores[self.current_frame], vk::Fence::null());

            match result {
                // _,suboptimal,bool
                Ok((image_index, _)) => image_index,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        self.recreate_swapchain(window);
                        return;
                    }
                    _ => panic!("Failed to acquire Swap Chain Image!"),
                },
            }
        };

        self.update_uniform_buffer(image_index as usize);

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffers[image_index as usize]];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

        let submit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr()
        }];

        unsafe {
            self.logical_device
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fences!");

            self.logical_device
                .queue_submit(self.graphics_queue, &submit_infos, self.in_flight_fences[self.current_frame])
                .expect("Failed to execute queue submit!");
        }

        let swapchains = [self.swapchain];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: signal_semaphores.len() as u32,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: swapchains.len() as u32,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: ptr::null_mut()
        };

        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
        };

        let changed = match result {
            Ok(_) => false,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => panic!("Failed to execute queue present!"),
            }
        };

        if self.is_framebuffer_resized || changed {
            self.is_framebuffer_resized = false;
            self.recreate_swapchain(window);
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHTS;
    }

    fn main_loop(mut self, event_loop: EventLoop<()>, window: winit::window::Window) {
        let mut destroying = false;
        let mut minimized = false;
        event_loop.run(move |event, _, control_flow| {
            match event {
                | Event::WindowEvent { event, .. } => {
                    match event {
                        | WindowEvent::CloseRequested => {
                            //'*'defer
                            destroying = true;
                            *control_flow = ControlFlow::Exit
                        },
                        | WindowEvent::Resized(size) => {
                            if size.width == 0 || size.height == 0 {
                                minimized = true;
                            } else {
                                minimized = false;
                                self.is_framebuffer_resized = true;
                            }
                        }
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
                | Event::MainEventsCleared if !destroying && !minimized => {
                    window.request_redraw();
                },
                | Event::RedrawRequested(window_id) => {
                    self.draw_frame(&window);
                },
                | Event::LoopDestroyed => {
                    unsafe {
                        self.logical_device.device_wait_idle()
                            .expect("Failed to wait device idle!");
                    }
                }
                _ => (),
            }
        })
    }
}

// clean up, order is opposite with init
impl Drop for VulkanApplication {
    fn drop(&mut self) {
        unsafe{
            for i in 0..MAX_FRAMES_IN_FLIGHTS {
                self.logical_device.destroy_semaphore(self.image_available_semaphores[i], None);
                self.logical_device.destroy_semaphore(self.render_finished_semaphores[i], None);
                self.logical_device.destroy_fence(self.in_flight_fences[i], None);
            }

            self.destroy_swapchain();
            self.logical_device.destroy_descriptor_set_layout(self.ubo_layout, None);

            self.logical_device.destroy_sampler(self.texture_sampler, None);

            self.logical_device.destroy_image_view(self.texture_imageview, None);
            self.logical_device.destroy_image(self.texture_image, None);
            self.logical_device.free_memory(self.texture_image_memory, None);

            self.logical_device.destroy_buffer(self.index_buffer, None);
            self.logical_device.free_memory(self.index_buffer_memory, None);
            self.logical_device.destroy_buffer(self.vertex_buffer, None);
            self.logical_device.free_memory(self.vertex_buffer_memory, None);

            self.logical_device.destroy_command_pool(self.command_pool, None);

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
