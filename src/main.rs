mod platform;
mod debug;
mod tools;
mod structures;
use ash::vk;
use ash::{Entry, Instance};
use debug::ValidationInfo;
use structures::QueueFamilyIndices;
use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use std::ffi::{CString, c_void, c_char};
use std::ptr;

const WINDOW_TITLE: &'static str = "Vulkan Renderer By Rust";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

const VALIDATION: ValidationInfo = ValidationInfo {
    is_enable: cfg!(debug_assertions),
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};

struct VulkanApplication {
    entry: Entry,
    instance: Instance,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    logical_device: ash::Device,
    graphics_queue: vk::Queue
}

impl VulkanApplication {
    pub fn new() -> VulkanApplication {
        let entry = unsafe { Entry::load() }.unwrap();
        let instance = Self::create_instance(&entry);
        let (debug_utils_loader, debug_messenger) = debug::setup_debug_utils(VALIDATION.is_enable, &entry, &instance);
        let physical_device = VulkanApplication::pick_physical_device(&instance);
        let (logical_device, graphics_queue) = VulkanApplication::create_logical_device(&instance, physical_device, &VALIDATION);
        VulkanApplication {
            entry,
            instance,
            debug_utils_loader,
            debug_messenger,
            physical_device,
            logical_device,
            graphics_queue
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
            .expect("Failed to create instance!")
        };
        
        instance
    }

    fn pick_physical_device(instance: &Instance) -> vk::PhysicalDevice {
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
            if VulkanApplication::is_physical_device_suitable(&instance, physical_device) {
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

    fn is_physical_device_suitable(instance: &Instance, physical_device: vk::PhysicalDevice) -> bool {
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

        let indices: QueueFamilyIndices = VulkanApplication::find_queue_family(&instance, physical_device);
        return indices.is_complete();
    }

    fn find_queue_family(instance: &Instance, physical_device: vk::PhysicalDevice) -> QueueFamilyIndices {
        let queue_families = unsafe { 
            instance.get_physical_device_queue_family_properties(physical_device) 
        };

        let mut queue_family_indices = QueueFamilyIndices {
            graphics_family: None,
        };

        let mut index = 0;
        for queue_family in queue_families.iter() {
            if queue_family.queue_count > 0 && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                queue_family_indices.graphics_family = Some(index);
            }

            if queue_family_indices.is_complete() {
                break;
            }

            index += 1;
        }

        queue_family_indices
    }

    fn create_logical_device(instance: &Instance, physical_device: vk::PhysicalDevice, validation: &ValidationInfo) -> (ash::Device, vk::Queue) {
        let indices = VulkanApplication::find_queue_family(instance, physical_device);
        // right now we are only interested in graphics queue
        let queue_priorities = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: indices.graphics_family.unwrap(),
            p_queue_priorities: queue_priorities.as_ptr(),
            queue_count: queue_priorities.len() as u32
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
            pp_enabled_extension_names: if validation.is_enable {
                enabled_layer_names.as_ptr()
            } else {
                ptr::null()
            },
            enabled_extension_count: 0,
            pp_enabled_layer_names: ptr::null(),
            p_enabled_features: & physical_device_features
        };

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create Logical Device!")
        };
        let graphics_queue = unsafe {
            device.get_device_queue(indices.graphics_family.unwrap(), 0)
        };

        (device, graphics_queue)
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
            self.logical_device.destroy_device(None);

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
    let vulkan_app = VulkanApplication::new();

    VulkanApplication::main_loop(vulkan_app,event_loop,window);
}
