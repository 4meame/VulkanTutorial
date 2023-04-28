mod platform;
mod debug;
mod tools;
use ash::vk;
use ash::{Entry, Instance};
use debug::ValidationInfo;
use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use std::ffi::{CString, c_void};
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
    debug_messenger: vk::DebugUtilsMessengerEXT
}

impl VulkanApplication {
    pub fn new() -> VulkanApplication {
        let entry = unsafe { Entry::load() }.unwrap();
        let instance = Self::create_instance(&entry);
        let (debug_utils_loader, debug_messenger) = debug::setup_debug_utils(VALIDATION.is_enable, &entry, &instance);
        VulkanApplication {
            entry,
            instance,
            debug_utils_loader,
            debug_messenger
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

//clean up, order is opposite with init
impl Drop for VulkanApplication {
    fn drop(&mut self) {
        unsafe{
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
