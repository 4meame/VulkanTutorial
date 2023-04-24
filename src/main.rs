mod platform;
use ash::vk;
use ash::{Entry, Instance};
use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use std::ffi::CString;

const WINDOW_TITLE: &'static str = "Vulkan Renderer By Rust";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

struct VulkanApplication {
    entry: Entry,
    instance: Instance
}

impl VulkanApplication {
    pub fn new() -> VulkanApplication {
        let entry = unsafe { Entry::load() }.unwrap();
        let instance = Self::create_instance(&entry);
        VulkanApplication {
            entry,
            instance
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
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let extensions_names = platform::required_extension_names();

        let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&extensions_names);

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

//clean up
impl Drop for VulkanApplication {
    fn drop(&mut self) {
        unsafe{
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
