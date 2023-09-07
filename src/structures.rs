use std::mem::size_of;
use ash::vk;


pub struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub present_family: Option<u32>
}

impl QueueFamilyIndices {
    pub fn new() -> Self {
        QueueFamilyIndices { 
            graphics_family: None, 
            present_family: None 
        }
    }

    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

pub struct SurfaceStuff {
    pub surface_loader: ash::extensions::khr::Surface,
    pub surface: vk::SurfaceKHR
}

pub struct DeviceExtension {
    pub names: [&'static str; 1]
}

pub struct SwapchainStuff {
    pub swapchain_loader: ash::extensions::khr::Swapchain,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D
}

pub struct SwapchainSupportDetail {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>
}

pub struct SyncObjects {
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub color: [f32; 3]
}

impl Vertex {
    pub fn new(pos: [f32; 2], color: [f32; 3]) -> Self {
        Vertex { 
            pos, 
            color
        }
    }

    pub fn get_binding_description() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX
        }]
    }

    pub fn get_attribute_description() -> [vk::VertexInputAttributeDescription; 2] {
        let pos = vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 0
            };
            
        let color = vk::VertexInputAttributeDescription {
            binding: 0,
            location: 1,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: size_of::<[f32; 2]>() as u32
            };

        [pos, color]
    }
}