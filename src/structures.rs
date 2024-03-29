use std::mem::size_of;
use ash::vk;
use cgmath::Matrix4;
use memoffset::offset_of;

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
    pub pos: [f32; 3],
    pub color: [f32; 3],
    pub texcoord: [f32; 2],
}

impl Vertex {
    pub fn new(pos: [f32; 3], color: [f32; 3], texcoord: [f32; 2]) -> Self {
        Vertex { 
            pos, 
            color,
            texcoord
        }
    }

    pub fn get_binding_description() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX
        }]
    }

    pub fn get_attribute_description() -> [vk::VertexInputAttributeDescription; 3] {
        let pos = vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(Self, pos) as u32
            };

        let color = vk::VertexInputAttributeDescription {
            binding: 0,
            location: 1,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(Self, color) as u32
            };

        let texcoord = vk::VertexInputAttributeDescription {
            binding: 0,
            location: 2,
            format: vk::Format::R32G32_SFLOAT,
            offset: offset_of!(Self, texcoord) as u32
            };

        [pos, color, texcoord]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    pub model: Matrix4<f32>,
    pub view: Matrix4<f32>, 
    pub proj: Matrix4<f32>
}