use std::{ffi::{c_void, CStr}};
use ash::vk;

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT, // a enum of the severity level of debug message
    message_type: vk::DebugUtilsMessageTypeFlagsEXT, // a enum of the type of debug message
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void //universal pointer
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]", // the most detailed
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]", // not important
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]", // important
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]", // useful info but less detailed than "Verbose"
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]", // general info
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]", // performance warning
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]", // validation error
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);
    vk::FALSE
}

pub struct ValidationInfo {
    pub is_enable: bool,
    pub required_validation_layers: [&'static str; 1],
}

pub fn check_validation_layer_support(
    entry: &ash::Entry,
    required_validation_layers: &Vec<&str>,
) -> bool {
    // if support validation layer, then return true

    let layer_properties = entry
        .enumerate_instance_layer_properties()
        .expect("Failed to enumerate Instance Layers Properties.");

    if layer_properties.len() <= 0 {
        eprintln!("No available layers.");
        return false;
    }

    for required_layer_name in required_validation_layers.iter() {
        let mut is_layer_found = false;

        for layer_property in layer_properties.iter() {
            let test_layer_name = super::tools::vk_str_to_string(&layer_property.layer_name);
            if (*required_layer_name) == test_layer_name {
                is_layer_found = true;
                break;
            }
        }

        if is_layer_found == false {
            return false;
        }
    }

    true
}

pub fn setup_debug_utils(
    is_enable_debug: bool,
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> (ash::extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT) {
    let debug_utils_loader = ash::extensions::ext::DebugUtils::new(entry, instance);

    if is_enable_debug == false {
        (debug_utils_loader, ash::vk::DebugUtilsMessengerEXT::null())
    } else {
        let messenger_ci = populate_debug_messenger_create_info();

        let utils_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&messenger_ci, None)
                .expect("Debug Utils Callback")
        };

        (debug_utils_loader, utils_messenger)
    }
}

pub fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(vulkan_debug_utils_callback))
        .build()
}