use std::ffi::{c_char, CStr};

// convert [c_char;size] to string
pub fn vk_str_to_string(raw_string_array: &[c_char]) -> String{
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };
    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}