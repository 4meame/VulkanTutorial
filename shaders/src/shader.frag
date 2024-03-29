#version 450

layout(location = 0) in vec3 vertColor;
layout(location = 1) in vec2 vertTexcoord;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 fragColor;

void main() {
    vec4 textureColor = texture(texSampler, vertTexcoord);
    fragColor = textureColor;
}