#version 450

layout(push_constant) uniform PushConstants {
    mat4 matrix;
    uint sampled_image_id;
    uint sampler_id;
};

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 col;

layout(location = 0) out vec2 f_uv;
layout(location = 1) out vec4 f_color;

void main() {
    f_uv = uv;
    f_color = col;
    gl_Position = matrix * vec4(pos.xy, 0, 1);
}
