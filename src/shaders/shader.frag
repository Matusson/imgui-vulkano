#version 450
#extension GL_EXT_nonuniform_qualifier : require

// TODO: This is not ideal. Ideally we'd include vulkano.glsl here and use its system,
// but I don't want to copy the whole file into the repo, and I have no clue how to make this
// work out-of-the box otherwise. It does work, though.
layout(set = 0, binding = 0) uniform sampler bindless_samplers[];
layout(set = 0, binding = 1) uniform texture2D bindless_textures[];

layout(push_constant) uniform FragPC {
    mat4 matrix;
    uint sampled_image_id;
    uint sampler_id;
};

layout(location = 0) in vec2 f_uv;
layout(location = 1) in vec4 f_color;

layout(location = 0) out vec4 Target0;

layout(constant_id = 0) const float OUT_GAMMA = 0.0;

void main() {
    vec4 tex_color = texture(
        sampler2D(bindless_textures[sampled_image_id], bindless_samplers[sampler_id]),
        f_uv.st
    );
    Target0 = pow(f_color * tex_color, vec4(vec3(OUT_GAMMA), 1.0));
}
