//! Shader modules for imgui rendering.

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/shader.vert",
        root_path_env: "CARGO_MANIFEST_DIR"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/shader.frag",
        root_path_env: "CARGO_MANIFEST_DIR"
    }
}
